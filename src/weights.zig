//! Sharded-safetensors loader for the Qwen3.8-27B checkpoint.
//!
//! `model.safetensors.index.json` gives the ordered shard list (plus
//! `mtp.safetensors`, which the index already references — no special case).
//! Each shard's header is parsed directly (`u64 LE` length + JSON) to validate
//! dtype/shape BEFORE `mlx_load_safetensors` pulls the shard in:
//! - BF16/F32 dense tensors load directly; U32/U8 tensors load as quant
//!   parts (`*.weight` packed + `*.scales` [+ `*.biases`]) and stay packed —
//!   `model.zig` runs them through `mlx_quantized_matmul`.
//! - any `conv1d.weight` whose last dim != 1 is `error.UnsupportedLayout`.
//! Text-only port: `vision_tower.*` / `model.visual.*` keys are dropped at
//! load (`--no-vision` is the only mode); the reference `sanitize` remaps
//! (`model.language_model.*` → strip the `model.` prefix, bare names gain a
//! `language_model.` prefix) are applied — all no-ops on this checkpoint,
//! which already ships sanitized `language_model.*` names.
//!
//! Ownership: the map owns every `mlx_array` handle; `get` returns a borrowed
//! handle (do NOT free it).

const std = @import("std");
const mlx = @import("mlx.zig");

pub const WeightsError = error{
    BadIndexJson,
    BadHeader,
    UnsupportedDtype,
    UnsupportedLayout,
    MissingTensor,
    DuplicateTensor,
    NonstandardQuant,
    InconsistentQuant,
    AffineUnsupported,
};

pub const DType = enum { bf16, f32, u32, u8 };

/// Per-checkpoint quantization, inferred from shapes (no config carries it):
/// U8 scales prove a microscaling mode — the packed/scales ratio `r`
/// determines it exactly (`r = group * bits / 32`): r=8 → mxfp8 (32, 8),
/// r=2 → nvfp4 (16, 4). BF16 scales mean affine quantization, which this
/// engine does not support (use a bf16/mxfp8/nvfp4 checkpoint instead).
pub const QuantMode = enum { none, mxfp8, nvfp4 };
pub const QuantSpec = struct {
    mode: QuantMode = .none,
    group_size: u32 = 0,
    bits: u32 = 0,

    pub fn modeStr(self: QuantSpec) [*:0]const u8 {
        return switch (self.mode) {
            .mxfp8 => "mxfp8",
            .nvfp4 => "nvfp4",
            .none => "mxfp8", // never used; qmm only runs when mode != .none
        };
    }
};
pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    shape: []i64,
};

pub fn freeTensorInfos(allocator: std.mem.Allocator, infos: []TensorInfo) void {
    for (infos) |t| {
        allocator.free(t.name);
        allocator.free(t.shape);
    }
    allocator.free(infos);
}

/// Parse + validate one safetensors header JSON object (the `n` bytes after
/// the `u64 LE` length). Returns owned infos; errors on dtype/layout.
pub fn parseHeaderJson(allocator: std.mem.Allocator, json_bytes: []const u8) ![]TensorInfo {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch
        return WeightsError.BadHeader;
    defer parsed.deinit();
    if (parsed.value != .object) return WeightsError.BadHeader;

    var out: std.ArrayList(TensorInfo) = .empty;
    errdefer {
        for (out.items) |t| {
            allocator.free(t.name);
            allocator.free(t.shape);
        }
        out.deinit(allocator);
    }
    var it = parsed.value.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
        if (e.value_ptr.* != .object) return WeightsError.BadHeader;
        const o = e.value_ptr.object;
        const dt_v = o.get("dtype") orelse return WeightsError.BadHeader;
        const sh_v = o.get("shape") orelse return WeightsError.BadHeader;
        if (dt_v != .string or sh_v != .array) return WeightsError.BadHeader;
        const dtype: DType = if (std.mem.eql(u8, dt_v.string, "BF16"))
            .bf16
        else if (std.mem.eql(u8, dt_v.string, "F32"))
            .f32
        else if (std.mem.eql(u8, dt_v.string, "U32"))
            .u32
        else if (std.mem.eql(u8, dt_v.string, "U8"))
            .u8
        else
            return WeightsError.UnsupportedDtype;
        var shape: std.ArrayList(i64) = .empty;
        errdefer shape.deinit(allocator);
        for (sh_v.array.items) |d| {
            if (d != .integer) return WeightsError.BadHeader;
            try shape.append(allocator, d.integer);
        }
        if (std.mem.indexOf(u8, e.key_ptr.*, "conv1d.weight") != null) {
            if (shape.items.len == 0 or shape.items[shape.items.len - 1] != 1)
                return WeightsError.UnsupportedLayout;
        }
        try out.append(allocator, .{
            .name = try allocator.dupe(u8, e.key_ptr.*),
            .dtype = dtype,
            .shape = try shape.toOwnedSlice(allocator),
        });
    }
    return out.toOwnedSlice(allocator);
}

fn findInfo(infos: []const TensorInfo, name: []const u8) ?TensorInfo {
    for (infos) |t| {
        if (std.mem.eql(u8, t.name, name)) return t;
    }
    return null;
}

/// Infer the checkpoint's quantization from tensor shapes. Pure (hermetic).
/// Returns `.none` when no `*.scales` tensors exist (dense bf16).
pub fn inferQuantSpec(infos: []const TensorInfo) !QuantSpec {
    var first: ?TensorInfo = null;
    for (infos) |t| {
        if (std.mem.endsWith(u8, t.name, ".scales")) {
            first = t;
            break;
        }
    }
    const sc = first orelse {
        for (infos) |t| {
            if (t.dtype == .u32 or t.dtype == .u8) return WeightsError.NonstandardQuant;
        }
        return .{};
    };
    if (sc.shape.len != 2) return WeightsError.NonstandardQuant;
    if (sc.dtype != .u8) return WeightsError.AffineUnsupported;
    // Sibling packed weight: "<prefix>.weight" U32 [out, packed].
    var pbuf: [1024]u8 = undefined;
    if (sc.name.len < ".scales".len + 1 or sc.name.len - ".scales".len > pbuf.len - ".weight".len)
        return WeightsError.NonstandardQuant;
    const prefix = sc.name[0 .. sc.name.len - ".scales".len];
    const wname = std.fmt.bufPrint(&pbuf, "{s}.weight", .{prefix}) catch return WeightsError.NonstandardQuant;
    const w = findInfo(infos, wname) orelse return WeightsError.NonstandardQuant;
    if (w.dtype != .u32 or w.shape.len != 2) return WeightsError.NonstandardQuant;
    const packed_d1: i64 = w.shape[1];
    const sgroups: i64 = sc.shape[1];
    if (packed_d1 <= 0 or sgroups <= 0 or @rem(packed_d1, sgroups) != 0) return WeightsError.NonstandardQuant;
    const r: u32 = @intCast(@divExact(packed_d1, sgroups)); // group * bits / 32
    const spec: QuantSpec = if (r == 8)
        .{ .mode = .mxfp8, .group_size = 32, .bits = 8 }
    else if (r == 2)
        .{ .mode = .nvfp4, .group_size = 16, .bits = 4 }
    else
        return WeightsError.NonstandardQuant;
    // Every scales tensor must agree (U8 dtype, same ratio); every packed
    // U32 weight must have scales (no orphan packed tensors).
    for (infos) |t| {
        if (std.mem.endsWith(u8, t.name, ".scales")) {
            if (t.dtype != .u8 or t.shape.len != 2) return WeightsError.InconsistentQuant;
            const p2: i64 = blk: {
                const wn = std.fmt.bufPrint(&pbuf, "{s}.weight", .{t.name[0 .. t.name.len - ".scales".len]}) catch return WeightsError.NonstandardQuant;
                const w2 = findInfo(infos, wn) orelse return WeightsError.NonstandardQuant;
                if (w2.dtype != .u32 or w2.shape.len != 2) return WeightsError.NonstandardQuant;
                break :blk w2.shape[1];
            };
            if (t.shape[1] <= 0 or @rem(p2, t.shape[1]) != 0 or @divExact(p2, t.shape[1]) != r) return WeightsError.InconsistentQuant;
        }
        if (t.dtype == .u32 and std.mem.endsWith(u8, t.name, ".weight")) {
            const wn = std.fmt.bufPrint(&pbuf, "{s}.scales", .{t.name[0 .. t.name.len - ".weight".len]}) catch return WeightsError.NonstandardQuant;
            if (findInfo(infos, wn) == null) return WeightsError.InconsistentQuant;
        }
    }
    return spec;
}
/// Reference-`sanitize` naming. Returns null for dropped (vision) keys,
/// otherwise an owned (possibly remapped) name.
pub fn sanitizeName(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    if (std.mem.startsWith(u8, name, "vision_tower") or
        std.mem.startsWith(u8, name, "model.visual"))
    {
        return null;
    }
    // Exact mirror of Model.sanitize: "model.language_model" -> "language_model.model".
    if (std.mem.startsWith(u8, name, "model.language_model")) {
        return try std.mem.concat(allocator, u8, &.{ "language_model.model", name["model.language_model".len..] });
    }
    if (std.mem.startsWith(u8, name, "language_model.")) {
        return try allocator.dupe(u8, name);
    }
    return try std.mem.concat(allocator, u8, &.{ "language_model.", name });
}
pub const WeightMap = struct {
    allocator: std.mem.Allocator,
    map: std.StringHashMap(mlx.mlx_array),
    quant: QuantSpec = .{},
    pub fn deinit(self: *WeightMap) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            self.allocator.free(e.key_ptr.*);
        }
        self.map.deinit();
    }

    /// Borrowed handle, or null when absent. Do NOT free the result.
    pub fn get(self: *const WeightMap, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    pub fn count(self: *const WeightMap) usize {
        return self.map.count();
    }

    fn readWholeFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
        var file = if (std.fs.path.isAbsolute(path))
            try std.Io.Dir.openFileAbsolute(io, path, .{})
        else
            try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const st = try file.stat(io);
        const size: usize = @intCast(st.size);
        const buf = try allocator.alloc(u8, size);
        errdefer allocator.free(buf);
        _ = try file.readPositionalAll(io, buf, 0);
        return buf;
    }

    /// Read + validate one shard header (first 8 + n bytes only — never the
    /// multi-GB data section; a single pread over it returns INVAL).
    fn checkShardFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]TensorInfo {
        var file = if (std.fs.path.isAbsolute(path))
            try std.Io.Dir.openFileAbsolute(io, path, .{})
        else
            try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var lenbuf: [8]u8 = undefined;
        _ = try file.readPositionalAll(io, &lenbuf, 0);
        const n = std.mem.readInt(u64, &lenbuf, .little);
        if (n == 0 or n > 256 * 1024 * 1024) return WeightsError.BadHeader;
        const hdr = try allocator.alloc(u8, n);
        defer allocator.free(hdr);
        _ = try file.readPositionalAll(io, hdr, 8);
        return parseHeaderJson(allocator, hdr);
    }

    /// Load every shard in index order. Slow path (tens of GB); caller logs it.
    pub fn load(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !WeightMap {
        const index_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors.index.json" });
        defer allocator.free(index_path);
        const index_bytes = try readWholeFile(allocator, io, index_path);
        defer allocator.free(index_bytes);
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, index_bytes, .{}) catch
            return WeightsError.BadIndexJson;
        defer parsed.deinit();
        if (parsed.value != .object) return WeightsError.BadIndexJson;
        const wm_v = parsed.value.object.get("weight_map") orelse return WeightsError.BadIndexJson;
        if (wm_v != .object) return WeightsError.BadIndexJson;

        // Ordered unique shard filenames (ObjectMap preserves file order).
        var shards: std.ArrayList([]const u8) = .empty;
        defer {
            for (shards.items) |s| allocator.free(s);
            shards.deinit(allocator);
        }
        var wmit = wm_v.object.iterator();
        while (wmit.next()) |e| {
            if (e.value_ptr.* != .string) return WeightsError.BadIndexJson;
            const shard = e.value_ptr.string;
            var seen = false;
            for (shards.items) |s| {
                if (std.mem.eql(u8, s, shard)) {
                    seen = true;
                    break;
                }
            }
            if (!seen) try shards.append(allocator, try allocator.dupe(u8, shard));
        }
        var self = WeightMap{
            .allocator = allocator,
            .map = std.StringHashMap(mlx.mlx_array).init(allocator),
        };
        errdefer self.deinit();
        // All headers, for checkpoint-wide quant inference after the loop.
        var all: std.ArrayList(TensorInfo) = .empty;
        defer {
            for (all.items) |t| {
                allocator.free(t.name);
                allocator.free(t.shape);
            }
            all.deinit(allocator);
        }

        // File-backed Load ops have no GPU eval in this libmlx: load on the
        // CPU stream (mmap + first-touch materialize); downstream GPU ops
        // move data on first use. GPU-stream load evals nothing, ever.
        const stream = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(stream);
        for (shards.items) |shard| {
            const shard_path = try std.fs.path.join(allocator, &.{ model_dir, shard });
            defer allocator.free(shard_path);
            const infos = try checkShardFile(allocator, io, shard_path);
            defer allocator.free(infos); // entries move into `all`
            try all.appendSlice(allocator, infos);

            const c_path_raw = try allocator.alloc(u8, shard_path.len + 1);
            @memcpy(c_path_raw[0..shard_path.len], shard_path);
            c_path_raw[shard_path.len] = 0;
            const c_path: [*:0]const u8 = @ptrCast(c_path_raw.ptr);
            defer allocator.free(c_path_raw);
            var arr_map = mlx.mlx_map_string_to_array_new();
            defer _ = mlx.mlx_map_string_to_array_free(arr_map);
            var str_map = mlx.mlx_map_string_to_string_new();
            defer _ = mlx.mlx_map_string_to_string_free(str_map);
            try mlx.check(mlx.mlx_load_safetensors(&arr_map, &str_map, c_path, stream));

            for (infos) |t| {
                var key_buf: [4096]u8 = undefined;
                if (t.name.len >= key_buf.len) return WeightsError.BadHeader;
                @memcpy(key_buf[0..t.name.len], t.name);
                key_buf[t.name.len] = 0;
                var val = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(val);
                const rc = mlx.mlx_map_string_to_array_get(&val, arr_map, @ptrCast(&key_buf));
                if (rc != 0) return WeightsError.MissingTensor;
                const kept = try sanitizeName(allocator, t.name);
                if (kept) |k| {
                    errdefer allocator.free(k);
                    if (self.map.contains(k)) {
                        _ = mlx.mlx_array_free(val);
                        allocator.free(k);
                        return WeightsError.DuplicateTensor;
                    }
                    try self.map.put(k, val);
                } else {
                    _ = mlx.mlx_array_free(val);
                }
            }
        }
        self.quant = try inferQuantSpec(all.items);
        return self;
    }
};

// ── tests (hermetic: no mlx calls — those live in mlx_test.zig) ────────────

test "sanitize drops vision, remaps, keeps" {
    const alloc = std.testing.allocator;
    // vision dropped
    try std.testing.expect(try sanitizeName(alloc, "vision_tower.blocks.0.attn.proj.weight") == null);
    try std.testing.expect(try sanitizeName(alloc, "model.visual.proj") == null);
    // unsanitized HF remap: model.language_model.X -> language_model.model.X
    {
        const k = try sanitizeName(alloc, "model.language_model.layers.0.q");
        defer alloc.free(k.?);
        try std.testing.expectEqualStrings("language_model.model.layers.0.q", k.?);
    }
    {
        const k = try sanitizeName(alloc, "embed_tokens.weight");
        defer alloc.free(k.?);
        try std.testing.expectEqualStrings("language_model.embed_tokens.weight", k.?);
    }
    // sanitized checkpoint names pass through
    {
        const k = try sanitizeName(alloc, "language_model.model.layers.3.self_attn.q_proj.weight");
        defer alloc.free(k.?);
        try std.testing.expectEqualStrings("language_model.model.layers.3.self_attn.q_proj.weight", k.?);
    }
}

test "header parse validates dtype and conv layout" {
    const alloc = std.testing.allocator;
    const good =
        \\{"__metadata__": {}, "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        \\ "w": {"dtype": "BF16", "shape": [4, 2], "data_offsets": [8, 24]},
        \\ "c": {"dtype": "BF16", "shape": [10240, 4, 1], "data_offsets": [24, 40]}}
    ;
    const infos = try parseHeaderJson(alloc, good);
    defer freeTensorInfos(alloc, infos);
    try std.testing.expectEqual(@as(usize, 3), infos.len);

    const bad_dtype =
        \\{"q": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}
    ;
    try std.testing.expectError(WeightsError.UnsupportedDtype, parseHeaderJson(alloc, bad_dtype));

    const bad_conv =
        \\{"m.conv1d.weight": {"dtype": "BF16", "shape": [8, 4, 2], "data_offsets": [0, 128]}}
    ;
    try std.testing.expectError(WeightsError.UnsupportedLayout, parseHeaderJson(alloc, bad_conv));

    try std.testing.expectError(WeightsError.BadHeader, parseHeaderJson(alloc, good[0 .. good.len - 20]));
    try std.testing.expectError(WeightsError.BadHeader, parseHeaderJson(alloc, "not json"));
}

test "missing index path is an open error, not a parse error" {
    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    // Pure-IO check: opening a missing file fails before any mlx call.
    try std.testing.expectError(error.FileNotFound, WeightMap.readWholeFile(alloc, io, "/tmp/mlx-runner-no-such-model-dir/x.json"));
}

/// Test-only sharded-fixture writer: deterministic narrow-range BF16
/// tensors + matching index. Hermetic-safe (std + IO only, no mlx calls).
pub const FixtureSpec = struct { name: []const u8, shape: [3]usize, ndim: usize };

pub fn fixtureNumel(sp: FixtureSpec) usize {
    var n: usize = 1;
    for (sp.shape[0..sp.ndim]) |d| n *= d;
    return n;
}

pub fn writeFixture(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, specs: []const FixtureSpec, seed0: u64) !void {
    std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    try std.Io.Dir.cwd().createDirPath(io, dir);
    errdefer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.appendSlice(allocator, "{\"__metadata__\":{},");
    var data: std.ArrayList(u16) = .empty;
    defer data.deinit(allocator);
    var off: usize = 0;
    var seed: u64 = seed0;
    for (specs, 0..) |sp, i| {
        const n = fixtureNumel(sp);
        var shape_json: std.ArrayList(u8) = .empty;
        defer shape_json.deinit(allocator);
        for (sp.shape[0..sp.ndim], 0..) |d, j| {
            const part = try std.fmt.allocPrint(allocator, "{s}{d}", .{ if (j == 0) "" else ",", d });
            defer allocator.free(part);
            try shape_json.appendSlice(allocator, part);
        }
        const entry = try std.fmt.allocPrint(allocator, "{s}\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{s}],\"data_offsets\":[{d},{d}]}}", .{
            if (i == 0) "" else ",",
            sp.name,
            shape_json.items,
            off,
            off + n * 2,
        });
        defer allocator.free(entry);
        try header.appendSlice(allocator, entry);
        var j: usize = 0;
        while (j < n) : (j += 1) {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            const u: f32 = @floatFromInt((seed >> 33) & 0x7FFFFFFF);
            const f = (u / 1073741823.5 - 1.0) * 0.2;
            try data.append(allocator, @as(u16, @intCast(@as(u32, @bitCast(f)) >> 16)));
        }
        off += n * 2;
    }
    try header.append(allocator, '}');
    var fb: std.ArrayList(u8) = .empty;
    defer fb.deinit(allocator);
    var lenbuf: [8]u8 = undefined;
    std.mem.writeInt(u64, &lenbuf, header.items.len, .little);
    try fb.appendSlice(allocator, &lenbuf);
    try fb.appendSlice(allocator, header.items);
    try fb.appendSlice(allocator, std.mem.sliceAsBytes(data.items));
    {
        const p = try std.fs.path.join(allocator, &.{ dir, "t.safetensors" });
        defer allocator.free(p);
        var f = try std.Io.Dir.createFileAbsolute(io, p, .{});
        defer f.close(io);
        try f.writePositionalAll(io, fb.items, 0);
    }
    var wm_json: std.ArrayList(u8) = .empty;
    defer wm_json.deinit(allocator);
    try wm_json.appendSlice(allocator, "{\"metadata\":{},\"weight_map\":{");
    for (specs, 0..) |sp, i| {
        const e = try std.fmt.allocPrint(allocator, "{s}\"{s}\":\"t.safetensors\"", .{ if (i == 0) "" else ",", sp.name });
        defer allocator.free(e);
        try wm_json.appendSlice(allocator, e);
    }
    try wm_json.appendSlice(allocator, "}}");
    {
        const p = try std.fs.path.join(allocator, &.{ dir, "model.safetensors.index.json" });
        defer allocator.free(p);
        var f = try std.Io.Dir.createFileAbsolute(io, p, .{});
        defer f.close(io);
        try f.writePositionalAll(io, wm_json.items, 0);
    }
}
test "inferQuantSpec reads mode/group/bits from shapes" {
    const alloc = std.testing.allocator;
    // nvfp4 file shapes: packed [O, I/8], scales U8 [O, I/16] (r=2).
    const nv =
        \\{"l.q_proj.weight": {"dtype": "U32", "shape": [8, 8], "data_offsets": [0, 256]},
        \\ "l.q_proj.scales": {"dtype": "U8", "shape": [8, 4], "data_offsets": [256, 288]},
        \\ "l.norm.weight": {"dtype": "BF16", "shape": [16], "data_offsets": [288, 320]}}
    ;
    const ni = try parseHeaderJson(alloc, nv);
    defer freeTensorInfos(alloc, ni);
    const nspec = try inferQuantSpec(ni);
    try std.testing.expectEqual(QuantMode.nvfp4, nspec.mode);
    try std.testing.expectEqual(@as(u32, 16), nspec.group_size);
    try std.testing.expectEqual(@as(u32, 4), nspec.bits);
    // mxfp8 file shapes: packed [O, I/4], scales U8 [O, I/32] (r=8).
    const mx =
        \\{"l.q_proj.weight": {"dtype": "U32", "shape": [8, 32], "data_offsets": [0, 1024]},
        \\ "l.q_proj.scales": {"dtype": "U8", "shape": [8, 4], "data_offsets": [1024, 1056]}}
    ;
    const mi = try parseHeaderJson(alloc, mx);
    defer freeTensorInfos(alloc, mi);
    const mspec = try inferQuantSpec(mi);
    try std.testing.expectEqual(QuantMode.mxfp8, mspec.mode);
    try std.testing.expectEqual(@as(u32, 32), mspec.group_size);
    try std.testing.expectEqual(@as(u32, 8), mspec.bits);
    // Dense checkpoint: no scales -> .none.
    const dn =
        \\{"l.norm.weight": {"dtype": "BF16", "shape": [16], "data_offsets": [0, 32]}}
    ;
    const di = try parseHeaderJson(alloc, dn);
    defer freeTensorInfos(alloc, di);
    try std.testing.expectEqual(QuantMode.none, (try inferQuantSpec(di)).mode);
    // BF16 scales (affine family) -> explicit rejection, not silent misread.
    const aff =
        \\{"l.q_proj.weight": {"dtype": "U32", "shape": [8, 32], "data_offsets": [0, 1024]},
        \\ "l.q_proj.scales": {"dtype": "BF16", "shape": [8, 2], "data_offsets": [1024, 1056]},
        \\ "l.q_proj.biases": {"dtype": "BF16", "shape": [8, 2], "data_offsets": [1056, 1088]}}
    ;
    const ai = try parseHeaderJson(alloc, aff);
    defer freeTensorInfos(alloc, ai);
    try std.testing.expectError(WeightsError.AffineUnsupported, inferQuantSpec(ai));
    // Orphan packed weight (no scales) -> inconsistent.
    const orph =
        \\{"l.q_proj.weight": {"dtype": "U32", "shape": [8, 32], "data_offsets": [0, 1024]}}
    ;
    const oi = try parseHeaderJson(alloc, orph);
    defer freeTensorInfos(alloc, oi);
    try std.testing.expectError(WeightsError.NonstandardQuant, inferQuantSpec(oi));
}
