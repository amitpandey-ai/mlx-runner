const std = @import("std");
const sampling = @import("sampling.zig");

pub const ModelSampling = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    min_p: ?f32 = null,
    seed: ?u64 = null,
    max_tokens: ?u32 = null,

    pub fn toSamplingParams(self: ModelSampling) sampling.SamplingParams {
        var p = sampling.SamplingParams{};
        if (self.temperature) |v| p.temp = v;
        if (self.top_p) |v| p.top_p = v;
        if (self.top_k) |v| p.top_k = v;
        if (self.min_p) |v| p.min_p = v;
        if (self.seed) |v| p.seed = v;
        return p.clamped();
    }
};

pub const ModelConfig = struct {
    ctx_size: ?u32 = null,
    mtp: ?bool = null,
    mtp_gamma: ?u32 = null,
};

pub const ModelEntry = struct {
    path: []const u8,
    alias: ?[]const u8 = null,
    sampling: ?ModelSampling = null,
    config: ?ModelConfig = null,
};

pub const ServerCfg = struct {
    host: ?[]const u8 = null,
    port: ?u16 = null,
    max_resident_models: ?usize = null,
};

pub const CacheCfg = struct {
    prefix_cache_entries: ?u32 = null,
    prefix_cache_mem: ?[]const u8 = null,
    prefix_cache_disk: ?[]const u8 = null,
    apc_disk: ?[]const u8 = null,
    apc_disk_dir: ?[]const u8 = null,
};

pub const FileConfig = struct {
    models: []ModelEntry = &.{},
    server: ?ServerCfg = null,
    cache: ?CacheCfg = null,

    pub fn deinit(self: *FileConfig, allocator: std.mem.Allocator) void {
        for (self.models) |*m| {
            allocator.free(m.path);
            if (m.alias) |a| allocator.free(a);
        }
        allocator.free(self.models);
        if (self.server) |s| {
            if (s.host) |h| allocator.free(h);
        }
        if (self.cache) |c| {
            if (c.prefix_cache_mem) |v| allocator.free(v);
            if (c.prefix_cache_disk) |v| allocator.free(v);
            if (c.apc_disk) |v| allocator.free(v);
            if (c.apc_disk_dir) |v| allocator.free(v);
        }
        self.* = .{};
    }
};

fn expandTilde(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (path.len > 1 and path[0] == '~' and path[1] == '/') {
        const home_c = std.c.getenv("HOME");
        const home = if (home_c) |p| std.mem.span(p) else return allocator.dupe(u8, path);
        return std.fs.path.join(allocator, &.{ home, path[2..] });
    }
    return allocator.dupe(u8, path);
}

fn parseModelSampling(v: std.json.Value) ModelSampling {
    var s = ModelSampling{};
    if (v != .object) return s;
    const o = v.object;
    if (o.get("temperature")) |x| s.temperature = switch (x) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
    if (o.get("top_p")) |x| s.top_p = switch (x) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
    if (o.get("top_k")) |x| s.top_k = switch (x) {
        .integer => |i| @intCast(i),
        .float => |f| @intFromFloat(f),
        else => null,
    };
    if (o.get("min_p")) |x| s.min_p = switch (x) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
    if (o.get("seed")) |x| s.seed = switch (x) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        .float => |f| if (f >= 0 and @trunc(f) == f) @intFromFloat(f) else null,
        else => null,
    };
    if (o.get("max_tokens")) |x| s.max_tokens = switch (x) {
        .integer => |i| if (i >= 0 and i <= 1_000_000) @intCast(i) else null,
        .float => |f| if (f >= 0 and @trunc(f) == f) @intFromFloat(f) else null,
        else => null,
    };
    return s;
}

fn parseModelConfig(v: std.json.Value) ModelConfig {
    var c = ModelConfig{};
    if (v != .object) return c;
    const o = v.object;
    if (o.get("ctx_size")) |x| c.ctx_size = switch (x) {
        .integer => |i| if (i >= 0 and i <= 10_000_000) @intCast(i) else null,
        .float => |f| if (f >= 0 and @trunc(f) == f) @intFromFloat(f) else null,
        else => null,
    };
    if (o.get("mtp")) |x| c.mtp = switch (x) {
        .bool => |b| b,
        else => null,
    };
    if (o.get("mtp_gamma")) |x| c.mtp_gamma = switch (x) {
        .integer => |i| if (i >= 1 and i <= 3) @intCast(i) else null,
        .float => |f| if (f >= 1 and f <= 3 and @trunc(f) == f) @intFromFloat(f) else null,
        else => null,
    };
    return c;
}

pub fn loadFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !FileConfig {
    var file = if (std.fs.path.isAbsolute(path))
        try std.Io.Dir.openFileAbsolute(io, path, .{})
    else
        try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const st = try file.stat(io);
    const bytes = try allocator.alloc(u8, @intCast(st.size));
    defer allocator.free(bytes);
    _ = try file.readPositionalAll(io, bytes, 0);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const root = parsed.value.object;
    var fc = FileConfig{};
    errdefer fc.deinit(allocator);

    // models
    if (root.get("models")) |mv| {
        if (mv == .array) {
            var list: std.ArrayList(ModelEntry) = .empty;
            errdefer {
                for (list.items) |*e| {
                    allocator.free(e.path);
                    if (e.alias) |a| allocator.free(a);
                }
                list.deinit(allocator);
            }
            for (mv.array.items) |item| {
                if (item != .object) continue;
                const o = item.object;
                const raw_path = if (o.get("path")) |p| (if (p == .string) p.string else continue) else continue;
                const path_exp = try expandTilde(allocator, raw_path);
                errdefer allocator.free(path_exp);
                var entry = ModelEntry{ .path = path_exp };
                if (o.get("alias")) |av| if (av == .string and av.string.len > 0) {
                    entry.alias = try allocator.dupe(u8, av.string);
                };
                if (o.get("sampling")) |sv| entry.sampling = parseModelSampling(sv);
                if (o.get("config")) |cv| entry.config = parseModelConfig(cv);
                try list.append(allocator, entry);
            }
            fc.models = try list.toOwnedSlice(allocator);
        }
    }

    // server
    if (root.get("server")) |sv| {
        if (sv == .object) {
            var s = ServerCfg{};
            errdefer if (s.host) |h| allocator.free(h);
            const o = sv.object;
            if (o.get("host")) |v| {
                if (v == .string) s.host = try allocator.dupe(u8, v.string);
            }
            if (o.get("port")) |v| s.port = switch (v) {
                .integer => |i| if (i > 0 and i < 65536) @intCast(i) else null,
                .float => |f| if (f > 0 and f < 65536 and @trunc(f) == f) @intFromFloat(f) else null,
                else => null,
            };
            if (o.get("max_resident_models")) |v| s.max_resident_models = switch (v) {
                .integer => |i| if (i > 0) @intCast(i) else null,
                .float => |f| if (f > 0 and @trunc(f) == f) @intFromFloat(f) else null,
                else => null,
            };
            fc.server = s;
        }
    }

    // cache
    if (root.get("cache")) |cv| {
        if (cv == .object) {
            var c = CacheCfg{};
            errdefer {
                if (c.prefix_cache_mem) |v| allocator.free(v);
                if (c.prefix_cache_disk) |v| allocator.free(v);
                if (c.apc_disk) |v| allocator.free(v);
                if (c.apc_disk_dir) |v| allocator.free(v);
            }
            const o = cv.object;
            if (o.get("prefix_cache_entries")) |v| c.prefix_cache_entries = switch (v) {
                .integer => |i| if (i >= 0) @intCast(i) else null,
                else => null,
            };
            if (o.get("prefix_cache_mem")) |v| {
                if (v == .string) c.prefix_cache_mem = try allocator.dupe(u8, v.string);
            }
            if (o.get("prefix_cache_disk")) |v| {
                if (v == .string) c.prefix_cache_disk = try allocator.dupe(u8, v.string);
            }
            if (o.get("apc_disk")) |v| {
                if (v == .string) c.apc_disk = try allocator.dupe(u8, v.string);
            }
            if (o.get("apc_disk_dir")) |v| {
                if (v == .string) c.apc_disk_dir = try allocator.dupe(u8, v.string);
            }
            fc.cache = c;
        }
    }

    return fc;
}

test "parse sampling" {
    const alloc = std.testing.allocator;
    var v = try std.json.parseFromSlice(std.json.Value, alloc,
        \\{"temperature":0.7,"top_p":0.9,"top_k":20,"seed":42}
    , .{});
    defer v.deinit();
    const s = parseModelSampling(v.value);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), s.temperature.?, 0.001);
    try std.testing.expectEqual(@as(u32, 20), s.top_k.?);
    try std.testing.expectEqual(@as(u64, 42), s.seed.?);
}

test "expand tilde" {
    const alloc = std.testing.allocator;
    const p = try expandTilde(alloc, "~/opt/models");
    defer alloc.free(p);
    try std.testing.expect(!std.mem.startsWith(u8, p, "~"));
}
