const std = @import("std");

pub const SamplingParams = struct {
    temp: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0,
    min_p: f32 = 0.0,
    seed: ?u64 = null,

    pub fn clamped(self: SamplingParams) SamplingParams {
        var out = self;
        out.temp = @max(0.0, out.temp);
        out.top_p = @min(1.0, @max(0.0, out.top_p));
        out.min_p = @min(1.0, @max(0.0, out.min_p));
        return out;
    }

    /// Priority: request > base. Only override when request field is non-null.
    pub fn mergedWithRequest(self: SamplingParams, req: RequestSampling) SamplingParams {
        var out = self;
        if (req.temperature) |v| out.temp = v;
        if (req.top_p) |v| out.top_p = v;
        if (req.top_k) |v| out.top_k = v;
        if (req.min_p) |v| out.min_p = v;
        if (req.seed) |v| out.seed = v;
        return out.clamped();
    }
};

pub const RequestSampling = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    min_p: ?f32 = null,
    seed: ?u64 = null,
};

/// Read generation_config.json if present, else hardcoded defaults.
/// Then overlay cli_override where cli fields differ from defaults.
pub fn fromGenerationConfig(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, cli_override: ?SamplingParams) !SamplingParams {
    var base = SamplingParams{};
    if (readGenerationConfig(allocator, io, model_dir) catch null) |file| {
        var f = file;
        defer f.deinit(allocator);
        if (f.temperature) |v| base.temp = v;
        if (f.top_p) |v| base.top_p = v;
        if (f.top_k) |v| base.top_k = v;
    }
    if (cli_override) |cli| {
        const def = SamplingParams{};
        if (cli.temp != def.temp) base.temp = cli.temp;
        if (cli.top_p != def.top_p) base.top_p = cli.top_p;
        if (cli.top_k != def.top_k) base.top_k = cli.top_k;
        if (cli.min_p != def.min_p) base.min_p = cli.min_p;
        if (cli.seed != null) base.seed = cli.seed;
    }
    return base.clamped();
}

pub const GenFile = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    eos: ?[]u32 = null,

    pub fn deinit(self: *GenFile, allocator: std.mem.Allocator) void {
        if (self.eos) |e| allocator.free(e);
        self.* = .{};
    }
};

fn numF32(v: std.json.Value) ?f32 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => null,
    };
}

fn readGenerationConfig(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !GenFile {
    const path = try std.fs.path.join(allocator, &.{ model_dir, "generation_config.json" });
    defer allocator.free(path);
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
    var out = GenFile{};
    errdefer out.deinit(allocator);
    if (parsed.value != .object) return out;
    const o = parsed.value.object;
    if (o.get("temperature")) |v| out.temperature = numF32(v);
    if (o.get("top_p")) |v| out.top_p = numF32(v);
    if (o.get("top_k")) |v| out.top_k = switch (v) {
        .integer => |i| @intCast(i),
        .float => |f| @intFromFloat(f),
        else => null,
    };
    if (o.get("eos_token_id")) |e| {
        switch (e) {
            .integer => |i| {
                const d = try allocator.alloc(u32, 1);
                d[0] = @intCast(i);
                out.eos = d;
            },
            .array => |a| {
                if (a.items.len > 0) {
                    const d = try allocator.alloc(u32, a.items.len);
                    errdefer allocator.free(d);
                    for (a.items, 0..) |v, i| {
                        d[i] = switch (v) {
                            .integer => |x| @intCast(x),
                            .float => |x| @intFromFloat(x),
                            else => return out,
                        };
                    }
                    out.eos = d;
                }
            },
            else => {},
        }
    }
    return out;
}

/// EOS ids from generation_config.json.
/// Falls back to the Qwen pair [248046, 248044] when absent/unreadable.
pub fn readEosIds(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) ![]u32 {
    if (readGenerationConfig(allocator, io, model_dir) catch null) |file| {
        var f = file;
        defer f.deinit(allocator);
        if (f.eos) |e| {
            const d = try allocator.alloc(u32, e.len);
            @memcpy(d, e);
            return d;
        }
    }
    const d = try allocator.alloc(u32, 2);
    d[0] = 248046;
    d[1] = 248044;
    return d;
}
// --- tests ---
test "defaults" {
    const p = SamplingParams{};
    try std.testing.expectEqual(@as(f32, 1.0), p.temp);
    try std.testing.expectEqual(@as(u32, 0), p.top_k);
}

test "mergedWithRequest overrides" {
    const base = SamplingParams{ .temp = 0.7, .top_p = 0.9, .top_k = 20 };
    const eff = base.mergedWithRequest(.{ .temperature = 0.2, .top_p = 0.95 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), eff.temp, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), eff.top_p, 0.001);
    try std.testing.expectEqual(@as(u32, 20), eff.top_k);
}

test "clamped" {
    const p = (SamplingParams{ .temp = -1, .top_p = 1.5, .min_p = 2.0 }).clamped();
    try std.testing.expectEqual(@as(f32, 0.0), p.temp);
    try std.testing.expectEqual(@as(f32, 1.0), p.top_p);
    try std.testing.expectEqual(@as(f32, 1.0), p.min_p);
}
