//! Fixed KV checkpoint — denfer ds4_kvstore binary format, not JSON.
//! Layout per denfer/ds4_kvstore.c:393 fill_header 48B `KVC\x01` + quant/reason/ext/model,
//! tokens/hits/ctx (LE32), payload_abi=2 at byte 20, created/last/payload (LE64),
//! then 4B LE text_bytes, then text, then payload (KV safetensors bytes).
//! File named SHA1(text).kv, dir budget 4096 MiB, eviction score (hits+1)*tokens/size with 6h half-life.
//! Cuts prefill on reuse: load finds longest byte-prefix where SHA1(prompt[0:n])==sha.

const std = @import("std");
const mlx = @import("mlx.zig");

pub const FIXED_HEADER: usize = 48;
pub const MAGIC0: u8 = 'K';
pub const MAGIC1: u8 = 'V';
pub const MAGIC2: u8 = 'C';
pub const VERSION: u8 = 1;
pub const PAYLOAD_ABI: u8 = 2;
pub const DEFAULT_MB: u64 = 4096;
pub const HIT_HALF_LIFE_S: u64 = 6 * 60 * 60;
pub const MIN_TOKENS: u32 = 512;
pub const BOUNDARY_ALIGN: u32 = 2048;

pub const Entry = struct {
    sha: [40]u8,
    path: []u8,
    quant_bits: u8,
    model_id: u8,
    reason: u8,
    ext_flags: u8,
    tokens: u32,
    hits: u32,
    ctx_size: u32,
    created_at: u64,
    last_used: u64,
    payload_bytes: u64,
    text_bytes: u32,
    file_size: u64,
};

fn lePut32(p: *[4]u8, v: u32) void {
    p[0] = @intCast(v & 0xff);
    p[1] = @intCast((v >> 8) & 0xff);
    p[2] = @intCast((v >> 16) & 0xff);
    p[3] = @intCast((v >> 24) & 0xff);
}
fn leGet32(p: *const [4]u8) u32 {
    return @as(u32, p[0]) | (@as(u32, p[1]) << 8) | (@as(u32, p[2]) << 16) | (@as(u32, p[3]) << 24);
}
fn lePut64(p: *[8]u8, v: u64) void {
    for (0..8) |i| p[i] = @intCast((v >> @intCast(8 * i)) & 0xff);
}
fn leGet64(p: *const [8]u8) u64 {
    var v: u64 = 0;
    for (0..8) |i| v |= @as(u64, p[i]) << @intCast(8 * i);
    return v;
}

pub fn fillHeader(h: *[FIXED_HEADER]u8, model_id: u8, quant_bits: u8, reason: u8, ext_flags: u8, tokens: u32, hits: u32, ctx_size: u32, created_at: u64, last_used: u64, payload_bytes: u64) void {
    @memset(h, 0);
    h[0] = MAGIC0;
    h[1] = MAGIC1;
    h[2] = MAGIC2;
    h[3] = VERSION;
    h[4] = quant_bits;
    h[5] = reason;
    h[6] = ext_flags;
    h[7] = model_id;
    lePut32(@ptrCast(h[8..12]), tokens);
    lePut32(@ptrCast(h[12..16]), hits);
    lePut32(@ptrCast(h[16..20]), ctx_size);
    h[20] = PAYLOAD_ABI;
    lePut64(@ptrCast(h[24..32]), created_at);
    lePut64(@ptrCast(h[32..40]), last_used);
    lePut64(@ptrCast(h[40..48]), payload_bytes);
}

pub fn sha1Hex(data: []const u8, out: *[40]u8) void {
    var hasher = std.crypto.hash.Sha1.init(.{});
    hasher.update(data);
    var digest: [20]u8 = undefined;
    hasher.final(&digest);
    const hex = "0123456789abcdef";
    for (digest, 0..) |b, i| {
        out[i * 2] = hex[b >> 4];
        out[i * 2 + 1] = hex[b & 0xf];
    }
}

fn isHexSha(name: []const u8) bool {
    if (name.len != 43) return false;
    if (!std.mem.eql(u8, name[40..43], ".kv")) return false;
    for (name[0..40]) |c| {
        if (!((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F'))) return false;
    }
    return true;
}

pub const KVStore = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: []u8,
    budget_bytes: u64,
    entries: std.ArrayList(Entry),
    // stats
    stores: u64 = 0,
    hits: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, budget_mb: u64) !KVStore {
        const budget = if (budget_mb == 0) DEFAULT_MB * 1024 * 1024 else budget_mb * 1024 * 1024;
        // ensure dir exists: use posix mkdir -p
        try std.Io.Dir.cwd().createDirPath(io, dir);
        const d = try allocator.dupe(u8, dir);
        var self = KVStore{
            .allocator = allocator,
            .io = io,
            .dir = d,
            .budget_bytes = budget,
            .entries = std.ArrayList(Entry).empty,
        };
        try self.scan();
        self.evictIfNeeded(0);
        return self;
    }

    pub fn deinit(self: *KVStore) void {
        for (self.entries.items) |*e| self.allocator.free(e.path);
        self.entries.deinit(self.allocator);
        self.allocator.free(self.dir);
    }

    fn pathForSha(self: *KVStore, sha: *const [40]u8) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}/{s}.kv", .{ self.dir, sha.* });
    }

    pub fn scan(self: *KVStore) !void {
        for (self.entries.items) |*e| self.allocator.free(e.path);
        self.entries.clearRetainingCapacity();
        var dir = std.Io.Dir.cwd().openDir(self.io, self.dir, .{ .iterate = true }) catch return;
        defer dir.close(self.io);
        var it = dir.iterate();
        while (try it.next(self.io)) |entry| {
            if (!isHexSha(entry.name)) continue;
            const path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.dir, entry.name });
            defer self.allocator.free(path);
            var e: Entry = undefined;
            if (readEntry(self.io, path, &e)) |ok| {
                if (!ok) continue;
                // copy sha from filename
                @memcpy(&e.sha, entry.name[0..40]);
                e.path = try self.allocator.dupe(u8, path);
                // get file size via stat
                const sz = fileSize(self.io, path) orelse e.file_size;
                e.file_size = sz;
                try self.entries.append(self.allocator, e);
            } else |_| continue;
        }
    }

    pub fn store(self: *KVStore, text: []const u8, tokens: u32, quant_bits: u8, model_id: u8, ctx_size: u32, payload: []const u8) !void {
        if (tokens < MIN_TOKENS) return;
        // Align like denfer: trim 32 and align 2048
        var store_tokens = tokens;
        const trim: u32 = 32;
        const boundary_align: u32 = BOUNDARY_ALIGN;
        if (tokens > MIN_TOKENS + trim) {
            var stable = tokens - trim;
            stable -= stable % boundary_align;
            if (stable >= MIN_TOKENS) store_tokens = stable;
        }
        // Use text prefix that corresponds to store_tokens? For v1, we store full text if tokens==text prefix; else truncate text to byte length that matches store_tokens ratio.
        // Simplify: store full text, tokens is logical; on load we match byte prefix.
        var sha: [40]u8 = undefined;
        sha1Hex(text, &sha);
        const path = try self.pathForSha(&sha);
        defer self.allocator.free(path);
        // If exists and compatible, just touch hits
        if (fileExists(self.io, path)) {
            // check compatible
            var existing: Entry = undefined;
            if (readEntry(self.io, path, &existing)) |ok| {
                if (ok and existing.model_id == model_id and existing.quant_bits == quant_bits) {
                    // touch
                    _ = touchFile(self.io, path, existing.hits + 1);
                    return;
                }
            } else |_| {}
            // incompatible: unlink
            std.Io.Dir.deleteFileAbsolute(self.io, path) catch {};
        }
        const now: u64 = nowSec(self.io);
        const payload_bytes: u64 = payload.len;
        const text_bytes: u32 = @intCast(text.len);
        const file_bytes: u64 = FIXED_HEADER + 4 + text_bytes + payload_bytes;
        const required = file_bytes + file_bytes / 100 + 1; // +1% slack
        if (required > self.budget_bytes) return; // too big even empty
        self.evictIfNeeded(required);
        // write tmp then rename
        const tmp = try std.fmt.allocPrint(self.allocator, "{s}.tmp.{d}", .{ path, now });
        defer self.allocator.free(tmp);
        {
            var file = try std.Io.Dir.createFileAbsolute(self.io, tmp, .{ .truncate = true });
            defer file.close(self.io);
            var h: [FIXED_HEADER]u8 = undefined;
            fillHeader(&h, model_id, quant_bits, 1, 0, store_tokens, 0, ctx_size, now, now, payload_bytes);
            var wbuf: [4096]u8 = undefined;
            var writer = file.writer(self.io, &wbuf);
            try writer.interface.writeAll(&h);
            var tb: [4]u8 = undefined;
            lePut32(&tb, text_bytes);
            try writer.interface.writeAll(&tb);
            try writer.interface.writeAll(text);
            try writer.interface.writeAll(payload);
            try writer.interface.flush();
        }
        // rename
        std.Io.Dir.renameAbsolute(tmp, path, self.io) catch |err| return err;
        self.stores += 1;
        // rescan to include new entry
        try self.scan();
    }

    /// Find longest byte-prefix hit where SHA1(prompt[0:text_bytes])==sha and tokens>=MIN_TOKENS.
    /// Returns entry index or null. If found, touches hits and returns payload.
    pub fn load(self: *KVStore, prompt: []const u8, quant_bits: u8, model_id: u8, ctx_size: u32, out_payload: *[]u8) !?Entry {
        var best_idx: ?usize = null;
        var best_text_bytes: u32 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.tokens < MIN_TOKENS) continue;
            if (e.model_id != model_id) continue;
            if (e.quant_bits != quant_bits) continue;
            if (e.ctx_size > ctx_size) continue;
            if (e.text_bytes > prompt.len) continue;
            if (e.text_bytes < best_text_bytes) continue;
            // SHA1(prompt[0:e.text_bytes]) must equal e.sha
            var sha: [40]u8 = undefined;
            sha1Hex(prompt[0..e.text_bytes], &sha);
            if (!std.mem.eql(u8, &sha, &e.sha)) continue;
            if (e.text_bytes == best_text_bytes and e.tokens <= (if (best_idx) |bi| self.entries.items[bi].tokens else 0)) continue;
            best_idx = i;
            best_text_bytes = e.text_bytes;
        }
        const idx = best_idx orelse return null;
        const e = &self.entries.items[idx];
        // read file and extract payload
        const payload = try readPayload(self.allocator, self.io, e.path, e.text_bytes, e.payload_bytes) orelse return null;
        errdefer self.allocator.free(payload);
        // touch hits
        e.hits += 1;
        e.last_used = nowSec(self.io);
        _ = touchFile(self.io, e.path, e.hits);
        self.hits += 1;
        out_payload.* = payload;
        return e.*;
    }

    fn evictIfNeeded(self: *KVStore, extra: u64) void {
        if (self.budget_bytes == 0) return;
        var total: u64 = 0;
        for (self.entries.items) |*e| total += e.file_size;
        const target = if (extra > self.budget_bytes) 0 else self.budget_bytes - extra;
        while (total > target and self.entries.items.len > 0) {
            var victim: usize = 0;
            var victim_score = evictionScore(&self.entries.items[0], self.io);
            for (self.entries.items[1..], 1..) |*e, i| {
                const s = evictionScore(e, self.io);
                if (s < victim_score or (s == victim_score and e.last_used < self.entries.items[victim].last_used)) {
                    victim = i;
                    victim_score = s;
                }
            }
            const e = self.entries.items[victim];
            std.Io.Dir.deleteFileAbsolute(self.io, e.path) catch {};
            total -|= e.file_size;
            self.allocator.free(e.path);
            _ = self.entries.orderedRemove(victim);
        }
    }
};

fn nowSec(io: std.Io) u64 {
    const ts = std.Io.Timestamp.now(io, .real);
    return @intCast(@divTrunc(ts.nanoseconds, std.time.ns_per_s));
}
fn evictionScore(e: *const Entry, io: std.Io) f64 {
    const now: u64 = nowSec(io);
    const elapsed: f64 = if (e.last_used > 0 and now > e.last_used) @floatFromInt(now - e.last_used) else 0;
    var eff_hits: f64 = @floatFromInt(e.hits);
    eff_hits *= std.math.exp2(-elapsed / @as(f64, @floatFromInt(HIT_HALF_LIFE_S)));
    if (eff_hits < 0.01) eff_hits = 0;
    const score = (eff_hits + 1.0) * @as(f64, @floatFromInt(e.tokens)) / @as(f64, @floatFromInt(@max(e.file_size, 1)));
    // anchor bonus like denfer for cold? simplified
    return score;
}

fn fileExists(io: std.Io, path: []const u8) bool {
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    f.close(io);
    return true;
}
fn fileSize(io: std.Io, path: []const u8) ?u64 {
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer file.close(io);
    const st = file.stat(io) catch return null;
    return @intCast(st.size);
}
fn readEntry(io: std.Io, path: []const u8, out: *Entry) !bool {
    var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    defer file.close(io);
    var h: [FIXED_HEADER]u8 = undefined;
    _ = file.readPositionalAll(io, &h, 0) catch return false;
    if (h[0] != MAGIC0 or h[1] != MAGIC1 or h[2] != MAGIC2 or h[3] != VERSION) return false;
    if (h[20] != PAYLOAD_ABI) return false;
    out.quant_bits = h[4];
    out.reason = h[5];
    out.ext_flags = h[6];
    out.model_id = h[7];
    out.tokens = leGet32(@ptrCast(h[8..12]));
    out.hits = leGet32(@ptrCast(h[12..16]));
    out.ctx_size = leGet32(@ptrCast(h[16..20]));
    out.created_at = leGet64(@ptrCast(h[24..32]));
    out.last_used = leGet64(@ptrCast(h[32..40]));
    out.payload_bytes = leGet64(@ptrCast(h[40..48]));
    var tb: [4]u8 = undefined;
    _ = file.readPositionalAll(io, &tb, FIXED_HEADER) catch return false;
    out.text_bytes = leGet32(&tb);
    if (out.tokens == 0) return false;
    if (out.quant_bits != 2 and out.quant_bits != 4 and out.quant_bits != 0 and out.quant_bits != 8) return false;
    if (file.stat(io) catch null) |st| out.file_size = @intCast(st.size) else out.file_size = FIXED_HEADER + 4 + out.text_bytes + out.payload_bytes;
    return true;
}
fn readPayload(allocator: std.mem.Allocator, io: std.Io, path: []const u8, text_bytes: u32, payload_bytes: u64) !?[]u8 {
    var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer file.close(io);
    const off: u64 = FIXED_HEADER + 4 + text_bytes;
    const len: usize = @intCast(payload_bytes);
    const payload = try allocator.alloc(u8, len);
    errdefer allocator.free(payload);
    _ = file.readPositionalAll(io, payload, off) catch {
        allocator.free(payload);
        return null;
    };
    return payload;
}
fn touchFile(io: std.Io, path: []const u8, hits: u32) bool {
    var file = std.Io.Dir.openFileAbsolute(io, path, .{ .mode = .read_write }) catch return false;
    defer file.close(io);
    var h: [FIXED_HEADER]u8 = undefined;
    _ = file.readPositionalAll(io, &h, 0) catch return false;
    lePut32(@ptrCast(h[12..16]), hits);
    const now: u64 = nowSec(io);
    lePut64(@ptrCast(h[32..40]), now);
    file.writePositionalAll(io, &h, 0) catch return false;
    return true;
}

// --- tests ---
test "header round-trip" {
    var h: [FIXED_HEADER]u8 = undefined;
    fillHeader(&h, 7, 4, 1, 0, 1024, 5, 262144, 1000, 2000, 12345);
    try std.testing.expectEqual(MAGIC0, h[0]);
    try std.testing.expectEqual(VERSION, h[3]);
    // read back via file using Io
    const tmp = "/tmp/mlx-test-header.kv";
    const io = std.Io.Threaded.global_single_threaded.io();
    {
        var file = try std.Io.Dir.createFileAbsolute(io, tmp, .{ .truncate = true });
        defer file.close(io);
        var wbuf: [4096]u8 = undefined;
        var writer = file.writer(io, &wbuf);
        try writer.interface.writeAll(&h);
        var tb: [4]u8 = undefined;
        lePut32(&tb, 11);
        try writer.interface.writeAll(&tb);
        try writer.interface.writeAll("hello world");
        try writer.interface.writeAll("payload");
        try writer.interface.flush();
    }
    defer std.Io.Dir.deleteFileAbsolute(io, tmp) catch {};
    var e: Entry = undefined;
    try std.testing.expect(try readEntry(io, tmp, &e));
    try std.testing.expectEqual(@as(u32, 1024), e.tokens);
    try std.testing.expectEqual(@as(u64, 12345), e.payload_bytes);
}

test "store and load cuts prefill" {
    const tmpdir = "/tmp/mlx-kv-test";
    const io = std.Io.Threaded.global_single_threaded.io();
    _ = std.Io.Dir.cwd().deleteTree(io, tmpdir) catch {};
    try std.Io.Dir.cwd().createDirPath(io, tmpdir);
    defer std.Io.Dir.cwd().deleteTree(io, tmpdir) catch {};
    const alloc = std.testing.allocator;
    var store = try KVStore.init(alloc, io, tmpdir, 10);
    defer store.deinit();
    const text = "The capital of France is Paris. The capital of Germany is Berlin. ";
    const payload = "dummy-kv-payload-for-512-tokens-safetensors-bytes";
    try store.store(text, 600, 4, 42, 262144, payload);
    // file should exist
    var sha: [40]u8 = undefined;
    sha1Hex(text, &sha);
    const path = try std.fmt.allocPrint(alloc, "{s}/{s}.kv", .{ tmpdir, sha });
    defer alloc.free(path);
    try std.testing.expect(fileExists(io, path));
    var out: []u8 = undefined;
    // Exact hit
    const hit = try store.load(text, 4, 42, 262144, &out);
    defer if (hit != null) alloc.free(out);
    try std.testing.expect(hit != null);
    try std.testing.expectEqualStrings(payload, out);
    try std.testing.expectEqual(@as(u64, 1), store.hits);
    // Prefix hits are found too (denfer style) — longer prompt shares prefix
    var out2: []u8 = undefined;
    const prompt2 = try std.fmt.allocPrint(alloc, "{s}extra suffix", .{text});
    defer alloc.free(prompt2);
    const hit2 = try store.load(prompt2, 4, 42, 262144, &out2);
    defer if (hit2 != null) alloc.free(out2);
    try std.testing.expect(hit2 != null);
    try std.testing.expectEqualStrings(payload, out2);
}
