const std = @import("std");
const kv_checkpoint = @import("kv_checkpoint.zig");

fn hashPrompt(prompt: []const u8) u64 {
    // FNV-1a 64-bit, then hex truncated — deterministic like Python's sha256[:16] but simpler.
    var h: u64 = 14695981039346656037;
    for (prompt) |b| {
        h ^= b;
        h *%= 1099511628211;
    }
    return h;
}

pub const CacheEntry = struct {
    prompt_hash: u64,
    prompt: []const u8,
    completion: []const u8,
    tokens: u32,
};

pub const PrefixCache = struct {
    allocator: std.mem.Allocator,
    max_entries: u32,
    max_bytes: u64,
    entries: std.ArrayList(CacheEntry),
    hits: u64 = 0,
    misses: u64 = 0,
    evictions: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, max_entries: u32, max_bytes: u64) PrefixCache {
        return .{
            .allocator = allocator,
            .max_entries = max_entries,
            .max_bytes = max_bytes,
            .entries = std.ArrayList(CacheEntry).empty,
        };
    }

    pub fn deinit(self: *PrefixCache) void {
        for (self.entries.items) |e| {
            self.allocator.free(e.prompt);
            self.allocator.free(e.completion);
        }
        self.entries.deinit(self.allocator);
    }

    pub fn get(self: *PrefixCache, prompt: []const u8) ?CacheEntry {
        const h = hashPrompt(prompt);
        for (self.entries.items, 0..) |e, idx| {
            if (e.prompt_hash == h and std.mem.eql(u8, e.prompt, prompt)) {
                self.hits += 1;
                // LRU bump: move to end
                const entry = self.entries.orderedRemove(idx);
                self.entries.append(self.allocator, entry) catch {};
                return entry;
            }
        }
        self.misses += 1;
        return null;
    }

    pub fn put(self: *PrefixCache, prompt: []const u8, completion: []const u8, tokens: u32) !void {
        if (self.max_entries == 0) return;
        const h = hashPrompt(prompt);
        // If exists, replace
        for (self.entries.items, 0..) |e, idx| {
            if (e.prompt_hash == h and std.mem.eql(u8, e.prompt, prompt)) {
                self.allocator.free(e.prompt);
                self.allocator.free(e.completion);
                _ = self.entries.orderedRemove(idx);
                break;
            }
        }
        while (self.entries.items.len >= self.max_entries) {
            const ev = self.entries.orderedRemove(0);
            self.allocator.free(ev.prompt);
            self.allocator.free(ev.completion);
            self.evictions += 1;
        }
        const approx_bytes: u64 = prompt.len + completion.len;
        if (self.max_bytes > 0 and approx_bytes > self.max_bytes) return;
        const p_dup = try self.allocator.dupe(u8, prompt);
        errdefer self.allocator.free(p_dup);
        const c_dup = try self.allocator.dupe(u8, completion);
        try self.entries.append(self.allocator, .{ .prompt_hash = h, .prompt = p_dup, .completion = c_dup, .tokens = tokens });
    }

    pub fn stats(self: *const PrefixCache) struct { entries: usize, max_entries: u32, hits: u64, misses: u64, evictions: u64 } {
        return .{ .entries = self.entries.items.len, .max_entries = self.max_entries, .hits = self.hits, .misses = self.misses, .evictions = self.evictions };
    }

    pub fn clear(self: *PrefixCache) void {
        for (self.entries.items) |e| {
            self.allocator.free(e.prompt);
            self.allocator.free(e.completion);
        }
        self.entries.clearRetainingCapacity();
    }
};

pub const DiskCache = struct {
    enabled: bool,
    writes: u64 = 0,
    hits: u64 = 0,
    store: ?kv_checkpoint.KVStore = null,
    allocator: std.mem.Allocator = undefined,
    // For model-specific header fields, cached at init
    model_id: u8 = 0,
    quant_bits: u8 = 0,
    ctx_size: u32 = 262144,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, budget_mb: u64, enabled: bool, model_id: u8, quant_bits: u8, ctx_size: u32) !DiskCache {
        if (!enabled) return .{ .enabled = false, .allocator = allocator };
        const kv = try kv_checkpoint.KVStore.init(allocator, io, dir, budget_mb);
        return .{
            .enabled = true,
            .allocator = allocator,
            .store = kv,
            .model_id = model_id,
            .quant_bits = quant_bits,
            .ctx_size = ctx_size,
        };
    }

    pub fn deinit(self: *DiskCache) void {
        if (self.store) |*s| s.deinit();
    }

    /// Lookup by byte prefix (denfer style). Returns owned payload (completion) or null.
    /// For completion payload, only exact hits are returned (prefix hits would need KV reuse).
    /// Caller must free with allocator.free when done.
    pub fn lookup(self: *DiskCache, prompt: []const u8) ?[]u8 {
        if (!self.enabled) return null;
        if (self.store == null) return null;
        var payload: []u8 = undefined;
        const hit = self.store.?.load(prompt, self.quant_bits, self.model_id, self.ctx_size, &payload) catch return null;
        const e = hit orelse return null;
        if (e.text_bytes != prompt.len) {
            // Prefix hit — would cut prefill if payload were KV. For completion, not reusable.
            std.log.info("[kv-disk] prefix hit stored {d} vs prompt {d} tokens {d} (exact required for completion)", .{ e.text_bytes, prompt.len, e.tokens });
            self.allocator.free(payload);
            return null;
        }
        self.hits += 1;
        return payload;
    }

    /// Store prompt+completion as fixed checkpoint. `tokens` is total tokens (prompt+gen) for header.
    pub fn storeCompletion(self: *DiskCache, prompt: []const u8, completion: []const u8, tokens: u32) void {
        if (!self.enabled) return;
        if (self.store == null) return;
        // Payload is completion bytes; text is prompt.
        self.store.?.store(prompt, tokens, self.quant_bits, self.model_id, self.ctx_size, completion) catch return;
        self.writes += 1;
    }

    // Backward compat for old calls (engine still calls store with 3 args)
    pub fn storeLegacy(self: *DiskCache, prompt: []const u8, completion: []const u8, tokens: u32) void {
        self.storeCompletion(prompt, completion, tokens);
    }
};

// --- tests ---
test "lru eviction" {
    var c = PrefixCache.init(std.testing.allocator, 2, 1 << 30);
    defer c.deinit();
    try c.put("a", "1", 1);
    try c.put("b", "2", 2);
    _ = c.get("a"); // bump a
    try c.put("c", "3", 3); // evicts b
    try std.testing.expect(c.get("b") == null);
    try std.testing.expect(c.get("a") != null);
    try std.testing.expect(c.get("c") != null);
    try std.testing.expectEqual(@as(u64, 1), c.evictions);
}

test "hit miss" {
    var c = PrefixCache.init(std.testing.allocator, 10, 1 << 30);
    defer c.deinit();
    try std.testing.expect(c.get("missing") == null);
    try std.testing.expectEqual(@as(u64, 1), c.misses);
    try c.put("hello", "world", 1);
    try std.testing.expect(c.get("hello") != null);
    try std.testing.expectEqual(@as(u64, 1), c.hits);
}

test "disabled" {
    var c = PrefixCache.init(std.testing.allocator, 0, 1 << 30);
    defer c.deinit();
    try c.put("a", "1", 1);
    try std.testing.expect(c.get("a") == null);
}
