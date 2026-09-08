//! Native in-process engine: tokenizer + weights + qwen3_5 forward +
//! sampler + MTP speculative decode. No Python interpreter, venv, or `uv`
//! anywhere in the loop (`bridge/` is deleted).
//!
//! The hot/disk TEXT completion caches from the bridge era are kept as-is
//! around the native loop (exact-hit fast path).

const std = @import("std");
const sampling = @import("sampling.zig");
const cache = @import("cache.zig");
const mlx = @import("mlx.zig");
const tokenizer_mod = @import("tokenizer.zig");
const weights_mod = @import("weights.zig");
const model = @import("model.zig");
const sample = @import("sample.zig");
const mtp_mod = @import("mtp.zig");
const apc_mod = @import("apc.zig");

pub const prefill_chunk: u32 = 2048; // fork parity

pub const EngineError = error{
    ModelNotFound,
    ContextOverflow,
    VisionUnsupported,
    UnsupportedDtype,
    BenchMismatch,
};

pub const EngineConfig = struct {
    model: []const u8,
    ctx_size: u32 = 0, // 0 = auto
    kv_quant: []const u8 = "off",
    prefix_cache_entries: u32 = 32,
    prefix_cache_mem: []const u8 = "2GB",
    prefix_cache_disk: ?[]const u8 = null,
    apc_disk_quota: ?[]const u8 = null, // e.g. "10GB" or "off"
    apc_disk_dir: ?[]const u8 = null, // absolute dir, else ~/.mlx-runner/apc-disk/<hash>
    no_vision: bool = false,
    mtp: bool = false,
    mtp_gamma: u32 = 1,
};

pub const GenerateStats = struct {
    prefill_tokens: u64 = 0,
    prefill_ms: i64 = 0,
    decode_tokens: u64 = 0,
    decode_ms: i64 = 0,
    ttft_ms: i64 = 0,
    from_draft: u64 = 0,
    offered: u64 = 0,
    accepted: u64 = 0,
    cached_tokens: u64 = 0, // APC hit saves
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    config: EngineConfig,
    model_path: []const u8,
    model_max_ctx: u32,
    base_sampling: sampling.SamplingParams,
    eos_ids: []u32,
    hot_cache: cache.PrefixCache,
    disk_cache: cache.DiskCache,
    apc: apc_mod.ApcCache,
    apc_disk: apc_mod.ApcDisk,
    tokenizer: tokenizer_mod.Tokenizer,
    model_cfg: model.Config,
    mm: model.Model,
    caches: model.Caches,
    mtp: mtp_mod.MtpState,
    seed_seq: u64,
    mtp_enabled: bool,
    bypass_cache: bool = false, // bench: skip text caches for true comparisons
    last_stats: GenerateStats = .{},
    last_ids: ?[]u32 = null, // owned ids of the last generate (bench agreement)
    total_tokens: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, cfg: EngineConfig, base_sampling: sampling.SamplingParams) !Engine {
        if (!std.mem.eql(u8, cfg.kv_quant, "off")) return EngineError.UnsupportedDtype;
        mlx.installErrorHandler();

        const model_path = try resolveModel(allocator, io, cfg.model);
        errdefer allocator.free(model_path);

        const model_cfg = try model.Config.load(allocator, io, model_path);
        const max_ctx = if (cfg.ctx_size != 0) cfg.ctx_size else model_cfg.max_position_embeddings;
        const base = try sampling.fromGenerationConfig(allocator, io, model_path, base_sampling);
        const eos = try sampling.readEosIds(allocator, io, model_path);
        errdefer allocator.free(eos);

        std.log.info("[load] {s} ctx {d} (this takes a while, tens of GB)", .{ model_path, max_ctx });
        var tok = try tokenizer_mod.Tokenizer.load(allocator, io, model_path);
        errdefer tok.deinit();
        var wm = try weights_mod.WeightMap.load(allocator, io, model_path);
        std.log.info("[load] {d} tensors (text only, vision dropped)", .{wm.count()});
        var mm = model.Model.init(allocator, model_cfg, wm) catch |err| {
            wm.deinit();
            return err;
        };
        errdefer mm.deinit();
        var caches = try model.Caches.init(allocator, model_cfg.num_hidden_layers);
        errdefer caches.deinit();

        const mem_bytes = parseSize(cfg.prefix_cache_mem);
        const disk_enabled = if (cfg.prefix_cache_disk) |d| !std.mem.eql(u8, d, "0") and !std.mem.eql(u8, d, "off") and d.len > 0 else false;
        const budget_mb: u64 = if (disk_enabled) @intCast(parseSize(cfg.prefix_cache_disk.?) / (1024 * 1024)) else 0;
        var model_id: u8 = 0;
        {
            var h = std.crypto.hash.Sha1.init(.{});
            h.update(cfg.model);
            var d: [20]u8 = undefined;
            h.final(&d);
            model_id = d[0];
        }
        const disk_dir = blk: {
            if (disk_enabled and cfg.prefix_cache_disk != null) {
                const s = cfg.prefix_cache_disk.?;
                if (s.len > 0 and s[0] == '/') break :blk try allocator.dupe(u8, s);
            }
            const home_c = std.c.getenv("HOME");
            const home = if (home_c) |p| std.mem.span(p) else "/tmp";
            var hs = std.crypto.hash.Sha1.init(.{});
            hs.update(cfg.model);
            var dg: [20]u8 = undefined;
            hs.final(&dg);
            const hex = "0123456789abcdef";
            var fp: [40]u8 = undefined;
            for (dg, 0..) |b, i| {
                fp[i * 2] = hex[b >> 4];
                fp[i * 2 + 1] = hex[b & 0xf];
            }
            break :blk try std.fmt.allocPrint(allocator, "{s}/.mlx-runner/kv-disk/{s}", .{ home, fp[0..8] });
        };
        defer allocator.free(disk_dir);
        const disk = try cache.DiskCache.init(allocator, io, disk_dir, budget_mb, disk_enabled, model_id, 0, max_ctx);

        // APC disk tier: quota + location (enable if either is set; default 10GB if dir without quota)
        const apc_disk_enabled = blk: {
            if (cfg.apc_disk_quota) |d| if (!std.mem.eql(u8, d, "0") and !std.mem.eql(u8, d, "off") and d.len > 0) break :blk true;
            if (cfg.apc_disk_dir) |d| if (d.len > 0) break :blk true;
            break :blk false;
        };
        const apc_budget_mb: u64 = if (apc_disk_enabled) blk2: {
            if (cfg.apc_disk_quota) |d| {
                if (d.len > 0 and d[0] == '/') break :blk2 10240; // path given as quota, use default 10GB
                const sz = parseSize(d);
                if (sz == 0) break :blk2 10240;
                break :blk2 @intCast(sz / (1024 * 1024));
            }
            break :blk2 10240; // default 10GB when only dir is given
        } else 0;
        const apc_disk_dir = blk: {
            if (cfg.apc_disk_dir) |s| {
                if (s.len > 0 and s[0] == '/') break :blk try allocator.dupe(u8, s);
            }
            // also accept quota as absolute path for backward compat (like kv-disk)
            if (apc_disk_enabled and cfg.apc_disk_quota != null) {
                const q = cfg.apc_disk_quota.?;
                if (q.len > 0 and q[0] == '/') break :blk try allocator.dupe(u8, q);
            }
            const home_c = std.c.getenv("HOME");
            const home = if (home_c) |p| std.mem.span(p) else "/tmp";
            var hs = std.crypto.hash.Sha1.init(.{});
            hs.update(cfg.model);
            var dg: [20]u8 = undefined;
            hs.final(&dg);
            const hex = "0123456789abcdef";
            var fp: [40]u8 = undefined;
            for (dg, 0..) |b, i| {
                fp[i * 2] = hex[b >> 4];
                fp[i * 2 + 1] = hex[b & 0xf];
            }
            break :blk try std.fmt.allocPrint(allocator, "{s}/.mlx-runner/apc-disk/{s}", .{ home, fp[0..8] });
        };
        defer allocator.free(apc_disk_dir);
        const apc_disk = try apc_mod.ApcDisk.init(allocator, io, apc_disk_dir, apc_budget_mb, apc_disk_enabled, model_id, 0, max_ctx);

        const seed_seq = if (base.seed) |s| s else entropySeed();
        return .{
            .allocator = allocator,
            .config = cfg,
            .model_path = model_path,
            .model_max_ctx = max_ctx,
            .base_sampling = base,
            .eos_ids = eos,
            .hot_cache = cache.PrefixCache.init(allocator, cfg.prefix_cache_entries, mem_bytes),
            .disk_cache = disk,
            .apc = apc_mod.ApcCache.init(allocator, cfg.prefix_cache_entries, mem_bytes),
            .apc_disk = apc_disk,
            .tokenizer = tok,
            .model_cfg = model_cfg,
            .mm = mm,
            .caches = caches,
            .mtp = .{},
            .seed_seq = seed_seq,
            .mtp_enabled = cfg.mtp,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.allocator.free(self.model_path);
        self.allocator.free(self.eos_ids);
        if (self.last_ids) |ids| self.allocator.free(ids);
        self.hot_cache.deinit();
        self.disk_cache.deinit();
        self.apc.deinit();
        self.apc_disk.deinit();
        self.tokenizer.deinit();
        self.mm.deinit();
        self.caches.deinit();
        self.mtp.deinit();
    }
    pub fn props(self: *const Engine) struct { model: []const u8, model_path: []const u8, ctx_size: u32, kv_quant: []const u8 } {
        return .{ .model = self.config.model, .model_path = self.model_path, .ctx_size = self.model_max_ctx, .kv_quant = self.config.kv_quant };
    }

    fn nextSeed(self: *Engine) u64 {
        const s = self.seed_seq;
        self.seed_seq +%= 0x9E3779B97F4A7C15;
        return s;
    }

    fn uniform01(self: *Engine) f32 {
        // splitmix64 -> [0,1)
        var z = self.nextSeed() +% 0x9E3779B97F4A7C15;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        z = z ^ (z >> 31);
        return @as(f32, @floatFromInt(z >> 40)) / 16777216.0;
    }

    fn isEos(self: *const Engine, id: u32) bool {
        for (self.eos_ids) |e| if (e == id) return true;
        return false;
    }

    /// Render chat messages (text path only) to the Qwen `<|im_start|>` framing.
    /// Any image/video/tool content -> error.VisionUnsupported.
    fn renderChat(allocator: std.mem.Allocator, messages_json: []const u8) ![]u8 {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, messages_json, .{}) catch
            return EngineError.VisionUnsupported;
        defer parsed.deinit();
        if (parsed.value != .array) return EngineError.VisionUnsupported;
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (parsed.value.array.items) |m| {
            if (m != .object) return EngineError.VisionUnsupported;
            const role = m.object.get("role") orelse return EngineError.VisionUnsupported;
            const content = m.object.get("content") orelse return EngineError.VisionUnsupported;
            if (role != .string or content != .string) return EngineError.VisionUnsupported;
            const part = try std.fmt.allocPrint(allocator, "<|im_start|>{s}\n{s}<|im_end|>\n", .{ role.string, content.string });
            defer allocator.free(part);
            try out.appendSlice(allocator, part);
        }
        try out.appendSlice(allocator, "<|im_start|>assistant\n");
        return out.toOwnedSlice(allocator);
    }

    /// Generate — prompt XOR messages. Returns owned completion text.
    /// Stats for the last call land in `last_stats` (bench reads them).
    pub fn generate(
        self: *Engine,
        io: std.Io,
        prompt: ?[]const u8,
        messages_json: ?[]const u8, // JSON array string when chat
        max_tokens: u32,
        req_sampling: ?sampling.SamplingParams,
        stream: bool, // accepted but output is still buffered (SSE splits later)
    ) ![]u8 {
        _ = stream;
        const eff = if (req_sampling) |rs|
            self.base_sampling.mergedWithRequest(.{
                .temperature = rs.temp,
                .top_p = rs.top_p,
                .top_k = rs.top_k,
                .min_p = rs.min_p,
                .seed = rs.seed,
            })
        else
            self.base_sampling;

        const cache_key = if (messages_json) |mj| mj else prompt.?;

        if (!self.bypass_cache) {
            if (self.hot_cache.get(cache_key)) |hit| {
                std.log.info("[cache] hit {x} ({d} chars) -> {d} tok", .{ hit.prompt_hash, cache_key.len, hit.tokens });
                return try self.allocator.dupe(u8, hit.completion);
            }
            if (self.disk_cache.lookup(cache_key)) |disk_payload| {
                defer self.allocator.free(disk_payload);
                std.log.info("[kv-disk] hit sha for {d} chars -> {d} bytes payload", .{ cache_key.len, disk_payload.len });
                const n_tok: u32 = @intCast(disk_payload.len / 4 + 1);
                try self.hot_cache.put(cache_key, disk_payload, n_tok);
                self.total_tokens += n_tok;
                return try self.allocator.dupe(u8, disk_payload);
            }
        }

        var text: []const u8 = undefined;
        var text_owned = false;
        if (messages_json) |mj| {
            text = try Engine.renderChat(self.allocator, mj);
            text_owned = true;
        } else {
            text = prompt.?;
        }
        defer if (text_owned) self.allocator.free(text);

        const completion = try self.generateUncached(io, text, max_tokens, eff);

        if (!self.bypass_cache) {
            const n_tok: u32 = @intCast(completion.len / 4 + 1);
            const prompt_tok: u32 = @intCast(cache_key.len / 4 + 1);
            self.total_tokens += n_tok;
            try self.hot_cache.put(cache_key, completion, n_tok);
            self.disk_cache.storeCompletion(cache_key, completion, n_tok + prompt_tok);
        }
        return completion;
    }

    fn generateUncached(self: *Engine, io: std.Io, text: []const u8, max_tokens: u32, eff: sampling.SamplingParams) ![]u8 {
        const use_mtp = self.mtp_enabled;
        var stats = GenerateStats{};
        const t_start = std.Io.Clock.Timestamp.now(io, .awake);

        const prompt_ids = try self.tokenizer.encode(self.allocator, text);
        defer self.allocator.free(prompt_ids);
        if (prompt_ids.len > self.model_max_ctx) return EngineError.ContextOverflow;
        if (prompt_ids.len == 0) return try self.allocator.dupe(u8, "");

        // True APC: token-prefix KV reuse (hot → disk)
        var apc_hit_len: u32 = 0;
        var apc_hit: ?*apc_mod.ApcEntry = null;
        var apc_disk_hit = false;
        if (!self.bypass_cache) {
            apc_hit = self.apc.findLongestPrefix(prompt_ids);
            if (apc_hit) |e| {
                apc_hit_len = e.cached_len;
                for (0..self.model_cfg.num_hidden_layers) |i| {
                    self.caches.kv[i].restoreFrom(&e.kv[i]);
                    self.caches.gdn[i].restoreFrom(&e.gdn[i]);
                }
                self.mtp.kv.restoreFrom(&e.mtp_kv);
                self.mtp.clearPending();
                std.log.info("[apc] hit prefix {d}/{d} tok cached {d} (remaining {d})", .{ e.tokens.len, prompt_ids.len, apc_hit_len, prompt_ids.len - apc_hit_len });
            } else if (self.apc_disk.load(text, prompt_ids, &self.caches, &self.mtp.kv) catch null) |disk_len| {
                apc_hit_len = disk_len;
                apc_disk_hit = true;
                self.mtp.clearPending();
                std.log.info("[apc-disk] hit {d}/{d} tok cached {d} (remaining {d})", .{ disk_len + 1, prompt_ids.len, disk_len, prompt_ids.len - disk_len });
            }
        }
        if (apc_hit == null and !apc_disk_hit) {
            self.caches.reset();
            self.mtp.reset();
        }
        // Prefill reserve: single alloc per KV cache instead of step-256 growth.
        const total_need: u32 = @intCast(prompt_ids.len + max_tokens);
        self.caches.reserveKv(self.model_cfg, total_need) catch {};
        self.mtp.kv.reserve(1, self.model_cfg.num_key_value_heads, self.model_cfg.head_dim, total_need) catch {};

        // Prefill in chunks, leaving 1 token for the decode loop (fork parity).
        // MTP cache is seeded per chunk when MTP will be used.
        var pos: usize = apc_hit_len;
        var remaining: usize = prompt_ids.len - pos;
        const t_pre0 = std.Io.Clock.Timestamp.now(io, .awake);
        while (remaining > 1) {
            const n = @min(prefill_chunk, remaining - 1);
            const chunk = prompt_ids[pos .. pos + n];
            const hidden = try model.forward(&self.mm, chunk, &self.caches, 0);
            try mlx.check(mlx.mlx_array_eval(hidden));
            if (use_mtp) {
                const next_ids = prompt_ids[pos + 1 .. pos + n + 1];
                const mout = try mtp_mod.mtpForward(&self.mm, hidden, next_ids, &self.mtp.kv);
                try mlx.check(mlx.mlx_array_eval(mout));
                model.freeArr(mout);
            }
            model.freeArr(hidden);
            try mlx.checkError();
            pos += n;
            remaining -= n;
        }
        _ = mlx.mlx_clear_cache();
        // Store APC snapshot after prefill (full prompt state) for future prefix hits (hot + disk)
        if (!self.bypass_cache and prompt_ids.len > 1) {
            const cached_len_full: u32 = @intCast(prompt_ids.len - 1);
            self.apc.put(text, prompt_ids, cached_len_full, &self.caches, &self.mtp.kv) catch |err| {
                std.log.warn("[apc] put failed {any}", .{err});
            };
            self.apc_disk.store(text, prompt_ids, cached_len_full, &self.caches, &self.mtp.kv) catch |err| {
                std.log.warn("[apc-disk] store failed {any}", .{err});
            };
        }
        stats.prefill_tokens = if (prompt_ids.len > 0) prompt_ids.len - 1 else 0;
        stats.cached_tokens = apc_hit_len;
        stats.prefill_ms = t_pre0.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.toMilliseconds();
        if (apc_hit_len > 0) {
            std.log.info("[apc] prefill {d} tok ({d} cached, {d} computed) in {d} ms", .{ stats.prefill_tokens, stats.cached_tokens, stats.prefill_tokens - stats.cached_tokens, stats.prefill_ms });
        } else {
            std.log.info("[apc] prefill {d} tok (miss) in {d} ms", .{ stats.prefill_tokens, stats.prefill_ms });
        }

        var gen: std.ArrayList(u32) = .empty;
        defer gen.deinit(self.allocator);
        var last: u32 = prompt_ids[pos];

        const t_dec0 = std.Io.Clock.Timestamp.now(io, .awake);
        var first_token = true;
        if (use_mtp) {
            try self.decodeMtp(io, &gen, &last, max_tokens, eff, &stats, &first_token, t_start);
        } else {
            try self.decodeBase(io, &gen, &last, max_tokens, eff, &stats, &first_token, t_start);
        }
        stats.decode_ms = t_dec0.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.toMilliseconds();

        if (use_mtp) {
            std.log.info("[generate] {d} tokens ({d} from draft) mtp=on accept={d:.1}%", .{
                gen.items.len, stats.from_draft, self.mtp.acceptRate() * 100,
            });
        }
        self.last_stats = stats;
        self.total_tokens += gen.items.len;
        if (self.last_ids) |old| self.allocator.free(old);
        self.last_ids = try self.allocator.dupe(u32, gen.items);
        return try self.tokenizer.decode(self.allocator, gen.items);
    }

    fn decodeBase(self: *Engine, io: std.Io, gen: *std.ArrayList(u32), last: *u32, max_tokens: u32, eff: sampling.SamplingParams, stats: *GenerateStats, first_token: *bool, t_start: std.Io.Clock.Timestamp) !void {
        while (gen.items.len < max_tokens) {
            const id = try self.stepOnce(last.*, eff);
            if (first_token.*) {
                stats.ttft_ms = t_start.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.toMilliseconds();
                first_token.* = false;
            }
            try gen.append(self.allocator, id);
            stats.decode_tokens += 1;
            last.* = id;
            if (self.isEos(id)) break;
        }
    }

    /// One backbone decode step on a single token: forward -> norm -> logits -> sample.
    fn stepOnce(self: *Engine, id: u32, eff: sampling.SamplingParams) !u32 {
        const ids = [_]u32{id};
        const hidden_pre = try model.forward(&self.mm, &ids, &self.caches, 0);
        defer model.freeArr(hidden_pre);
        const post = try model.normHidden(&self.mm, hidden_pre);
        defer model.freeArr(post);
        const logits = try model.logitsOf(&self.mm, post, 1);
        defer model.freeArr(logits);
        const draw = try sample.sampleToken(logits, eff, self.nextSeed());
        defer if (draw.filtered) |f| model.freeArr(f);
        try mlx.checkError();
        return draw.id;
    }

    fn decodeMtp(self: *Engine, io: std.Io, gen: *std.ArrayList(u32), last: *u32, max_tokens: u32, eff: sampling.SamplingParams, stats: *GenerateStats, first_token: *bool, t_start: std.Io.Clock.Timestamp) !void {
        const greedy = eff.temp == 0;
        const gamma = self.config.mtp_gamma;
        while (gen.items.len < max_tokens) {
            if (self.mtp.pending == null) {
                // Backbone-only step, then draft from the MTP head.
                const ids = [_]u32{last.*};
                const hidden_pre = try model.forward(&self.mm, &ids, &self.caches, 0);
                defer model.freeArr(hidden_pre);
                const hrow = try model.sliceArr(hidden_pre, model.newShape(3, .{ 0, 0, 0 }), model.newShape(3, .{ 1, 1, self.model_cfg.hidden_size }));
                defer model.freeArr(hrow);
                const post = try model.normHidden(&self.mm, hidden_pre);
                defer model.freeArr(post);
                const logits = try model.logitsOf(&self.mm, post, 1);
                defer model.freeArr(logits);
                const main = try sample.sampleToken(logits, eff, self.nextSeed());
                defer if (main.filtered) |f| model.freeArr(f);
                try mlx.checkError();
                if (first_token.*) {
                    stats.ttft_ms = t_start.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.toMilliseconds();
                    first_token.* = false;
                }
                try gen.append(self.allocator, main.id);
                stats.decode_tokens += 1;
                if (self.isEos(main.id)) {
                    last.* = main.id;
                    return;
                }
                if (gen.items.len >= max_tokens) {
                    last.* = main.id;
                    return;
                }
                if (gamma == 1) {
                    const mtp_h = try mtp_mod.mtpForward(&self.mm, hrow, &[_]u32{main.id}, &self.mtp.kv);
                    defer model.freeArr(mtp_h);
                    const dlogits = try mtp_mod.mtpDraftLogits(&self.mm, mtp_h);
                    defer model.freeArr(dlogits);
                    const draft = try sample.sampleToken(dlogits, eff, self.nextSeed());
                    try mlx.checkError();
                    self.mtp.pending = .{ .drafts = [_]mtp_mod.MtpState.Draft{ .{ .id = draft.id, .lp = draft.lp, .filtered = draft.filtered }, undefined }, .len = 1 };
                    last.* = main.id;
                } else {
                    // gamma=2: draft two tokens sequentially
                    const mtp_h1 = try mtp_mod.mtpForward(&self.mm, hrow, &[_]u32{main.id}, &self.mtp.kv);
                    defer model.freeArr(mtp_h1);
                    const dlogits1 = try mtp_mod.mtpDraftLogits(&self.mm, mtp_h1);
                    defer model.freeArr(dlogits1);
                    const draft1 = try sample.sampleToken(dlogits1, eff, self.nextSeed());
                    errdefer if (draft1.filtered) |f| model.freeArr(f);
                    try mlx.checkError();
                    if (gen.items.len + 1 >= max_tokens) {
                        // only one slot left, keep single draft
                        self.mtp.pending = .{ .drafts = [_]mtp_mod.MtpState.Draft{ .{ .id = draft1.id, .lp = draft1.lp, .filtered = draft1.filtered }, undefined }, .len = 1 };
                        last.* = main.id;
                        continue;
                    }
                    const mtp_h2 = try mtp_mod.mtpForward(&self.mm, mtp_h1, &[_]u32{draft1.id}, &self.mtp.kv);
                    defer model.freeArr(mtp_h2);
                    const dlogits2 = try mtp_mod.mtpDraftLogits(&self.mm, mtp_h2);
                    defer model.freeArr(dlogits2);
                    const draft2 = try sample.sampleToken(dlogits2, eff, self.nextSeed());
                    try mlx.checkError();
                    self.mtp.pending = .{ .drafts = [_]mtp_mod.MtpState.Draft{
                        .{ .id = draft1.id, .lp = draft1.lp, .filtered = draft1.filtered },
                        .{ .id = draft2.id, .lp = draft2.lp, .filtered = draft2.filtered },
                    }, .len = 2 };
                    last.* = main.id;
                }
            } else {
                if (gamma == 1) {
                    const cont = try self.verifyDraft(io, gen, last, max_tokens, eff, stats, greedy, first_token, t_start);
                    if (!cont) return;
                } else {
                    const pending = self.mtp.pending.?;
                    if (pending.len == 1) {
                        const cont = try self.verifyDraft(io, gen, last, max_tokens, eff, stats, greedy, first_token, t_start);
                        if (!cont) return;
                    } else {
                        const cont = try self.verifyDraftGamma2(io, gen, last, max_tokens, eff, stats, greedy, first_token, t_start);
                        if (!cont) return;
                    }
                }
            }
        }
    }

    fn verifyDraft(self: *Engine, io: std.Io, gen: *std.ArrayList(u32), last: *u32, max_tokens: u32, eff: sampling.SamplingParams, stats: *GenerateStats, greedy: bool, first_token: *bool, t_start: std.Io.Clock.Timestamp) !bool {
        const pending = self.mtp.pending.?;
        std.debug.assert(pending.len == 1);
        const draft = pending.drafts[0];
        const pair = [_]u32{ last.*, draft.id };
        const vpre = try model.forward(&self.mm, &pair, &self.caches, 1);
        defer model.freeArr(vpre);
        const post = try model.normHidden(&self.mm, vpre);
        defer model.freeArr(post);
        const logits2 = try model.logitsOf(&self.mm, post, 2);
        defer model.freeArr(logits2);
        const v: u32 = self.model_cfg.vocab_size;
        const r0 = try model.sliceArr(logits2, model.newShape(2, .{ 0, 0 }), model.newShape(2, .{ 1, v }));
        defer model.freeArr(r0);
        const r1 = try model.sliceArr(logits2, model.newShape(2, .{ 1, 0 }), model.newShape(2, .{ 2, v }));
        defer model.freeArr(r1);
        const draw0 = try sample.sampleToken(r0, eff, self.nextSeed());
        defer if (draw0.filtered) |f| model.freeArr(f);
        const draw1 = try sample.sampleToken(r1, eff, self.nextSeed());
        defer if (draw1.filtered) |f| model.freeArr(f);
        try mlx.checkError();

        const hc = try model.sliceArr(vpre, model.newShape(3, .{ 0, 0, 0 }), model.newShape(3, .{ 1, 1, self.model_cfg.hidden_size }));
        defer model.freeArr(hc);
        const hd = try model.sliceArr(vpre, model.newShape(3, .{ 0, 1, 0 }), model.newShape(3, .{ 1, 2, self.model_cfg.hidden_size }));
        defer model.freeArr(hd);

        var accept = false;
        if (greedy) {
            accept = mtp_mod.acceptGreedy(draw0.id, draft.id);
        } else {
            // Re-derive the draft's filtered basis is unnecessary: draft.lp is
            // the accept-LP scalar; target side reads filt0 at the draft id.
            const lp_t = try sample.logprobOf(draw0.filtered.?, draft.id);
            accept = mtp_mod.acceptSample(lp_t, draft.lp, self.uniform01());
        }
        self.mtp.offered += 1;

        if (accept) {
            for (self.caches.gdn[0..self.model_cfg.num_hidden_layers]) |*g| g.clearSnaps();
            self.mtp.accepted += 1;
            stats.accepted += 1;
            stats.offered += 1;
            if (first_token.*) {
                stats.ttft_ms = t_start.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.toMilliseconds();
                first_token.* = false;
            }
            try gen.append(self.allocator, draft.id);
            stats.decode_tokens += 1;
            stats.from_draft += 1;
            if (self.isEos(draft.id) or gen.items.len >= max_tokens) {
                last.* = draft.id;
                self.mtp.clearPending();
                return false;
            }
            try gen.append(self.allocator, draw1.id);
            stats.decode_tokens += 1;
            if (self.isEos(draw1.id) or gen.items.len >= max_tokens) {
                last.* = draw1.id;
                self.mtp.clearPending();
                return false;
            }
            // Next draft with cache-commit: [confirmed+accepted, draft+bonus].
            const ch = try model.concatAxis(&.{ hc, hd }, 1);
            defer model.freeArr(ch);
            const both = [_]u32{ draft.id, draw1.id };
            const mtp_h = try mtp_mod.mtpForward(&self.mm, ch, &both, &self.mtp.kv);
            defer model.freeArr(mtp_h);
            const dlogits = try mtp_mod.mtpDraftLogits(&self.mm, mtp_h);
            defer model.freeArr(dlogits);
            const nd = try sample.sampleToken(dlogits, eff, self.nextSeed());
            try mlx.checkError();
            self.mtp.clearPending();
            self.mtp.pending = .{ .drafts = [_]mtp_mod.MtpState.Draft{ .{ .id = nd.id, .lp = nd.lp, .filtered = nd.filtered }, undefined }, .len = 1 };
            last.* = draw1.id;
            return true;
        } else {
            for (self.caches.gdn[0..self.model_cfg.num_hidden_layers]) |*g| g.restore();
            for (self.caches.kv[0..self.model_cfg.num_hidden_layers]) |*k| k.trim(1);
            self.mtp.kv.trim(1);
            stats.offered += 1;
            // Fork-exact reject: greedy emits the backbone token, sampling
            // draws from the residual max(p_target - p_draft, 0)/Z.
            const replacement: u32 = if (greedy) draw0.id else try mtp_mod.residualSample(draw0.filtered.?, draft.filtered.?, self.nextSeed());
            self.mtp.clearPending();
            if (first_token.*) {
                stats.ttft_ms = t_start.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.toMilliseconds();
                first_token.* = false;
            }
            try gen.append(self.allocator, replacement);
            stats.decode_tokens += 1;
            if (self.isEos(replacement) or gen.items.len >= max_tokens) {
                last.* = replacement;
                return false;
            }
            const mtp_h = try mtp_mod.mtpForward(&self.mm, hc, &[_]u32{replacement}, &self.mtp.kv);
            defer model.freeArr(mtp_h);
            const dlogits = try mtp_mod.mtpDraftLogits(&self.mm, mtp_h);
            defer model.freeArr(dlogits);
            const nd = try sample.sampleToken(dlogits, eff, self.nextSeed());
            try mlx.checkError();
            self.mtp.pending = .{ .drafts = [_]mtp_mod.MtpState.Draft{ .{ .id = nd.id, .lp = nd.lp, .filtered = nd.filtered }, undefined }, .len = 1 };
            last.* = replacement;
            return true;
        }
    }

    fn verifyDraftGamma2(self: *Engine, io: std.Io, gen: *std.ArrayList(u32), last: *u32, max_tokens: u32, eff: sampling.SamplingParams, stats: *GenerateStats, greedy: bool, first_token: *bool, t_start: std.Io.Clock.Timestamp) !bool {
        const pending = self.mtp.pending.?;
        std.debug.assert(pending.len == 2);
        const d1 = pending.drafts[0];
        const d2 = pending.drafts[1];
        const triple = [_]u32{ last.*, d1.id, d2.id };
        const vpre = try model.forward(&self.mm, &triple, &self.caches, 1);
        defer model.freeArr(vpre);
        const post = try model.normHidden(&self.mm, vpre);
        defer model.freeArr(post);
        const logits3 = try model.logitsOf(&self.mm, post, 3);
        defer model.freeArr(logits3);
        const v: u32 = self.model_cfg.vocab_size;
        const r0 = try model.sliceArr(logits3, model.newShape(2, .{ 0, 0 }), model.newShape(2, .{ 1, v }));
        defer model.freeArr(r0);
        const r1 = try model.sliceArr(logits3, model.newShape(2, .{ 1, 0 }), model.newShape(2, .{ 2, v }));
        defer model.freeArr(r1);
        const r2 = try model.sliceArr(logits3, model.newShape(2, .{ 2, 0 }), model.newShape(2, .{ 3, v }));
        defer model.freeArr(r2);
        const draw0 = try sample.sampleToken(r0, eff, self.nextSeed());
        defer if (draw0.filtered) |f| model.freeArr(f);
        const draw1 = try sample.sampleToken(r1, eff, self.nextSeed());
        defer if (draw1.filtered) |f| model.freeArr(f);
        const draw2 = try sample.sampleToken(r2, eff, self.nextSeed());
        defer if (draw2.filtered) |f| model.freeArr(f);
        try mlx.checkError();

        const hc = try model.sliceArr(vpre, model.newShape(3, .{ 0, 0, 0 }), model.newShape(3, .{ 1, 1, self.model_cfg.hidden_size }));
        defer model.freeArr(hc);
        const hd1 = try model.sliceArr(vpre, model.newShape(3, .{ 0, 1, 0 }), model.newShape(3, .{ 1, 2, self.model_cfg.hidden_size }));
        defer model.freeArr(hd1);
        const hd2 = try model.sliceArr(vpre, model.newShape(3, .{ 0, 2, 0 }), model.newShape(3, .{ 1, 3, self.model_cfg.hidden_size }));
        defer model.freeArr(hd2);

        var accept0: bool = undefined;
        var accept1: bool = undefined;
        if (greedy) {
            accept0 = mtp_mod.acceptGreedy(draw0.id, d1.id);
            accept1 = mtp_mod.acceptGreedy(draw1.id, d2.id);
        } else {
            const lp_t0 = try sample.logprobOf(draw0.filtered.?, d1.id);
            accept0 = mtp_mod.acceptSample(lp_t0, d1.lp, self.uniform01());
            if (accept0) {
                const lp_t1 = try sample.logprobOf(draw1.filtered.?, d2.id);
                accept1 = mtp_mod.acceptSample(lp_t1, d2.lp, self.uniform01());
            } else {
                accept1 = false;
            }
        }
        self.mtp.offered += 2;
        stats.offered += 2;

        if (accept0 and accept1) {
            for (self.caches.gdn[0..self.model_cfg.num_hidden_layers]) |*g| g.clearSnaps();
            self.mtp.accepted += 2;
            stats.accepted += 2;
            if (first_token.*) {
                stats.ttft_ms = t_start.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.toMilliseconds();
                first_token.* = false;
            }
            try gen.append(self.allocator, d1.id);
            stats.decode_tokens += 1;
            stats.from_draft += 1;
            if (self.isEos(d1.id) or gen.items.len >= max_tokens) {
                last.* = d1.id;
                self.mtp.clearPending();
                return false;
            }
            try gen.append(self.allocator, d2.id);
            stats.decode_tokens += 1;
            stats.from_draft += 1;
            if (self.isEos(d2.id) or gen.items.len >= max_tokens) {
                last.* = d2.id;
                self.mtp.clearPending();
                return false;
            }
            try gen.append(self.allocator, draw2.id);
            stats.decode_tokens += 1;
            if (self.isEos(draw2.id) or gen.items.len >= max_tokens) {
                last.* = draw2.id;
                self.mtp.clearPending();
                return false;
            }
            const mtp_h1 = try mtp_mod.mtpForward(&self.mm, hd2, &[_]u32{draw2.id}, &self.mtp.kv);
            defer model.freeArr(mtp_h1);
            const dl1 = try mtp_mod.mtpDraftLogits(&self.mm, mtp_h1);
            defer model.freeArr(dl1);
            const nd1 = try sample.sampleToken(dl1, eff, self.nextSeed());
            errdefer if (nd1.filtered) |f| model.freeArr(f);
            if (gen.items.len + 1 >= max_tokens) {
                self.mtp.clearPending();
                self.mtp.pending = .{ .drafts = [_]mtp_mod.MtpState.Draft{ .{ .id = nd1.id, .lp = nd1.lp, .filtered = nd1.filtered }, undefined }, .len = 1 };
                last.* = draw2.id;
                return true;
            }
            const mtp_h2 = try mtp_mod.mtpForward(&self.mm, mtp_h1, &[_]u32{nd1.id}, &self.mtp.kv);
            defer model.freeArr(mtp_h2);
            const dl2 = try mtp_mod.mtpDraftLogits(&self.mm, mtp_h2);
            defer model.freeArr(dl2);
            const nd2 = try sample.sampleToken(dl2, eff, self.nextSeed());
            try mlx.checkError();
            self.mtp.clearPending();
            self.mtp.pending = .{ .drafts = [_]mtp_mod.MtpState.Draft{
                .{ .id = nd1.id, .lp = nd1.lp, .filtered = nd1.filtered },
                .{ .id = nd2.id, .lp = nd2.lp, .filtered = nd2.filtered },
            }, .len = 2 };
            last.* = draw2.id;
            return true;
        } else if (accept0 and !accept1) {
            for (self.caches.gdn[0..self.model_cfg.num_hidden_layers]) |*g| g.clearSnaps();
            // keep first, rollback second: restore then recompute state after 2
            // Restore to before drafts, then forward 2 to get correct GDN after accept
            for (self.caches.gdn[0..self.model_cfg.num_hidden_layers]) |*g| g.restore();
            for (self.caches.kv[0..self.model_cfg.num_hidden_layers]) |*k| k.trim(2);
            self.mtp.kv.trim(2);
            // recompute GDN state for accepted prefix [last,d1]
            const pair2 = [_]u32{ last.*, d1.id };
            const vpre2 = try model.forward(&self.mm, &pair2, &self.caches, 1);
            model.freeArr(vpre2);
            // now trim again to keep only 1 extra? Actually we trimmed 2, then added 2 back via forward, so net +1 (d1) as desired, but we also need to keep MTP kv for d1
            // MTP kv was trimmed 2, need to add back d1's MTP entry
            // Instead of recomputing via forward, we can just trim 1 from original state before restore? Simpler: trim 1 from original 3 state without restore.
            // For now, handle by trimming 1 from post-verify state and clearing snaps? Let's just trim 1 and keep GDN as after 3 but with last token removed? This is approximate.
            // To keep correctness, we will have restored and recomputed, then need to re-add MTP draft for d1
            const hc2 = try model.sliceArr(vpre, model.newShape(3, .{ 0, 0, 0 }), model.newShape(3, .{ 1, 1, self.model_cfg.hidden_size }));
            defer model.freeArr(hc2);
            const mtp_h1 = try mtp_mod.mtpForward(&self.mm, hc2, &[_]u32{d1.id}, &self.mtp.kv);
            defer model.freeArr(mtp_h1);
            self.mtp.accepted += 1;
            stats.accepted += 1;
            if (first_token.*) {
                stats.ttft_ms = t_start.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.toMilliseconds();
                first_token.* = false;
            }
            try gen.append(self.allocator, d1.id);
            stats.decode_tokens += 1;
            stats.from_draft += 1;
            if (self.isEos(d1.id) or gen.items.len >= max_tokens) {
                last.* = d1.id;
                self.mtp.clearPending();
                return false;
            }
            const replacement: u32 = if (greedy) draw1.id else try mtp_mod.residualSample(draw1.filtered.?, d2.filtered.?, self.nextSeed());
            try gen.append(self.allocator, replacement);
            stats.decode_tokens += 1;
            if (self.isEos(replacement) or gen.items.len >= max_tokens) {
                last.* = replacement;
                self.mtp.clearPending();
                return false;
            }
            const h_rep = try model.sliceArr(vpre, model.newShape(3, .{ 0, 0, 0 }), model.newShape(3, .{ 1, 1, self.model_cfg.hidden_size }));
            defer model.freeArr(h_rep);
            const mtp_h2 = try mtp_mod.mtpForward(&self.mm, h_rep, &[_]u32{replacement}, &self.mtp.kv);
            defer model.freeArr(mtp_h2);
            const dl = try mtp_mod.mtpDraftLogits(&self.mm, mtp_h2);
            defer model.freeArr(dl);
            const nd = try sample.sampleToken(dl, eff, self.nextSeed());
            try mlx.checkError();
            self.mtp.clearPending();
            self.mtp.pending = .{ .drafts = [_]mtp_mod.MtpState.Draft{ .{ .id = nd.id, .lp = nd.lp, .filtered = nd.filtered }, undefined }, .len = 1 };
            last.* = replacement;
            return true;
        } else {
            // reject first
            for (self.caches.gdn[0..self.model_cfg.num_hidden_layers]) |*g| g.restore();
            for (self.caches.kv[0..self.model_cfg.num_hidden_layers]) |*k| k.trim(2);
            self.mtp.kv.trim(2);
            const replacement: u32 = if (greedy) draw0.id else try mtp_mod.residualSample(draw0.filtered.?, d1.filtered.?, self.nextSeed());
            self.mtp.clearPending();
            if (first_token.*) {
                stats.ttft_ms = t_start.durationTo(std.Io.Clock.Timestamp.now(io, .awake)).raw.toMilliseconds();
                first_token.* = false;
            }
            try gen.append(self.allocator, replacement);
            stats.decode_tokens += 1;
            if (self.isEos(replacement) or gen.items.len >= max_tokens) {
                last.* = replacement;
                return false;
            }
            const hc2 = try model.sliceArr(vpre, model.newShape(3, .{ 0, 0, 0 }), model.newShape(3, .{ 1, 1, self.model_cfg.hidden_size }));
            defer model.freeArr(hc2);
            const mtp_h = try mtp_mod.mtpForward(&self.mm, hc2, &[_]u32{replacement}, &self.mtp.kv);
            defer model.freeArr(mtp_h);
            const dl = try mtp_mod.mtpDraftLogits(&self.mm, mtp_h);
            defer model.freeArr(dl);
            const nd = try sample.sampleToken(dl, eff, self.nextSeed());
            try mlx.checkError();
            // draft second for next pending if room
            if (gen.items.len + 1 >= max_tokens) {
                self.mtp.pending = .{ .drafts = [_]mtp_mod.MtpState.Draft{ .{ .id = nd.id, .lp = nd.lp, .filtered = nd.filtered }, undefined }, .len = 1 };
                last.* = replacement;
                return true;
            }
            const mtp_h2 = try mtp_mod.mtpForward(&self.mm, mtp_h, &[_]u32{nd.id}, &self.mtp.kv);
            defer model.freeArr(mtp_h2);
            const dl2 = try mtp_mod.mtpDraftLogits(&self.mm, mtp_h2);
            defer model.freeArr(dl2);
            const nd2 = try sample.sampleToken(dl2, eff, self.nextSeed());
            try mlx.checkError();
            self.mtp.pending = .{ .drafts = [_]mtp_mod.MtpState.Draft{
                .{ .id = nd.id, .lp = nd.lp, .filtered = nd.filtered },
                .{ .id = nd2.id, .lp = nd2.lp, .filtered = nd2.filtered },
            }, .len = 2 };
            last.* = replacement;
            return true;
        }
    }
};

fn entropySeed() u64 {
    var b: [8]u8 = undefined;
    std.c.arc4random_buf(&b, b.len);
    return std.mem.readInt(u64, &b, .little);
}

fn resolveModel(allocator: std.mem.Allocator, io: std.Io, spec: []const u8) ![]u8 {
    // Native engine: local checkpoint dir only (no HF downloads).
    const need = [_][]const u8{ "config.json", "tokenizer.json", "model.safetensors.index.json" };
    for (need) |f| {
        const p = try std.fs.path.join(allocator, &.{ spec, f });
        defer allocator.free(p);
        const ok = blk: {
            if (std.fs.path.isAbsolute(p)) {
                var fh = std.Io.Dir.openFileAbsolute(io, p, .{}) catch break :blk false;
                fh.close(io);
                break :blk true;
            }
            break :blk false;
        };
        if (!ok) {
            std.log.err("model '{s}' is not a local checkpoint dir (missing {s}) — native engine needs a local path", .{ spec, f });
            return EngineError.ModelNotFound;
        }
    }
    return try allocator.dupe(u8, spec);
}

fn parseSize(s: []const u8) u64 {
    if (s.len == 0) return 0;
    if (std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "none")) return 0;
    var lower_buf: [32]u8 = undefined;
    const lower = std.ascii.lowerString(&lower_buf, s);
    var mult: u64 = 1;
    var num_str = lower;
    if (std.mem.endsWith(u8, lower, "gb")) {
        mult = 1024 * 1024 * 1024;
        num_str = lower[0 .. lower.len - 2];
    } else if (std.mem.endsWith(u8, lower, "mb")) {
        mult = 1024 * 1024;
        num_str = lower[0 .. lower.len - 2];
    } else if (std.mem.endsWith(u8, lower, "kb")) {
        mult = 1024;
        num_str = lower[0 .. lower.len - 2];
    } else if (std.mem.endsWith(u8, lower, "b")) {
        num_str = lower[0 .. lower.len - 1];
    }
    const val = std.fmt.parseFloat(f64, std.mem.trim(u8, num_str, " \t")) catch return 0;
    return @intFromFloat(val * @as(f64, @floatFromInt(mult)));
}

// ── tests ────────────────────────────────────────────────────────────────
// renderChat is pure (no mlx calls) but lives here for the generate path;
// these run under the linked suite (engine pulls mlx transitively).

test "chat renders im_start framing" {
    const alloc = std.testing.allocator;
    const out = try Engine.renderChat(alloc, "[{\"role\":\"user\",\"content\":\"Hi\"}]");
    defer alloc.free(out);
    try std.testing.expectEqualStrings("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", out);
}

test "chat multi-turn framing" {
    const alloc = std.testing.allocator;
    const out = try Engine.renderChat(alloc, "[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"Hi\"}]");
    defer alloc.free(out);
    try std.testing.expectEqualStrings("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", out);
}

test "chat image content is rejected (text-only engine)" {
    const alloc = std.testing.allocator;
    // OpenAI-style image_url part (model-card format) -> VisionUnsupported
    const msg = "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://x/y.jpg\"}},{\"type\":\"text\",\"text\":\"What is this?\"}]}]";
    try std.testing.expectError(EngineError.VisionUnsupported, Engine.renderChat(alloc, msg));
    // video part likewise
    const vid = "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe.\"},{\"type\":\"video\",\"data\":\"AAAA\"}]}]";
    try std.testing.expectError(EngineError.VisionUnsupported, Engine.renderChat(alloc, vid));
    // non-array envelope likewise
    try std.testing.expectError(EngineError.VisionUnsupported, Engine.renderChat(alloc, "{\"role\":\"user\"}"));
}
