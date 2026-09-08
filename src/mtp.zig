//! Native MTP speculative decoding (γ=1), mirror of the feat/mtp-native fork:
//! MTPDecoderLayer (full attention only, own KVCache), MTPModule
//! (`pre_fc_norm_embedding(embed(t+1))` + `pre_fc_norm_hidden(h_t)` concat →
//! `fc` (2H→H) → 1× decoder → `norm` → shared `lm_head`), `mtp_forward` over
//! N positions, greedy accept, Leviathan sampling accept, residual sampling.
//!
//! One deliberate deviation: on reject the fork keeps the stale draft
//! position in the MTP KV cache; we `trim(1)` it so every MTP position stays
//! aligned to an accepted (hidden, token) pair. Accept math is unaffected.
//!
//! The token loop itself lives in the engine (EOS, budgets, streaming);
//! this file owns the head, the verify math, and the per-sequence MTP state.

const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const sample = @import("sample.zig");
const weights = @import("weights.zig");
const stream = model.stream;
const freeArr = model.freeArr;
const check = model.check;

pub const MtpState = struct {
    kv: model.KVCache = .{},
    pending: ?Pending = null,
    offered: u64 = 0,
    accepted: u64 = 0,

    /// A pending draft owns its filtered accept-LP basis (for residual
    /// sampling on reject); null on the greedy path. Freed on consume.
    pub const Draft = struct { id: u32, lp: f32, filtered: ?mlx.mlx_array };
    pub const Pending = struct { drafts: [2]Draft, len: u32 = 1 };

    pub fn deinit(self: *MtpState) void {
        self.clearPending();
        self.kv.deinit();
    }

    pub fn reset(self: *MtpState) void {
        self.clearPending();
        self.kv.reset();
        self.offered = 0;
        self.accepted = 0;
    }

    pub fn clearPending(self: *MtpState) void {
        if (self.pending) |p| {
            for (p.drafts[0..p.len]) |d| {
                if (d.filtered) |f| model.freeArr(f);
            }
        }
        self.pending = null;
    }

    pub fn acceptRate(self: *const MtpState) f32 {
        if (self.offered == 0) return 0;
        return @as(f32, @floatFromInt(self.accepted)) / @as(f32, @floatFromInt(self.offered));
    }
};

fn mtpW(m: *const model.Model, comptime suffix: []const u8) !mlx.mlx_array {
    var kb: [160]u8 = undefined;
    const k = try std.fmt.bufPrint(&kb, "language_model.mtp.{s}", .{suffix});
    return m.wm.get(k) orelse model.ModelError.MissingWeight;
}

fn mtpLinearQ(m: *const model.Model, x: mlx.mlx_array, comptime suffix: []const u8) !mlx.mlx_array {
    var kb: [160]u8 = undefined;
    const k = try std.fmt.bufPrint(&kb, "language_model.mtp.{s}", .{suffix});
    return model.namedLinearQ(m, x, k);
}

fn mtpAttention(m: *const model.Model, x: mlx.mlx_array, cache: *model.KVCache) !mlx.mlx_array {
    const cfg = m.cfg;
    const shape = mlx.getShape(x);
    const b: u32 = @intCast(shape[0]);
    const t: u32 = @intCast(shape[1]);
    const nh = cfg.num_attention_heads;
    const nkv = cfg.num_key_value_heads;
    const hd = cfg.head_dim;

    const wqn = try mtpW(m, "layers.0.self_attn.q_norm.weight");
    const wkn = try mtpW(m, "layers.0.self_attn.k_norm.weight");

    const q2 = try mtpLinearQ(m, x, "layers.0.self_attn.q_proj.weight");
    defer freeArr(q2);
    const qrsh = try model.reshape(q2, model.newShape(4, .{ b, t, nh, 2 * hd }));
    defer freeArr(qrsh);
    var halves: [2]mlx.mlx_array = undefined;
    try model.splitEqual(qrsh, 2, 3, &halves);
    defer freeArr(halves[0]);
    defer freeArr(halves[1]);
    const gate = try model.reshape(halves[1], model.newShape(3, .{ b, t, nh * hd }));
    defer freeArr(gate);

    const kl = try mtpLinearQ(m, x, "layers.0.self_attn.k_proj.weight");
    defer freeArr(kl);
    const kr = try model.reshape(kl, model.newShape(4, .{ b, t, nkv, hd }));
    defer freeArr(kr);
    const vl = try mtpLinearQ(m, x, "layers.0.self_attn.v_proj.weight");
    defer freeArr(vl);
    const vr = try model.reshape(vl, model.newShape(4, .{ b, t, nkv, hd }));
    defer freeArr(vr);
    const qn = try model.rmsNorm(halves[0], wqn, cfg.rms_norm_eps);
    defer freeArr(qn);
    const kn = try model.rmsNorm(kr, wkn, cfg.rms_norm_eps);
    defer freeArr(kn);

    const ax = model.newShape(4, .{ 0, 2, 1, 3 });
    const qt = try model.transposeAxes(qn, ax);
    defer freeArr(qt);
    const kt = try model.transposeAxes(kn, ax);
    defer freeArr(kt);
    const vt = try model.transposeAxes(vr, ax);
    defer freeArr(vt);

    const rqt = try model.rope(qt, cfg.rope_dims, cfg.rope_theta, cache.len);
    defer freeArr(rqt);
    const rkt = try model.rope(kt, cfg.rope_dims, cfg.rope_theta, cache.len);
    defer freeArr(rkt);
    const kv = try cache.update(rkt, vt, nkv, hd);

    const sdt = try model.sdpa(rqt, kv.kk, kv.vv, t, 1.0 / @sqrt(@as(f32, @floatFromInt(hd))));
    defer freeArr(sdt);
    const back = try model.transposeAxes(sdt, ax);
    defer freeArr(back);
    const flat = try model.reshape(back, model.newShape(3, .{ b, t, nh * hd }));
    defer freeArr(flat);
    const sg = try model.sigmoid(gate);
    defer freeArr(sg);
    const gated = try model.mul(sg, flat);
    defer freeArr(gated);
    return mtpLinearQ(m, gated, "layers.0.self_attn.o_proj.weight");
}
/// MTP head over N positions: fused = fc([norm(emb(ids)), norm(hiddenPre)])
/// `mtp_use_dedicated_embeddings=false`, so drafts embed through the backbone table.
pub fn mtpForward(m: *const model.Model, hidden_pre: mlx.mlx_array, ids: []const u32, cache: *model.KVCache) !mlx.mlx_array {
    const cfg = m.cfg;
    const hs = mlx.getShape(hidden_pre);
    const t: u32 = @intCast(hs[1]);
    const hdim = cfg.hidden_size;
    std.debug.assert(ids.len == t);

    const tsh = model.newShape(1, .{t});
    const idxarr = mlx.mlx_array_new_data(ids.ptr, &tsh, 1, .uint32);
    defer freeArr(idxarr);
    const emb = try model.takeAxis(m.w_embed, idxarr);
    defer freeArr(emb);
    const embeds = try model.reshape(emb, model.newShape(3, .{ 1, t, hdim }));
    defer freeArr(embeds);

    const e = try model.rmsNorm(embeds, try mtpW(m, "pre_fc_norm_embedding.weight"), cfg.rms_norm_eps);
    defer freeArr(e);
    const h = try model.rmsNorm(hidden_pre, try mtpW(m, "pre_fc_norm_hidden.weight"), cfg.rms_norm_eps);
    defer freeArr(h);
    const cat = try model.concatAxis(&.{ e, h }, 2);
    defer freeArr(cat);
    const fused = try mtpLinearQ(m, cat, "fc.weight");
    defer freeArr(fused);

    const ln1 = try model.rmsNorm(fused, try mtpW(m, "layers.0.input_layernorm.weight"), cfg.rms_norm_eps);
    const r = try mtpAttention(m, ln1, cache);
    defer freeArr(r);
    const h2 = try model.add(fused, r);
    defer freeArr(h2);
    const ln2 = try model.rmsNorm(h2, try mtpW(m, "layers.0.post_attention_layernorm.weight"), cfg.rms_norm_eps);
    defer freeArr(ln2);
    const mp = try model.mlpForwardQ(
        m,
        ln2,
        "language_model.mtp.layers.0.mlp.gate_proj.weight",
        "language_model.mtp.layers.0.mlp.down_proj.weight",
        "language_model.mtp.layers.0.mlp.up_proj.weight",
    );
    defer freeArr(mp);
    const out = try model.add(h2, mp);
    defer freeArr(out);
    return model.rmsNorm(out, try mtpW(m, "norm.weight"), cfg.rms_norm_eps);
}

/// MTP draft logits: head output for the last position only -> [V] (owned).
pub fn mtpDraftLogits(m: *const model.Model, mtp_hidden: mlx.mlx_array) !mlx.mlx_array {
    const hs = mlx.getShape(mtp_hidden);
    const t: u32 = @intCast(hs[1]);
    const hdim: u32 = @intCast(hs[2]);
    const row = try model.sliceArr(mtp_hidden, model.newShape(3, .{ 0, t - 1, 0 }), model.newShape(3, .{ 1, t, hdim }));
    defer freeArr(row);
    const flat = try model.reshape(row, model.newShape(2, .{ 1, hdim }));
    defer freeArr(flat);
    return model.namedLinearQ(m, flat, "language_model.lm_head.weight");
}

pub fn acceptGreedy(pred: u32, draft: u32) bool {
    return pred == draft;
}

/// Leviathan sampling accept: u < min(1, p_target/p_draft), in log space.
/// `u` must be uniform in [0,1).
pub fn acceptSample(lp_target_draft: f32, lp_draft: f32, u: f32) bool {
    const log_accept = lp_target_draft - lp_draft;
    return log_accept >= 0 or u < std.math.exp(log_accept);
}

/// Fork residual sampling on reject (temp>0): draw from
/// max(p_target - p_draft, 0)/Z (falls back to p_target when Z == 0).
/// Inputs are the temp-normalized accept-LP vectors.
pub fn residualSample(accept_t: mlx.mlx_array, accept_d: mlx.mlx_array, seed: u64) !u32 {
    const s = stream();
    const pt = try model.expArr(accept_t);
    defer freeArr(pt);
    const pd = try model.expArr(accept_d);
    defer freeArr(pd);
    const diff = try model.sub(pt, pd);
    defer freeArr(diff);
    const n = mlx.mlx_array_size(diff);
    const zsh = [_]c_int{@intCast(n)};
    var zeros = mlx.mlx_array_new();
    defer freeArr(zeros);
    try check(mlx.mlx_zeros(&zeros, &zsh, zsh.len, .float32, s));
    var res = mlx.mlx_array_new();
    defer freeArr(res);
    try check(mlx.mlx_maximum(&res, diff, zeros, s));
    const z = try model.sumAxis(res, 0);
    defer freeArr(z);
    const zv = try sample.readScalarF32(z);
    var dist = mlx.mlx_array_new();
    defer freeArr(dist);
    if (zv > 0) {
        try check(mlx.mlx_divide(&dist, res, z, s));
    } else {
        try check(mlx.mlx_array_set(&dist, pt));
    }
    var ld = mlx.mlx_array_new();
    defer freeArr(ld);
    try check(mlx.mlx_log(&ld, dist, s));
    var key = mlx.mlx_array_new();
    defer freeArr(key);
    try check(mlx.mlx_random_key(&key, seed));
    var drawn = mlx.mlx_array_new();
    defer freeArr(drawn);
    try check(mlx.mlx_random_categorical(&drawn, ld, 0, key, s));
    try check(mlx.mlx_array_eval(drawn));
    var ival: i32 = 0;
    try check(mlx.mlx_array_item_int32(&ival, drawn));
    return @intCast(ival);
}

// ── tests (linked; run under `zig build test-mlx`) ───────────────────────

test "greedy accept is exact id match" {
    try std.testing.expect(acceptGreedy(7, 7));
    try std.testing.expect(!acceptGreedy(7, 8));
}

test "sampling accept matches Leviathan rule" {
    // log_accept = 0.5 - 1.0 = -0.5 -> accept iff u < e^-0.5 ≈ 0.607
    try std.testing.expect(acceptSample(0.5, 1.0, 0.6));
    try std.testing.expect(!acceptSample(0.5, 1.0, 0.7));
    try std.testing.expect(acceptSample(1.0, 0.5, 0.999));
}

test "residual sampling draws the forced token" {
    const alloc = std.testing.allocator;
    // target puts all mass on 3, draft on 5 -> residual forces 3
    const v: u32 = 16;
    const mk = struct {
        fn f(a: std.mem.Allocator, hot: u32) !mlx.mlx_array {
            const vals = try a.alloc(f32, v);
            defer a.free(vals);
            @memset(vals, -100.0);
            vals[hot] = 0.0;
            const sh = [_]c_int{@intCast(v)};
            return mlx.mlx_array_new_data(vals.ptr, &sh, sh.len, .float32);
        }
    }.f;
    const at = try mk(alloc, 3);
    defer freeArr(at);
    const ad = try mk(alloc, 5);
    defer freeArr(ad);
    // normalize to accept-LPs (log-softmax) like the real path
    const norm = struct {
        fn f(x: mlx.mlx_array) !mlx.mlx_array {
            var lse = mlx.mlx_array_new();
            defer freeArr(lse);
            try check(mlx.mlx_logsumexp_axis(&lse, x, 0, false, stream()));
            // broadcast subtract: [V] - scalar
            var y = mlx.mlx_array_new();
            errdefer freeArr(y);
            try check(mlx.mlx_subtract(&y, x, lse, stream()));
            return y;
        }
    }.f;
    const nt = try norm(at);
    defer freeArr(nt);
    const nd = try norm(ad);
    defer freeArr(nd);
    const id = try residualSample(nt, nd, 42);
    try std.testing.expectEqual(@as(u32, 3), id);
}

test "mtp head forward: shapes, dtype, no NaN, cache carry, trim" {
    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = "/tmp/mlx-runner-tiny-mtp";
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    // MTP head needs only embed + its own 15 tensors (+ norm/head for init).
    var specs: std.ArrayList(model.TinySpec) = .empty;
    defer specs.deinit(alloc);
    for (model.tiny_specs) |sp| {
        if (std.mem.eql(u8, sp.name, "language_model.model.embed_tokens.weight") or
            std.mem.eql(u8, sp.name, "language_model.model.norm.weight") or
            std.mem.eql(u8, sp.name, "language_model.lm_head.weight") or
            std.mem.startsWith(u8, sp.name, "language_model.mtp."))
        {
            try specs.append(alloc, sp);
        }
    }
    try std.testing.expectEqual(@as(usize, 18), specs.items.len);
    try weights.writeFixture(alloc, io, dir, specs.items, 0xABCD);
    var wm = try weights.WeightMap.load(alloc, io, dir);
    var mm = model.Model.init(alloc, model.tiny_cfg, wm) catch |err| {
        wm.deinit();
        return err;
    };
    defer mm.deinit();

    var cache = model.KVCache{};
    defer cache.deinit();

    // random pre-norm hidden [1,2,32] (bf16) as the backbone would emit
    var seed: u64 = 99;
    var hvals: [64]f32 = undefined;
    for (&hvals) |*x| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        x.* = @as(f32, @floatFromInt((seed >> 33) & 0xFF)) / 128.0 - 1.0;
    }
    const hsh = model.newShape(3, .{ 1, 2, 32 });
    const hf = mlx.mlx_array_new_data(&hvals, &hsh, @intCast(hsh.len), .float32);
    defer freeArr(hf);
    const hidden = try model.astype(hf, .bfloat16);
    defer freeArr(hidden);

    const out = try mtpForward(&mm, hidden, &[_]u32{ 7, 8 }, &cache);
    defer freeArr(out);
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ 1, 2, 32 }, mlx.getShape(out));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(out));
    try check(mlx.mlx_array_eval(out));
    const n = mlx.mlx_array_size(out);
    const p = mlx.mlx_array_data_bfloat16(out).?;
    for (p[0..n]) |bits| try std.testing.expect(((bits >> 10) & 0x1F) != 0x1F);
    try std.testing.expectEqual(@as(u32, 2), cache.len);

    // T=1 carry then trim (the reject path)
    const hidden1 = try model.sliceArr(hidden, model.newShape(3, .{ 0, 1, 0 }), model.newShape(3, .{ 1, 2, 32 }));
    defer freeArr(hidden1);
    const out2 = try mtpForward(&mm, hidden1, &[_]u32{9}, &cache);
    defer freeArr(out2);
    try std.testing.expectEqual(@as(u32, 3), cache.len);
    cache.trim(1);
    try std.testing.expectEqual(@as(u32, 2), cache.len);

    // draft logits share lm_head: [1,V]
    const dl = try mtpDraftLogits(&mm, out);
    defer freeArr(dl);
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ 1, 64 }, mlx.getShape(dl));
    try mlx.checkError();
}
