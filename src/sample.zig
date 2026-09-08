//! Native sampler: temperature + top-k + top-p + min-p filter chain
//! (the `make_sampler_chain` order minus XTC, which is out of scope),
//! then `mlx_random_categorical` with `mlx_random_key(seed++)`.
//!
//! Idioms mirror the proven mlx-c patterns (mask in place, sample in the
//! original index space): top-k keeps `logits >= min(topk)`, top-p keeps
//! `logits >= min(nucleus)` via ascending sort + cumsum, min-p keeps
//! `probs >= max(probs) * min_p` — all masked to `-inf`, then categorical.
//! `temp == 0` short-circuits to argmax and ignores the seed.
//!
//! Logprobs stay on-device until read: `sampleToken` returns the drawn id
//! plus its logprob under the FILTERED distribution; `logprobOf` reads any
//! id's logprob (the MTP accept step needs the draft id under target logits).

const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const sampling = @import("sampling.zig");

const stream = model.stream;
const freeArr = model.freeArr;
const check = model.check;

/// A draw plus its accept-LP basis. `filtered` is the owned temp-scaled
/// filtered logits (`logprobOf(filtered, id) == lp`); null on the greedy
/// path. The MTP verify step keeps the target position's `filtered` alive
/// to read `lp_target[draft]`; all other callers free it immediately.
pub const Draw = struct { id: u32, lp: f32, filtered: ?mlx.mlx_array };

fn retain(a: mlx.mlx_array) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_array_set(&y, a));
    return y;
}

fn asVec(logits: mlx.mlx_array) !mlx.mlx_array {
    const n = mlx.mlx_array_size(logits);
    const sh = [_]c_int{@intCast(n)};
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_reshape(&y, logits, &sh, sh.len, stream()));
    return y;
}

pub fn readScalarF32(a: mlx.mlx_array) !f32 {
    try check(mlx.mlx_array_eval(a));
    return switch (mlx.mlx_array_dtype(a)) {
        .float32 => mlx.mlx_array_data_float32(a).?[0],
        .bfloat16 => blk: {
            const bits = mlx.mlx_array_data_bfloat16(a).?[0];
            break :blk @bitCast(@as(u32, bits) << 16);
        },
        else => error.UnsupportedDtype,
    };
}

fn slice1(a: mlx.mlx_array, id: u32) !mlx.mlx_array {
    const st = [_]c_int{@intCast(id)};
    const en = [_]c_int{@intCast(id + 1)};
    var strides = [_]c_int{1};
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_slice(&y, a, &st, 1, &en, 1, &strides, 1, stream()));
    return y;
}

/// log-softmax(id) read back as f32.
pub fn logprobOf(logits_v: mlx.mlx_array, id: u32) !f32 {
    const s = stream();
    var lse = mlx.mlx_array_new();
    defer freeArr(lse);
    try check(mlx.mlx_logsumexp_axis(&lse, logits_v, 0, false, s));
    const li = try slice1(logits_v, id);
    defer freeArr(li);
    const a = try readScalarF32(li);
    const b = try readScalarF32(lse);
    return a - b;
}

fn applyTopK(logits_v: mlx.mlx_array, k: u32) !mlx.mlx_array {
    const s = stream();
    var topk = mlx.mlx_array_new();
    defer freeArr(topk);
    try check(mlx.mlx_topk_axis(&topk, logits_v, @intCast(k), 0, s));
    var cutoff = mlx.mlx_array_new();
    defer freeArr(cutoff);
    try check(mlx.mlx_min_axis(&cutoff, topk, 0, true, s));
    var mask = mlx.mlx_array_new();
    defer freeArr(mask);
    try check(mlx.mlx_greater_equal(&mask, logits_v, cutoff, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer freeArr(neg_inf);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_where(&y, mask, logits_v, neg_inf, s));
    return y;
}

fn applyTopP(logits_v: mlx.mlx_array, top_p: f32) !mlx.mlx_array {
    const s = stream();
    var sorted = mlx.mlx_array_new();
    defer freeArr(sorted);
    try check(mlx.mlx_sort_axis(&sorted, logits_v, 0, s));
    var sprob = mlx.mlx_array_new();
    defer freeArr(sprob);
    try check(mlx.mlx_softmax_axis(&sprob, sorted, 0, true, s));
    var cum = mlx.mlx_array_new();
    defer freeArr(cum);
    try check(mlx.mlx_cumsum(&cum, sprob, 0, false, true, s));
    const thresh = mlx.mlx_array_new_float(1.0 - top_p);
    defer freeArr(thresh);
    var in_nucleus = mlx.mlx_array_new();
    defer freeArr(in_nucleus);
    try check(mlx.mlx_greater(&in_nucleus, cum, thresh, s));
    const pos_inf = mlx.mlx_array_new_float(std.math.inf(f32));
    defer freeArr(pos_inf);
    var nuc = mlx.mlx_array_new();
    defer freeArr(nuc);
    try check(mlx.mlx_where(&nuc, in_nucleus, sorted, pos_inf, s));
    var min_val = mlx.mlx_array_new();
    defer freeArr(min_val);
    try check(mlx.mlx_min_axis(&min_val, nuc, 0, true, s));
    var keep = mlx.mlx_array_new();
    defer freeArr(keep);
    try check(mlx.mlx_greater_equal(&keep, logits_v, min_val, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer freeArr(neg_inf);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_where(&y, keep, logits_v, neg_inf, s));
    return y;
}

fn applyMinP(logits_v: mlx.mlx_array, min_p: f32) !mlx.mlx_array {
    const s = stream();
    var probs = mlx.mlx_array_new();
    defer freeArr(probs);
    try check(mlx.mlx_softmax_axis(&probs, logits_v, 0, true, s));
    var mx = mlx.mlx_array_new();
    defer freeArr(mx);
    try check(mlx.mlx_max_axis(&mx, probs, 0, true, s));
    const scaled = mlx.mlx_array_new_float(min_p);
    defer freeArr(scaled);
    var cutoff = mlx.mlx_array_new();
    defer freeArr(cutoff);
    try check(mlx.mlx_multiply(&cutoff, mx, scaled, s));
    var keep = mlx.mlx_array_new();
    defer freeArr(keep);
    try check(mlx.mlx_greater_equal(&keep, probs, cutoff, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer freeArr(neg_inf);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_where(&y, keep, logits_v, neg_inf, s));
    return y;
}

/// Draw one token from [V] (or [1,V]) logits. Returns id + its logprob under
/// the filtered distribution. `seed` builds `mlx_random_key(seed)`; the
/// caller advances it (`--seed` or OS entropy once at startup).
pub fn sampleToken(logits: mlx.mlx_array, params: sampling.SamplingParams, seed: u64) !Draw {
    const s = stream();
    const vec = try asVec(logits);
    defer freeArr(vec);
    const n: u32 = @intCast(mlx.mlx_array_size(vec));

    if (params.temp == 0) {
        var am = mlx.mlx_array_new();
        defer freeArr(am);
        try check(mlx.mlx_argmax_axis(&am, vec, 0, false, s));
        try check(mlx.mlx_array_eval(am));
        const id: u32 = mlx.mlx_array_data_uint32(am).?[0];
        return .{ .id = id, .lp = try logprobOf(vec, id), .filtered = null };
    }

    var cur = vec;
    var cur_owned = false;
    defer if (cur_owned) freeArr(cur);
    if (params.temp != 1.0) {
        const scaled = try model.scaleBy(cur, 1.0 / params.temp);
        cur = scaled;
        cur_owned = true;
    }
    if (params.top_k > 0 and params.top_k < n) {
        const f = try applyTopK(cur, params.top_k);
        if (cur_owned) freeArr(cur);
        cur = f;
        cur_owned = true;
    }
    // Fork parity: top_p == 0 means OFF (not "keep nothing").
    if (params.top_p > 0.0 and params.top_p < 1.0) {
        const f = try applyTopP(cur, params.top_p);
        if (cur_owned) freeArr(cur);
        cur = f;
        cur_owned = true;
    }
    if (params.min_p > 0.0) {
        const f = try applyMinP(cur, params.min_p);
        if (cur_owned) freeArr(cur);
        cur = f;
        cur_owned = true;
    }

    var key = mlx.mlx_array_new();
    defer freeArr(key);
    try check(mlx.mlx_random_key(&key, seed));
    var drawn = mlx.mlx_array_new();
    defer freeArr(drawn);
    try check(mlx.mlx_random_categorical(&drawn, cur, 0, key, s));
    try check(mlx.mlx_array_eval(drawn));
    var ival: i32 = 0;
    try check(mlx.mlx_array_item_int32(&ival, drawn));
    const id: u32 = @intCast(ival);
    const lp = try logprobOf(cur, id);
    // Transfer: the caller owns `filtered` (frees after the accept read).
    const filt = if (cur_owned) cur else try retain(cur);
    cur_owned = false;
    return .{ .id = id, .lp = lp, .filtered = filt };
}

// ── tests (linked; run under `zig build test-mlx`) ───────────────────────

fn testLogits(alloc: std.mem.Allocator, v: u32, peak: u32) !mlx.mlx_array {
    // peaked distribution: peak id has logit 4, rest ramp down
    const vals = try alloc.alloc(f32, v);
    defer alloc.free(vals);
    for (vals, 0..) |*x, i| x.* = 4.0 - @as(f32, @floatFromInt(if (i <= peak) peak - i else i - peak));
    const sh = [_]c_int{@intCast(v)};
    return mlx.mlx_array_new_data(vals.ptr, &sh, sh.len, .float32);
}

test "temp 0 is argmax and ignores seed" {
    const alloc = std.testing.allocator;
    const l = try testLogits(alloc, 64, 7);
    defer freeArr(l);
    const a = try sampleToken(l, .{ .temp = 0 }, 1);
    const b = try sampleToken(l, .{ .temp = 0 }, 999);
    try std.testing.expectEqual(@as(u32, 7), a.id);
    try std.testing.expectEqual(a.id, b.id);
}

test "same seed twice draws the same token" {
    const alloc = std.testing.allocator;
    const l = try testLogits(alloc, 64, 7);
    defer freeArr(l);
    const a = try sampleToken(l, .{ .temp = 0.8, .top_p = 0.95 }, 12345);
    defer if (a.filtered) |f| freeArr(f);
    const b = try sampleToken(l, .{ .temp = 0.8, .top_p = 0.95 }, 12345);
    defer if (b.filtered) |f| freeArr(f);
    try std.testing.expectEqual(a.id, b.id);
    try std.testing.expectApproxEqAbs(a.lp, b.lp, 1e-6);
}

test "top-k 1 equals argmax" {
    const alloc = std.testing.allocator;
    const l = try testLogits(alloc, 64, 7);
    defer freeArr(l);
    const a = try sampleToken(l, .{ .temp = 1.0, .top_k = 1 }, 321);
    defer if (a.filtered) |f| freeArr(f);
    try std.testing.expectEqual(@as(u32, 7), a.id);
}

test "logprobOf uniform is -ln V" {
    const alloc = std.testing.allocator;
    const v: u32 = 32;
    const vals = try alloc.alloc(f32, v);
    defer alloc.free(vals);
    @memset(vals, 0.0);
    const sh = [_]c_int{@intCast(v)};
    const l = mlx.mlx_array_new_data(vals.ptr, &sh, sh.len, .float32);
    defer freeArr(l);
    const lp = try logprobOf(l, 3);
    try std.testing.expectApproxEqAbs(-@log(@as(f32, @floatFromInt(v))), lp, 1e-4);
}
