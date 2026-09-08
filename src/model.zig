//! Native qwen3_5 text forward pass (Qwen3.8-27B, bf16, batch-1).
//!
//! Reference, top to bottom (reread before touching this file):
//! - `mlx_lm/models/qwen3_next.py`: Attention, Qwen3NextMLP, RMSNormGated
//! - `mlx_lm/models/qwen3_5.py`: GDN, DecoderLayer, TextModel, sanitize
//! - `mlx_lm/models/gated_delta.py`: `gated_delta_update` + ops fallback
//! - `mlx_lm/models/base.py`: `create_attention_mask` / SDPA wrapper
//! - `mlx_lm/models/cache.py`: KVCache (step-256 prealloc) / ArraysCache
//!
//! Config (64 layers, hidden 5120, 24/4 heads, head_dim 256, rope 64 @ 1e7,
//! interval 4, intermediate 17408, vocab 248320, tie_word_embeddings=false)
//! is parsed from `config.json`/`text_config` at load; tests use literals.
//!
//! Masks: batch-1, no padding, no window — the mask is always None here.
//! Decode (`T==1`) passes no mask (single query over cache is causal by
//! construction); any forward with `T>1` uses SDPA `causal` mode, which is
//! exactly right because keys are always the full prefix `[:len+T]` and
//! queries its last `T` rows (offset math falls out of `kL-qL`).
//!
//! Logits are NEVER materialized for full prefill: `forward` returns the
//! final hidden states `[1,T,H]` and the engine slices the rows it needs
//! before `lm_head` (a 32K×248320 bf16 matmul would be 16 GB of logits).
//!
//! Ownership: every helper returns an OWNED handle; inputs are BORROWED
//! unless named `*_owned`. Weights and cache contents are borrowed; caches
//! own their storage and the current views.

const std = @import("std");
const mlx = @import("mlx.zig");
const weights_mod = @import("weights.zig");

pub const ModelError = error{
    BadConfig,
    MissingWeight,
    BadWeightShape,
};

// ── config ───────────────────────────────────────────────────────────────

pub const Config = struct {
    num_hidden_layers: u32 = 64,
    hidden_size: u32 = 5120,
    intermediate_size: u32 = 17408,
    num_attention_heads: u32 = 24,
    num_key_value_heads: u32 = 4,
    head_dim: u32 = 256,
    full_attention_interval: u32 = 4,
    rope_dims: u32 = 64,
    rope_theta: f32 = 1e7,
    rms_norm_eps: f32 = 1e-6,
    vocab_size: u32 = 248320,
    tie_word_embeddings: bool = false,
    max_position_embeddings: u32 = 262144,
    // GDN dims
    linear_num_key_heads: u32 = 16,
    linear_num_value_heads: u32 = 48,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,

    pub fn isLinear(self: Config, idx: u32) bool {
        return (idx + 1) % self.full_attention_interval != 0;
    }
    pub fn keyDim(self: Config) u32 {
        return self.linear_num_key_heads * self.linear_key_head_dim;
    }
    pub fn valueDim(self: Config) u32 {
        return self.linear_num_value_heads * self.linear_value_head_dim;
    }
    pub fn convDim(self: Config) u32 {
        return self.keyDim() * 2 + self.valueDim();
    }

    fn u32Field(o: std.json.ObjectMap, name: []const u8) !u32 {
        const v = o.get(name) orelse return ModelError.BadConfig;
        if (v != .integer) return ModelError.BadConfig;
        return @intCast(v.integer);
    }
    fn f32Field(o: std.json.ObjectMap, name: []const u8) !f32 {
        const v = o.get(name) orelse return ModelError.BadConfig;
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| @floatCast(f),
            else => ModelError.BadConfig,
        };
    }

    /// Parse the `text_config` object (already unwrapped from config.json).
    pub fn fromTextConfig(tv: std.json.Value) !Config {
        if (tv != .object) return ModelError.BadConfig;
        const o = tv.object;
        var c = Config{};
        c.hidden_size = try u32Field(o, "hidden_size");
        c.intermediate_size = try u32Field(o, "intermediate_size");
        c.num_hidden_layers = try u32Field(o, "num_hidden_layers");
        c.num_attention_heads = try u32Field(o, "num_attention_heads");
        c.num_key_value_heads = try u32Field(o, "num_key_value_heads");
        c.head_dim = try u32Field(o, "head_dim");
        c.full_attention_interval = try u32Field(o, "full_attention_interval");
        c.rms_norm_eps = try f32Field(o, "rms_norm_eps");
        c.vocab_size = try u32Field(o, "vocab_size");
        c.max_position_embeddings = try u32Field(o, "max_position_embeddings");
        c.linear_num_key_heads = try u32Field(o, "linear_num_key_heads");
        c.linear_num_value_heads = try u32Field(o, "linear_num_value_heads");
        c.linear_key_head_dim = try u32Field(o, "linear_key_head_dim");
        c.linear_value_head_dim = try u32Field(o, "linear_value_head_dim");
        c.linear_conv_kernel_dim = try u32Field(o, "linear_conv_kernel_dim");
        if (o.get("tie_word_embeddings")) |t| {
            if (t != .bool) return ModelError.BadConfig;
            c.tie_word_embeddings = t.bool;
        }
        if (o.get("rope_parameters")) |rp| {
            if (rp != .object) return ModelError.BadConfig;
            const theta = rp.object.get("rope_theta") orelse return ModelError.BadConfig;
            c.rope_theta = switch (theta) {
                .integer => |i| @floatFromInt(i),
                .float => |f| @floatCast(f),
                else => return ModelError.BadConfig,
            };
            const prf = rp.object.get("partial_rotary_factor") orelse return ModelError.BadConfig;
            const factor: f32 = switch (prf) {
                .integer => |i| @floatFromInt(i),
                .float => |f| @floatCast(f),
                else => return ModelError.BadConfig,
            };
            c.rope_dims = @intFromFloat(@as(f32, @floatFromInt(c.head_dim)) * factor);
        }
        return c;
    }

    pub fn load(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !Config {
        const path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
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
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch
            return ModelError.BadConfig;
        defer parsed.deinit();
        if (parsed.value != .object) return ModelError.BadConfig;
        const tc = parsed.value.object.get("text_config") orelse return ModelError.BadConfig;
        return fromTextConfig(tc);
    }
};
pub fn stream() mlx.mlx_stream {
    return mlx.gpuStream();
}

pub fn freeArr(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

pub fn newShape(comptime n: usize, dims: [n]u32) [n]c_int {
    var s: [n]c_int = undefined;
    for (dims, 0..) |d, i| s[i] = @intCast(d);
    return s;
}

pub fn nullArr() mlx.mlx_array {
    return .{ .ctx = null };
}

pub fn check(rc: c_int) !void {
    try mlx.check(rc);
}

pub fn reshape(a: mlx.mlx_array, shape: anytype) !mlx.mlx_array {
    // shape: N-element c_int array (passed by value; no caller temp needed).
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_reshape(&y, a, &shape, shape.len, stream()));
    return y;
}

pub fn transposeAxes(a: mlx.mlx_array, axes: anytype) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_transpose_axes(&y, a, &axes, axes.len, stream()));
    return y;
}

pub fn sliceArr(a: mlx.mlx_array, start: anytype, stop: anytype) !mlx.mlx_array {
    const ST = @typeInfo(@TypeOf(start)).array;
    const SP = @typeInfo(@TypeOf(stop)).array;
    comptime std.debug.assert(ST.len == SP.len);
    comptime std.debug.assert(ST.child == c_int and SP.child == c_int);
    var strides: [ST.len]c_int = undefined;
    @memset(&strides, 1);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_slice(&y, a, &start, ST.len, &stop, SP.len, &strides, ST.len, stream()));
    return y;
}
/// y = x @ W.T for W [out,in] (transpose is a lazy view, no copy).
pub fn linear(x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
    const s = stream();
    var wt = mlx.mlx_array_new();
    errdefer freeArr(wt);
    try check(mlx.mlx_transpose(&wt, w, s));
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_matmul(&y, x, wt, s));
    freeArr(wt);
    return y;
}

/// y = x @ W.T with checkpoint quantization honored: packed U32 weights go
/// through `mlx_quantized_matmul` (transpose, per-map group/bits/mode);
/// anything without `*.scales` falls back to dense `linear` (norms-adjacent
/// weights, mxfp8 `lm_head`). A packed weight missing its scales is
/// `MissingWeight`, never silent garbage.
pub fn namedLinearQ(m: *const Model, x: mlx.mlx_array, name: []const u8) !mlx.mlx_array {
    const w = m.wm.get(name) orelse return ModelError.MissingWeight;
    const q = m.wm.quant;
    if (q.mode == .none) return linear(x, w);
    if (!std.mem.endsWith(u8, name, ".weight")) return ModelError.MissingWeight;
    const prefix = name[0 .. name.len - ".weight".len];
    var sb: [1056]u8 = undefined;
    const sname = try std.fmt.bufPrint(&sb, "{s}.scales", .{prefix});
    const sc = m.wm.get(sname) orelse {
        if (mlx.mlx_array_dtype(w) == .uint32) return ModelError.MissingWeight;
        return linear(x, w);
    };
    // Microscaling modes (mxfp8/nvfp4) are bias-less; affine never reaches
    // here (the loader rejects BF16-scales checkpoints outright).
    return qlinear(x, w, sc, null, q);
}

pub fn layerLinearQ(m: *const Model, x: mlx.mlx_array, idx: u32, rel: []const u8) !mlx.mlx_array {
    var kb: [160]u8 = undefined;
    const name = try std.fmt.bufPrint(&kb, "language_model.model.layers.{d}.{s}", .{ idx, rel });
    return namedLinearQ(m, x, name);
}
pub fn layerName(buf: []u8, idx: u32, rel: []const u8) ![]u8 {
    return std.fmt.bufPrint(buf, "language_model.model.layers.{d}.{s}", .{ idx, rel });
}

fn qlinear(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: ?mlx.mlx_array, q: weights_mod.QuantSpec) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_quantized_matmul(
        &y,
        x,
        w,
        sc,
        bi orelse nullArr(),
        true, // packed [out, in*bits/32], same as nn.QuantizedLinear
        mlx.mlx_optional_int.some(@intCast(q.group_size)),
        mlx.mlx_optional_int.some(@intCast(q.bits)),
        q.modeStr(),
        stream(),
    ));
    return y;
}
/// Packed GDN recurrence kernel (vendored from mlx_lm's gated_delta.py packed
/// path, `T` adapted to a uint32[1] input since mlx-c takes array inputs
/// only). One launch replaces ~20 graph nodes x T positions of tiny f32
/// ops per layer. Handle is compiled once per process and cached.
var gdn_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn gdnKernel() !mlx.mlx_fast_metal_kernel {
    if (gdn_kernel) |k| return k;
    const src = @embedFile("gdn_packed.metal");
    const in_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const inv = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(inv);
    const out_names = [_][*:0]const u8{ "y", "state_out" };
    const outv = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(outv);
    const k = mlx.mlx_fast_metal_kernel_new("gated_delta_step_packed_btree", inv, outv, src, "", true, false);
    gdn_kernel = k;
    return k;
}

const GdnKernelOut = struct { y: mlx.mlx_array, state: mlx.mlx_array };

/// Single fused recurrence over T positions. All inputs borrowed; outputs owned.
fn gdnKernelRun(b: u32, t: u32, hk: u32, hv: u32, dk: u32, dv: u32, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, st: mlx.mlx_array, ydt: mlx.mlx_dtype) !GdnKernelOut {
    const ker = try gdnKernel();
    var tc: u32 = t;
    var tsh = newShape(1, .{1});
    const tarr = mlx.mlx_array_new_data(@ptrCast(&tc), &tsh, 1, .uint32);
    defer freeArr(tarr);
    const inits = [_]mlx.mlx_array{ q, k, v, g, beta, st, tarr };
    const invec = mlx.mlx_vector_array_new_data(&inits, inits.len);
    defer _ = mlx.mlx_vector_array_free(invec);
    const cfgh = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfgh);
    var ysh = newShape(4, .{ b, t, hv, dv });
    try check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfgh, &ysh, ysh.len, ydt));
    var ssh = newShape(4, .{ b, hv, dv, dk });
    try check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfgh, &ssh, ssh.len, .float32));
    try check(mlx.mlx_fast_metal_kernel_config_set_grid(cfgh, 32, @intCast(dv / 8), @intCast(b * hv)));
    try check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfgh, 32, 2, 1));
    try check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfgh, "InT", ydt));
    try check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfgh, "StT", .float32));
    try check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfgh, "Dk", @intCast(dk)));
    try check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfgh, "Dv", @intCast(dv)));
    try check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfgh, "Hk", @intCast(hk)));
    try check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfgh, "Hv", @intCast(hv)));
    var outvec = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(outvec);
    try check(mlx.mlx_fast_metal_kernel_apply(&outvec, ker, invec, cfgh, stream()));
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_vector_array_get(&y, outvec, 0));
    var nst = mlx.mlx_array_new();
    errdefer freeArr(nst);
    try check(mlx.mlx_vector_array_get(&nst, outvec, 1));
    _ = mlx.mlx_vector_array_free(outvec);
    return .{ .y = y, .state = nst };
}

/// Mirror of the reference `packed_eligible`: scalar gate, Dk == 128,
/// Dv % 8 == 0, f32 gate and state. Unmasked by construction (we never mask).
fn fusedEligible(m: *const Model, dk: u32, dv: u32, g: mlx.mlx_array, st: mlx.mlx_array) bool {
    if (!m.fused_gdn) return false;
    if (dk != 128 or dv % 8 != 0) return false;
    if (mlx.mlx_array_ndim(g) != 3) return false;
    if (mlx.mlx_array_dtype(g) != .float32) return false;
    if (mlx.mlx_array_dtype(st) != .float32) return false;
    return true;
}

fn sliceT4(a: mlx.mlx_array, b: u32, s: u32, e: u32, d2: u32, d3: u32) !mlx.mlx_array {
    return sliceArr(a, newShape(4, .{ 0, s, 0, 0 }), newShape(4, .{ b, e, d2, d3 }));
}

fn sliceT3(a: mlx.mlx_array, b: u32, s: u32, e: u32, d2: u32) !mlx.mlx_array {
    return sliceArr(a, newShape(3, .{ 0, s, 0 }), newShape(3, .{ b, e, d2 }));
}

/// Fused recurrence over a T-range, preserving the ops path's MTP snapshot
/// contract: with a confirmed prefix, run prefix + rest separately and
/// snapshot the conv slice + mid-chunk state exactly like the per-position
/// loop does at `ti + 1 == n_confirmed`.
fn gdnRecurFused(b: u32, t: u32, hk: u32, hv: u32, dk: u32, dv: u32, qs: mlx.mlx_array, ks: mlx.mlx_array, v4: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, st: mlx.mlx_array, cache: *GDNCache, n_confirmed: u32, cin: mlx.mlx_array, kk: u32, cd: u32) !GdnKernelOut {
    const ydt = mlx.mlx_array_dtype(v4);
    if (n_confirmed == 0 or n_confirmed >= t) {
        return gdnKernelRun(b, t, hk, hv, dk, dv, qs, ks, v4, g, beta, st, ydt);
    }
    const nc = n_confirmed;
    const q0 = try sliceT4(qs, b, 0, nc, hk, dk);
    defer freeArr(q0);
    const k0 = try sliceT4(ks, b, 0, nc, hk, dk);
    defer freeArr(k0);
    const v0 = try sliceT4(v4, b, 0, nc, hv, dv);
    defer freeArr(v0);
    const g0 = try sliceT3(g, b, 0, nc, hv);
    defer freeArr(g0);
    const b0 = try sliceT3(beta, b, 0, nc, hv);
    defer freeArr(b0);
    const pre = try gdnKernelRun(b, nc, hk, hv, dk, dv, q0, k0, v0, g0, b0, st, ydt);
    errdefer {
        freeArr(pre.y);
        freeArr(pre.state);
    }
    const sc = try sliceArr(cin, newShape(3, .{ 0, nc, 0 }), newShape(3, .{ b, nc + kk - 1, cd }));
    defer freeArr(sc);
    const scc = try contiguous(sc);
    try cache.snapshotFrom(scc, pre.state);
    freeArr(scc);
    const q1 = try sliceT4(qs, b, nc, t, hk, dk);
    defer freeArr(q1);
    const k1 = try sliceT4(ks, b, nc, t, hk, dk);
    defer freeArr(k1);
    const v1 = try sliceT4(v4, b, nc, t, hv, dv);
    defer freeArr(v1);
    const g1 = try sliceT3(g, b, nc, t, hv);
    defer freeArr(g1);
    const b1 = try sliceT3(beta, b, nc, t, hv);
    defer freeArr(b1);
    const rest = try gdnKernelRun(b, t - nc, hk, hv, dk, dv, q1, k1, v1, g1, b1, pre.state, ydt);
    errdefer {
        freeArr(rest.y);
        freeArr(rest.state);
    }
    freeArr(pre.state);
    var pair = [_]mlx.mlx_array{ pre.y, rest.y };
    const y = try concatAxis(&pair, 1);
    freeArr(pre.y);
    freeArr(rest.y);
    return .{ .y = y, .state = rest.state };
}
pub fn rmsNorm(x: mlx.mlx_array, weight: ?mlx.mlx_array, eps: f32) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_fast_rms_norm(&y, x, weight orelse nullArr(), eps, stream()));
    return y;
}

pub fn sigmoid(x: mlx.mlx_array) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_sigmoid(&y, x, stream()));
    return y;
}

pub fn silu(x: mlx.mlx_array) !mlx.mlx_array {
    const s = stream();
    var g = mlx.mlx_array_new();
    errdefer freeArr(g);
    try check(mlx.mlx_sigmoid(&g, x, s));
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_multiply(&y, x, g, s));
    freeArr(g);
    return y;
}

/// Multiply by a float scalar WITHOUT widening: the scalar is rounded to the
/// input dtype first (matches MLX weak-scalar promotion).
pub fn scaleBy(x: mlx.mlx_array, f: f32) !mlx.mlx_array {
    const s = stream();
    const dt = mlx.mlx_array_dtype(x);
    const sc = mlx.mlx_array_new_float(f);
    defer freeArr(sc);
    var sc_t = mlx.mlx_array_new();
    defer freeArr(sc_t);
    if (dt == .float32) {
        try check(mlx.mlx_array_set(&sc_t, sc));
    } else {
        try check(mlx.mlx_astype(&sc_t, sc, dt, s));
    }
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_multiply(&y, x, sc_t, s));
    return y;
}

pub fn astype(a: mlx.mlx_array, dt: mlx.mlx_dtype) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_astype(&y, a, dt, stream()));
    return y;
}

pub fn add(a: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_add(&y, a, b, stream()));
    return y;
}

pub fn mul(a: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_multiply(&y, a, b, stream()));
    return y;
}

pub fn expArr(a: mlx.mlx_array) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_exp(&y, a, stream()));
    return y;
}


pub fn concatAxis(arrays: []const mlx.mlx_array, axis: c_int) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_concatenate_axis(&y, vec, axis, stream()));
    return y;
}

pub fn repeatAxis(a: mlx.mlx_array, repeats: c_int, axis: c_int) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_repeat_axis(&y, a, repeats, axis, stream()));
    return y;
}

pub fn expandDim(a: mlx.mlx_array, axis: c_int) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_expand_dims(&y, a, axis, stream()));
    return y;
}

pub fn sumAxis(a: mlx.mlx_array, axis: c_int) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_sum_axis(&y, a, axis, false, stream()));
    return y;
}
pub fn contiguous(a: mlx.mlx_array) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_contiguous(&y, a, false, stream()));
    return y;
}

pub fn splitEqual(a: mlx.mlx_array, n: c_int, axis: c_int, out: []mlx.mlx_array) !void {
    std.debug.assert(out.len == @as(usize, @intCast(n)));
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try check(mlx.mlx_split(&vec, a, n, axis, stream()));
    std.debug.assert(mlx.mlx_vector_array_size(vec) == out.len);
    for (out, 0..) |*slot, i| {
        slot.* = mlx.mlx_array_new();
        errdefer freeArr(slot.*);
        try check(mlx.mlx_vector_array_get(&slot.*, vec, i));
    }
}

pub fn softplusF32(a: mlx.mlx_array) !mlx.mlx_array {
    const s = stream();
    var e = mlx.mlx_array_new();
    defer freeArr(e);
    try check(mlx.mlx_exp(&e, a, s));
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_log1p(&y, e, s));
    return y;
}

// ── caches ───────────────────────────────────────────────────────────────

pub const KVCache = struct {
    keys: ?mlx.mlx_array = null,
    values: ?mlx.mlx_array = null,
    kview: ?mlx.mlx_array = null,
    vview: ?mlx.mlx_array = null,
    len: u32 = 0,
    cap: u32 = 0,

    pub fn deinit(self: *KVCache) void {
        if (self.keys) |a| freeArr(a);
        if (self.values) |a| freeArr(a);
        if (self.kview) |a| freeArr(a);
        if (self.vview) |a| freeArr(a);
        self.* = .{};
    }

    pub fn reset(self: *KVCache) void {
        self.deinit();
    }

    const step: u32 = 256;

    /// Append T rows; returns borrowed (cache-owned) full views [B,H,len,D].
    /// Views stay valid until the next update/reset/deinit.
    pub fn update(self: *KVCache, k: mlx.mlx_array, v: mlx.mlx_array, n_kv_heads: u32, head_dim: u32) !struct { kk: mlx.mlx_array, vv: mlx.mlx_array } {
        const s = stream();
        const kshape = mlx.getShape(k);
        const t: u32 = @intCast(kshape[2]);
        const b: u32 = @intCast(kshape[0]);
        if (self.keys == null or self.len + t > self.cap) {
            const newcap = ((self.len + t + step - 1) / step) * step;
            const ksh = newShape(4, .{ b, n_kv_heads, newcap, head_dim });
            // Nullable locals: transfer to self.* on success (null cancels the defer).
            var nk: ?mlx.mlx_array = mlx.mlx_array_new();
            defer if (nk) |a| freeArr(a);
            try check(mlx.mlx_zeros(&nk.?, &ksh, ksh.len, .bfloat16, s));
            var nv: ?mlx.mlx_array = mlx.mlx_array_new();
            defer if (nv) |a| freeArr(a);
            try check(mlx.mlx_zeros(&nv.?, &ksh, ksh.len, .bfloat16, s));
            if (self.keys) |old_k| {
                const old_v = self.values.?;
                if (self.len > 0) {
                    const e = newShape(4, .{ b, n_kv_heads, self.len, head_dim });
                    const z = newShape(4, .{ 0, 0, 0, 0 });
                    const tk = try sliceArr(old_k, z, e);
                    defer freeArr(tk);
                    const tv = try sliceArr(old_v, z, e);
                    defer freeArr(tv);
                    const ck = try concatAxis(&.{ tk, nk.? }, 2);
                    const cv = try concatAxis(&.{ tv, nv.? }, 2);
                    freeArr(nk.?);
                    freeArr(nv.?);
                    nk = ck;
                    nv = cv;
                }
                freeArr(old_k);
                freeArr(old_v);
            }
            self.keys = nk;
            self.values = nv;
            nk = null;
            nv = null;
            self.cap = newcap;
        }
        // write rows [len, len+T)
        const off = self.len + t;
        const st = newShape(4, .{ 0, 0, self.len, 0 });
        const en = newShape(4, .{ b, n_kv_heads, off, head_dim });
        const strides = newShape(4, .{ 1, 1, 1, 1 });
        var upd_k: ?mlx.mlx_array = mlx.mlx_array_new();
        defer if (upd_k) |a| freeArr(a);
        try check(mlx.mlx_slice_update(&upd_k.?, self.keys.?, k, &st, st.len, &en, en.len, &strides, strides.len, s));
        freeArr(self.keys.?);
        self.keys = upd_k;
        upd_k = null;
        var upd_v: ?mlx.mlx_array = mlx.mlx_array_new();
        defer if (upd_v) |a| freeArr(a);
        try check(mlx.mlx_slice_update(&upd_v.?, self.values.?, v, &st, st.len, &en, en.len, &strides, strides.len, s));
        freeArr(self.values.?);
        self.values = upd_v;
        upd_v = null;
        self.len = off;
        // refresh full-prefix views
        const ve = newShape(4, .{ b, n_kv_heads, off, head_dim });
        const z = newShape(4, .{ 0, 0, 0, 0 });
        const nkv = try sliceArr(self.keys.?, z, ve);
        errdefer freeArr(nkv);
        const nvv = try sliceArr(self.values.?, z, ve);
        errdefer freeArr(nvv);
        if (self.kview) |a| freeArr(a);
        if (self.vview) |a| freeArr(a);
        self.kview = nkv;
        self.vview = nvv;
        return .{ .kk = nkv, .vv = nvv };
    }

    pub fn trim(self: *KVCache, n: u32) void {
        const m = @min(self.len, n);
        self.len -= m;
    }

    pub fn reserve(self: *KVCache, b: u32, n_kv_heads: u32, head_dim: u32, need: u32) !void {
        if (need <= self.cap) return;
        const s = stream();
        const newcap = ((need + step - 1) / step) * step;
        const ksh = newShape(4, .{ b, n_kv_heads, newcap, head_dim });
        var nk: ?mlx.mlx_array = mlx.mlx_array_new();
        defer if (nk) |a| freeArr(a);
        try check(mlx.mlx_zeros(&nk.?, &ksh, ksh.len, .bfloat16, s));
        var nv: ?mlx.mlx_array = mlx.mlx_array_new();
        defer if (nv) |a| freeArr(a);
        try check(mlx.mlx_zeros(&nv.?, &ksh, ksh.len, .bfloat16, s));
        if (self.keys) |old_k| {
            const old_v = self.values.?;
            if (self.len > 0) {
                const e = newShape(4, .{ b, n_kv_heads, self.len, head_dim });
                const z = newShape(4, .{ 0, 0, 0, 0 });
                const tk = try sliceArr(old_k, z, e);
                defer freeArr(tk);
                const tv = try sliceArr(old_v, z, e);
                defer freeArr(tv);
                const ck = try concatAxis(&.{ tk, nk.? }, 2);
                const cv = try concatAxis(&.{ tv, nv.? }, 2);
                freeArr(nk.?);
                freeArr(nv.?);
                nk = ck;
                nv = cv;
            }
            freeArr(old_k);
            freeArr(old_v);
            if (self.kview) |a| freeArr(a);
            if (self.vview) |a| freeArr(a);
            self.kview = null;
            self.vview = null;
        }
        self.keys = nk;
        self.values = nv;
        nk = null;
        nv = null;
        self.cap = newcap;
    }

    fn retainArray(a: mlx.mlx_array) mlx.mlx_array {
        if (a.ctx == null) return a;
        var b = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&b, a);
        return b;
    }

    pub fn clone(self: *const KVCache) KVCache {
        var out: KVCache = .{ .len = self.len, .cap = self.cap };
        if (self.keys) |a| out.keys = retainArray(a);
        if (self.values) |a| out.values = retainArray(a);
        if (self.kview) |a| out.kview = retainArray(a);
        if (self.vview) |a| out.vview = retainArray(a);
        return out;
    }

    pub fn restoreFrom(self: *KVCache, src: *const KVCache) void {
        self.deinit();
        self.len = src.len;
        self.cap = src.cap;
        if (src.keys) |a| self.keys = retainArray(a);
        if (src.values) |a| self.values = retainArray(a);
        if (src.kview) |a| self.kview = retainArray(a);
        if (src.vview) |a| self.vview = retainArray(a);
    }
};
// ── part 2: GDN cache, model, attention, GDN, layers, forward ──

pub fn sub(a: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_subtract(&y, a, b, stream()));
    return y;
}

// ── GDN cache ────────────────────────────────────────────────────────────

pub const GDNCache = struct {
    conv: ?mlx.mlx_array = null, // [B,K-1,convDim] bf16
    ssm: ?mlx.mlx_array = null, // [B,Hv,Dv,Dk] f32
    snap_conv: ?mlx.mlx_array = null,
    snap_ssm: ?mlx.mlx_array = null,

    pub fn deinit(self: *GDNCache) void {
        if (self.conv) |a| freeArr(a);
        if (self.ssm) |a| freeArr(a);
        self.clearSnaps();
        self.* = .{};
    }

    pub fn reset(self: *GDNCache) void {
        self.deinit();
    }
    /// Retain the given states as the rollback point (shared_ptr copies).
    /// `conv_owned` stays caller-owned; `ssm_borrowed` is only retained.
    pub fn snapshotFrom(self: *GDNCache, conv_owned: mlx.mlx_array, ssm_borrowed: mlx.mlx_array) !void {
        self.clearSnaps();
        var sc = mlx.mlx_array_new();
        errdefer freeArr(sc);
        try check(mlx.mlx_array_set(&sc, conv_owned));
        self.snap_conv = sc;
        var ss = mlx.mlx_array_new();
        errdefer freeArr(ss);
        try check(mlx.mlx_array_set(&ss, ssm_borrowed));
        self.snap_ssm = ss;
    }

    pub fn clearSnaps(self: *GDNCache) void {
        if (self.snap_conv) |a| freeArr(a);
        if (self.snap_ssm) |a| freeArr(a);
        self.snap_conv = null;
        self.snap_ssm = null;
    }

    /// Retain current states for a possible MTP rollback (shared_ptr copy —
    /// no data moves). Caller must clearSnaps() or restore() exactly once.
    pub fn snapshot(self: *GDNCache) !void {
        self.clearSnaps();
        if (self.conv) |c| {
            var sc = mlx.mlx_array_new();
            errdefer freeArr(sc);
            try check(mlx.mlx_array_set(&sc, c));
            self.snap_conv = sc;
        }
        if (self.ssm) |ssm| {
            var ss = mlx.mlx_array_new();
            errdefer freeArr(ss);
            try check(mlx.mlx_array_set(&ss, ssm));
            self.snap_ssm = ss;
        }
    }

    /// Reject path: drop post-verify states, reinstate the snapshot.
    pub fn restore(self: *GDNCache) void {
        if (self.conv) |a| freeArr(a);
        if (self.ssm) |a| freeArr(a);
        self.conv = self.snap_conv;
        self.ssm = self.snap_ssm;
        self.snap_conv = null;
        self.snap_ssm = null;
    }

    /// Take state ownership out of the cache (leaves null behind).
    pub fn takeSsm(self: *GDNCache) ?mlx.mlx_array {
        const s = self.ssm;
        self.ssm = null;
        return s;
    }

    fn retainArrayGdn(a: mlx.mlx_array) mlx.mlx_array {
        if (a.ctx == null) return a;
        var b = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&b, a);
        return b;
    }

    pub fn clone(self: *const GDNCache) GDNCache {
        var out: GDNCache = .{};
        if (self.conv) |a| out.conv = retainArrayGdn(a);
        if (self.ssm) |a| out.ssm = retainArrayGdn(a);
        // snaps are transient (MTP rollback), never cached
        return out;
    }

    pub fn restoreFrom(self: *GDNCache, src: *const GDNCache) void {
        self.deinit();
        if (src.conv) |a| self.conv = retainArrayGdn(a);
        if (src.ssm) |a| self.ssm = retainArrayGdn(a);
    }
};

pub const Caches = struct {
    alloc: std.mem.Allocator,
    kv: []KVCache,
    gdn: []GDNCache,
    n_layers: u32,

    pub fn init(alloc: std.mem.Allocator, n_layers: u32) !Caches {
        const kv = try alloc.alloc(KVCache, n_layers);
        errdefer alloc.free(kv);
        const gdn = try alloc.alloc(GDNCache, n_layers);
        errdefer alloc.free(gdn);
        for (kv) |*c| c.* = .{};
        for (gdn) |*c| c.* = .{};
        return .{ .alloc = alloc, .kv = kv, .gdn = gdn, .n_layers = n_layers };
    }

    pub fn deinit(self: *Caches) void {
        for (self.kv) |*c| c.deinit();
        for (self.gdn) |*c| c.deinit();
        self.alloc.free(self.kv);
        self.alloc.free(self.gdn);
    }

    pub fn reset(self: *Caches) void {
        for (self.kv) |*c| c.reset();
        for (self.gdn) |*c| c.reset();
    }

    pub fn reserveKv(self: *Caches, cfg: Config, need: u32) !void {
        for (self.kv) |*c| {
            try c.reserve(1, cfg.num_key_value_heads, cfg.head_dim, need);
        }
        // also reserve MTP? caller handles separately
    }
};

// ── model ────────────────────────────────────────────────────────────────

pub const Model = struct {
    alloc: std.mem.Allocator,
    cfg: Config,
    wm: weights_mod.WeightMap,
    w_embed: mlx.mlx_array,
    w_norm: mlx.mlx_array,
    w_head: mlx.mlx_array,
    fused_gdn: bool = true, // packed metal kernel when shapes qualify; ops fallback otherwise (validation hatch)

    /// Takes ownership of `wm` (moved, not copied — the map must live in
    /// Model so `wm` outlives any frame; a borrowed pointer dangles once
    /// the loader returns).
    pub fn init(alloc: std.mem.Allocator, cfg: Config, wm: weights_mod.WeightMap) !Model {
        if (cfg.tie_word_embeddings) return ModelError.BadConfig;
        var self = Model{
            .alloc = alloc,
            .cfg = cfg,
            .wm = wm,
            .w_embed = undefined,
            .w_norm = undefined,
            .w_head = undefined,
        };
        errdefer self.wm.deinit();
        self.w_embed = self.wm.get("language_model.model.embed_tokens.weight") orelse return ModelError.MissingWeight;
        self.w_norm = self.wm.get("language_model.model.norm.weight") orelse return ModelError.MissingWeight;
        self.w_head = self.wm.get("language_model.lm_head.weight") orelse return ModelError.MissingWeight;
        return self;
    }

    pub fn deinit(self: *Model) void {
        self.wm.deinit();
    }

    fn lw(self: *const Model, idx: u32, comptime suffix: []const u8) !mlx.mlx_array {
        var kb: [160]u8 = undefined;
        const k = try std.fmt.bufPrint(&kb, "language_model.model.layers.{d}.{s}", .{ idx, suffix });
        return self.wm.get(k) orelse ModelError.MissingWeight;
    }

    fn attnW(self: *const Model, idx: u32, comptime name: []const u8) !mlx.mlx_array {
        return self.lw(idx, "self_attn." ++ name);
    }

    fn gdnW(self: *const Model, idx: u32, comptime name: []const u8) !mlx.mlx_array {
        return self.lw(idx, "linear_attn." ++ name);
    }

    fn mlpW(self: *const Model, idx: u32, comptime name: []const u8) !mlx.mlx_array {
        return self.lw(idx, "mlp." ++ name);
    }
};

pub fn rope(x: mlx.mlx_array, dims: u32, theta: f32, offset: u32) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_fast_rope(&y, x, @intCast(dims), false, mlx.mlx_optional_float.some(theta), 1.0, @intCast(offset), nullArr(), stream()));
    return y;
}

pub fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, t: u32, scale: f32) !mlx.mlx_array {
    const mode: [*:0]const u8 = if (t > 1) "causal" else "";
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_fast_scaled_dot_product_attention(&y, q, k, v, scale, mode, nullArr(), nullArr(), false, stream()));
    return y;
}
pub fn mlpForwardQ(m: *const Model, x: mlx.mlx_array, wg_name: []const u8, wd_name: []const u8, wu_name: []const u8) !mlx.mlx_array {
    const g = try namedLinearQ(m, x, wg_name);
    defer freeArr(g);
    const u = try namedLinearQ(m, x, wu_name);
    defer freeArr(u);
    const s = try silu(g);
    defer freeArr(s);
    const mm = try mul(s, u);
    defer freeArr(mm);
    return namedLinearQ(m, mm, wd_name);
}


pub fn attentionForward(cfg: Config, m: *const Model, idx: u32, x: mlx.mlx_array, cache: *KVCache) !mlx.mlx_array {
    const shape = mlx.getShape(x);
    const b: u32 = @intCast(shape[0]);
    const t: u32 = @intCast(shape[1]);
    const nh = cfg.num_attention_heads;
    const nkv = cfg.num_key_value_heads;
    const hd = cfg.head_dim;

    const wqn = try m.attnW(idx, "q_norm.weight");
    const wkn = try m.attnW(idx, "k_norm.weight");

    const q2 = try layerLinearQ(m, x, idx, "self_attn.q_proj.weight");
    defer freeArr(q2);
    const qrsh = try reshape(q2, newShape(4, .{ b, t, nh, 2 * hd }));
    defer freeArr(qrsh);
    var halves: [2]mlx.mlx_array = undefined;
    try splitEqual(qrsh, 2, 3, &halves);
    defer freeArr(halves[0]);
    defer freeArr(halves[1]);
    const gate = try reshape(halves[1], newShape(3, .{ b, t, nh * hd }));
    defer freeArr(gate);

    const kl = try layerLinearQ(m, x, idx, "self_attn.k_proj.weight");
    defer freeArr(kl);
    const kr = try reshape(kl, newShape(4, .{ b, t, nkv, hd }));
    defer freeArr(kr);
    const vl = try layerLinearQ(m, x, idx, "self_attn.v_proj.weight");
    defer freeArr(vl);
    const vr = try reshape(vl, newShape(4, .{ b, t, nkv, hd }));
    defer freeArr(vr);

    const qn = try rmsNorm(halves[0], wqn, cfg.rms_norm_eps);
    defer freeArr(qn);
    const kn = try rmsNorm(kr, wkn, cfg.rms_norm_eps);
    defer freeArr(kn);

    const ax = newShape(4, .{ 0, 2, 1, 3 });
    const qt = try transposeAxes(qn, ax);
    defer freeArr(qt);
    const kt = try transposeAxes(kn, ax);
    defer freeArr(kt);
    const vt = try transposeAxes(vr, ax);
    defer freeArr(vt);

    // rope at the pre-update offset, then append K/V (reference parity)
    const rqt = try rope(qt, cfg.rope_dims, cfg.rope_theta, cache.len);
    defer freeArr(rqt);
    const rkt = try rope(kt, cfg.rope_dims, cfg.rope_theta, cache.len);
    defer freeArr(rkt);
    const kv = try cache.update(rkt, vt, nkv, hd);

    const sdt = try sdpa(rqt, kv.kk, kv.vv, t, 1.0 / @sqrt(@as(f32, @floatFromInt(hd))));
    defer freeArr(sdt);
    const back = try transposeAxes(sdt, ax);
    defer freeArr(back);
    const flat = try reshape(back, newShape(3, .{ b, t, nh * hd }));
    defer freeArr(flat);
    const sg = try sigmoid(gate);
    defer freeArr(sg);
    const gated = try mul(sg, flat);
    defer freeArr(gated);
    return layerLinearQ(m, gated, idx, "self_attn.o_proj.weight");
}

pub fn zeros3(b: u32, d1: u32, d2: u32, dt: mlx.mlx_dtype) !mlx.mlx_array {
    const sh = newShape(3, .{ b, d1, d2 });
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_zeros(&y, &sh, sh.len, dt, stream()));
    return y;
}

pub fn zeros4(b: u32, d1: u32, d2: u32, d3: u32, dt: mlx.mlx_dtype) !mlx.mlx_array {
    const sh = newShape(4, .{ b, d1, d2, d3 });
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_zeros(&y, &sh, sh.len, dt, stream()));
    return y;
}

pub fn conv1dFwd(x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_conv1d(&y, x, w, 1, 0, 1, mlx.getShape(w)[0], stream()));
    return y;
}

/// One recurrence step. Consumes st_owned, returns {y bf16, state f32}.
pub fn gdnStepOwned(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, st_owned: mlx.mlx_array) !struct { y: mlx.mlx_array, state: mlx.mlx_array } {
    // q,k: [B,Hv,Dk], v: [B,Hv,Dv], g/beta: [B,Hv], st: [B,Hv,Dv,Dk]
    // dst is explicitly freed; the optional + errdefer keeps error paths safe.
    var dst_opt: ?mlx.mlx_array = null;
    errdefer if (dst_opt) |a| freeArr(a);
    const g3 = try expandDim(g, 2);
    defer freeArr(g3);
    const g4 = try expandDim(g3, 3);
    defer freeArr(g4);
    dst_opt = try mul(st_owned, g4);
    freeArr(st_owned);
    const dst = dst_opt.?;
    const ke = try expandDim(k, 2); // [B,Hv,1,Dk]
    defer freeArr(ke);
    const kvm = try mul(dst, ke);
    defer freeArr(kvm);
    const kv = try sumAxis(kvm, 3); // [B,Hv,Dv]
    defer freeArr(kv);
    const be3 = try expandDim(beta, 2); // [B,Hv,1]
    defer freeArr(be3);
    const vmkv = try sub(v, kv);
    defer freeArr(vmkv);
    const delta = try mul(vmkv, be3);
    defer freeArr(delta);
    const de = try expandDim(delta, 3); // [B,Hv,Dv,1]
    defer freeArr(de);
    const kdd = try mul(ke, de);
    defer freeArr(kdd);
    const state = try add(dst, kdd);
    errdefer freeArr(state);
    dst_opt = null;
    freeArr(dst);
    const qe = try expandDim(q, 2); // [B,Hv,1,Dk]
    defer freeArr(qe);
    const sym = try mul(state, qe);
    defer freeArr(sym);
    const yf = try sumAxis(sym, 3);
    defer freeArr(yf);
    const y = try astype(yf, .bfloat16);
    return .{ .y = y, .state = state };
}

pub fn gdnForward(cfg: Config, m: *const Model, idx: u32, x: mlx.mlx_array, cache: *GDNCache, n_confirmed: u32) !mlx.mlx_array {
    const shape = mlx.getShape(x);
    const b: u32 = @intCast(shape[0]);
    const t: u32 = @intCast(shape[1]);
    const hk = cfg.linear_num_key_heads;
    const hv = cfg.linear_num_value_heads;
    const dk = cfg.linear_key_head_dim;
    const dv = cfg.linear_value_head_dim;
    const kd = cfg.keyDim();
    const vd = cfg.valueDim();
    const cd = cfg.convDim();
    const kk = cfg.linear_conv_kernel_dim; // 4

    const wconv = try m.gdnW(idx, "conv1d.weight");
    const alog = try m.gdnW(idx, "A_log");
    const dtb = try m.gdnW(idx, "dt_bias");
    const wnorm = try m.gdnW(idx, "norm.weight");

    const qkv = try layerLinearQ(m, x, idx, "linear_attn.in_proj_qkv.weight");
    defer freeArr(qkv);
    const zl = try layerLinearQ(m, x, idx, "linear_attn.in_proj_z.weight");
    defer freeArr(zl);
    const z = try reshape(zl, newShape(4, .{ b, t, hv, dv }));
    defer freeArr(z);
    const bl = try layerLinearQ(m, x, idx, "linear_attn.in_proj_b.weight");
    defer freeArr(bl);
    const al = try layerLinearQ(m, x, idx, "linear_attn.in_proj_a.weight");
    defer freeArr(al);

    // conv cache: prepend state (zeros when fresh), keep last K-1 rows
    const cs: mlx.mlx_array = cache.conv orelse try zeros3(b, kk - 1, cd, .bfloat16);
    const cs_fresh = cache.conv == null;
    defer if (cs_fresh) freeArr(cs);
    const cin = try concatAxis(&.{ cs, qkv }, 1);
    defer freeArr(cin);
    {
        const ns = try sliceArr(cin, newShape(3, .{ 0, t, 0 }), newShape(3, .{ b, t + kk - 1, cd }));
        defer freeArr(ns);
        const nc = try contiguous(ns);
        if (cache.conv) |old| freeArr(old);
        cache.conv = nc;
    }
    const cw = try conv1dFwd(cin, wconv);
    defer freeArr(cw);
    const cout = try silu(cw);
    defer freeArr(cout);

    // split q | k | v
    const qs = try sliceArr(cout, newShape(3, .{ 0, 0, 0 }), newShape(3, .{ b, t, kd }));
    defer freeArr(qs);
    const ks = try sliceArr(cout, newShape(3, .{ 0, 0, kd }), newShape(3, .{ b, t, 2 * kd }));
    defer freeArr(ks);
    const vs = try sliceArr(cout, newShape(3, .{ 0, 0, 2 * kd }), newShape(3, .{ b, t, cd }));
    defer freeArr(vs);
    const q4 = try reshape(qs, newShape(4, .{ b, t, hk, dk }));
    defer freeArr(q4);
    const k4 = try reshape(ks, newShape(4, .{ b, t, hk, dk }));
    defer freeArr(k4);
    const v4 = try reshape(vs, newShape(4, .{ b, t, hv, dv }));
    defer freeArr(v4);

    // q·Dk^-1, k·Dk^-0.5 with weightless RMSNorm (reference fold-in)
    const qrn = try rmsNorm(q4, null, 1e-6);
    defer freeArr(qrn);
    const krn = try rmsNorm(k4, null, 1e-6);
    defer freeArr(krn);
    const inv: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dk)));
    const qsc = try scaleBy(qrn, inv * inv);
    defer freeArr(qsc);
    const ksc = try scaleBy(krn, inv);
    defer freeArr(ksc);

    // g (f32), beta (bf16, reference dtype)
    const adt = try add(al, dtb);
    defer freeArr(adt);
    const adt_f = try astype(adt, .float32);
    defer freeArr(adt_f);
    const sp = try softplusF32(adt_f);
    defer freeArr(sp);
    const alog_f = try astype(alog, .float32);
    defer freeArr(alog_f);
    const eal = try expArr(alog_f);
    defer freeArr(eal);
    const m1 = try mul(eal, sp);
    defer freeArr(m1);
    const neg = try scaleBy(m1, -1.0);
    defer freeArr(neg);
    const g = try expArr(neg);
    defer freeArr(g);
    const beta = try sigmoid(bl);
    defer freeArr(beta);

    // heads: q,k 16 -> 48
    const rep: c_int = @intCast(hv / hk);
    const qr = try repeatAxis(qsc, rep, 2);
    defer freeArr(qr);
    const kr = try repeatAxis(ksc, rep, 2);
    defer freeArr(kr);

    // state ownership: take out of cache, hand back at the end
    var st = cache.takeSsm() orelse try zeros4(b, hv, dv, dk, .float32);
    errdefer {
        if (cache.ssm == null) {
            cache.ssm = st;
        } else {
            freeArr(st);
        }
    }

    var ycat: mlx.mlx_array = undefined;
    if (fusedEligible(m, dk, dv, g, st)) {
        // NOTE: the kernel fans Hk(16) out to Hv(48) itself via hk_idx, so
        // it takes the pre-repeat qsc/ksc [B,T,Hk,Dk], unlike the ops path.
        const fr = try gdnRecurFused(b, t, hk, hv, dk, dv, qsc, ksc, v4, g, beta, st, cache, n_confirmed, cin, kk, cd);
        freeArr(st);
        ycat = fr.y;
        st = fr.state;
    } else {
        // sequential recurrence over T, one lazy position at a time
        var ys: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (ys.items) |a| freeArr(a);
            ys.deinit(m.alloc);
        }
        var ti: u32 = 0;
        while (ti < t) : (ti += 1) {
            const sq = try sliceArr(qr, newShape(4, .{ 0, ti, 0, 0 }), newShape(4, .{ b, ti + 1, hv, dk }));
            defer freeArr(sq);
            const q_t = try reshape(sq, newShape(3, .{ b, hv, dk }));
            defer freeArr(q_t);
            const sk = try sliceArr(kr, newShape(4, .{ 0, ti, 0, 0 }), newShape(4, .{ b, ti + 1, hv, dk }));
            defer freeArr(sk);
            const k_t = try reshape(sk, newShape(3, .{ b, hv, dk }));
            defer freeArr(k_t);
            const sv = try sliceArr(v4, newShape(4, .{ 0, ti, 0, 0 }), newShape(4, .{ b, ti + 1, hv, dv }));
            defer freeArr(sv);
            const v_t = try reshape(sv, newShape(3, .{ b, hv, dv }));
            defer freeArr(v_t);
            const sg = try sliceArr(g, newShape(3, .{ 0, ti, 0 }), newShape(3, .{ b, ti + 1, hv }));
            defer freeArr(sg);
            const g_t = try reshape(sg, newShape(2, .{ b, hv }));
            defer freeArr(g_t);
            const sb = try sliceArr(beta, newShape(3, .{ 0, ti, 0 }), newShape(3, .{ b, ti + 1, hv }));
            defer freeArr(sb);
            const be_t = try reshape(sb, newShape(2, .{ b, hv }));
            defer freeArr(be_t);
            const pair = try gdnStepOwned(q_t, k_t, v_t, g_t, be_t, st);
            st = pair.state;
            // Fork parity (n_confirmed split): retain conv/ssm as of the end of
            // the confirmed prefix so a draft rejection restores exactly.
            if (n_confirmed != 0 and ti + 1 == n_confirmed and n_confirmed < t) {
                const sc = try sliceArr(cin, newShape(3, .{ 0, n_confirmed, 0 }), newShape(3, .{ b, n_confirmed + kk - 1, cd }));
                defer freeArr(sc);
                const scc = try contiguous(sc);
                try cache.snapshotFrom(scc, st);
                freeArr(scc);
            }
            // restore the T-dim so the concat over axis 1 rebuilds [B,T,Hv,Dv]
            const y4 = try reshape(pair.y, newShape(4, .{ b, 1, hv, dv }));
            freeArr(pair.y);
            try ys.append(m.alloc, y4);
        }
        ycat = try concatAxis(ys.items, 1);
    }
    defer freeArr(ycat);
    cache.ssm = st;

    // output gate: silu(z.f32) * rms(y) -> bf16
    const r = try rmsNorm(ycat, wnorm, 1e-6);
    defer freeArr(r);
    const rf = try astype(r, .float32);
    defer freeArr(rf);
    const zf = try astype(z, .float32);
    defer freeArr(zf);
    const sz = try silu(zf);
    defer freeArr(sz);
    const mm = try mul(sz, rf);
    defer freeArr(mm);
    const out = try astype(mm, .bfloat16);
    defer freeArr(out);
    const flat = try reshape(out, newShape(3, .{ b, t, vd }));
    defer freeArr(flat);
    return layerLinearQ(m, flat, idx, "linear_attn.out_proj.weight");
}

pub fn takeAxis(a: mlx.mlx_array, idx: mlx.mlx_array) !mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try check(mlx.mlx_take_axis(&y, a, idx, 0, stream()));
    return y;
}

pub fn layerForward(cfg: Config, m: *const Model, idx: u32, x: mlx.mlx_array, caches: *Caches, n_confirmed: u32) !mlx.mlx_array {
    const ln1 = try m.lw(idx, "input_layernorm.weight");
    const ln2 = try m.lw(idx, "post_attention_layernorm.weight");
    const h1 = try rmsNorm(x, ln1, cfg.rms_norm_eps);
    defer freeArr(h1);
    var r: mlx.mlx_array = undefined;
    if (cfg.isLinear(idx)) {
        r = try gdnForward(cfg, m, idx, h1, &caches.gdn[idx], n_confirmed);
    } else {
        r = try attentionForward(cfg, m, idx, h1, &caches.kv[idx]);
    }
    defer freeArr(r);
    const h = try add(x, r);
    defer freeArr(h);
    const h2 = try rmsNorm(h, ln2, cfg.rms_norm_eps);
    defer freeArr(h2);
    var gb: [160]u8 = undefined;
    var db: [160]u8 = undefined;
    var ub: [160]u8 = undefined;
    const mp = try mlpForwardQ(
        m,
        h2,
        try layerName(&gb, idx, "mlp.gate_proj.weight"),
        try layerName(&db, idx, "mlp.down_proj.weight"),
        try layerName(&ub, idx, "mlp.up_proj.weight"),
    );
    defer freeArr(mp);
    return add(h, mp);
}

/// Full forward: ids[T] -> PRE-norm hidden [1,T,H] (owned). The fork's
/// `return_hidden` path feeds this straight to the MTP head; the engine
/// applies `normHidden` before `logitsOf`. `n_confirmed` (verify path only)
/// snapshots GDN state after that many leading positions for rollback.
/// Eval once per call at the engine level.
pub fn forward(m: *const Model, ids: []const u32, caches: *Caches, n_confirmed: u32) !mlx.mlx_array {
    std.debug.assert(ids.len > 0);
    const t: u32 = @intCast(ids.len);
    const hdim = m.cfg.hidden_size;
    const tsh = newShape(1, .{t});
    const idxarr = mlx.mlx_array_new_data(ids.ptr, &tsh, 1, .uint32);
    defer freeArr(idxarr);
    const emb = try takeAxis(m.w_embed, idxarr);
    defer freeArr(emb);
    var h = try reshape(emb, newShape(3, .{ 1, t, hdim }));
    errdefer freeArr(h);
    for (0..m.cfg.num_hidden_layers) |li| {
        const nh = try layerForward(m.cfg, m, @intCast(li), h, caches, n_confirmed);
        freeArr(h);
        h = nh;
    }
    return h;
}

/// Final RMSNorm: pre-norm hidden -> post-norm hidden (owned).
pub fn normHidden(m: *const Model, hidden_pre: mlx.mlx_array) !mlx.mlx_array {
    return rmsNorm(hidden_pre, m.w_norm, m.cfg.rms_norm_eps);
}

/// logits for the last k hidden rows: hidden [1,T,H] -> [k,V] (owned).
pub fn logitsOf(m: *const Model, hidden: mlx.mlx_array, take_last: u32) !mlx.mlx_array {
    const hs = mlx.getShape(hidden);
    const t: u32 = @intCast(hs[1]);
    const hdim: u32 = @intCast(hs[2]);
    std.debug.assert(take_last <= t and take_last > 0);
    const rows = try sliceArr(hidden, newShape(3, .{ 0, t - take_last, 0 }), newShape(3, .{ 1, t, hdim }));
    defer freeArr(rows);
    const flat = try reshape(rows, newShape(2, .{ take_last, hdim }));
    defer freeArr(flat);
    return namedLinearQ(m, flat, "language_model.lm_head.weight");
}

test "config parses from text_config json" {
    const alloc = std.testing.allocator;
    const doc =
        \\{"hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
        \\ "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
        \\ "full_attention_interval": 4, "rms_norm_eps": 0.000001, "vocab_size": 248320,
        \\ "max_position_embeddings": 262144, "tie_word_embeddings": false,
        \\ "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        \\ "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        \\ "linear_conv_kernel_dim": 4,
        \\ "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, doc, .{});
    defer parsed.deinit();
    const c = try Config.fromTextConfig(parsed.value);
    try std.testing.expectEqual(@as(u32, 64), c.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 64), c.rope_dims);
    try std.testing.expectApproxEqAbs(@as(f32, 1e7), c.rope_theta, 1.0);
    try std.testing.expect(c.isLinear(0));
    try std.testing.expect(!c.isLinear(3));
    try std.testing.expectEqual(@as(u32, 2048), c.keyDim());
    try std.testing.expectEqual(@as(u32, 6144), c.valueDim());
    try std.testing.expectEqual(@as(u32, 10240), c.convDim());
}
// ── linked tests (need libmlx; run under `zig build test-mlx`) ──
// These also force analysis of the forward path in the linked binary.

pub fn newF32(data: []const f32, shape: []const c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
}

pub fn evalF32(a: mlx.mlx_array) ![]const f32 {
    try check(mlx.mlx_array_eval(a));
    const p = mlx.mlx_array_data_float32(a).?;
    return p[0..mlx.mlx_array_size(a)];
}

/// Eval a bf16 array and return an owned f32 copy (bit-exact widening).
pub fn evalBf16ToF32(alloc: std.mem.Allocator, a: mlx.mlx_array) ![]f32 {
    try check(mlx.mlx_array_eval(a));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(a));
    const n = mlx.mlx_array_size(a);
    const p = mlx.mlx_array_data_bfloat16(a).?;
    const out = try alloc.alloc(f32, n);
    errdefer alloc.free(out);
    for (p[0..n], 0..) |bits, i| out[i] = @bitCast(@as(u32, bits) << 16);
    return out;
}

pub fn expectNoNaNBf16(a: mlx.mlx_array) !void {
    try check(mlx.mlx_array_eval(a));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(a));
    const n = mlx.mlx_array_size(a);
    const p = mlx.mlx_array_data_bfloat16(a).?;
    for (p[0..n]) |bits| {
        // exponent all-ones (inf or NaN) is never valid here
        try std.testing.expect(((bits >> 10) & 0x1F) != 0x1F);
    }
}

test "rope matches closed-form sin/cos" {
    // dims=4, theta=1e4: freqs = [1, 0.01]. Verified against mx.fast.rope:
    // traditional=false rotates half-interleaved pairs (x[i], x[i+2]) and
    // lays out [first-halves, second-halves].
    const sh = newShape(4, .{ 1, 1, 2, 4 });
    const vals = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    const x = newF32(&vals, &sh);
    defer freeArr(x);
    const y = try rope(x, 4, 10000.0, 0);
    defer freeArr(y);
    const got = try evalF32(y);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), got[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), got[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), got[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), got[3], 1e-4);
    const c1: f32 = @floatCast(@cos(1.0));
    const s1: f32 = @floatCast(@sin(1.0));
    const c2: f32 = @floatCast(@cos(0.01));
    const s2: f32 = @floatCast(@sin(0.01));
    try std.testing.expectApproxEqAbs(c1 - s1, got[4], 1e-4);
    try std.testing.expectApproxEqAbs(c2 - s2, got[5], 1e-4);
    try std.testing.expectApproxEqAbs(s1 + c1, got[6], 1e-4);
    try std.testing.expectApproxEqAbs(s2 + c2, got[7], 1e-4);
}

test "rope leaves passthrough dims alone and honors offset" {
    // D=8, dims=4: last 4 untouched; offset shifts positions.
    const sh = newShape(4, .{ 1, 1, 1, 8 });
    const vals = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    const x = newF32(&vals, &sh);
    defer freeArr(x);
    const y = try rope(x, 4, 10000.0, 2);
    defer freeArr(y);
    const got = try evalF32(y);
    const c: f32 = @floatCast(@cos(2.0));
    const s: f32 = @floatCast(@sin(2.0));
    const c2: f32 = @floatCast(@cos(0.02));
    const s2: f32 = @floatCast(@sin(0.02));
    try std.testing.expectApproxEqAbs(c - s, got[0], 1e-4);
    try std.testing.expectApproxEqAbs(c2 - s2, got[1], 1e-4);
    try std.testing.expectApproxEqAbs(s + c, got[2], 1e-4);
    try std.testing.expectApproxEqAbs(s2 + c2, got[3], 1e-4);
    for (got[4..8]) |v| try std.testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);
}

test "gdn step matches hand-computed f64 case" {
    // 1 head, Dk=2, Dv=2: q=[1,2] k=[3,4] v=[5,6] g=0.5 beta=1 state=0
    // -> y=[55,66], state=[[15,20],[18,24]]
    const s3 = newShape(3, .{ 1, 1, 2 });
    const s2 = newShape(2, .{ 1, 1 });
    const q = newF32(&[_]f32{ 1, 2 }, &s3);
    defer freeArr(q);
    const k = newF32(&[_]f32{ 3, 4 }, &s3);
    defer freeArr(k);
    const v = newF32(&[_]f32{ 5, 6 }, &s3);
    defer freeArr(v);
    const g = newF32(&[_]f32{0.5}, &s2);
    defer freeArr(g);
    const beta = newF32(&[_]f32{1.0}, &s2);
    defer freeArr(beta);
    const st = try zeros4(1, 1, 2, 2, .float32);
    const pair = try gdnStepOwned(q, k, v, g, beta, st);
    defer freeArr(pair.y);
    defer freeArr(pair.state);
    const y = try evalBf16ToF32(std.testing.allocator, pair.y);
    defer std.testing.allocator.free(y);
    try std.testing.expectApproxEqAbs(@as(f32, 55.0), y[0], 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 66.0), y[1], 0.5);
    try check(mlx.mlx_array_eval(pair.state));
    const spp = mlx.mlx_array_data_float32(pair.state).?;
    const sp = spp[0..4];
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), sp[0], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), sp[1], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), sp[2], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), sp[3], 0.1);
}

// Tiny 2-layer model (1 GDN + 1 full) with file-backed random weights:
// validates shapes, dtypes, NaN-freedom, and cache carry across steps.
pub const tiny_cfg = Config{
    .num_hidden_layers = 2,
    .hidden_size = 32,
    .intermediate_size = 64,
    .num_attention_heads = 2,
    .num_key_value_heads = 1,
    .head_dim = 16,
    .full_attention_interval = 2,
    .rope_dims = 4,
    .rope_theta = 1e4,
    .rms_norm_eps = 1e-6,
    .vocab_size = 64,
    .max_position_embeddings = 1024,
    .linear_num_key_heads = 1,
    .linear_num_value_heads = 2,
    .linear_key_head_dim = 8,
    .linear_value_head_dim = 8,
    .linear_conv_kernel_dim = 4,
};

pub const TinySpec = weights_mod.FixtureSpec;

pub const tiny_specs: []const TinySpec = &.{
    .{ .name = "language_model.model.embed_tokens.weight", .shape = .{ 64, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.norm.weight", .shape = .{ 32, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.lm_head.weight", .shape = .{ 64, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.0.input_layernorm.weight", .shape = .{ 32, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.model.layers.0.post_attention_layernorm.weight", .shape = .{ 32, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.model.layers.0.linear_attn.in_proj_qkv.weight", .shape = .{ 32, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.0.linear_attn.in_proj_z.weight", .shape = .{ 16, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.0.linear_attn.in_proj_b.weight", .shape = .{ 2, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.0.linear_attn.in_proj_a.weight", .shape = .{ 2, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.0.linear_attn.out_proj.weight", .shape = .{ 32, 16, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.0.linear_attn.conv1d.weight", .shape = .{ 32, 4, 1 }, .ndim = 3 },
    .{ .name = "language_model.model.layers.0.linear_attn.A_log", .shape = .{ 2, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.model.layers.0.linear_attn.dt_bias", .shape = .{ 2, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.model.layers.0.linear_attn.norm.weight", .shape = .{ 8, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.model.layers.0.mlp.gate_proj.weight", .shape = .{ 64, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.0.mlp.down_proj.weight", .shape = .{ 32, 64, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.0.mlp.up_proj.weight", .shape = .{ 64, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.1.input_layernorm.weight", .shape = .{ 32, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.model.layers.1.post_attention_layernorm.weight", .shape = .{ 32, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.model.layers.1.self_attn.q_proj.weight", .shape = .{ 64, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.1.self_attn.k_proj.weight", .shape = .{ 16, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.1.self_attn.v_proj.weight", .shape = .{ 16, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.1.self_attn.o_proj.weight", .shape = .{ 32, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.1.self_attn.q_norm.weight", .shape = .{ 16, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.model.layers.1.self_attn.k_norm.weight", .shape = .{ 16, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.model.layers.1.mlp.gate_proj.weight", .shape = .{ 64, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.1.mlp.down_proj.weight", .shape = .{ 32, 64, 0 }, .ndim = 2 },
    .{ .name = "language_model.model.layers.1.mlp.up_proj.weight", .shape = .{ 64, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.mtp.pre_fc_norm_embedding.weight", .shape = .{ 32, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.mtp.pre_fc_norm_hidden.weight", .shape = .{ 32, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.mtp.fc.weight", .shape = .{ 32, 64, 0 }, .ndim = 2 },
    .{ .name = "language_model.mtp.layers.0.input_layernorm.weight", .shape = .{ 32, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.mtp.layers.0.post_attention_layernorm.weight", .shape = .{ 32, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.mtp.layers.0.self_attn.q_proj.weight", .shape = .{ 64, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.mtp.layers.0.self_attn.k_proj.weight", .shape = .{ 16, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.mtp.layers.0.self_attn.v_proj.weight", .shape = .{ 16, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.mtp.layers.0.self_attn.o_proj.weight", .shape = .{ 32, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.mtp.layers.0.self_attn.q_norm.weight", .shape = .{ 16, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.mtp.layers.0.self_attn.k_norm.weight", .shape = .{ 16, 0, 0 }, .ndim = 1 },
    .{ .name = "language_model.mtp.layers.0.mlp.gate_proj.weight", .shape = .{ 64, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.mtp.layers.0.mlp.down_proj.weight", .shape = .{ 32, 64, 0 }, .ndim = 2 },
    .{ .name = "language_model.mtp.layers.0.mlp.up_proj.weight", .shape = .{ 64, 32, 0 }, .ndim = 2 },
    .{ .name = "language_model.mtp.norm.weight", .shape = .{ 32, 0, 0 }, .ndim = 1 },
};

test "tiny model forward: shapes, dtype, no NaN, cache carry" {
    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = "/tmp/mlx-runner-tiny-model";
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    try weights_mod.writeFixture(alloc, io, dir, tiny_specs, 0x12345678);
    var wm = try weights_mod.WeightMap.load(alloc, io, dir);
    try std.testing.expectEqual(tiny_specs.len, wm.count());
    var mm = Model.init(alloc, tiny_cfg, wm) catch |err| {
        wm.deinit();
        return err;
    };
    defer mm.deinit();
    var caches = try Caches.init(alloc, tiny_cfg.num_hidden_layers);
    defer caches.deinit();

    // prefill T=3 then decode T=1 (offset path, conv streaming, state carry).
    // forward returns PRE-norm hidden (MTP head input); norm explicitly.
    const h1 = try forward(&mm, &[_]u32{ 1, 2, 3 }, &caches, 0);
    defer freeArr(h1);
    try std.testing.expectEqual(@as(usize, 3), mlx.mlx_array_ndim(h1));
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ 1, 3, 32 }, mlx.getShape(h1));
    try expectNoNaNBf16(h1);
    const h1n = try normHidden(&mm, h1);
    defer freeArr(h1n);
    const l1 = try logitsOf(&mm, h1n, 1);
    defer freeArr(l1);
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ 1, 64 }, mlx.getShape(l1));
    try expectNoNaNBf16(l1);

    const h2 = try forward(&mm, &[_]u32{4}, &caches, 0);
    defer freeArr(h2);
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ 1, 1, 32 }, mlx.getShape(h2));
    try expectNoNaNBf16(h2);
    try std.testing.expectEqual(@as(u32, 4), caches.kv[1].len);
    try mlx.checkError();
}

// Synthetic quant wiring proof (linked suite only): MLX encodes a known f32
// matrix, then `qlinear`-equivalent qmm must match a CPU matmul over MLX's
// own dequantize. A wiring bug (mode/transpose/scales misalignment) misses
// by orders of magnitude; tolerance stays loose on purpose.
test "quantized matmul matches dense over dequantize (nvfp4/mxfp8)" {
    const cases = [_]struct { mode: [*:0]const u8, group: c_int, bits: c_int }{
        .{ .mode = "nvfp4", .group = 16, .bits = 4 },
        .{ .mode = "mxfp8", .group = 32, .bits = 8 },
    };
    var wdata: [8 * 32]f32 = undefined;
    for (&wdata, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(i % 7)) - 3));
    var xdata: [2 * 32]f32 = undefined;
    for (&xdata, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(i % 5)) - 2));
    const s = stream();
    for (cases) |c| {
        var wshape = newShape(2, .{ 8, 32 });
        const w = mlx.mlx_array_new_data(@ptrCast(&wdata), &wshape, 2, .float32);
        defer freeArr(w);
        var vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        try check(mlx.mlx_quantize(&vec, w, mlx.mlx_optional_int.some(c.group), mlx.mlx_optional_int.some(c.bits), c.mode, nullArr(), s));
        var q = mlx.mlx_array_new();
        defer freeArr(q);
        try check(mlx.mlx_vector_array_get(&q, vec, 0));
        var sc = mlx.mlx_array_new();
        defer freeArr(sc);
        try check(mlx.mlx_vector_array_get(&sc, vec, 1));
        var xshape = newShape(2, .{ 2, 32 });
        const x = mlx.mlx_array_new_data(@ptrCast(&xdata), &xshape, 2, .float32);
        defer freeArr(x);
        var y = mlx.mlx_array_new();
        defer freeArr(y);
        try check(mlx.mlx_quantized_matmul(&y, x, q, sc, nullArr(), true, mlx.mlx_optional_int.some(c.group), mlx.mlx_optional_int.some(c.bits), c.mode, s));
        try check(mlx.mlx_array_eval(y));
        var dq = mlx.mlx_array_new();
        defer freeArr(dq);
        try check(mlx.mlx_dequantize(&dq, q, sc, nullArr(), mlx.mlx_optional_int.some(c.group), mlx.mlx_optional_int.some(c.bits), c.mode, nullArr(), .{ .value = .float32, .has_value = true }, s));
        try check(mlx.mlx_array_eval(dq));
        const yp = mlx.mlx_array_data_float32(y).?;
        const dqp = mlx.mlx_array_data_float32(dq).?;
        var max_err: f32 = 0;
        for (0..2) |r| {
            for (0..8) |o| {
                var ref: f32 = 0;
                for (0..32) |k| ref += xdata[r * 32 + k] * dqp[o * 32 + k];
                const e = @abs(yp[r * 8 + o] - ref);
                if (e > max_err) max_err = e;
            }
        }
        try std.testing.expect(max_err < 0.05);
        try mlx.checkError();
    }
}

// Permanent wiring proof for the packed GDN kernel (linked suite only):
// hand-rolled f32 recurrence in-test vs one gdnKernelRun call. Catches
// shape/grid/template/T-array regressions; math fidelity vs the ops path is
// covered by the greedy-agreement gate (see bench).
test "fused gdn kernel matches scalar recurrence" {
    const hv: u32 = 2;
    const hk: u32 = 1;
    const dk: u32 = 128;
    const dv: u32 = 8;
    const t: u32 = 3;
    // host inputs: small deterministic patterns
    var qh: [1 * 3 * 1 * 128]f32 = undefined;
    var kh: [1 * 3 * 1 * 128]f32 = undefined;
    var vh: [1 * 3 * 2 * 8]f32 = undefined;
    var gh: [1 * 3 * 2]f32 = undefined;
    var bh: [1 * 3 * 2]f32 = undefined;
    for (&qh, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(i % 13)) - 6)) * 0.05;
    for (&kh, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(i % 11)) - 5)) * 0.05;
    for (&vh, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(i % 7)) - 3)) * 0.1;
    for (&gh, 0..) |*v, i| v.* = 0.9 + @as(f32, @floatFromInt(i % 3)) * 0.03;
    for (&bh, 0..) |*v, i| v.* = 0.4 + @as(f32, @floatFromInt(i % 2)) * 0.1;
    var sh0: [1 * 2 * 8 * 128]f32 = undefined;
    for (&sh0, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(i % 5)) - 2)) * 0.01;
    // CPU reference recurrence (matches the kernel's scalar-gate math)
    var st: [2 * 8 * 128]f32 = undefined;
    @memcpy(&st, &sh0);
    var yref: [3 * 2 * 8]f32 = undefined;
    // CPU reference recurrence (matches the kernel's scalar-gate math)
    for (0..t) |ti| {
        for (0..hv) |h| {
            const gt = gh[(ti * hv + h)];
            const bt = bh[(ti * hv + h)];
            // per (hv,dv) row: decay, kv over dk, delta, update, out
            for (0..dv) |r| {
                var kvv: f32 = 0;
                for (0..dk) |d| {
                    const si = (h * dv + r) * dk + d;
                    st[si] *= gt;
                    kvv += st[si] * kh[(ti * hk + h / (hv / hk)) * dk + d];
                }
                const delta = (vh[(ti * hv + h) * dv + r] - kvv) * bt;
                var o: f32 = 0;
                for (0..dk) |d| {
                    const si = (h * dv + r) * dk + d;
                    st[si] += kh[(ti * hk + h / (hv / hk)) * dk + d] * delta;
                    o += st[si] * qh[(ti * hk + h / (hv / hk)) * dk + d];
                }
                yref[(ti * hv + h) * dv + r] = o;
            }
        }
    }
    // device run (bf16 y like production via the InT cast; state f32)
    var qb = newShape(4, .{ 1, t, hk, dk });
    const q = mlx.mlx_array_new_data(@ptrCast(&qh), &qb, 4, .float32);
    defer freeArr(q);
    var kb = newShape(4, .{ 1, t, hk, dk });
    const kk = mlx.mlx_array_new_data(@ptrCast(&kh), &kb, 4, .float32);
    defer freeArr(kk);
    var vb = newShape(4, .{ 1, t, hv, dv });
    const vv = mlx.mlx_array_new_data(@ptrCast(&vh), &vb, 4, .float32);
    defer freeArr(vv);
    var gb = newShape(3, .{ 1, t, hv });
    const gg = mlx.mlx_array_new_data(@ptrCast(&gh), &gb, 3, .float32);
    defer freeArr(gg);
    var bb = newShape(3, .{ 1, t, hv });
    const be = mlx.mlx_array_new_data(@ptrCast(&bh), &bb, 3, .float32);
    defer freeArr(be);
    var sb = newShape(4, .{ 1, hv, dv, dk });
    const s0 = mlx.mlx_array_new_data(@ptrCast(&sh0), &sb, 4, .float32);
    defer freeArr(s0);
    const out = try gdnKernelRun(1, t, hk, hv, dk, dv, q, kk, vv, gg, be, s0, .bfloat16);
    defer freeArr(out.y);
    defer freeArr(out.state);
    try check(mlx.mlx_array_eval(out.y));
    try check(mlx.mlx_array_eval(out.state));
    const yp = mlx.mlx_array_data_bfloat16(out.y).?;
    const sp = mlx.mlx_array_data_float32(out.state).?;
    var max_y: f32 = 0;
    for (yref, 0..) |ref, i| {
        const got: f32 = @bitCast(@as(u32, yp[i]) << 16); // bf16 -> f32 is a left shift
        const e = @abs(got - ref);
        if (e > max_y) max_y = e;
    }
    var max_s: f32 = 0;
    for (st, 0..) |ref, i| {
        const e = @abs(sp[i] - ref);
        if (e > max_s) max_s = e;
    }
    try std.testing.expect(max_y < 0.05);
    try std.testing.expect(max_s < 1e-3);
    try mlx.checkError();
}
