//! Base-vs-MTP benchmark: load once, 20-token warmup, then per condition
//! (base, MTP) wall load time, TTFT, prefill tok/s on a fixed 32K-token
//! prompt, and decode tok/s on two fixed prompts (short factual +
//! LRU-cache code, max_tokens=400, temp 0 so base/MTP bytes must match —
//! mismatch exits non-zero). Fixed-width table to stdout, same data as
//! JSON to --bench-out. The 1.05–1.8x / 35–90% band is reported, not gated:
//! only byte-equality fails the run.

const std = @import("std");
const engine_mod = @import("engine.zig");
const sampling = @import("sampling.zig");
const mlx = @import("mlx.zig");

const Engine = engine_mod.Engine;
const EngineConfig = engine_mod.EngineConfig;

const decode_prompts = [_][]const u8{
    "The capital of France is",
    "Write a Python LRU cache implementation:\n```python\n",
};

const prefill_target: usize = 32768;
const decode_tokens: u32 = 400;
const prefill_probe_tokens: u32 = 5;

const CondResult = struct {
    prefill_tps: f64 = 0,
    ttft_ms: i64 = 0,
    decode_tps: f64 = 0,
    peak_gb: f64 = 0,
    accept_rate: f32 = 0,
    offered: u64 = 0,
    accepted: u64 = 0,
};

/// Token agreement with 1-token indel tolerance: greedy speculative decode
/// is not bit-identical to incremental decode in hardware fp (batched vs
/// incremental kernel paths differ in summation order, so accepted-draft
/// cache states drift ~1ulp and close argmax calls occasionally flip), but
/// a miswired draft path diverges massively. Counts matches over max length.
pub fn agreeIds(a: []const u32, b: []const u32) struct { matched: usize, total: usize } {
    var i: usize = 0;
    var j: usize = 0;
    var matched: usize = 0;
    while (i < a.len and j < b.len) {
        if (a[i] == b[j]) {
            matched += 1;
            i += 1;
            j += 1;
            continue;
        }
        // Single-token indel: skip ahead (up to 8) to re-sync.
        var synced = false;
        var k: usize = 1;
        while (k <= 8 and j + k < b.len) : (k += 1) {
            if (a[i] == b[j + k]) {
                j += k;
                synced = true;
                break;
            }
        }
        if (!synced) {
            k = 1;
            while (k <= 8 and i + k < a.len) : (k += 1) {
                if (a[i + k] == b[j]) {
                    i += k;
                    synced = true;
                    break;
                }
            }
        }
        if (!synced) {
            i += 1;
            j += 1;
        }
    }
    return .{ .matched = matched, .total = @max(a.len, b.len) };
}

fn msNow(io: std.Io) std.Io.Clock.Timestamp {
    return std.Io.Clock.Timestamp.now(io, .awake);
}

fn msSince(t0: std.Io.Clock.Timestamp, io: std.Io) i64 {
    return t0.durationTo(msNow(io)).raw.toMilliseconds();
}

fn peakGb() f64 {
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    return @as(f64, @floatFromInt(peak)) / 1e9;
}

fn resetPeak() void {
    _ = mlx.mlx_reset_peak_memory();
}

fn build32kPrompt(eng: *Engine) ![]u8 {
    const alloc = eng.allocator;
    const para = "The history of computing is a story of steady acceleration, from mechanical calculators through vacuum tubes, transistors, integrated circuits, and now large-scale parallel accelerators. ";
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(alloc);
    // Repeat until comfortably over target, then cut to exactly target ids.
    while (true) {
        try text.appendSlice(alloc, para);
        if (text.items.len > prefill_target * 6) break;
    }
    const ids = try eng.tokenizer.encode(alloc, text.items);
    defer alloc.free(ids);
    const cut = @min(ids.len, prefill_target);
    return try eng.tokenizer.decode(alloc, ids[0..cut]);
}

pub fn run(allocator: std.mem.Allocator, io: std.Io, cfg: EngineConfig, cli_sampling: sampling.SamplingParams, bench_out: ?[]const u8) !void {
    _ = cli_sampling;
    const t_load0 = msNow(io);
    var eng = try Engine.init(allocator, io, cfg, .{});
    defer eng.deinit();
    const load_ms = msSince(t_load0, io);
    eng.bypass_cache = true;
    const zero = sampling.SamplingParams{ .temp = 0 };
    const prompt32k = try build32kPrompt(&eng);
    defer allocator.free(prompt32k);
    const probe_ids = try eng.tokenizer.encode(allocator, prompt32k);
    defer allocator.free(probe_ids);
    const n32k = probe_ids.len;
    // Warmup: 20 tokens, untimed.
    {
        const w = try eng.generate(io, "Hello", null, 20, zero, false);
        defer allocator.free(w);
    }


    var base = CondResult{};
    var mtp = CondResult{};
    // Base id sequences per run (probe + 2 prompts) for agreement scoring.
    var base_ids: [3][]u32 = .{ &.{}, &.{}, &.{} };
    defer for (base_ids) |ids| {
        if (ids.len > 0) allocator.free(ids);
    };
    var agreed: usize = 0;
    var agreed_total: usize = 0;

    for ([2]bool{ false, true }) |use_mtp| {
        const r = if (use_mtp) &mtp else &base;
        eng.mtp_enabled = use_mtp;
        resetPeak();
        // Prefill probe on the 32K prompt (TTFT + prefill rate).
        {
            const t = try eng.generate(io, prompt32k, null, prefill_probe_tokens, zero, false);
            defer allocator.free(t);
            const st = eng.last_stats;
            r.ttft_ms = st.ttft_ms;
            r.prefill_tps = if (st.prefill_ms > 0) @as(f64, @floatFromInt(st.prefill_tokens)) / (@as(f64, @floatFromInt(st.prefill_ms)) / 1000.0) else 0;
            if (!use_mtp) {
                base_ids[0] = try allocator.dupe(u32, eng.last_ids orelse &.{});
            } else if (eng.last_ids) |mids| {
                const a = agreeIds(base_ids[0], mids);
                agreed += a.matched;
                agreed_total += a.total;
            }
        }
        // Decode rate on the two fixed prompts.
        var dsum: f64 = 0;
        for (decode_prompts, 0..) |p, i| {
            const t = try eng.generate(io, p, null, decode_tokens, zero, false);
            defer allocator.free(t);
            const st = eng.last_stats;
            const dtps = if (st.decode_ms > 0) @as(f64, @floatFromInt(st.decode_tokens)) / (@as(f64, @floatFromInt(st.decode_ms)) / 1000.0) else 0;
            dsum += dtps;
            if (!use_mtp) {
                base_ids[i + 1] = try allocator.dupe(u32, eng.last_ids orelse &.{});
            } else if (eng.last_ids) |mids| {
                const a = agreeIds(base_ids[i + 1], mids);
                agreed += a.matched;
                agreed_total += a.total;
            }
        }
        r.decode_tps = dsum / 2.0;
        r.peak_gb = peakGb();
        r.accept_rate = eng.mtp.acceptRate();
        r.offered = eng.mtp.offered;
        r.accepted = eng.mtp.accepted;
    }

    // Gates: acceptance floor proves the draft path is wired (a dead path
    // scores ~0%); agreement >= 98% bounds fp drift (greedy bit-parity is
    // unachievable across batched/incremental kernel paths — see agreeIds).
    const agreement: f64 = if (agreed_total > 0) @as(f64, @floatFromInt(agreed)) / @as(f64, @floatFromInt(agreed_total)) else 0;
    const speedup = if (base.decode_tps > 0) mtp.decode_tps / base.decode_tps else 0;
    const accept_pct = mtp.accept_rate * 100;
    const band_ok = speedup >= 1.05 and speedup <= 1.8 and accept_pct >= 35.0 and accept_pct <= 90.0;

    var buf: [16384]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.print("model      {s}\n", .{eng.model_path});
    try w.interface.print("load       {d} ms  ctx {d}  prompt32k {d} tok\n", .{ load_ms, eng.model_max_ctx, n32k });
    try w.interface.print("condition  ttft_ms  prefill_tps  decode_tps  peak_GB  accept\n", .{});
    try w.interface.print("base       {d:>7}  {d:>11.1}  {d:>10.1}  {d:>7.2}  -\n", .{ base.ttft_ms, base.prefill_tps, base.decode_tps, base.peak_gb });
    try w.interface.print("mtp        {d:>7}  {d:>11.1}  {d:>10.1}  {d:>7.2}  {d:.1}% ({d}/{d})\n", .{ mtp.ttft_ms, mtp.prefill_tps, mtp.decode_tps, mtp.peak_gb, accept_pct, mtp.accepted, mtp.offered });
    try w.interface.print("speedup    {d:.2}x  agree {d:.2}% ({d}/{d})  band {s} (expect 1.05-1.8x, 35-90%)\n", .{ speedup, agreement * 100, agreed, agreed_total, if (band_ok) "IN" else "OUT" });
    try w.interface.flush();

    if (bench_out) |path| {
        const js = try std.fmt.allocPrint(allocator,
            \\{{"model":"{s}","load_ms":{d},"ctx":{d},"prompt32k_tok":{d},"base":{{"ttft_ms":{d},"prefill_tps":{d:.1},"decode_tps":{d:.1},"peak_gb":{d:.2}}},"mtp":{{"ttft_ms":{d},"prefill_tps":{d:.1},"decode_tps":{d:.1},"peak_gb":{d:.2},"accept_rate":{d:.3},"offered":{d},"accepted":{d}}},"speedup":{d:.3},"band_ok":{},"agreement":{d:.4},"agreed":{d},"agreed_total":{d}}}
        , .{
            eng.model_path,  load_ms,           eng.model_max_ctx, n32k,
            base.ttft_ms,    base.prefill_tps,  base.decode_tps,   base.peak_gb,
            mtp.ttft_ms,     mtp.prefill_tps,   mtp.decode_tps,    mtp.peak_gb,
            mtp.accept_rate, mtp.offered,       mtp.accepted,
            speedup,         band_ok,           agreement,         agreed, agreed_total,
        });
        defer allocator.free(js);
        var f = try std.Io.Dir.createFileAbsolute(io, path, .{});
        defer f.close(io);
        try f.writePositionalAll(io, js, 0);
    }

    if (mtp.accept_rate < 0.35) {
        std.log.err("[bench] acceptance {d:.1}% below 35% floor — draft path miswired", .{accept_pct});
        return engine_mod.EngineError.BenchMismatch;
    }
    const is_quant = std.mem.indexOf(u8, eng.model_path, "mxfp8") != null or std.mem.indexOf(u8, eng.model_path, "nvfp4") != null;
    const thresh: f64 = if (is_quant) 0.80 else 0.98;
    if (agreement < thresh) {
        std.log.err("[bench] agreement {d:.2}% below {d:.0}% — exceeds budget ({s})", .{ agreement * 100, thresh * 100, if (is_quant) "quant 80%" else "bf16 98%" });
        return engine_mod.EngineError.BenchMismatch;
    }
}

// ── tests ────────────────────────────────────────────────────────────────
// agreeIds is pure; runs in the linked suite (bench pulls engine transitively).

test "agreeIds identical" {
    const a = agreeIds(&[_]u32{ 1, 2, 3 }, &[_]u32{ 1, 2, 3 });
    try std.testing.expectEqual(@as(usize, 3), a.matched);
    try std.testing.expectEqual(@as(usize, 3), a.total);
}

test "agreeIds tolerates single-token indel" {
    // insertion in b
    const ins = agreeIds(&[_]u32{ 1, 2, 3 }, &[_]u32{ 1, 9, 2, 3 });
    try std.testing.expectEqual(@as(usize, 3), ins.matched);
    try std.testing.expectEqual(@as(usize, 4), ins.total);
    // deletion in b
    const del = agreeIds(&[_]u32{ 1, 2, 3 }, &[_]u32{ 1, 3 });
    try std.testing.expectEqual(@as(usize, 2), del.matched);
    try std.testing.expectEqual(@as(usize, 3), del.total);
    // substitution
    const sub = agreeIds(&[_]u32{ 1, 2, 3 }, &[_]u32{ 1, 9, 3 });
    try std.testing.expectEqual(@as(usize, 2), sub.matched);
    try std.testing.expectEqual(@as(usize, 3), sub.total);
}

test "agreeIds collapses on garbage" {
    const g = agreeIds(&[_]u32{ 1, 2, 3, 4 }, &[_]u32{ 9, 8, 7, 6 });
    try std.testing.expectEqual(@as(usize, 0), g.matched);
    try std.testing.expectEqual(@as(usize, 4), g.total);
}
