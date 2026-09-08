const std = @import("std");

test {
    std.testing.refAllDecls(@import("sampling.zig"));
    std.testing.refAllDecls(@import("cache.zig"));
    std.testing.refAllDecls(@import("tokenizer.zig"));
    std.testing.refAllDecls(@import("weights.zig"));
    std.testing.refAllDecls(@import("http_api.zig"));
    std.testing.refAllDecls(@import("kv_checkpoint.zig"));
}

test "sampling priority" {
    const s = @import("sampling.zig");
    const base = s.SamplingParams{ .temp = 0.7, .top_p = 0.9 };
    const eff = base.mergedWithRequest(.{ .temperature = 0.2 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), eff.temp, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), eff.top_p, 0.001);
}
