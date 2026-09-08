const std = @import("std");
const mlx = @import("mlx.zig");
const weights = @import("weights.zig");

test {
    // Force full analysis so model/sample/mtp/engine/bench tests join this (linked) runner.
    std.testing.refAllDecls(@import("model.zig"));
    std.testing.refAllDecls(@import("sample.zig"));
    std.testing.refAllDecls(@import("mtp.zig"));
    std.testing.refAllDecls(@import("engine.zig"));
    std.testing.refAllDecls(@import("bench.zig"));
    std.testing.refAllDecls(@import("apc.zig"));
}

test "mlx metal is available and simple op works" {
    var avail: bool = false;
    try mlx.check(mlx.mlx_metal_is_available(&avail));
    // On Apple Silicon, should be true; on CI (Linux) false is okay — just test the call succeeds
    std.debug.print("mlx_metal_is_available={}\n", .{avail});

    // Simple array test: 1.5 + 2.5 = 4.0
    const s = mlx.gpuStream();
    const shape = [_]c_int{2};
    const data = [_]f32{ 1.5, 2.5 };
    const a = mlx.mlx_array_new_data(@ptrCast(&data), &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(a);
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_sum(&sum, a, false, s));
    try mlx.check(mlx.mlx_array_eval(sum));
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, sum));
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out, 1e-6);
    try mlx.checkError();
}

test "mlx version string" {
    var str = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(str);
    try mlx.check(mlx.mlx_version(&str));
    const cstr = mlx.mlx_string_data(str);
    const slice = std.mem.span(cstr);
    std.debug.print("mlx version: {s}\n", .{slice});
    try std.testing.expect(slice.len > 0);
}

test "weights sharded fixture roundtrips shapes and values" {
    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = "/tmp/mlx-runner-weights-fixture";
    std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    try std.Io.Dir.cwd().createDirPath(io, dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    // 3 kept F32 tensors + 1 dropped vision tensor, header JSON by hand.
    const header =
        "{\"__metadata__\":{}," ++
        "\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}," ++
        "\"b\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[8,24]}," ++
        "\"c\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[24,28]}," ++
        "\"vision_tower.q\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[28,32]}}";
    var file_bytes: std.ArrayList(u8) = .empty;
    defer file_bytes.deinit(alloc);
    var lenbuf: [8]u8 = undefined;
    std.mem.writeInt(u64, &lenbuf, header.len, .little);
    try file_bytes.appendSlice(alloc, &lenbuf);
    try file_bytes.appendSlice(alloc, header);
    const vals = [_]f32{ 1.5, -2.0, 1, 2, 3, 4, 7.0, 9.0 };
    try file_bytes.appendSlice(alloc, std.mem.asBytes(&vals));
    {
        var f = try std.Io.Dir.createFileAbsolute(io, dir ++ "/t.safetensors", .{});
        defer f.close(io);
        try f.writePositionalAll(io, file_bytes.items, 0);
    }
    const index =
        "{\"metadata\":{},\"weight_map\":{" ++
        "\"a\":\"t.safetensors\",\"b\":\"t.safetensors\"," ++
        "\"c\":\"t.safetensors\",\"vision_tower.q\":\"t.safetensors\"}}";
    {
        var f = try std.Io.Dir.createFileAbsolute(io, dir ++ "/model.safetensors.index.json", .{});
        defer f.close(io);
        try f.writePositionalAll(io, index, 0);
    }

    var wm = try weights.WeightMap.load(alloc, io, dir);
    defer wm.deinit();
    // vision dropped, bare names gain the language_model. prefix
    try std.testing.expectEqual(@as(usize, 3), wm.count());
    try std.testing.expect(wm.get("vision_tower.q") == null);
    try std.testing.expect(wm.get("language_model.vision_tower.q") == null);

    const a = wm.get("language_model.a") orelse return error.MissingTensor;
    try mlx.check(mlx.mlx_array_eval(a));
    try std.testing.expectEqual(@as(usize, 1), mlx.mlx_array_ndim(a));
    try std.testing.expectEqual(@as(c_int, 2), mlx.getShape(a)[0]);
    const ap = mlx.mlx_array_data_float32(a).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), ap[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), ap[1], 1e-6);

    const b = wm.get("language_model.b") orelse return error.MissingTensor;
    try mlx.check(mlx.mlx_array_eval(b));
    try std.testing.expectEqual(@as(usize, 2), mlx.mlx_array_ndim(b));
    try std.testing.expectEqual(@as(c_int, 2), mlx.getShape(b)[0]);
    try std.testing.expectEqual(@as(c_int, 2), mlx.getShape(b)[1]);
    const bp = mlx.mlx_array_data_float32(b).?;
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), bp[2], 1e-6);

    const c = wm.get("language_model.c") orelse return error.MissingTensor;
    try mlx.check(mlx.mlx_array_eval(c));
    const cp = mlx.mlx_array_data_float32(c).?;
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), cp[0], 1e-6);
    try mlx.checkError();
}

test "weights load fails cleanly on missing index" {
    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.testing.expectError(error.FileNotFound, weights.WeightMap.load(alloc, io, "/tmp/mlx-runner-no-such-model-dir"));
}
