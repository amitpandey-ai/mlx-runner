const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "mlx-runner requires Zig 0.17 (have {d}.{d}.{d})",
            .{ builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
    }
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } },
        },
    });
    const optimize = b.standardOptimizeOption(.{});

    // Verify lib/mlx stage (pip-based, non-NAX, like mlx-infer's check but no NAX assert)
    if (target.result.os.tag == .macos) verifyMlxStage(b);

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
    });
    addMlxLib(b, exe_mod, target);

    const exe = b.addExecutable(.{
        .name = "mlx-runner",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Run mlx-runner");
    run_step.dependOn(&run_cmd.step);

    // hermetic tests (no MLX) — sampling, cache, chat template
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    const unit_tests = b.addTest(.{ .root_module = test_mod });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // mlx-c integration test (needs lib/mlx)
    const mlx_test_mod = b.createModule(.{
        .root_source_file = b.path("src/mlx_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
    });
    addMlxLib(b, mlx_test_mod, target);
    const mlx_tests = b.addTest(.{ .root_module = mlx_test_mod });
    const run_mlx_tests = b.addRunArtifact(mlx_tests);
    const mlx_test_step = b.step("test-mlx", "Run mlx-c integration tests (Metal)");
    mlx_test_step.dependOn(&run_mlx_tests.step);
}

fn addMlxLib(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    _ = target;
    mod.addIncludePath(b.path("lib/mlx/include"));
    mod.addLibraryPath(b.path("lib/mlx/lib"));
    mod.linkSystemLibrary("mlx", .{ .use_pkg_config = .no });
    mod.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    mod.addRPath(b.path("lib/mlx/lib"));
    // Also add @loader_path rpaths like mlx-infer (for zig-out + .zig-cache)
    mod.addRPath(.{ .cwd_relative = "@loader_path/../../lib/mlx/lib" });
    mod.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/mlx/lib" });
    // Frameworks required by libmlx (Metal)
    // Resolve SDK frameworks path
    const frameworks: ?[]const u8 = blk: {
        var code: u8 = undefined;
        const stdout = b.runAllowFail(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }, &code, .inherit) catch break :blk null;
        const sdk = std.mem.trim(u8, stdout, " \n\r\t");
        if (sdk.len == 0) break :blk null;
        break :blk b.fmt("{s}/System/Library/Frameworks", .{sdk});
    };
    if (frameworks) |fw| mod.addFrameworkPath(.{ .cwd_relative = fw });
    mod.linkFramework("Metal", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("IOKit", .{});
    mod.linkFramework("IOSurface", .{});
    mod.linkFramework("CoreFoundation", .{});
}

fn buildRootHandle(b: *std.Build) std.Io.Dir {
    return b.root.root_dir.handle;
}
fn verifyMlxStage(b: *std.Build) void {
    const io = b.graph.io;
    const root = buildRootHandle(b);
    const ok = blk: {
        root.access(io, "lib/mlx/lib/libmlxc.dylib", .{}) catch break :blk false;
        root.access(io, "lib/mlx/lib/libmlx.dylib", .{}) catch break :blk false;
        root.access(io, "lib/mlx/lib/mlx.metallib", .{}) catch break :blk false;
        break :blk true;
    };
    if (!ok) {
        std.debug.print("\n[mlx-runner] lib/mlx not staged — run: bash scripts/build-mlx.sh\n\n", .{});
        std.process.exit(1);
    }
}
