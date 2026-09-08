const std = @import("std");
const sampling = @import("sampling.zig");
const engine_mod = @import("engine.zig");
const server = @import("server.zig");
const mlx = @import("mlx.zig");
const config_mod = @import("config.zig");

const VERSION = "0.1.1-zig";

fn printUsage(io: std.Io) !void {
    var buf: [8192]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.writeAll(
        \\mlx-runner — MLX-native Metal LLM engine (Zig 0.17, Qwen3.5 GDN, no Python)
        \\
        \\Usage: mlx-runner <command> [options]
        \\       mlx-runner [options]
        \\
        \\Commands:
        \\  run <model>         Use a local checkpoint dir and chat (alias for --model)
        \\
        \\Options:
        \\  --model <dir>       Local checkpoint dir, repeatable for --serve (default: Qwen3.8-27B; ids are dir basenames)
        \\  --prompt <text>     Single prompt (non-interactive)
        \\  --chat              Chat REPL (stdin)
        \\  --serve             Start HTTP server (OpenAI + Anthropic compat)
        \\  --host <ip>         Bind host (default: 127.0.0.1)
        \\  --port <n>          Bind port (default: 11234)
        \\  --max-resident-models <n>  Serve: live engines before LRU evict (default: 1)
        \\  --temp <f>          Temperature (default: from generation_config or 1.0)
        \\  --top-p <f>         Top-p nucleus (default 1.0)
        \\  --top-k <n>         Top-k (default 0 = off)
        \\  --min-p <f>         Min-p (default 0.0)
        \\  --seed <n>          RNG seed
        \\  --max-tokens <n>    Max tokens to generate (default: 256)
        \\  --ctx-size <n>      Max context length (0 = model max, 262144 for Qwen3.8)
        \\  --kv-quant <mode>   Accepted for CLI stability; only off is supported (else UnsupportedDtype)
        \\  --stream            Stream tokens (prompt mode, v1 buffered)
        \\  --prefix-cache-entries <n>  Hot prefix cache LRU capacity (default: 32)
        \\  --prefix-cache-mem <size>   Hot cache byte budget (default: 2GB, e.g. 512MB)
        \\  --prefix-cache-disk <size>  SSD tier budget (default: off, e.g. 10GB)
        \\  --apc-disk <size>           APC disk quota (default: off, e.g. 10GB)
        \\  --apc-disk-dir <path>       APC disk location (default: ~/.mlx-runner/apc-disk/<hash>)
        \\  --no-vision         Accepted no-op (text-only engine; vision always off)
        \\  --mtp               Enable native MTP speculative decoding (default: off)
        \\  --no-mtp            Disable MTP speculative decoding
        \\  --bench             Base-vs-MTP benchmark (needs --model); implies temp 0
        \\  --bench-out <f>     Write bench JSON to f (table always goes to stdout)
        \\  --tp <n>            Tensor parallel shards (stub v1, default 1)
        \\  --pipeline <n>      Pipeline stages (stub v1, default 1)
        \\  --config <path>     Per-model config JSON (alias + sampling + per-model ctx/mtp; see README)
        \\  --version           Print version and exit
        \\  --help              Show this help
        \\
        \\Sampling priority: request > CLI > per-model config > generation_config.json > hardcoded (1.0/1.0/0/0)
        \\Qwen3.8 defaults from generation_config.json: temp 1.0, top_p 0.95, top_k 20.
        \\Examples:
        \\  mlx-runner --model ~/opt/models/mlx_models/Qwen3.8-27B --prompt "The capital of France is" --temp 0 --max-tokens 10
        \\  mlx-runner --model ~/opt/models/mlx_models/Qwen3.8-27B --prompt "Hello" --mtp --stream
        \\  mlx-runner --model ~/opt/models/mlx_models/Qwen3.8-27B --bench --bench-out bench.json
        \\  mlx-runner --config models.json --serve --port 11234
        \\  # models.json: {"models":[{"path":"~/opt/.../Qwen3.8-27B","alias":"qwen-main","sampling":{"temperature":0.7}}]}
        \\
    );
    try w.interface.flush();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    // Materialize args from Init
    var args_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_iter.deinit();
    var args_list: std.ArrayList([]const u8) = .empty;
    defer {
        for (args_list.items) |a| allocator.free(a);
        args_list.deinit(allocator);
    }
    while (args_iter.next()) |arg| {
        try args_list.append(allocator, try allocator.dupe(u8, arg));
    }
    const args = args_list.items;

    if (args.len == 1) {
        try printUsage(io);
        return;
    }

    var models: std.ArrayList([]const u8) = .empty;
    defer models.deinit(allocator);
    var prompt: ?[]const u8 = null;
    var chat = false;
    var serve = false;
    var host: []const u8 = "127.0.0.1";
    var port: u16 = 11234;
    var max_resident: usize = 1;
    var max_tokens: u32 = 256;
    var ctx_size: u32 = 0;
    var kv_quant: []const u8 = "off";
    var stream = false;
    var prefix_cache_entries: u32 = 32;
    var prefix_cache_mem: []const u8 = "2GB";
    var prefix_cache_disk: ?[]const u8 = null;
    var apc_disk_quota: ?[]const u8 = null;
    var apc_disk_dir: ?[]const u8 = null;
    var no_vision = false;
    var mtp: ?bool = null;
    var mtp_gamma: u32 = 1;
    var bench = false;
    var bench_out: ?[]const u8 = null;
    var tp: u32 = 1;
    var pipeline: u32 = 1;
    var metal_check = false;
    var config_path: ?[]const u8 = null;

    var cli_sampling = sampling.SamplingParams{};

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printUsage(io);
            return;
        } else if (std.mem.eql(u8, arg, "--version")) {
            var buf: [512]u8 = undefined;
            var w = std.Io.File.stdout().writer(io, &buf);
            try w.interface.print("mlx-runner {s}\n", .{VERSION});
            try w.interface.print("zig 0.17, mlx 0.32.2, model: qwen3_5 27B (GDN, bf16, native, no Python)\n", .{});
            try w.interface.flush();
            return;
        } else if (std.mem.eql(u8, arg, "--model") and i + 1 < args.len) {
            i += 1;
            try models.append(allocator, args[i]);
        } else if (std.mem.eql(u8, arg, "--max-resident-models") and i + 1 < args.len) {
            i += 1;
            max_resident = std.fmt.parseInt(usize, args[i], 10) catch 1;
        } else if (std.mem.eql(u8, arg, "--prompt") and i + 1 < args.len) {
            i += 1;
            prompt = args[i];
        } else if (std.mem.eql(u8, arg, "--chat")) {
            chat = true;
        } else if (std.mem.eql(u8, arg, "--serve")) {
            serve = true;
        } else if (std.mem.eql(u8, arg, "--host") and i + 1 < args.len) {
            i += 1;
            host = args[i];
        } else if (std.mem.eql(u8, arg, "--port") and i + 1 < args.len) {
            i += 1;
            port = std.fmt.parseInt(u16, args[i], 10) catch 11234;
        } else if (std.mem.eql(u8, arg, "--temp") and i + 1 < args.len) {
            i += 1;
            cli_sampling.temp = std.fmt.parseFloat(f32, args[i]) catch 1.0;
        } else if (std.mem.eql(u8, arg, "--top-p") and i + 1 < args.len) {
            i += 1;
            cli_sampling.top_p = std.fmt.parseFloat(f32, args[i]) catch 1.0;
        } else if (std.mem.eql(u8, arg, "--top-k") and i + 1 < args.len) {
            i += 1;
            cli_sampling.top_k = std.fmt.parseInt(u32, args[i], 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--min-p") and i + 1 < args.len) {
            i += 1;
            cli_sampling.min_p = std.fmt.parseFloat(f32, args[i]) catch 0.0;
        } else if (std.mem.eql(u8, arg, "--seed") and i + 1 < args.len) {
            i += 1;
            cli_sampling.seed = std.fmt.parseInt(u64, args[i], 10) catch null;
        } else if (std.mem.eql(u8, arg, "--max-tokens") and i + 1 < args.len) {
            i += 1;
            max_tokens = std.fmt.parseInt(u32, args[i], 10) catch 256;
        } else if (std.mem.eql(u8, arg, "--ctx-size") and i + 1 < args.len) {
            i += 1;
            ctx_size = std.fmt.parseInt(u32, args[i], 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--kv-quant") and i + 1 < args.len) {
            i += 1;
            kv_quant = args[i];
        } else if (std.mem.eql(u8, arg, "--stream")) {
            stream = true;
        } else if (std.mem.eql(u8, arg, "--no-stream")) {
            stream = false;
        } else if (std.mem.eql(u8, arg, "--prefix-cache-entries") and i + 1 < args.len) {
            i += 1;
            prefix_cache_entries = std.fmt.parseInt(u32, args[i], 10) catch 32;
        } else if (std.mem.eql(u8, arg, "--prefix-cache-mem") and i + 1 < args.len) {
            i += 1;
            prefix_cache_mem = args[i];
        } else if (std.mem.eql(u8, arg, "--prefix-cache-disk") and i + 1 < args.len) {
            i += 1;
            prefix_cache_disk = args[i];
        } else if ((std.mem.eql(u8, arg, "--apc-disk") or std.mem.eql(u8, arg, "--apc-cache-disk") or std.mem.eql(u8, arg, "--apc-disk-quota")) and i + 1 < args.len) {
            i += 1;
            apc_disk_quota = args[i];
        } else if ((std.mem.eql(u8, arg, "--apc-disk-dir") or std.mem.eql(u8, arg, "--apc-cache-dir")) and i + 1 < args.len) {
            i += 1;
            apc_disk_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--no-vision")) {
            no_vision = true;
        } else if (std.mem.eql(u8, arg, "--mtp")) {
            mtp = true;
        } else if (std.mem.eql(u8, arg, "--no-mtp")) {
            mtp = false;
        } else if (std.mem.eql(u8, arg, "--mtp-gamma") and i + 1 < args.len) {
            i += 1;
            mtp_gamma = std.fmt.parseInt(u32, args[i], 10) catch 1;
            if (mtp_gamma < 1) mtp_gamma = 1;
            if (mtp_gamma > 2) mtp_gamma = 2;
        } else if (std.mem.eql(u8, arg, "--bench")) {
            bench = true;
        } else if (std.mem.eql(u8, arg, "--bench-out") and i + 1 < args.len) {
            i += 1;
            bench_out = args[i];
        } else if (std.mem.eql(u8, arg, "--tp") and i + 1 < args.len) {
            i += 1;
            tp = std.fmt.parseInt(u32, args[i], 10) catch 1;
        } else if (std.mem.eql(u8, arg, "--pipeline") and i + 1 < args.len) {
            i += 1;
            pipeline = std.fmt.parseInt(u32, args[i], 10) catch 1;
        } else if (std.mem.eql(u8, arg, "--config") and i + 1 < args.len) {
            i += 1;
            config_path = args[i];
        } else if (std.mem.eql(u8, arg, "--metal-check")) {
            metal_check = true;
        } else if (arg.len > 0 and arg[0] == '-') {
            std.log.warn("unknown flag {s}", .{arg});
        } else {
            if (std.mem.eql(u8, arg, "run") and i + 1 < args.len) {
                i += 1;
                try models.append(allocator, args[i]);
            }
        }
    }

    // Load per-model config file if --config given
    var file_cfg: ?config_mod.FileConfig = null;
    defer if (file_cfg) |*fc| fc.deinit(allocator);
    if (config_path) |cp| {
        file_cfg = config_mod.loadFile(allocator, io, cp) catch |e| {
            std.log.err("failed to load config {s}: {any}", .{ cp, e });
            return e;
        };
        // Apply file-level server/cache overrides if CLI didn't set them explicitly
        if (file_cfg.?.server) |s| {
            if (s.host) |h| host = h;
            if (s.port) |p| port = p;
            if (s.max_resident_models) |m| max_resident = m;
        }
        if (file_cfg.?.cache) |c| {
            if (c.prefix_cache_entries) |v| prefix_cache_entries = v;
            if (c.prefix_cache_mem) |v| prefix_cache_mem = v;
            if (c.prefix_cache_disk) |v| prefix_cache_disk = v;
            if (c.apc_disk) |v| apc_disk_quota = v;
            if (c.apc_disk_dir) |v| apc_disk_dir = v;
        }
    }

    // Build final model specs: config file models + CLI --model flags
    var specs: std.ArrayList(server.ModelSpec) = .empty;
    defer specs.deinit(allocator);
    // From config file
    if (file_cfg) |fc| {
        for (fc.models) |m| {
            var sp: ?sampling.SamplingParams = null;
            var mt: ?u32 = null;
            if (m.sampling) |s| {
                sp = s.toSamplingParams();
                mt = s.max_tokens;
            }
            var sc = server.ModelSpec{ .dir = m.path, .alias = m.alias, .sampling = sp, .max_tokens = mt };
            if (m.config) |c| {
                sc.ctx_size = c.ctx_size;
                sc.mtp = c.mtp;
                sc.mtp_gamma = c.mtp_gamma;
            }
            try specs.append(allocator, sc);
        }
    }
    // From CLI --model flags (use global cli_sampling as per-model sampling if non-default)
    const cli_has_sampling = blk: {
        const def = sampling.SamplingParams{};
        break :blk cli_sampling.temp != def.temp or cli_sampling.top_p != def.top_p or cli_sampling.top_k != def.top_k or cli_sampling.min_p != def.min_p or cli_sampling.seed != null;
    };
    for (models.items) |d| {
        var sp: ?sampling.SamplingParams = null;
        if (cli_has_sampling) sp = cli_sampling;
        try specs.append(allocator, .{ .dir = d, .alias = null, .sampling = sp });
    }
    // If still empty, use default model
    if (specs.items.len == 0) {
        const def_dir = if (std.c.getenv("MLX_RUNNER_MODEL")) |p| try allocator.dupe(u8, std.mem.span(p)) else if (std.c.getenv("HOME")) |h| blk: {
            const home = std.mem.span(h);
            break :blk try std.fs.path.join(allocator, &.{ home, "opt/models/mlx_models/Qwen3.8-27B" });
        } else try allocator.dupe(u8, "models/Qwen3.8-27B");
        try specs.append(allocator, .{ .dir = def_dir });
    }
    // Validate alias/basename uniqueness (alias if present else basename)
    for (specs.items, 0..) |a, x| {
        const aid = a.alias orelse std.fs.path.basename(a.dir);
        for (specs.items[0..x]) |b| {
            const bid = b.alias orelse std.fs.path.basename(b.dir);
            if (std.mem.eql(u8, aid, bid)) {
                std.log.err("duplicate model id {s} (alias or basename must differ)", .{aid});
                return error.DuplicateModelId;
            }
        }
    }
    // For non-serve single-model run, we need models list for compat
    models.clearRetainingCapacity();
    for (specs.items) |s| try models.append(allocator, s.dir);
    if (metal_check) {
        // Direct mlx-c test — proves zig links libmlxc.dylib (Metal)
        var avail: bool = false;
        try mlx.check(mlx.mlx_metal_is_available(&avail));
        var buf: [256]u8 = undefined;
        var w = std.Io.File.stdout().writer(io, &buf);
        try w.interface.print("mlx-c metal_is_available={}\n", .{avail});
        // Simple Metal op: 1.5 + 2.5 via mlx_sum (runs on GPU stream)
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
        try w.interface.print("mlx_sum [1.5,2.5] = {d} (Metal via mlx-c)\n", .{out});
        try w.interface.flush();
        // Also install error handler like mlx-infer
        mlx.installErrorHandler();
        var str = mlx.mlx_string_new();
        defer _ = mlx.mlx_string_free(str);
        try mlx.check(mlx.mlx_version(&str));
        const ver = std.mem.span(mlx.mlx_string_data(str));
        std.log.info("mlx-c version {s}, metal {s}", .{ ver, if (avail) "yes" else "no" });
        return;
    }

    if (tp > 1 or pipeline > 1) {
        std.log.info("[distributed] tp={d} pipeline={d} — v1 stub: single-device", .{ tp, pipeline });
    }

    // Build base EngineConfig from CLI + file global overrides (ctx-size 0 = model max, assume RAM sufficient)
    const base_cfg = engine_mod.EngineConfig{
        .model = specs.items[0].dir,
        .ctx_size = specs.items[0].ctx_size orelse ctx_size,
        .kv_quant = kv_quant,
        .prefix_cache_entries = prefix_cache_entries,
        .prefix_cache_mem = prefix_cache_mem,
        .prefix_cache_disk = prefix_cache_disk,
        .apc_disk_quota = apc_disk_quota,
        .apc_disk_dir = apc_disk_dir,
        .no_vision = no_vision,
        .mtp = specs.items[0].mtp orelse (mtp orelse false),
        .mtp_gamma = specs.items[0].mtp_gamma orelse mtp_gamma,
    };
    const base_sampling = if (specs.items[0].sampling) |s| s else cli_sampling;

    // Per-model sampling for single-model runs (prompt/chat/bench) uses first spec
    const cfg = base_cfg;
    const eff_sampling = base_sampling;

    if (bench) {
        try @import("bench.zig").run(allocator, io, cfg, eff_sampling, bench_out);
        return;
    }

    if (serve) {
        // Build per-model registry from specs (alias + per-model sampling/config)
        var reg = try server.ModelRegistry.initWithSpecs(allocator, io, base_cfg, cli_sampling, specs.items, max_resident);
        defer reg.deinit();
        for (specs.items) |sp| {
            const id = sp.alias orelse std.fs.path.basename(sp.dir);
            std.log.info("mlx-runner model {s} ({s})", .{ id, sp.dir });
            if (sp.sampling) |s| std.log.info("  sampling {s}: temp={d} top_p={d} top_k={d} min_p={d} seed={?d}", .{ id, s.temp, s.top_p, s.top_k, s.min_p, s.seed });
        }
        std.log.info("mlx-runner serving on http://{s}:{d} (max {d} resident)", .{ host, port, max_resident });
        std.log.info("  kv-quant {s}  sampling temp={d} top_p={d} top_k={d} min_p={d}", .{ kv_quant, eff_sampling.temp, eff_sampling.top_p, eff_sampling.top_k, eff_sampling.min_p });
        std.log.info("  hot cache {d} entries  disk {s}  apc-disk {s} ({s})  distributed tp={d} pipeline={d}", .{ prefix_cache_entries, if (prefix_cache_disk) |d| d else "off", if (apc_disk_quota) |q| q else "off", if (apc_disk_dir) |d| d else "default", tp, pipeline });
        try server.serve(allocator, io, &reg, host, port);
        return;
    }

    if (prompt) |p| {
        var eng = try engine_mod.Engine.init(allocator, io, cfg, eff_sampling);
        defer eng.deinit();
        std.log.info("model {s} ctx {d} temp {d} top_p {d} top_k {d}", .{ eng.config.model, eng.model_max_ctx, eng.base_sampling.temp, eng.base_sampling.top_p, eng.base_sampling.top_k });
        if (stream) {
            const out = try eng.generate(io, p, null, max_tokens, null, false);
            defer allocator.free(out);
            var buf: [4096]u8 = undefined;
            var w = std.Io.File.stdout().writer(io, &buf);
            for (out) |ch| try w.interface.writeByte(ch);
            try w.interface.writeByte('\n');
            try w.interface.flush();
            std.log.info("[stream] <- {d} chars", .{out.len});
        } else {
            const out = try eng.generate(io, p, null, max_tokens, null, false);
            defer allocator.free(out);
            var buf: [4096]u8 = undefined;
            var w = std.Io.File.stdout().writer(io, &buf);
            try w.interface.writeAll(out);
            try w.interface.writeByte('\n');
            try w.interface.flush();
        }
        return;
    }

    if (chat) {
        var eng = try engine_mod.Engine.init(allocator, io, cfg, eff_sampling);
        defer eng.deinit();
        var out_buf: [4096]u8 = undefined;
        var out_w = std.Io.File.stdout().writer(io, &out_buf);
        try out_w.interface.print("mlx-runner chat — model {s} ctx {d} — /exit to quit\n", .{ specs.items[0].dir, eng.model_max_ctx });
        try out_w.interface.flush();
        var messages: std.ArrayList([]const u8) = .empty;
        defer {
            for (messages.items) |m| allocator.free(m);
            messages.deinit(allocator);
        }
        var in_buf: [4096]u8 = undefined;
        var stdin = std.Io.File.stdin().reader(io, &in_buf);
        while (true) {
            var prompt_buf: [128]u8 = undefined;
            var prompt_w = std.Io.File.stdout().writer(io, &prompt_buf);
            try prompt_w.interface.writeAll("\n> ");
            try prompt_w.interface.flush();
            const line = stdin.interface.takeDelimiter('\n') catch break orelse break;
            const trimmed = std.mem.trim(u8, line, " \r\n\t");
            if (trimmed.len == 0) continue;
            if (std.mem.eql(u8, trimmed, "/exit") or std.mem.eql(u8, trimmed, "/quit") or std.mem.eql(u8, trimmed, ":q")) break;
            if (std.mem.eql(u8, trimmed, "/clear")) {
                for (messages.items) |m| allocator.free(m);
                messages.clearRetainingCapacity();
                var clear_buf: [128]u8 = undefined;
                var clear_w = std.Io.File.stdout().writer(io, &clear_buf);
                try clear_w.interface.writeAll("[cleared]\n");
                try clear_w.interface.flush();
                continue;
            }
            const user_json = try std.fmt.allocPrint(allocator, "{{\"role\":\"user\",\"content\":\"{s}\"}}", .{trimmed});
            defer allocator.free(user_json);
            try messages.append(allocator, try allocator.dupe(u8, user_json));
            var json_buf: std.ArrayList(u8) = .empty;
            defer json_buf.deinit(allocator);
            try json_buf.append(allocator, '[');
            for (messages.items, 0..) |m, idx| {
                if (idx > 0) try json_buf.append(allocator, ',');
                try json_buf.appendSlice(allocator, m);
            }
            try json_buf.append(allocator, ']');
            const out = try eng.generate(io, null, json_buf.items, max_tokens, null, false);
            defer allocator.free(out);
            var resp_buf: [8192]u8 = undefined;
            var resp_w = std.Io.File.stdout().writer(io, &resp_buf);
            try resp_w.interface.print("\n{s}\n", .{out});
            try resp_w.interface.flush();
            const assistant_json = try std.fmt.allocPrint(allocator, "{{\"role\":\"assistant\",\"content\":\"{s}\"}}", .{out});
            try messages.append(allocator, assistant_json);
        }
        return;
    }

    try printUsage(io);
}
