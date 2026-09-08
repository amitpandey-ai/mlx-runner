const std = @import("std");
const engine_mod = @import("engine.zig");
const sampling = @import("sampling.zig");
const api = @import("http_api.zig");

/// OpenAI + Anthropic compatible HTTP server (sequential accept loop; the
/// Metal queue serializes inference anyway, so one in-flight request max).
/// Routes: GET /health, GET /v1/models, POST /v1/chat/completions,
/// POST /v1/completions, POST /v1/messages.
/// NOTE: generation itself is monolithic (engine.generate returns full text),
/// so `stream:true` emits format-compatible SSE after the fact, split into
/// small UTF-8-safe chunks. True token-by-token streaming needs an engine
/// token-callback refactor (tracked, not this file).

/// Multi-model registry: one Engine per configured checkpoint dir, lazy-init
/// on first request, LRU-evicted past `max_resident` (27B bf16 is ~55 GB
/// resident; quants ~30 GB — two 27Bs do not fit this 128 GB box together).
/// ids borrows the caller's model-dir slices (main blocks in serve).
pub const ModelSpec = struct {
    dir: []const u8,
    alias: ?[]const u8 = null,
    sampling: ?sampling.SamplingParams = null,
    max_tokens: ?u32 = null,
    ctx_size: ?u32 = null,
    mtp: ?bool = null,
    mtp_gamma: ?u32 = null,

    pub fn id(self: ModelSpec) []const u8 {
        return self.alias orelse std.fs.path.basename(self.dir);
    }
};

pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    base_cfg: engine_mod.EngineConfig,
    base_sampling: sampling.SamplingParams,
    specs: []const ModelSpec,
    engines: std.StringHashMap(*engine_mod.Engine),
    lru: std.ArrayList([]const u8), // resident ids, front = oldest
    max_resident: usize,

    pub const Resolved = struct { eng: *engine_mod.Engine, id: []const u8, max_tokens: ?u32 = null };
    pub const RegError = error{ UnknownModel, NoModelsConfigured };

    pub fn init(allocator: std.mem.Allocator, io: std.Io, base_cfg: engine_mod.EngineConfig, base_sampling: sampling.SamplingParams, dirs: []const []const u8, max_resident: usize) ModelRegistry {
        // Backward compat: dirs without per-model overrides
        const specs = allocator.alloc(ModelSpec, dirs.len) catch @panic("oom");
        for (dirs, 0..) |d, i| specs[i] = .{ .dir = d };
        return .{
            .allocator = allocator,
            .io = io,
            .base_cfg = base_cfg,
            .base_sampling = base_sampling,
            .specs = specs,
            .engines = std.StringHashMap(*engine_mod.Engine).init(allocator),
            .lru = .empty,
            .max_resident = @max(1, max_resident),
        };
    }

    pub fn initWithSpecs(allocator: std.mem.Allocator, io: std.Io, base_cfg: engine_mod.EngineConfig, base_sampling: sampling.SamplingParams, specs: []const ModelSpec, max_resident: usize) !ModelRegistry {
        const dup = try allocator.dupe(ModelSpec, specs);
        return .{
            .allocator = allocator,
            .io = io,
            .base_cfg = base_cfg,
            .base_sampling = base_sampling,
            .specs = dup,
            .engines = std.StringHashMap(*engine_mod.Engine).init(allocator),
            .lru = .empty,
            .max_resident = @max(1, max_resident),
        };
    }

    pub fn deinit(self: *ModelRegistry) void {
        var it = self.engines.valueIterator();
        while (it.next()) |ep| {
            ep.*.deinit();
            self.allocator.destroy(ep.*);
        }
        self.engines.deinit();
        self.lru.deinit(self.allocator);
        self.allocator.free(self.specs);
    }

    pub fn defaultId(self: *const ModelRegistry) []const u8 {
        return self.specs[0].id();
    }

    /// Resolve a request `model` name (null = default) to a live engine,
    /// loading and evicting as needed. Returned id borrows `specs`.
    pub fn resolve(self: *ModelRegistry, name: ?[]const u8) !Resolved {
        if (self.specs.len == 0) return RegError.NoModelsConfigured;
        const want = name orelse self.defaultId();
        var spec: ?ModelSpec = null;
        var id: []const u8 = undefined;
        for (self.specs) |s| {
            const mid = s.id();
            if (std.mem.eql(u8, mid, want)) {
                spec = s;
                id = mid;
                break;
            }
        }
        const found = spec orelse return RegError.UnknownModel;
        const model_dir = found.dir;
        if (self.engines.get(id)) |e| {
            self.touch(id);
            return .{ .eng = e, .id = id, .max_tokens = found.max_tokens };
        }
        while (self.engines.count() >= self.max_resident) {
            const old = self.lru.orderedRemove(0);
            if (self.engines.fetchRemove(old)) |kv| {
                std.log.info("[registry] evicting {s}", .{old});
                kv.value.deinit();
                self.allocator.destroy(kv.value);
            }
        }
        std.log.info("[registry] loading {s} ({s})", .{ id, model_dir });
        var cfg = self.base_cfg;
        cfg.model = model_dir;
        if (found.ctx_size) |v| cfg.ctx_size = v;
        if (found.mtp) |v| cfg.mtp = v;
        if (found.mtp_gamma) |v| cfg.mtp_gamma = v;
        const samp = found.sampling orelse self.base_sampling;
        const e = try self.allocator.create(engine_mod.Engine);
        errdefer self.allocator.destroy(e);
        e.* = try engine_mod.Engine.init(self.allocator, self.io, cfg, samp);
        errdefer e.deinit();
        std.log.info("[registry] {s} ready: ctx {d} temp={d} top_p={d} top_k={d} quant={s}", .{ id, e.model_max_ctx, e.base_sampling.temp, e.base_sampling.top_p, e.base_sampling.top_k, @tagName(e.mm.wm.quant.mode) });
        try self.engines.put(id, e);
        try self.lru.append(self.allocator, id);
        return .{ .eng = e, .id = id, .max_tokens = found.max_tokens };
    }

    fn touch(self: *ModelRegistry, id: []const u8) void {
        for (self.lru.items, 0..) |v, i| {
            if (v.ptr == id.ptr) {
                _ = self.lru.orderedRemove(i);
                self.lru.append(self.allocator, id) catch {};
                return;
            }
        }
    }
};

pub fn serve(allocator: std.mem.Allocator, io: std.Io, reg: *ModelRegistry, host: []const u8, port: u16) !void {
    const addr = try std.Io.net.IpAddress.parse(host, port);
    var listener = try addr.listen(io, .{});
    std.log.info("mlx-runner serving {d} model(s) (OpenAI + Anthropic compat)", .{reg.specs.len});
    var ids: u64 = 0;
    while (true) {
        const stream = listener.accept(io) catch continue;
        handleConn(allocator, io, reg, stream, &ids);
        stream.close(io);
    }
}

fn handleConn(allocator: std.mem.Allocator, io: std.Io, reg: *ModelRegistry, stream: std.Io.net.Stream, ids: *u64) void {
    var head_buf: [65536]u8 = undefined;
    var in_reader = stream.reader(io, &head_buf);
    var out_buf: [65536]u8 = undefined;
    var out_writer = stream.writer(io, &out_buf);
    var srv = std.http.Server.init(&in_reader.interface, &out_writer.interface);
    while (true) {
        var req = srv.receiveHead() catch break;
        dispatch(allocator, io, reg, &req, ids);
    }
}

const max_body: usize = 8 * 1024 * 1024;
const default_max_tokens: u32 = 1024;

const json_ct = std.http.Header{ .name = "content-type", .value = "application/json" };
const sse_ct = std.http.Header{ .name = "content-type", .value = "text/event-stream" };
const cors = std.http.Header{ .name = "access-control-allow-origin", .value = "*" };
const no_cache = std.http.Header{ .name = "cache-control", .value = "no-cache" };

fn dispatch(allocator: std.mem.Allocator, io: std.Io, reg: *ModelRegistry, req: *std.http.Server.Request, ids: *u64) void {
    const method = req.head.method;
    const target = req.head.target;
    const path = target[0 .. (std.mem.indexOfScalar(u8, target, '?') orelse target.len)];
    if (method == .OPTIONS) {
        req.respond("", .{ .status = .ok, .extra_headers = &.{cors} }) catch {};
        return;
    }
    const anthropic = std.mem.startsWith(u8, path, "/v1/messages");
    if (method == .GET and std.mem.eql(u8, path, "/health")) {
        jsonOk(req, "{\"status\":\"ok\"}");
        return;
    }
    if (method == .GET and std.mem.eql(u8, path, "/v1/models")) {
        var names: std.ArrayList([]const u8) = .empty;
        defer names.deinit(allocator);
        for (reg.specs) |s| names.append(allocator, s.id()) catch
            return failReq(req, false, .internal_server_error, "failed to build models response");
        const body = api.buildModelsList(allocator, names.items) catch
            return failReq(req, false, .internal_server_error, "failed to build models response");
        defer allocator.free(body);
        jsonOk(req, body);
        return;
    }
    if (method == .POST and std.mem.eql(u8, path, "/v1/chat/completions")) {
        chatRoute(allocator, io, reg, req, ids);
        return;
    }
    if (method == .POST and std.mem.eql(u8, path, "/v1/completions")) {
        completionsRoute(allocator, io, reg, req, ids);
        return;
    }
    if (method == .POST and std.mem.eql(u8, path, "/v1/messages")) {
        messagesRoute(allocator, io, reg, req, ids);
        return;
    }
    if (method != .GET and method != .POST) {
        failReq(req, anthropic, .method_not_allowed, "method not allowed");
        return;
    }
    failReq(req, anthropic, .not_found, "unknown route");
}

fn jsonOk(req: *std.http.Server.Request, body: []const u8) void {
    req.respond(body, .{ .status = .ok, .extra_headers = &.{ json_ct, cors } }) catch {};
}

fn failReq(req: *std.http.Server.Request, anthropic: bool, status: std.http.Status, msg: []const u8) void {
    // Fixed buffer is enough: all messages here are short controlled literals.
    var buf: [1024]u8 = undefined;
    const body = if (anthropic)
        std.fmt.bufPrint(&buf, "{{\"type\":\"error\",\"error\":{{\"type\":\"invalid_request_error\",\"message\":\"{s}\"}}}}", .{msg}) catch return
    else
        std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"invalid_request_error\"}}}}", .{msg}) catch return;
    req.respond(body, .{ .status = status, .extra_headers = &.{ json_ct, cors } }) catch {};
}

fn modelId(model_dir: []const u8) []const u8 {
    return std.fs.path.basename(model_dir);
}

/// Resolve the request's `model` field (null = default) to a live engine.
/// Unknown names get a 400 here; load failures flow to genError (500).
fn resolveRoute(reg: *ModelRegistry, req: *std.http.Server.Request, v: std.json.Value, anthropic: bool) ?ModelRegistry.Resolved {
    const r = reg.resolve(api.optString(v, "model")) catch |e| {
        if (e == error.UnknownModel) {
            failReq(req, anthropic, .bad_request, "unknown model");
            return null;
        }
        genError(req, anthropic, e);
        return null;
    };
    return r;
}

// --- request body ---

fn readBody(allocator: std.mem.Allocator, req: *std.http.Server.Request) ![]u8 {
    if (req.head.expect) |e| {
        if (!std.mem.eql(u8, e, "100-continue")) return error.BadExpectation;
        try req.writeExpectContinue();
    }
    var tmp: [8192]u8 = undefined;
    const r = req.readerExpectNone(&tmp);
    if (r == std.Io.Reader.ending) return try allocator.dupe(u8, "");
    if (req.head.content_length) |n| {
        if (n > max_body) return error.BodyTooLarge;
        const buf = try allocator.alloc(u8, n);
        errdefer allocator.free(buf);
        r.readSliceAll(buf) catch return error.BodyRead;
        return buf;
    }
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(allocator);
    var chunk: [16384]u8 = undefined;
    while (list.items.len < max_body) {
        const n = r.readSliceShort(chunk[0..]) catch return error.BodyRead;
        if (n == 0) break;
        try list.appendSlice(allocator, chunk[0..n]);
    } else return error.BodyTooLarge;
    return list.toOwnedSlice(allocator);
}

// --- generation driver shared by all POST routes ---

const GenOut = struct {
    text: []u8, // owned, stop-trimmed + max-fitted
    prompt_tokens: u64,
    completion_tokens: u64,
    hit_max: bool, // true => length/max_tokens finish
    stopped: bool, // a stop string cut the text
};

fn runGen(
    allocator: std.mem.Allocator,
    io: std.Io,
    eng: *engine_mod.Engine,
    prompt: ?[]const u8,
    messages_json: ?[]const u8,
    max_tokens: u32,
    req_sampling: ?sampling.RequestSampling,
    stops: []const []const u8,
) !GenOut {
    const eff = if (req_sampling) |rs| eng.base_sampling.mergedWithRequest(rs) else null;
    const raw = try eng.generate(io, prompt, messages_json, max_tokens, eff, false);
    defer allocator.free(raw);
    const cut = api.applyStops(raw, stops);
    // What we actually return, at a token boundary (the hot text cache is
    // keyed without max_tokens, so a cached completion may run longer).
    const ret_ids = try eng.tokenizer.encode(allocator, cut.text);
    defer allocator.free(ret_ids);
    const over = ret_ids.len > max_tokens;
    const text = if (over)
        try eng.tokenizer.decode(allocator, ret_ids[0..max_tokens])
    else
        try allocator.dupe(u8, cut.text);
    errdefer allocator.free(text);
    // Usage is the engine's own token count: re-encoding decoded text can
    // merge across boundaries (e.g. "!" runs) and undercount generated ids.
    const gen_tokens = eng.last_stats.decode_tokens;
    return .{
        .text = text,
        .prompt_tokens = eng.last_stats.prefill_tokens + 1,
        // NOTE: on a hot-cache hit last_stats is the previous call's; usage
        // is then approximate. Output text is always exact.
        .completion_tokens = if (over) max_tokens else gen_tokens,
        .hit_max = over or gen_tokens >= max_tokens,
        .stopped = cut.stopped,
    };
}

fn genError(req: *std.http.Server.Request, anthropic: bool, err: anyerror) void {
    switch (err) {
        error.VisionUnsupported => failReq(req, anthropic, .bad_request, "image/video content is not supported (text-only engine)"),
        error.ToolsUnsupported => failReq(req, anthropic, .bad_request, "tool use is not supported"),
        error.BadMessages => failReq(req, anthropic, .bad_request, "invalid messages"),
        error.BadExpectation => failReq(req, anthropic, .bad_request, "unsupported expect header"),
        error.BodyTooLarge => failReq(req, anthropic, .payload_too_large, "request body too large"),
        error.BodyRead => failReq(req, anthropic, .bad_request, "failed to read request body"),
        error.ContextOverflow => failReq(req, anthropic, .bad_request, "prompt exceeds context window"),
        error.UnsupportedDtype => failReq(req, anthropic, .internal_server_error, "model dtype not supported"),
        error.AffineUnsupported => failReq(req, anthropic, .internal_server_error, "affine quantization not supported (use bf16/mxfp8/nvfp4)"),
        error.NonstandardQuant, error.InconsistentQuant => failReq(req, anthropic, .internal_server_error, "model quantization not recognized"),
        error.OutOfMemory => failReq(req, anthropic, .internal_server_error, "out of memory"),
        else => failReq(req, anthropic, .internal_server_error, "inference failed"),
    }
}

fn chatRoute(allocator: std.mem.Allocator, io: std.Io, reg: *ModelRegistry, req: *std.http.Server.Request, ids: *u64) void {
    chatRouteInner(allocator, io, reg, req, ids) catch |e| genError(req, false, e);
}

fn chatRouteInner(allocator: std.mem.Allocator, io: std.Io, reg: *ModelRegistry, req: *std.http.Server.Request, ids: *u64) !void {
    const body = try readBody(allocator, req);
    defer allocator.free(body);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.BadMessages;
    defer parsed.deinit();
    const v = parsed.value;
    if (v != .object) return error.BadMessages;
    const messages = api.get(v, "messages") orelse return error.BadMessages;
    const r = resolveRoute(reg, req, v, false) orelse return;
    const eng = r.eng;
    const mj = try api.normalizeOpenAI(allocator, messages);
    defer allocator.free(mj);
    if (api.get(v, "n")) |n| {
        if (n != .integer or n.integer != 1) {
            failReq(req, false, .bad_request, "only n=1 is supported");
            return;
        }
    }
    const max_tokens = api.optU32(v, "max_tokens") orelse api.optU32(v, "max_completion_tokens") orelse r.max_tokens orelse default_max_tokens;
    const stops = try api.collectStops(allocator, v, "stop");
    defer allocator.free(stops);
    const rs = sampling.RequestSampling{
        .temperature = api.optF32(v, "temperature"),
        .top_p = api.optF32(v, "top_p"),
        .seed = api.optU64(v, "seed"),
    };
    const out = try runGen(allocator, io, eng, null, mj, max_tokens, rs, stops);
    defer allocator.free(out.text);
    ids.* += 1;
    const created = std.Io.Clock.real.now(io).toSeconds();
    const args = api.ChatArgs{
        .id = ids.*,
        .created = created,
        .model = r.id,
        .text = out.text,
        .finish = api.openaiFinish(out.hit_max),
        .prompt_tokens = out.prompt_tokens,
        .completion_tokens = out.completion_tokens,
    };
    if (api.optBool(v, "stream") orelse false) {
        try streamOpenAI(allocator, r.id, req, args);
    } else {
        const resp = try api.buildChatResponse(allocator, args);
        defer allocator.free(resp);
        req.respond(resp, .{ .status = .ok, .extra_headers = &.{ json_ct, cors } }) catch {};
    }
}

fn completionsRoute(allocator: std.mem.Allocator, io: std.Io, reg: *ModelRegistry, req: *std.http.Server.Request, ids: *u64) void {
    completionsRouteInner(allocator, io, reg, req, ids) catch |e| genError(req, false, e);
}

fn completionsRouteInner(allocator: std.mem.Allocator, io: std.Io, reg: *ModelRegistry, req: *std.http.Server.Request, ids: *u64) !void {
    const body = try readBody(allocator, req);
    defer allocator.free(body);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.BadMessages;
    defer parsed.deinit();
    const v = parsed.value;
    if (v != .object) return error.BadMessages;
    const prompt = api.optString(v, "prompt") orelse return error.BadMessages;
    const r = resolveRoute(reg, req, v, false) orelse return;
    const eng = r.eng;
    const max_tokens = api.optU32(v, "max_tokens") orelse r.max_tokens orelse default_max_tokens;
    const stops = try api.collectStops(allocator, v, "stop");
    defer allocator.free(stops);
    const rs = sampling.RequestSampling{
        .temperature = api.optF32(v, "temperature"),
        .top_p = api.optF32(v, "top_p"),
        .seed = api.optU64(v, "seed"),
    };
    const out = try runGen(allocator, io, eng, prompt, null, max_tokens, rs, stops);
    defer allocator.free(out.text);
    ids.* += 1;
    const created = std.Io.Clock.real.now(io).toSeconds();
    const resp = try api.buildCompletionsResponse(allocator, .{
        .id = ids.*,
        .created = created,
        .model = r.id,
        .text = out.text,
        .finish = api.openaiFinish(out.hit_max),
        .prompt_tokens = out.prompt_tokens,
        .completion_tokens = out.completion_tokens,
    });
    defer allocator.free(resp);
    req.respond(resp, .{ .status = .ok, .extra_headers = &.{ json_ct, cors } }) catch {};
}

fn streamOpenAI(allocator: std.mem.Allocator, model: []const u8, req: *std.http.Server.Request, a: api.ChatArgs) !void {
    var sbuf: [8192]u8 = undefined;
    var bw = req.respondStreaming(&sbuf, .{ .respond_options = .{ .status = .ok, .extra_headers = &.{ sse_ct, cors, no_cache } } }) catch return error.BodyRead;
    const ctx = api.OACtx{ .id = a.id, .created = a.created, .model = model };
    // First chunk carries the role (OpenAI convention).
    const first = try api.buildOAFirst(allocator, a.id, a.created, ctx.model);
    defer allocator.free(first);
    bw.writer.writeAll(first) catch return error.BodyRead;
    bw.flush() catch return error.BodyRead;
    var acc: std.Io.Writer.Allocating = .init(allocator);
    defer acc.deinit();
    api.writeChunks(&acc, a.text, 64, ctx, api.emitOAChunk) catch return error.BodyRead;
    bw.writer.writeAll(acc.writer.buffered()) catch return error.BodyRead;
    bw.flush() catch return error.BodyRead;
    const last = try api.buildOALast(allocator, a);
    defer allocator.free(last);
    bw.writer.writeAll(last) catch return error.BodyRead;
    bw.end() catch {};
}

// --- Anthropic route ---

fn messagesRoute(allocator: std.mem.Allocator, io: std.Io, reg: *ModelRegistry, req: *std.http.Server.Request, ids: *u64) void {
    messagesRouteInner(allocator, io, reg, req, ids) catch |e| genError(req, true, e);
}

fn messagesRouteInner(allocator: std.mem.Allocator, io: std.Io, reg: *ModelRegistry, req: *std.http.Server.Request, ids: *u64) !void {
    const body = try readBody(allocator, req);
    defer allocator.free(body);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.BadMessages;
    defer parsed.deinit();
    const v = parsed.value;
    if (v != .object) return error.BadMessages;
    if (api.get(v, "tools")) |t| {
        if (t == .array and t.array.items.len > 0) return error.ToolsUnsupported;
    }
    const messages = api.get(v, "messages") orelse return error.BadMessages;
    const r = resolveRoute(reg, req, v, true) orelse return;
    const eng = r.eng;
    const mj = try api.normalizeAnthropic(allocator, api.get(v, "system"), messages);
    defer allocator.free(mj);
    const max_tokens = api.optU32(v, "max_tokens") orelse r.max_tokens orelse default_max_tokens;
    const stops = try api.collectStops(allocator, v, "stop_sequences");
    defer allocator.free(stops);
    const rs = sampling.RequestSampling{
        .temperature = api.optF32(v, "temperature"),
        .top_p = api.optF32(v, "top_p"),
        .top_k = api.optU32(v, "top_k"),
    };
    const out = try runGen(allocator, io, eng, null, mj, max_tokens, rs, stops);
    defer allocator.free(out.text);
    ids.* += 1;
    const args = api.MsgArgs{
        .id = ids.*,
        .model = r.id,
        .text = out.text,
        .stop = api.anthropicStop(out.hit_max, out.stopped),
        .input_tokens = out.prompt_tokens,
        .output_tokens = out.completion_tokens,
    };
    if (api.optBool(v, "stream") orelse false) {
        try streamAnthropic(allocator, r.id, req, args);
    } else {
        const resp = try api.buildMessagesResponse(allocator, args);
        defer allocator.free(resp);
        req.respond(resp, .{ .status = .ok, .extra_headers = &.{ json_ct, cors } }) catch {};
    }
}

fn streamAnthropic(allocator: std.mem.Allocator, model: []const u8, req: *std.http.Server.Request, a: api.MsgArgs) !void {
    var sbuf: [8192]u8 = undefined;
    var bw = req.respondStreaming(&sbuf, .{ .respond_options = .{ .status = .ok, .extra_headers = &.{ sse_ct, cors, no_cache } } }) catch return error.BodyRead;
    const head = try api.buildAHead(allocator, a.id, model, a.input_tokens);
    defer allocator.free(head);
    bw.writer.writeAll(head) catch return error.BodyRead;
    bw.flush() catch return error.BodyRead;
    var acc: std.Io.Writer.Allocating = .init(allocator);
    defer acc.deinit();
    api.writeChunks(&acc, a.text, 64, api.ACtx{}, api.emitACDelta) catch return error.BodyRead;
    bw.writer.writeAll(acc.writer.buffered()) catch return error.BodyRead;
    bw.flush() catch return error.BodyRead;
    const tail = try api.buildATail(allocator, a.stop, a.output_tokens);
    defer allocator.free(tail);
    bw.writer.writeAll(tail) catch return error.BodyRead;
    bw.end() catch {};
}
