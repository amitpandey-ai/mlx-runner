const std = @import("std");

/// Pure OpenAI/Anthropic protocol builders: no engine/mlx imports, so this
/// file's tests run in the hermetic suite. The engine-coupled glue
/// (sockets, dispatch, generation) lives in server.zig.

pub const MsgError = error{ BadMessages, VisionUnsupported, ToolsUnsupported, OutOfMemory };

/// JSON string escaping (std's printStringEscaped emits Zig `\x..`, invalid JSON).
pub fn writeJsonStr(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    const hex = "0123456789abcdef";
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        0x08 => try w.writeAll("\\b"),
        0x0C => try w.writeAll("\\f"),
        else => {
            if (c < 0x20) {
                var tmp: [6]u8 = .{ '\\', 'u', '0', '0', hex[c >> 4], hex[c & 0xF] };
                try w.writeAll(&tmp);
            } else try w.writeByte(c);
        },
    };
    try w.writeByte('"');
}

// --- JSON value helpers (request parsing) ---

pub fn get(v: std.json.Value, name: []const u8) ?std.json.Value {
    if (v != .object) return null;
    return v.object.get(name);
}

pub fn optBool(v: std.json.Value, name: []const u8) ?bool {
    const f = get(v, name) orelse return null;
    return if (f == .bool) f.bool else null;
}

pub fn optF32(v: std.json.Value, name: []const u8) ?f32 {
    const f = get(v, name) orelse return null;
    return switch (f) {
        .float => |x| @floatCast(x),
        .integer => |x| @floatFromInt(x),
        else => null,
    };
}

pub fn optU32(v: std.json.Value, name: []const u8) ?u32 {
    const f = get(v, name) orelse return null;
    switch (f) {
        .integer => |x| {
            if (x < 0 or x > std.math.maxInt(u32)) return null;
            return @intCast(x);
        },
        .float => |x| {
            if (x < 0 or x > 4294967295.0 or @trunc(x) != x) return null;
            return @intFromFloat(x);
        },
        else => return null,
    }
}

pub fn optU64(v: std.json.Value, name: []const u8) ?u64 {
    const f = get(v, name) orelse return null;
    switch (f) {
        .integer => |x| {
            if (x < 0) return null;
            return @intCast(x);
        },
        .float => |x| {
            if (x < 0 or @trunc(x) != x) return null;
            return @intFromFloat(x);
        },
        else => return null,
    }
}

pub fn optString(v: std.json.Value, name: []const u8) ?[]const u8 {
    const f = get(v, name) orelse return null;
    return if (f == .string) f.string else null;
}

// --- message normalization to engine [{"role","content"}] JSON ---

/// Extract plain text from a string content value (null assistant content -> "").
fn textContent(v: std.json.Value) MsgError![]const u8 {
    switch (v) {
        .string => |s| return s,
        .null => return "",
        else => return error.BadMessages,
    }
}

/// Append concatenated text parts; image/tool parts are rejected (text-only engine).
fn appendPartText(parts: std.json.Value, w: *std.Io.Writer, first: *bool) MsgError!void {
    for (parts.array.items) |p| {
        if (p != .object) return error.BadMessages;
        const t = get(p, "type") orelse return error.BadMessages;
        if (t != .string) return error.BadMessages;
        if (std.mem.eql(u8, t.string, "text")) {
            const s = get(p, "text") orelse return error.BadMessages;
            if (s != .string) return error.BadMessages;
            if (!first.*) w.writeAll("\n") catch return error.OutOfMemory;
            first.* = false;
            w.writeAll(s.string) catch return error.OutOfMemory;
        } else if (std.mem.eql(u8, t.string, "image_url") or std.mem.eql(u8, t.string, "image")) {
            return error.VisionUnsupported;
        } else if (std.mem.eql(u8, t.string, "tool_use") or std.mem.eql(u8, t.string, "tool_result")) {
            return error.ToolsUnsupported;
        } else return error.BadMessages;
    }
}

fn appendMsgJson(w: *std.Io.Writer, role: []const u8, content: []const u8) !void {
    try w.writeAll("{\"role\":");
    try writeJsonStr(w, role);
    try w.writeAll(",\"content\":");
    try writeJsonStr(w, content);
    try w.writeByte('}');
}

/// OpenAI `messages` array -> engine messages_json.
pub fn normalizeOpenAI(allocator: std.mem.Allocator, messages: std.json.Value) MsgError![]u8 {
    if (messages != .array or messages.array.items.len == 0) return error.BadMessages;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    w.writeByte('[') catch return error.OutOfMemory;
    var first_msg = true;
    for (messages.array.items) |m| {
        if (m != .object) return error.BadMessages;
        const role = get(m, "role") orelse return error.BadMessages;
        if (role != .string) return error.BadMessages;
        const content = get(m, "content") orelse return error.BadMessages;
        if (!first_msg) w.writeByte(',') catch return error.OutOfMemory;
        first_msg = false;
        if (content == .array) {
            var tmp: std.Io.Writer.Allocating = .init(allocator);
            defer tmp.deinit();
            var first = true;
            try appendPartText(content, &tmp.writer, &first);
            const s = tmp.toOwnedSlice() catch return error.OutOfMemory;
            defer allocator.free(s);
            appendMsgJson(w, role.string, s) catch return error.OutOfMemory;
        } else {
            appendMsgJson(w, role.string, try textContent(content)) catch return error.OutOfMemory;
        }
    }
    w.writeByte(']') catch return error.OutOfMemory;
    return aw.toOwnedSlice() catch error.OutOfMemory;
}

/// Anthropic `system` + `messages` -> engine messages_json (system first).
pub fn normalizeAnthropic(allocator: std.mem.Allocator, system: ?std.json.Value, messages: std.json.Value) MsgError![]u8 {
    if (messages != .array or messages.array.items.len == 0) return error.BadMessages;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    w.writeByte('[') catch return error.OutOfMemory;
    var first_msg = true;
    if (system) |s| {
        var tmp: std.Io.Writer.Allocating = .init(allocator);
        defer tmp.deinit();
        if (s == .array) {
            var first = true;
            try appendPartText(s, &tmp.writer, &first);
        } else {
            tmp.writer.writeAll(try textContent(s)) catch return error.OutOfMemory;
        }
        const st = tmp.toOwnedSlice() catch return error.OutOfMemory;
        defer allocator.free(st);
        appendMsgJson(w, "system", st) catch return error.OutOfMemory;
        first_msg = false;
    }
    for (messages.array.items) |m| {
        if (m != .object) return error.BadMessages;
        const role = get(m, "role") orelse return error.BadMessages;
        if (role != .string) return error.BadMessages;
        if (!std.mem.eql(u8, role.string, "user") and !std.mem.eql(u8, role.string, "assistant")) return error.BadMessages;
        const content = get(m, "content") orelse return error.BadMessages;
        if (!first_msg) w.writeByte(',') catch return error.OutOfMemory;
        first_msg = false;
        if (content == .array) {
            var tmp: std.Io.Writer.Allocating = .init(allocator);
            defer tmp.deinit();
            var first = true;
            try appendPartText(content, &tmp.writer, &first);
            const s = tmp.toOwnedSlice() catch return error.OutOfMemory;
            defer allocator.free(s);
            appendMsgJson(w, role.string, s) catch return error.OutOfMemory;
        } else {
            appendMsgJson(w, role.string, try textContent(content)) catch return error.OutOfMemory;
        }
    }
    w.writeByte(']') catch return error.OutOfMemory;
    return aw.toOwnedSlice() catch error.OutOfMemory;
}

// --- stops ---

/// Collect stop strings: OpenAI `stop` (string | array) / Anthropic `stop_sequences`.
pub fn collectStops(allocator: std.mem.Allocator, v: std.json.Value, name: []const u8) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    const f = get(v, name) orelse return out.toOwnedSlice(allocator);
    switch (f) {
        .string => |s| if (s.len > 0) try out.append(allocator, s),
        .array => for (f.array.items) |it| {
            if (it == .string and it.string.len > 0) try out.append(allocator, it.string);
        },
        else => {},
    }
    return out.toOwnedSlice(allocator);
}

/// Truncate text at the earliest stop-string occurrence. Returns truncated flag.
pub fn applyStops(text: []const u8, stops: []const []const u8) struct { text: []const u8, stopped: bool } {
    var best: ?usize = null;
    for (stops) |s| {
        if (std.mem.indexOf(u8, text, s)) |i| {
            if (best == null or i < best.?) best = i;
        }
    }
    if (best) |i| return .{ .text = text[0..i], .stopped = true };
    return .{ .text = text, .stopped = false };
}

// --- finish reasons ---

pub fn openaiFinish(hit_max: bool) []const u8 {
    return if (hit_max) "length" else "stop";
}

pub fn anthropicStop(hit_max: bool, stopped: bool) []const u8 {
    return if (hit_max) "max_tokens" else if (stopped) "stop_sequence" else "end_turn";
}

// --- response builders (all owned) ---

pub const ChatArgs = struct {
    id: u64,
    created: i64,
    model: []const u8,
    text: []const u8,
    finish: []const u8,
    prompt_tokens: u64,
    completion_tokens: u64,
};

pub fn buildChatResponse(allocator: std.mem.Allocator, a: ChatArgs) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.print("{{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion\",\"created\":{d},\"model\":", .{ a.id, a.created });
    try writeJsonStr(w, a.model);
    try w.writeAll(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    try writeJsonStr(w, a.text);
    try w.print("}},\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}", .{ a.finish, a.prompt_tokens, a.completion_tokens, a.prompt_tokens + a.completion_tokens });
    return aw.toOwnedSlice();
}

pub fn buildCompletionsResponse(allocator: std.mem.Allocator, a: ChatArgs) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.print("{{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":", .{ a.id, a.created });
    try writeJsonStr(w, a.model);
    try w.writeAll(",\"choices\":[{\"text\":");
    try writeJsonStr(w, a.text);
    try w.print(",\"index\":0,\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}", .{ a.finish, a.prompt_tokens, a.completion_tokens, a.prompt_tokens + a.completion_tokens });
    return aw.toOwnedSlice();
}

pub const MsgArgs = struct {
    id: u64,
    model: []const u8,
    text: []const u8,
    stop: []const u8,
    input_tokens: u64,
    output_tokens: u64,
};

pub fn buildMessagesResponse(allocator: std.mem.Allocator, a: MsgArgs) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.print("{{\"id\":\"msg_{d}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", .{a.id});
    try writeJsonStr(w, a.model);
    try w.writeAll(",\"content\":[{\"type\":\"text\",\"text\":");
    try writeJsonStr(w, a.text);
    try w.print("}}],\"stop_reason\":\"{s}\",\"usage\":{{\"input_tokens\":{d},\"output_tokens\":{d}}}}}", .{ a.stop, a.input_tokens, a.output_tokens });
    return aw.toOwnedSlice();
}

/// Error body in the route's API shape. msg must be a controlled literal.
pub fn failBody(allocator: std.mem.Allocator, anthropic: bool, msg: []const u8) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    if (anthropic) {
        try w.writeAll("{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":");
        try writeJsonStr(w, msg);
        try w.writeAll("}}");
    } else {
        try w.writeAll("{\"error\":{\"message\":");
        try writeJsonStr(w, msg);
        try w.writeAll(",\"type\":\"invalid_request_error\"}}");
    }
    return aw.toOwnedSlice();
}

// --- SSE builders ---

/// Split text into ~budget-byte pieces without cutting a UTF-8 sequence.
pub fn writeChunks(aw: *std.Io.Writer.Allocating, text: []const u8, budget: usize, ctx: anytype, comptime emit: fn (@TypeOf(ctx), *std.Io.Writer, []const u8) anyerror!void) !void {
    const w = &aw.writer;
    var start: usize = 0;
    while (start < text.len) {
        var end: usize = @min(start + budget, text.len);
        while (end > start and end < text.len and (text[end] & 0xC0) == 0x80) end -= 1; // back off continuation bytes
        if (end == start) end = @min(start + budget, text.len); // degenerate: raw byte fallback
        try emit(ctx, w, text[start..end]);
        start = end;
    }
}

pub const OACtx = struct { id: u64, created: i64, model: []const u8 };

pub fn emitOAChunk(ctx: OACtx, w: *std.Io.Writer, piece: []const u8) !void {
    try w.print("data: {{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":", .{ ctx.id, ctx.created });
    try writeJsonStr(w, ctx.model);
    try w.writeAll(",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
    try writeJsonStr(w, piece);
    try w.writeAll(",\"finish_reason\":null}]}}\n\n");
}

pub fn buildOAFirst(allocator: std.mem.Allocator, id: u64, created: i64, model: []const u8) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.print("data: {{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":", .{ id, created });
    try writeJsonStr(w, model);
    try w.writeAll(",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}}\n\n");
    return aw.toOwnedSlice();
}

pub fn buildOALast(allocator: std.mem.Allocator, a: ChatArgs) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.print("data: {{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":", .{ a.id, a.created });
    try writeJsonStr(w, a.model);
    try w.print(",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}\n\ndata: [DONE]\n\n", .{ a.finish, a.prompt_tokens, a.completion_tokens, a.prompt_tokens + a.completion_tokens });
    return aw.toOwnedSlice();
}

pub const ACtx = struct {};

pub fn emitACDelta(ctx: ACtx, w: *std.Io.Writer, piece: []const u8) !void {
    _ = ctx;
    try w.writeAll("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":");
    try writeJsonStr(w, piece);
    try w.writeAll("}}\n\n");
}

pub fn buildAHead(allocator: std.mem.Allocator, id: u64, model: []const u8, input_tokens: u64) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.print("event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_{d}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", .{id});
    try writeJsonStr(w, model);
    try w.print(",\"content\":[],\"stop_reason\":null,\"usage\":{{\"input_tokens\":{d},\"output_tokens\":0}}}}}}\n\nevent: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n", .{input_tokens});
    return aw.toOwnedSlice();
}

pub fn buildATail(allocator: std.mem.Allocator, stop: []const u8, output_tokens: u64) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.print("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\nevent: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{s}\"}},\"usage\":{{\"output_tokens\":{d}}}}}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n", .{ stop, output_tokens });
    return aw.toOwnedSlice();
}

test "normalizeOpenAI joins string and part-array content" {
    const alloc = std.testing.allocator;
    const body =
        \\[{"role":"system","content":"Be brief."},{"role":"user","content":[{"type":"text","text":"Hi"},{"type":"text","text":"there"}]}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{});
    defer parsed.deinit();
    const mj = try normalizeOpenAI(alloc, parsed.value);
    defer alloc.free(mj);
    try std.testing.expectEqualStrings(
        "[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"Hi\\nthere\"}]",
        mj,
    );
}

test "normalizeOpenAI rejects image content" {
    const alloc = std.testing.allocator;
    const body =
        \\[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}}]}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{});
    defer parsed.deinit();
    try std.testing.expectError(error.VisionUnsupported, normalizeOpenAI(alloc, parsed.value));
}

test "normalizeAnthropic prepends system and joins blocks" {
    const alloc = std.testing.allocator;
    var sys = try std.json.parseFromSlice(std.json.Value, alloc, "\"You are terse.\"", .{});
    defer sys.deinit();
    var msgs = try std.json.parseFromSlice(std.json.Value, alloc, "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hi\"}]}]", .{});
    defer msgs.deinit();
    const mj = try normalizeAnthropic(alloc, sys.value, msgs.value);
    defer alloc.free(mj);
    try std.testing.expectEqualStrings(
        "[{\"role\":\"system\",\"content\":\"You are terse.\"},{\"role\":\"user\",\"content\":\"Hi\"}]",
        mj,
    );
}

test "applyStops cuts at earliest stop" {
    try std.testing.expectEqualStrings("ab", applyStops("abXYcdEF", &.{ "EF", "XY" }).text);
    try std.testing.expect(!applyStops("abcd", &.{ "XY" }).stopped);
    try std.testing.expect(applyStops("abXY", &.{ "XY" }).stopped);
    try std.testing.expectEqualStrings("abcd", applyStops("abcd", &.{"ZZ"}).text);
}

test "writeJsonStr escapes" {
    const alloc = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(alloc);
    defer aw.deinit();
    try writeJsonStr(&aw.writer, "a\"b\\c\nd\x01e日本語");
    const s = try aw.toOwnedSlice();
    defer alloc.free(s);
    try std.testing.expectEqualStrings("\"a\\\"b\\\\c\\nd\\u0001e日本語\"", s);
}

test "buildChatResponse pins wire format" {
    const alloc = std.testing.allocator;
    const s = try buildChatResponse(alloc, .{ .id = 7, .created = 1000, .model = "m", .text = "hi\"x", .finish = "stop", .prompt_tokens = 3, .completion_tokens = 2 });
    defer alloc.free(s);
    try std.testing.expectEqualStrings(
        "{\"id\":\"chatcmpl-7\",\"object\":\"chat.completion\",\"created\":1000,\"model\":\"m\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"hi\\\"x\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}",
        s,
    );
}

test "buildMessagesResponse pins wire format" {
    const alloc = std.testing.allocator;
    const s = try buildMessagesResponse(alloc, .{ .id = 9, .model = "m", .text = "yo", .stop = "end_turn", .input_tokens = 4, .output_tokens = 1 });
    defer alloc.free(s);
    try std.testing.expectEqualStrings(
        "{\"id\":\"msg_9\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[{\"type\":\"text\",\"text\":\"yo\"}],\"stop_reason\":\"end_turn\",\"usage\":{\"input_tokens\":4,\"output_tokens\":1}}",
        s,
    );
}

test "failBody shapes per API" {
    const alloc = std.testing.allocator;
    const o = try failBody(alloc, false, "nope");
    defer alloc.free(o);
    try std.testing.expectEqualStrings("{\"error\":{\"message\":\"nope\",\"type\":\"invalid_request_error\"}}", o);
    const a = try failBody(alloc, true, "nope");
    defer alloc.free(a);
    try std.testing.expectEqualStrings("{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"nope\"}}", a);
}

test "writeChunks never splits UTF-8" {
    const alloc = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(alloc);
    defer aw.deinit();
    const Ctx = struct {};
    const emit = struct {
        fn f(_: Ctx, w: *std.Io.Writer, piece: []const u8) !void {
            try w.print("[{s}]", .{piece});
        }
    }.f;
    try writeChunks(&aw, "ab日本語cd", 4, Ctx{}, emit);
    const s = try aw.toOwnedSlice();
    defer alloc.free(s);
    // Reassembly is exact and no piece cuts a multibyte char.
    var joined: std.Io.Writer.Allocating = .init(alloc);
    defer joined.deinit();
    // pieces are bracketed; strip brackets to verify exact reassembly.
    for (s) |c| if (c != '[' and c != ']') try joined.writer.writeByte(c);

    const j = try joined.toOwnedSlice();
    defer alloc.free(j);
    try std.testing.expectEqualStrings("ab日本語cd", j);
}

pub fn buildModelsList(allocator: std.mem.Allocator, models: []const []const u8) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.writeAll("{\"object\":\"list\",\"data\":[");
    for (models, 0..) |m, i| {
        if (i > 0) try w.writeByte(',');
        try w.writeAll("{\"id\":");
        try writeJsonStr(w, m);
        try w.writeAll(",\"object\":\"model\",\"owned_by\":\"mlx-runner\"}");
    }
    try w.writeAll("]}");
    return aw.toOwnedSlice();
}

test "buildModelsList enumerates ids" {
    const alloc = std.testing.allocator;
    const s = try buildModelsList(alloc, &.{ "a", "b\"c" });
    defer alloc.free(s);
    try std.testing.expectEqualStrings("{\"object\":\"list\",\"data\":[{\"id\":\"a\",\"object\":\"model\",\"owned_by\":\"mlx-runner\"},{\"id\":\"b\\\"c\",\"object\":\"model\",\"owned_by\":\"mlx-runner\"}]}", s);
    const e = try buildModelsList(alloc, &.{});
    defer alloc.free(e);
    try std.testing.expectEqualStrings("{\"object\":\"list\",\"data\":[]}", e);
}
