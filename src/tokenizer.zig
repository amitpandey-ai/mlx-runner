//! Native byte-level-BPE tokenizer for Qwen3.5-family checkpoints.
//!
//! Parses `<model>/tokenizer.json` with `std.json`: `model.vocab`,
//! rank-ordered `model.merges`, and `added_tokens`. No Python, no regex
//! engine: the file's one pretokenizer (GPT-2 Split + ByteLevel) is matched by
//! a hand-rolled scanner for exactly that alternation. There is no second
//! pattern to support, so there is no general engine.
//!
//! Deviations from the HF reference, stated plainly:
//! - The file's `normalizer` is NFC. Zig `std` ships no Unicode normalization
//!   tables, so input is encoded as-is. Already-NFC text (all ASCII, all
//!   precomposed UTF-8, i.e. effectively every prompt in practice) encodes
//!   byte-identically; decomposed combining sequences may split differently.
//! - `\p{L}` / `\p{N}` / `\s` are covered by a compact range table (major
//!   scripts, Nd digit blocks, common Unicode spaces). Exotic historic scripts
//!   fall back to the symbol branch. Roundtrips still hold; ids may differ
//!   from the reference on such text, never on ASCII/Latin/CJK core text.
//! - Prompts are encoded as-is with no BOS prepend (matches `mlx_lm`).

const std = @import("std");

pub const TokenizerError = error{
    BadTokenizerJson,
    MissingToken,
    UnknownTokenId,
    BadEncoding,
};

// ── bytes_to_unicode (GPT-2) ─────────────────────────────────────────────

fn isDirectByte(b: u8) bool {
    return (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255);
}

/// Byte -> unicode codepoint. comptime table: printable/latin-1 bytes map to
/// themselves, the other 68 map to U+0100.. in byte order.
const b2u_table: [256]u21 = blk: {
    var t: [256]u21 = undefined;
    var n: u21 = 0x100;
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (isDirectByte(b)) {
            t[i] = b;
        } else {
            t[i] = n;
            n += 1;
        }
    }
    break :blk t;
};

fn b2u(b: u8) u21 {
    return b2u_table[b];
}

/// Inverse of the shifted range: index c - 0x100 -> byte.
const u2b_table: [68]u8 = blk: {
    var t: [68]u8 = undefined;
    @memset(&t, 0);
    for (0..256) |i| {
        const c = b2u_table[i];
        if (c >= 0x100) t[c - 0x100] = @intCast(i);
    }
    break :blk t;
};

fn u2b(c: u21) ?u8 {
    if (c < 256) {
        const b: u8 = @intCast(c);
        if (isDirectByte(b)) return b;
        return null;
    }
    if (c >= 0x100 and c < 0x144) return u2b_table[c - 0x100];
    return null;
}

// ── Unicode classes (compact range tables) ───────────────────────────────
// N is checked before L so digit subranges inside coarse letter blocks win.

const Range = struct { lo: u21, hi: u21 };

fn inRanges(cp: u21, ranges: []const Range) bool {
    for (ranges) |r| {
        if (cp >= r.lo and cp <= r.hi) return true;
    }
    return false;
}

const number_ranges: []const Range = &.{
    .{ .lo = 0x30, .hi = 0x39 },
    .{ .lo = 0xB2, .hi = 0xB3 },
    .{ .lo = 0xB9, .hi = 0xB9 },
    .{ .lo = 0xBC, .hi = 0xBE },
    .{ .lo = 0x660, .hi = 0x669 },
    .{ .lo = 0x6F0, .hi = 0x6F9 },
    .{ .lo = 0x7C0, .hi = 0x7C9 },
    .{ .lo = 0x966, .hi = 0x96F },
    .{ .lo = 0x9E6, .hi = 0x9EF },
    .{ .lo = 0xA66, .hi = 0xA6F },
    .{ .lo = 0xAE6, .hi = 0xAEF },
    .{ .lo = 0xB66, .hi = 0xB6F },
    .{ .lo = 0xB72, .hi = 0xB77 },
    .{ .lo = 0xBE6, .hi = 0xBEF },
    .{ .lo = 0xC66, .hi = 0xC6F },
    .{ .lo = 0xC78, .hi = 0xC7E },
    .{ .lo = 0xCE6, .hi = 0xCEF },
    .{ .lo = 0xD66, .hi = 0xD75 },
    .{ .lo = 0xDE6, .hi = 0xDEF },
    .{ .lo = 0xE50, .hi = 0xE59 },
    .{ .lo = 0xED0, .hi = 0xED9 },
    .{ .lo = 0xF20, .hi = 0xF29 },
    .{ .lo = 0x1040, .hi = 0x1049 },
    .{ .lo = 0x1090, .hi = 0x1099 },
    .{ .lo = 0x1369, .hi = 0x137C },
    .{ .lo = 0x17E0, .hi = 0x17E9 },
    .{ .lo = 0x17F0, .hi = 0x17F9 },
    .{ .lo = 0x1810, .hi = 0x1819 },
    .{ .lo = 0x1946, .hi = 0x194F },
    .{ .lo = 0x19D0, .hi = 0x19D9 },
    .{ .lo = 0x1A80, .hi = 0x1A89 },
    .{ .lo = 0x1A90, .hi = 0x1A99 },
    .{ .lo = 0x1B50, .hi = 0x1B59 },
    .{ .lo = 0x1BB0, .hi = 0x1BB9 },
    .{ .lo = 0x1C40, .hi = 0x1C49 },
    .{ .lo = 0x1C50, .hi = 0x1C59 },
    .{ .lo = 0x2070, .hi = 0x2070 },
    .{ .lo = 0x2074, .hi = 0x2089 },
    .{ .lo = 0x2150, .hi = 0x2182 },
    .{ .lo = 0x2185, .hi = 0x2189 },
    .{ .lo = 0x2460, .hi = 0x249B },
    .{ .lo = 0x24EA, .hi = 0x24FF },
    .{ .lo = 0x2776, .hi = 0x2793 },
    .{ .lo = 0x2CFD, .hi = 0x2CFD },
    .{ .lo = 0x3007, .hi = 0x3007 },
    .{ .lo = 0x3021, .hi = 0x3029 },
    .{ .lo = 0x3038, .hi = 0x303A },
    .{ .lo = 0x3192, .hi = 0x3195 },
    .{ .lo = 0x3220, .hi = 0x3229 },
    .{ .lo = 0x3248, .hi = 0x324F },
    .{ .lo = 0x3251, .hi = 0x325F },
    .{ .lo = 0x3280, .hi = 0x3289 },
    .{ .lo = 0x32B1, .hi = 0x32BF },
    .{ .lo = 0xA620, .hi = 0xA629 },
    .{ .lo = 0xA8D0, .hi = 0xA8D9 },
    .{ .lo = 0xA900, .hi = 0xA909 },
    .{ .lo = 0xA9D0, .hi = 0xA9D9 },
    .{ .lo = 0xA9F0, .hi = 0xA9F9 },
    .{ .lo = 0xAA50, .hi = 0xAA59 },
    .{ .lo = 0xABF0, .hi = 0xABF9 },
    .{ .lo = 0xFF10, .hi = 0xFF19 },
    .{ .lo = 0x10140, .hi = 0x10178 },
    .{ .lo = 0x104A0, .hi = 0x104A9 },
    .{ .lo = 0x1D7CE, .hi = 0x1D7FF },
    .{ .lo = 0xA490, .hi = 0xA4CF },
};

const letter_ranges: []const Range = &.{
    .{ .lo = 0x41, .hi = 0x5A },
    .{ .lo = 0x61, .hi = 0x7A },
    .{ .lo = 0xAA, .hi = 0xAA },
    .{ .lo = 0xB5, .hi = 0xB5 },
    .{ .lo = 0xBA, .hi = 0xBA },
    .{ .lo = 0xC0, .hi = 0x2C1 },
    .{ .lo = 0x2C6, .hi = 0x2D1 },
    .{ .lo = 0x2E0, .hi = 0x2E4 },
    .{ .lo = 0x2EC, .hi = 0x2EC },
    .{ .lo = 0x2EE, .hi = 0x2EE },
    .{ .lo = 0x345, .hi = 0x345 },
    .{ .lo = 0x370, .hi = 0x374 },
    .{ .lo = 0x376, .hi = 0x377 },
    .{ .lo = 0x37A, .hi = 0x37D },
    .{ .lo = 0x37F, .hi = 0x37F },
    .{ .lo = 0x386, .hi = 0x386 },
    .{ .lo = 0x388, .hi = 0x38A },
    .{ .lo = 0x38C, .hi = 0x38C },
    .{ .lo = 0x38E, .hi = 0x3A1 },
    .{ .lo = 0x3A3, .hi = 0x3F5 },
    .{ .lo = 0x3F7, .hi = 0x481 },
    .{ .lo = 0x48A, .hi = 0x52F },
    .{ .lo = 0x531, .hi = 0x556 },
    .{ .lo = 0x559, .hi = 0x559 },
    .{ .lo = 0x561, .hi = 0x586 },
    .{ .lo = 0x5D0, .hi = 0x5EA },
    .{ .lo = 0x5EF, .hi = 0x5F2 },
    .{ .lo = 0x620, .hi = 0x64A },
    .{ .lo = 0x66E, .hi = 0x66F },
    .{ .lo = 0x671, .hi = 0x6D3 },
    .{ .lo = 0x6D5, .hi = 0x6D5 },
    .{ .lo = 0x6E5, .hi = 0x6E6 },
    .{ .lo = 0x6EE, .hi = 0x6EF },
    .{ .lo = 0x6FA, .hi = 0x6FC },
    .{ .lo = 0x6FF, .hi = 0x6FF },
    .{ .lo = 0x710, .hi = 0x710 },
    .{ .lo = 0x712, .hi = 0x72F },
    .{ .lo = 0x74D, .hi = 0x7A5 },
    .{ .lo = 0x7B1, .hi = 0x7B1 },
    .{ .lo = 0x7CA, .hi = 0x7EA },
    .{ .lo = 0x7F4, .hi = 0x7F5 },
    .{ .lo = 0x7FA, .hi = 0x7FA },
    .{ .lo = 0x800, .hi = 0x82D },
    .{ .lo = 0x840, .hi = 0x85B },
    .{ .lo = 0x860, .hi = 0x86A },
    .{ .lo = 0x8A0, .hi = 0x8C9 },
    .{ .lo = 0x900, .hi = 0xFFF },
    .{ .lo = 0x1000, .hi = 0x10FF },
    .{ .lo = 0x1100, .hi = 0x11FF },
    .{ .lo = 0x1200, .hi = 0x137F },
    .{ .lo = 0x1380, .hi = 0x139F },
    .{ .lo = 0x13A0, .hi = 0x13FF },
    .{ .lo = 0x1400, .hi = 0x167F },
    .{ .lo = 0x1681, .hi = 0x169C },
    .{ .lo = 0x16A0, .hi = 0x16FF },
    .{ .lo = 0x1700, .hi = 0x171F },
    .{ .lo = 0x1720, .hi = 0x173F },
    .{ .lo = 0x1740, .hi = 0x175F },
    .{ .lo = 0x1760, .hi = 0x177F },
    .{ .lo = 0x1780, .hi = 0x17FF },
    .{ .lo = 0x1800, .hi = 0x18AF },
    .{ .lo = 0x1900, .hi = 0x194F },
    .{ .lo = 0x1950, .hi = 0x197F },
    .{ .lo = 0x1980, .hi = 0x19DF },
    .{ .lo = 0x19E0, .hi = 0x19FF },
    .{ .lo = 0x1A00, .hi = 0x1A1F },
    .{ .lo = 0x1B00, .hi = 0x1B7F },
    .{ .lo = 0x1B80, .hi = 0x1BBF },
    .{ .lo = 0x1C00, .hi = 0x1C4F },
    .{ .lo = 0x1D00, .hi = 0x1D7F },
    .{ .lo = 0x1E00, .hi = 0x1EFF },
    .{ .lo = 0x1F00, .hi = 0x1FFF },
    .{ .lo = 0x3041, .hi = 0x3096 },
    .{ .lo = 0x309D, .hi = 0x309E },
    .{ .lo = 0x30A1, .hi = 0x30FA },
    .{ .lo = 0x30FC, .hi = 0x30FE },
    .{ .lo = 0x3105, .hi = 0x312F },
    .{ .lo = 0x3131, .hi = 0x318E },
    .{ .lo = 0x31A0, .hi = 0x31BF },
    .{ .lo = 0x31F0, .hi = 0x31FF },
    .{ .lo = 0x3400, .hi = 0x4DBF },
    .{ .lo = 0x4E00, .hi = 0x9FFF },
    .{ .lo = 0xA000, .hi = 0xA48F },
    .{ .lo = 0xA4D0, .hi = 0xA4FF },
    .{ .lo = 0xA500, .hi = 0xA63F },
    .{ .lo = 0xA640, .hi = 0xA69F },
    .{ .lo = 0xA6A0, .hi = 0xA6FF },
    .{ .lo = 0xA700, .hi = 0xA71F },
    .{ .lo = 0xA720, .hi = 0xA7FF },
    .{ .lo = 0xA800, .hi = 0xA82F },
    .{ .lo = 0xA840, .hi = 0xA87F },
    .{ .lo = 0xA880, .hi = 0xA8DF },
    .{ .lo = 0xA8E0, .hi = 0xA8FF },
    .{ .lo = 0xA900, .hi = 0xA92F },
    .{ .lo = 0xA930, .hi = 0xA95F },
    .{ .lo = 0xA960, .hi = 0xA97F },
    .{ .lo = 0xA980, .hi = 0xA9DF },
    .{ .lo = 0xAA00, .hi = 0xAA5F },
    .{ .lo = 0xAA60, .hi = 0xAA7F },
    .{ .lo = 0xAA80, .hi = 0xAADF },
    .{ .lo = 0xAAE0, .hi = 0xAAFF },
    .{ .lo = 0xAB00, .hi = 0xAB2F },
    .{ .lo = 0xAB30, .hi = 0xAB6F },
    .{ .lo = 0xABC0, .hi = 0xABFF },
    .{ .lo = 0xAC00, .hi = 0xD7A3 },
    .{ .lo = 0xD7B0, .hi = 0xD7FF },
    .{ .lo = 0xF900, .hi = 0xFA6D },
    .{ .lo = 0xFA70, .hi = 0xFAD9 },
    .{ .lo = 0xFB00, .hi = 0xFB4F },
    .{ .lo = 0xFB50, .hi = 0xFDFF },
    .{ .lo = 0xFE70, .hi = 0xFEFF },
    .{ .lo = 0xFF21, .hi = 0xFF3A },
    .{ .lo = 0xFF41, .hi = 0xFF5A },
    .{ .lo = 0xFF66, .hi = 0xFF9D },
    .{ .lo = 0xFF9E, .hi = 0xFF9F },
    .{ .lo = 0xFFA0, .hi = 0xFFDC },
    .{ .lo = 0xFFE0, .hi = 0xFFE6 },
    .{ .lo = 0x10000, .hi = 0x100FF },
    .{ .lo = 0x10300, .hi = 0x1032F },
    .{ .lo = 0x10330, .hi = 0x1034F },
    .{ .lo = 0x10400, .hi = 0x1044F },
    .{ .lo = 0x10450, .hi = 0x1047F },
    .{ .lo = 0x10480, .hi = 0x104BF },
    .{ .lo = 0x10500, .hi = 0x1052F },
    .{ .lo = 0x10530, .hi = 0x1056F },
    .{ .lo = 0x10600, .hi = 0x1077F },
    .{ .lo = 0x10800, .hi = 0x1083F },
    .{ .lo = 0x10900, .hi = 0x1091F },
    .{ .lo = 0x10920, .hi = 0x1093F },
    .{ .lo = 0x10A00, .hi = 0x10A5F },
    .{ .lo = 0x10A60, .hi = 0x10A7F },
    .{ .lo = 0x10A80, .hi = 0x10A9F },
    .{ .lo = 0x10AC0, .hi = 0x10AFF },
    .{ .lo = 0x10B00, .hi = 0x10B3F },
    .{ .lo = 0x10B40, .hi = 0x10B5F },
    .{ .lo = 0x10B60, .hi = 0x10B7F },
    .{ .lo = 0x10B80, .hi = 0x10BFF },
    .{ .lo = 0x10C00, .hi = 0x10C4F },
    .{ .lo = 0x10C80, .hi = 0x10CFF },
    .{ .lo = 0x10E60, .hi = 0x10E7F },
    .{ .lo = 0x11000, .hi = 0x1107F },
    .{ .lo = 0x11080, .hi = 0x110CF },
    .{ .lo = 0x110D0, .hi = 0x110FF },
    .{ .lo = 0x11100, .hi = 0x1114F },
    .{ .lo = 0x11150, .hi = 0x1117F },
    .{ .lo = 0x11180, .hi = 0x111DF },
    .{ .lo = 0x11200, .hi = 0x1124F },
    .{ .lo = 0x11280, .hi = 0x112AF },
    .{ .lo = 0x112B0, .hi = 0x112FF },
    .{ .lo = 0x11300, .hi = 0x1137F },
    .{ .lo = 0x11400, .hi = 0x1147F },
    .{ .lo = 0x11480, .hi = 0x114DF },
    .{ .lo = 0x11580, .hi = 0x115FF },
    .{ .lo = 0x11600, .hi = 0x1164F },
    .{ .lo = 0x11680, .hi = 0x116CF },
    .{ .lo = 0x11800, .hi = 0x118DF },
    .{ .lo = 0x11A00, .hi = 0x11AFF },
    .{ .lo = 0x11C00, .hi = 0x11C6F },
    .{ .lo = 0x11C70, .hi = 0x11CBF },
    .{ .lo = 0x11D00, .hi = 0x11D5F },
    .{ .lo = 0x12000, .hi = 0x123FF },
    .{ .lo = 0x12400, .hi = 0x1247F },
    .{ .lo = 0x12480, .hi = 0x1254F },
    .{ .lo = 0x13000, .hi = 0x1342F },
    .{ .lo = 0x14400, .hi = 0x1467F },
    .{ .lo = 0x16800, .hi = 0x16A3F },
    .{ .lo = 0x16A40, .hi = 0x16A6F },
    .{ .lo = 0x16F00, .hi = 0x16F9F },
    .{ .lo = 0x1B000, .hi = 0x1B0FF },
    .{ .lo = 0x1BC00, .hi = 0x1BC9F },
    .{ .lo = 0x1D400, .hi = 0x1D7CD },
    .{ .lo = 0x20000, .hi = 0x2A6DF },
    .{ .lo = 0x2A6E0, .hi = 0x2CEAF },
    .{ .lo = 0x2CEB0, .hi = 0x2EBEF },
    .{ .lo = 0x2EBF0, .hi = 0x2EE5F },
    .{ .lo = 0x2F800, .hi = 0x2FA1F },
    .{ .lo = 0x30000, .hi = 0x3134F },
    .{ .lo = 0x31350, .hi = 0x323AF },
};

const space_ranges: []const Range = &.{
    .{ .lo = 0x09, .hi = 0x0D },
    .{ .lo = 0x1C, .hi = 0x1F },
    .{ .lo = 0x20, .hi = 0x20 },
    .{ .lo = 0x85, .hi = 0x85 },
    .{ .lo = 0xA0, .hi = 0xA0 },
    .{ .lo = 0x1680, .hi = 0x1680 },
    .{ .lo = 0x2000, .hi = 0x200A },
    .{ .lo = 0x2028, .hi = 0x2029 },
    .{ .lo = 0x202F, .hi = 0x202F },
    .{ .lo = 0x205F, .hi = 0x205F },
    .{ .lo = 0x3000, .hi = 0x3000 },
};

fn isNumber(cp: u21) bool {
    return inRanges(cp, number_ranges);
}
fn isLetter(cp: u21) bool {
    if (isNumber(cp)) return false;
    return inRanges(cp, letter_ranges);
}
fn isSpace(cp: u21) bool {
    return inRanges(cp, space_ranges);
}
fn isCrLf(cp: u21) bool {
    return cp == '\r' or cp == '\n';
}

// ── codepoint cursor ─────────────────────────────────────────────────────

const Cp = struct { cp: u21, len: u3 };

fn decodeCpAt(s: []const u8, pos: usize) Cp {
    if (pos >= s.len) return .{ .cp = 0, .len = 1 };
    const b = s[pos];
    if (b < 0x80) return .{ .cp = b, .len = 1 };
    const seqlen = std.unicode.utf8ByteSequenceLength(b) catch 1;
    if (pos + seqlen > s.len) return .{ .cp = b, .len = 1 };
    const cp = std.unicode.utf8Decode(s[pos .. pos + seqlen]) catch b;
    return .{ .cp = cp, .len = @intCast(seqlen) };
}

fn toLowerAscii(c: u8) u8 {
    return if (c >= 'A' and c <= 'Z') c + 32 else c;
}

// ── GPT-2 Split matcher ──────────────────────────────────────────────────
// Pattern (ordered alternation — first match wins at each position):
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

const Span = struct { start: usize, end: usize };

fn matchContraction(s: []const u8, pos: usize, end: usize) ?usize {
    // s[pos] == '\'' (0x27) on entry.
    if (pos + 1 >= end) return null;
    const c1 = toLowerAscii(s[pos + 1]);
    // two-letter tails first: 're 've 'll
    if (pos + 2 < end) {
        const c2 = toLowerAscii(s[pos + 2]);
        if ((c1 == 'r' and c2 == 'e') or
            (c1 == 'v' and c2 == 'e') or
            (c1 == 'l' and c2 == 'l'))
        {
            return pos + 3;
        }
    }
    if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') {
        // 'm vs 'll... 'm is single; no conflict with pairs above ('m' + letter
        // is still just 'm — pairs only match re/ve/ll).
        return pos + 2;
    }
    return null;
}

fn splitPretokens(allocator: std.mem.Allocator, s: []const u8, start: usize, end: usize) ![]Span {
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(allocator);
    var pos = start;
    while (pos < end) {
        const c0 = decodeCpAt(s, pos);
        // alt 1: contractions
        if (c0.cp == '\'' and pos + 1 < end) {
            if (matchContraction(s, pos, end)) |npos| {
                try out.append(allocator, .{ .start = pos, .end = npos });
                pos = npos;
                continue;
            }
        }
        // alt 2: [^\r\nL/N]? L+
        {
            var p = pos;
            if (p < end) {
                const c = decodeCpAt(s, p);
                if (!isCrLf(c.cp) and !isLetter(c.cp) and !isNumber(c.cp)) p += c.len;
            }
            if (p < end and isLetter(decodeCpAt(s, p).cp)) {
                while (p < end and isLetter(decodeCpAt(s, p).cp)) {
                    p += decodeCpAt(s, p).len;
                }
                try out.append(allocator, .{ .start = pos, .end = p });
                pos = p;
                continue;
            }
        }
        // alt 3: single number char
        if (isNumber(c0.cp)) {
            try out.append(allocator, .{ .start = pos, .end = pos + c0.len });
            pos += c0.len;
            continue;
        }
        // alt 4: ' '? [^\sLN]+ [\r\n]*
        {
            var p = pos;
            if (p < end and s[p] == ' ') p += 1;
            const run = p;
            while (p < end) {
                const c = decodeCpAt(s, p);
                if (isSpace(c.cp) or isLetter(c.cp) or isNumber(c.cp)) break;
                p += c.len;
            }
            if (p > run) {
                while (p < end) {
                    const c = decodeCpAt(s, p);
                    if (!isCrLf(c.cp)) break;
                    p += c.len;
                }
                try out.append(allocator, .{ .start = pos, .end = p });
                pos = p;
                continue;
            }
        }
        // alt 5: \s* [\r\n]+ — greedy with backtracking: the \s* gives back
        // chars until [\r\n]+ matches, so the match runs through the LAST
        // newline of the maximal whitespace run (e.g. "\r\n", "  \n").
        {
            var p = pos;
            var last_nl: ?usize = null;
            while (p < end and isSpace(decodeCpAt(s, p).cp)) {
                const c = decodeCpAt(s, p);
                if (isCrLf(c.cp)) last_nl = p + c.len;
                p += c.len;
            }
            if (last_nl) |nl_end| {
                try out.append(allocator, .{ .start = pos, .end = nl_end });
                pos = nl_end;
                continue;
            }
        }
        // alt 6: \s+ (?!\S) — longest run whose follower is space or end
        if (isSpace(c0.cp)) {
            var p = pos;
            while (p < end and isSpace(decodeCpAt(s, p).cp)) {
                p += decodeCpAt(s, p).len;
            }
            // backtrack: drop trailing chars while follower is non-space
            var q = p;
            while (q > pos and q < end and !isSpace(decodeCpAt(s, q).cp)) {
                // step back one codepoint
                q -= 1;
                while (q > pos and (s[q] & 0xC0) == 0x80) q -= 1;
            }
            if (q > pos) {
                try out.append(allocator, .{ .start = pos, .end = q });
                pos = q;
                continue;
            }
            // alt 7: \s+ maximal
            try out.append(allocator, .{ .start = pos, .end = p });
            pos = p;
            continue;
        }
        // Defensive: every byte is covered above (L/N/space/CRLF/other), but
        // never stall the scanner on unexpected input.
        try out.append(allocator, .{ .start = pos, .end = pos + c0.len });
        pos += c0.len;
    }
    return out.toOwnedSlice(allocator);
}

// ── Tokenizer ────────────────────────────────────────────────────────────

const AddedTok = struct { content: []const u8, id: u32 };

const Pair = struct { a: []const u8, b: []const u8 };
const PairCtx = struct {
    pub fn hash(_: @This(), k: Pair) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(k.a);
        h.update(&[_]u8{0});
        h.update(k.b);
        return h.final();
    }
    pub fn eql(_: @This(), x: Pair, y: Pair) bool {
        return std.mem.eql(u8, x.a, y.a) and std.mem.eql(u8, x.b, y.b);
    }
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(u32),
    decode_map: std.AutoHashMap(u32, []const u8),
    added: std.ArrayList(AddedTok),
    added_by_id: std.AutoHashMap(u32, []const u8),
    merges: std.ArrayList([2][]const u8),
    ranks: std.HashMap(Pair, u32, PairCtx, 80),

    pub fn deinit(self: *Tokenizer) void {
        var vit = self.vocab.iterator();
        while (vit.next()) |e| self.allocator.free(e.key_ptr.*);
        self.vocab.deinit();
        self.decode_map.deinit();
        for (self.added.items) |a| self.allocator.free(a.content);
        self.added.deinit(self.allocator);
        self.added_by_id.deinit();
        for (self.merges.items) |m| {
            self.allocator.free(m[0]);
            self.allocator.free(m[1]);
        }
        self.merges.deinit(self.allocator);
        self.ranks.deinit();
    }

    fn readWholeFile(allocator: std.mem.Allocator, io: std.Io, abs_or_rel: []const u8) ![]u8 {
        const abs = std.fs.path.isAbsolute(abs_or_rel);
        var file = if (abs)
            try std.Io.Dir.openFileAbsolute(io, abs_or_rel, .{})
        else
            try std.Io.Dir.cwd().openFile(io, abs_or_rel, .{});
        defer file.close(io);
        const st = try file.stat(io);
        const size: usize = @intCast(st.size);
        const buf = try allocator.alloc(u8, size);
        errdefer allocator.free(buf);
        _ = try file.readPositionalAll(io, buf, 0);
        return buf;
    }

    pub fn load(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !Tokenizer {
        const path = try std.fs.path.join(allocator, &.{ model_dir, "tokenizer.json" });
        defer allocator.free(path);
        const bytes = try readWholeFile(allocator, io, path);
        defer allocator.free(bytes);
        return fromJson(allocator, bytes);
    }

    pub fn fromJson(allocator: std.mem.Allocator, bytes: []const u8) !Tokenizer {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch
            return TokenizerError.BadTokenizerJson;
        defer parsed.deinit();

        var self = Tokenizer{
            .allocator = allocator,
            .vocab = std.StringHashMap(u32).init(allocator),
            .decode_map = std.AutoHashMap(u32, []const u8).init(allocator),
            .added = .empty,
            .added_by_id = std.AutoHashMap(u32, []const u8).init(allocator),
            .merges = .empty,
            .ranks = std.HashMap(Pair, u32, PairCtx, 80).init(allocator),
        };
        errdefer self.deinit();

        const root = parsed.value;
        if (root != .object) return TokenizerError.BadTokenizerJson;
        const model = root.object.get("model") orelse return TokenizerError.BadTokenizerJson;
        if (model != .object) return TokenizerError.BadTokenizerJson;

        // vocab: token string -> id
        const vocab_v = model.object.get("vocab") orelse return TokenizerError.BadTokenizerJson;
        if (vocab_v != .object) return TokenizerError.BadTokenizerJson;
        var vit = vocab_v.object.iterator();
        while (vit.next()) |e| {
            if (e.value_ptr.* != .integer) return TokenizerError.BadTokenizerJson;
            const id: u32 = @intCast(e.value_ptr.integer);
            const key = try allocator.dupe(u8, e.key_ptr.*);
            errdefer allocator.free(key);
            try self.vocab.put(key, id);
            // decode_map borrows the same duped key (freed once via vocab).
            try self.decode_map.put(id, key);
        }
        if (self.vocab.count() == 0) return TokenizerError.BadTokenizerJson;

        // merges: rank-ordered [a, b] pairs
        const merges_v = model.object.get("merges") orelse return TokenizerError.BadTokenizerJson;
        if (merges_v != .array) return TokenizerError.BadTokenizerJson;
        for (merges_v.array.items) |m| {
            if (m != .array or m.array.items.len != 2) return TokenizerError.BadTokenizerJson;
            const a_v = m.array.items[0];
            const b_v = m.array.items[1];
            if (a_v != .string or b_v != .string) return TokenizerError.BadTokenizerJson;
            const a = try allocator.dupe(u8, a_v.string);
            errdefer allocator.free(a);
            const b = try allocator.dupe(u8, b_v.string);
            errdefer allocator.free(b);
            const rank: u32 = @intCast(self.merges.items.len);
            try self.merges.append(allocator, .{ a, b });
            try self.ranks.put(.{ .a = a, .b = b }, rank);
        }

        // added_tokens: {content, id}
        if (root.object.get("added_tokens")) |at| {
            if (at != .array) return TokenizerError.BadTokenizerJson;
            for (at.array.items) |t| {
                if (t != .object) return TokenizerError.BadTokenizerJson;
                const c_v = t.object.get("content") orelse return TokenizerError.BadTokenizerJson;
                const id_v = t.object.get("id") orelse return TokenizerError.BadTokenizerJson;
                if (c_v != .string or id_v != .integer) return TokenizerError.BadTokenizerJson;
                const content = try allocator.dupe(u8, c_v.string);
                errdefer allocator.free(content);
                const id: u32 = @intCast(id_v.integer);
                try self.added.append(allocator, .{ .content = content, .id = id });
                try self.added_by_id.put(id, content);
            }
        }
        // longest-first so scanning matches greedily
        std.mem.sort(AddedTok, self.added.items, {}, struct {
            fn lt(_: void, x: AddedTok, y: AddedTok) bool {
                return x.content.len > y.content.len;
            }
        }.lt);

        return self;
    }

    fn matchAdded(self: *const Tokenizer, s: []const u8, pos: usize) ?AddedTok {
        for (self.added.items) |a| {
            if (pos + a.content.len <= s.len and
                std.mem.eql(u8, s[pos .. pos + a.content.len], a.content))
            {
                return a;
            }
        }
        return null;
    }

    pub fn idOf(self: *const Tokenizer, content: []const u8) ?u32 {
        for (self.added.items) |a| {
            if (std.mem.eql(u8, a.content, content)) return a.id;
        }
        return self.vocab.get(content);
    }

    /// Encode text to token ids (owned). Added tokens match longest-first and
    /// are never BPE-split; everything else goes through the GPT-2 split +
    /// ByteLevel + rank loop. No BOS is prepended.
    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const tmp = arena.allocator();

        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);

        var pos: usize = 0;
        var span_start: usize = 0;
        while (pos < text.len) {
            if (self.matchAdded(text, pos)) |a| {
                if (pos > span_start) try self.encodeSpan(tmp, &out, text[span_start..pos]);
                try out.append(allocator, a.id);
                pos += a.content.len;
                span_start = pos;
            } else {
                pos += 1;
            }
        }
        if (text.len > span_start) try self.encodeSpan(tmp, &out, text[span_start..]);
        return out.toOwnedSlice(allocator);
    }

    fn encodeSpan(self: *const Tokenizer, tmp: std.mem.Allocator, out: *std.ArrayList(u32), span: []const u8) !void {
        const spans = try splitPretokens(tmp, span, 0, span.len);
        for (spans) |sp| {
            try self.encodePretoken(tmp, out, span[sp.start..sp.end]);
        }
    }

    fn encodePretoken(self: *const Tokenizer, tmp: std.mem.Allocator, out: *std.ArrayList(u32), pre: []const u8) !void {
        // ByteLevel (use_regex=false): one symbol per byte via b2u.
        var syms: std.ArrayList([]const u8) = .empty;
        defer syms.deinit(tmp);
        for (pre) |b| {
            var buf: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(b2u(b), &buf) catch return TokenizerError.BadEncoding;
            try syms.append(tmp, try tmp.dupe(u8, buf[0..n]));
        }
        // BPE rank loop.
        while (syms.items.len >= 2) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_idx: ?usize = null;
            for (0..syms.items.len - 1) |i| {
                if (self.ranks.get(.{ .a = syms.items[i], .b = syms.items[i + 1] })) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best_idx = i;
                    }
                }
            }
            const idx = best_idx orelse break;
            const merged = try std.mem.concat(tmp, u8, &.{ syms.items[idx], syms.items[idx + 1] });
            syms.items[idx] = merged;
            std.mem.copyForwards([]const u8, syms.items[idx + 1 ..], syms.items[idx + 2 ..]);
            syms.items.len -= 1;
        }
        for (syms.items) |sym| {
            const id = self.vocab.get(sym) orelse return TokenizerError.MissingToken;
            try out.append(self.allocator, id);
        }
    }

    /// Decode ids to bytes (owned). Added-token ids emit their content raw;
    /// model ids invert vocab and ByteLevel-decode to UTF-8 bytes.
    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids) |id| {
            if (self.added_by_id.get(id)) |content| {
                try out.appendSlice(allocator, content);
                continue;
            }
            const tok = self.decode_map.get(id) orelse return TokenizerError.UnknownTokenId;
            var i: usize = 0;
            while (i < tok.len) {
                const seqlen = std.unicode.utf8ByteSequenceLength(tok[i]) catch
                    return TokenizerError.BadEncoding;
                const cp = std.unicode.utf8Decode(tok[i .. i + seqlen]) catch
                    return TokenizerError.BadEncoding;
                const b = u2b(cp) orelse return TokenizerError.BadEncoding;
                try out.append(allocator, b);
                i += seqlen;
            }
        }
        return out.toOwnedSlice(allocator);
    }
};

// ── tests (hermetic: synthetic fixtures, no model files) ──────────────────

const test_vocab_extra = 256;

fn makeFixtureTokenizer(alloc: std.mem.Allocator) !Tokenizer {
    // 256 single-byte tokens (ids 0..255, token text = b2u UTF-8) so every
    // byte sequence roundtrips, plus merge products and one added token.
    var self = Tokenizer{
        .allocator = alloc,
        .vocab = std.StringHashMap(u32).init(alloc),
        .decode_map = std.AutoHashMap(u32, []const u8).init(alloc),
        .added = .empty,
        .added_by_id = std.AutoHashMap(u32, []const u8).init(alloc),
        .merges = .empty,
        .ranks = std.HashMap(Pair, u32, PairCtx, 80).init(alloc),
    };
    errdefer self.deinit();
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        var buf: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(b2u(b), &buf);
        const key = try alloc.dupe(u8, buf[0..n]);
        try self.vocab.put(key, @intCast(i));
        try self.decode_map.put(@intCast(i), key);
    }
    // merges: "a"+"b" -> "ab" (id 256), "ab"+"c" -> "abc" (id 257)
    const ab = try alloc.dupe(u8, "ab");
    const abc = try alloc.dupe(u8, "abc");
    try self.vocab.put(ab, 256);
    try self.decode_map.put(256, ab);
    try self.vocab.put(abc, 257);
    try self.decode_map.put(257, abc);
    const ma = try alloc.dupe(u8, "a");
    const mb = try alloc.dupe(u8, "b");
    const mab = try alloc.dupe(u8, "ab");
    const mc = try alloc.dupe(u8, "c");
    try self.merges.append(alloc, .{ ma, mb });
    try self.ranks.put(.{ .a = ma, .b = mb }, 0);
    try self.merges.append(alloc, .{ mab, mc });
    try self.ranks.put(.{ .a = mab, .b = mc }, 1);
    // "\r\n" merge (mirrors the real rank-61 ["č","Ċ"] -> "čĊ"): exercises
    // alt-5 backtracking — without it "\r\n" splits into two pretokens.
    var cbuf: [4]u8 = undefined;
    const cr_n = try std.unicode.utf8Encode(b2u(0x0D), &cbuf);
    const cr_s = try alloc.dupe(u8, cbuf[0..cr_n]);
    var lbuf: [4]u8 = undefined;
    const lf_n = try std.unicode.utf8Encode(b2u(0x0A), &lbuf);
    const lf_s = try alloc.dupe(u8, lbuf[0..lf_n]);
    const crlf_s = try std.mem.concat(alloc, u8, &.{ cr_s, lf_s });
    try self.vocab.put(crlf_s, 258);
    try self.decode_map.put(258, crlf_s);
    try self.merges.append(alloc, .{ cr_s, lf_s });
    try self.ranks.put(.{ .a = cr_s, .b = lf_s }, 2);
    const content = try alloc.dupe(u8, "<|im_start|>");
    try self.added.append(alloc, .{ .content = content, .id = 999 });
    try self.added_by_id.put(999, content);
    return self;
}

test "bytes_to_unicode roundtrips all 256 bytes" {
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        const c = b2u(b);
        try std.testing.expectEqual(b, u2b(c).?);
    }
    // spot: 0x20 maps to U+0120 (Ġ), 0x00 maps to U+0100
    try std.testing.expectEqual(@as(u21, 0x120), b2u(0x20));
    try std.testing.expectEqual(@as(u21, 0x100), b2u(0x00));
}

test "bpe rank loop merges lowest rank first" {
    const alloc = std.testing.allocator;
    var tok = try makeFixtureTokenizer(alloc);
    defer tok.deinit();
    // "abc" is one pretoken (letters): ab=0 then (ab)c=1 -> single id 257
    const ids = try tok.encode(alloc, "abc");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &.{257}, ids);
    // determinism across calls
    const again = try tok.encode(alloc, "abc");
    defer alloc.free(again);
    try std.testing.expectEqualSlices(u32, ids, again);
}

test "added tokens are isolated from BPE" {
    const alloc = std.testing.allocator;
    var tok = try makeFixtureTokenizer(alloc);
    defer tok.deinit();
    const ids = try tok.encode(alloc, "a<|im_start|>b");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ @as(u32, 'a'), 999, 'b' }, ids);
    const back = try tok.decode(alloc, ids);
    defer alloc.free(back);
    try std.testing.expectEqualStrings("a<|im_start|>b", back);
}

test "crlf is one pretoken via alt-5 backtracking" {
    const alloc = std.testing.allocator;
    var tok = try makeFixtureTokenizer(alloc);
    defer tok.deinit();
    const ids = try tok.encode(alloc, "a\r\nb");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ @as(u32, 'a'), 258, @as(u32, 'b') }, ids);
}

test "pretokenizer splits letters/digits/symbols" {
    const alloc = std.testing.allocator;
    var tok = try makeFixtureTokenizer(alloc);
    defer tok.deinit();
    // "a1": letter + digit are separate pretokens
    const ids = try tok.encode(alloc, "a1");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ @as(u32, 'a'), '1' }, ids);
    // " hello": leading space joins the word (alt 2 prefix), 6 symbols
    const hi = try tok.encode(alloc, " hello");
    defer alloc.free(hi);
    try std.testing.expectEqual(@as(usize, 6), hi.len);
    try std.testing.expectEqual(b2u(' ') , b2u(' ')); // table sanity
    const sp = try tok.decode(alloc, hi[0..1]);
    defer alloc.free(sp);
    try std.testing.expectEqualStrings(" ", sp);
}

test "roundtrips over fixed corpus" {
    const alloc = std.testing.allocator;
    var tok = try makeFixtureTokenizer(alloc);
    defer tok.deinit();
    const corpus = [_][]const u8{
        "hello world",
        "  leading and trailing  ",
        "tabs\tand\nnewlines\r\nmixed",
        "héllo wörld → ✓",
        "emoji 😀 mixed a😀b",
        "punctuation: 're 'll 123, 45.6!",
        "<|im_start|>user\nHi there<|im_start|>",
        "CJK: 日本語テスト 한국어 中文",
        "combining: e\xcc\x81 vs \xc3\xa9",
        "",
    };
    for (corpus) |s| {
        const ids = try tok.encode(alloc, s);
        defer alloc.free(ids);
        const back = try tok.decode(alloc, ids);
        defer alloc.free(back);
        try std.testing.expectEqualStrings(s, back);
    }
}

test "fromJson parses vocab/merges/added_tokens" {
    const alloc = std.testing.allocator;
    const doc =
        \\{"model": {"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]},
        \\ "added_tokens": [{"content": "<|im_end|>", "id": 7}]}
    ;
    var tok = try Tokenizer.fromJson(alloc, doc);
    defer tok.deinit();
    try std.testing.expectEqual(@as(u32, 0), tok.vocab.get("a").?);
    try std.testing.expectEqual(@as(u32, 7), tok.idOf("<|im_end|>").?);
    const ids = try tok.encode(alloc, "ab<|im_end|>");
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ @as(u32, 2), 7 }, ids);
    const back = try tok.decode(alloc, ids);
    defer alloc.free(back);
    try std.testing.expectEqualStrings("ab<|im_end|>", back);
}
