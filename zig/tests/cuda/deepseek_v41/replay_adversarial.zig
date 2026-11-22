//! Finite replay-input edits: real recorded shapes/layouts, with no synthetic replacement fixture.
const std = @import("std");
const Fixture = @import("../fixture.zig").Fixture;
const Value = std.json.Value;

const Edit = struct { storage: []const u8, offset: usize, bytes: []const u8 };

pub const Plant = struct {
    zeroes: std.ArrayList([]const u8) = .empty,
    edits: std.ArrayList(Edit) = .empty,

    /// Apply edits on upload; zeroed caches zero values, and overlays tie chunks 0/1 before chunk 2 raises the maximum.
    pub fn apply(self: *const Plant, storage: []const u8, host: []u8) !void {
        for (self.zeroes.items) |name| if (std.mem.eql(u8, name, storage)) @memset(host, 0);
        for (self.edits.items) |change| {
            if (!std.mem.eql(u8, change.storage, storage)) continue;
            if (change.offset > host.len or change.bytes.len > host.len - change.offset) return error.AdversarialRange;
            @memcpy(host[change.offset..][0..change.bytes.len], change.bytes);
        }
    }
};

const Host = struct {
    data: []u8,
    tensor: Value,

    fn load(gpu: anytype, fx: Fixture, a: std.mem.Allocator, tensor: Value) !Host {
        _ = gpu;
        const name = try std.fmt.allocPrint(a, "{s}_before", .{storageOf(tensor)});
        return .{ .data = try fx.bytes(name), .tensor = tensor };
    }

    fn deinit(self: *Host, gpu: anytype) void {
        gpu.gpa.free(self.data);
    }

    fn at(self: *const Host, index: i64) !i64 {
        if (index < 0) return error.AdversarialRange;
        const size = elementSize(self.tensor);
        const base = try byteOffset(self.tensor);
        const offset = std.math.add(usize, base, std.math.mul(usize, @intCast(index), size) catch return error.AdversarialRange) catch return error.AdversarialRange;
        if (offset > self.data.len or size > self.data.len - offset) return error.AdversarialRange;
        return switch (size) {
            4 => std.mem.readInt(i32, self.data[offset..][0..4], .little),
            8 => std.mem.readInt(i64, self.data[offset..][0..8], .little),
            else => error.AdversarialDtype,
        };
    }
};

fn object(v: Value) std.json.ObjectMap {
    return v.object;
}

fn args(call: Value) []const Value {
    return object(call).get("args").?.array.items;
}

fn storageOf(v: Value) []const u8 {
    return object(v).get("t").?.string;
}

fn byteOffset(v: Value) !usize {
    const raw = object(v).get("off").?.integer;
    if (raw < 0) return error.AdversarialRange;
    return @intCast(raw);
}

fn shape(v: Value) []const Value {
    return object(v).get("shape").?.array.items;
}

fn stride(v: Value) []const Value {
    return object(v).get("stride").?.array.items;
}

fn numel(v: Value) usize {
    var n: usize = 1;
    for (shape(v)) |x| n *= @intCast(x.integer);
    return n;
}

fn elementSize(v: Value) usize {
    const name = object(v).get("dtype").?.string;
    const map = .{ .{ "uint8", 1 }, .{ "int8", 1 }, .{ "bool", 1 }, .{ "bfloat16", 2 }, .{ "float16", 2 }, .{ "int16", 2 }, .{ "float32", 4 }, .{ "int32", 4 }, .{ "float64", 8 }, .{ "int64", 8 } };
    inline for (map) |entry| if (std.mem.eql(u8, name, entry[0])) return entry[1];
    unreachable;
}

fn scalar(v: Value) i64 {
    return object(v).get("i").?.integer;
}

fn zero(p: *Plant, a: std.mem.Allocator, v: Value) !void {
    const name = storageOf(v);
    for (p.zeroes.items) |old| if (std.mem.eql(u8, old, name)) return;
    try p.zeroes.append(a, name);
}

fn edit(p: *Plant, a: std.mem.Allocator, v: Value, offset: usize, bytes: []const u8) !void {
    try p.edits.append(a, .{ .storage = storageOf(v), .offset = offset, .bytes = try a.dupe(u8, bytes) });
}

fn query(p: *Plant, a: std.mem.Allocator, v: Value) !void {
    if (elementSize(v) != 2 or shape(v).len != 3 or stride(v).len != 3) return error.AdversarialQueryLayout;
    try zero(p, a, v);
    const bits = [_]u16{ 0x3f80, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f };
    const base = try byteOffset(v);
    const rows: usize = @intCast(shape(v)[0].integer);
    const heads: usize = @intCast(shape(v)[1].integer);
    for (0..rows) |r| for (0..heads) |h| {
        const at = base + 2 * (r * @as(usize, @intCast(stride(v)[0].integer)) + h * @as(usize, @intCast(stride(v)[1].integer)));
        const data = try a.alloc(u8, 12);
        for (bits, 0..) |word, i| std.mem.writeInt(u16, data[2 * i ..][0..2], word, .little);
        try edit(p, a, v, at, data);
    };
}

fn cache(p: *Plant, a: std.mem.Allocator, v: Value, row: i64, score: u8) !void {
    if (row < 0 or numel(v) == 0) return;
    if (elementSize(v) != 1 or stride(v).len == 0) return error.AdversarialCacheLayout;
    const base = try byteOffset(v);
    const row_stride: usize = @intCast(stride(v)[0].integer);
    const at = std.math.add(usize, base, std.math.mul(usize, @intCast(row), row_stride) catch return error.AdversarialRange) catch return error.AdversarialRange;
    // E4M3 1/2, signed zero, least/largest signed subnormals, then two value lanes.
    const bytes = [_]u8{ score, 0x80, 0x01, 0x81, 0x07, 0x87, score, score };
    try edit(p, a, v, at, &bytes);
}

/// Edit recorded tensors: cache scales are one; Q has one unit score, signed zeros and extreme BF16 subnormals.
pub fn attention(gpu: anytype, fx: Fixture, a: std.mem.Allocator, call: Value) !Plant {
    var p: Plant = .{};
    const av = args(call);
    if (av.len != 23) return error.AdversarialArguments;
    const q = av[0];
    const cv = av[1];
    const tok = av[3];
    const sv = av[5];
    const hi = av[8];
    const sl = av[10];
    const pt = av[11];
    try query(&p, a, q);
    if (numel(cv) != 0) {
        try zero(&p, a, cv);
        try zero(&p, a, av[2]);
    }
    if (numel(sv) != 0) {
        try zero(&p, a, sv);
        try zero(&p, a, av[6]);
    }
    var tok_h = try Host.load(gpu, fx, a, tok);
    defer tok_h.deinit(gpu);
    var cnt_h = try Host.load(gpu, fx, a, av[4]);
    defer cnt_h.deinit(gpu);
    var lo_h = try Host.load(gpu, fx, a, av[7]);
    defer lo_h.deinit(gpu);
    var hi_h = try Host.load(gpu, fx, a, hi);
    defer hi_h.deinit(gpu);
    var pos_h = try Host.load(gpu, fx, a, av[9]);
    defer pos_h.deinit(gpu);
    var sl_h = try Host.load(gpu, fx, a, sl);
    defer sl_h.deinit(gpu);
    var pt_h = try Host.load(gpu, fx, a, pt);
    defer pt_h.deinit(gpu);

    const rows: i64 = shape(q)[0].integer;
    const ring = scalar(av[19]);
    const psh = scalar(av[20]);
    const pts = scalar(av[21]);
    const token_stride = stride(tok)[0].integer;
    for (0..@intCast(rows)) |r| {
        const ri: i64 = @intCast(r);
        const count = try cnt_h.at(ri);
        const slot = if (numel(sl) == 0) 0 else try sl_h.at(ri);
        if (numel(cv) != 0) for (0..4) |chunk| {
            const begin: i64 = @intCast(chunk * 128);
            const end = @min(count, begin + 128);
            if (end <= begin) continue;
            for (@intCast(begin)..@intCast(end)) |index| {
                var row = try tok_h.at(ri * token_stride + @as(i64, @intCast(index)));
                if (row < 0) continue;
                if (psh > 0) {
                    const page = try pt_h.at(slot * pts + (row >> @intCast(psh)));
                    row = (page << @intCast(psh)) + (row & ((@as(i64, 1) << @intCast(psh)) - 1));
                }
                try cache(&p, a, cv, row, if (chunk < 2) 0x38 else 0x40);
            }
        };
        if (numel(sv) != 0) {
            if (ring <= 0) return error.AdversarialRange;
            const end = if (numel(hi) == 0) (if (numel(sl) == 0) try pos_h.at(0) + ri else try pos_h.at(ri)) else try hi_h.at(ri);
            const first = @max(try lo_h.at(ri), end - 127);
            if (end < first) continue;
            for (@intCast(first)..@intCast(end + 1)) |pos| {
                if (pos < 0) continue;
                const row = @mod(@as(i64, @intCast(pos)), ring) + slot * ring;
                try cache(&p, a, sv, row, 0x40);
            }
        }
    }
    return p;
}
