//! Isolated exceptional V: finite zero queries/keys, ordinary values, and one later key per fixture.
const std = @import("std");
const ct = @import("contract");
const c = ct.spec;

pub const Poison = enum { inf, nan };
pub const Layout = enum { token, head };
pub const Geometry = struct {
    heads: usize,
    kv: usize,
    dim: usize,
    cap: usize = 2048,
    first: usize = 16,
    window: usize = 0,
    ring: usize = 0,
    cache: Layout = .token,
    query: Layout = .token,
    output: Layout = .token,
    gated: bool = false,
    latent: bool = false,
    poison_key: ?usize = null,
    skip_zero_first: bool = false,
    selected_prefix: ?usize = null,
    selected_shift_first: usize = 0,
    sink: bool = false,
    scale: ?f32 = null,
    identity_all_masked: bool = true,
};

pub const Data = struct {
    geo: Geometry,
    rows: usize,
    poisoned: bool,
    poison: Poison,
    identity: bool = false,
    q: []u16,
    k: []u16,
    v: []u16,

    pub fn init(a: std.mem.Allocator, geo: Geometry, rows: usize, poisoned: bool, poison: Poison) !Data {
        const d: Data = .{ .geo = geo, .rows = rows, .poisoned = poisoned, .poison = poison, .q = try a.alloc(u16, @max(64, rows) * geo.heads * geo.dim), .k = try a.alloc(u16, geo.cap * geo.kv * geo.dim), .v = try a.alloc(u16, geo.cap * geo.kv * geo.dim) };
        @memset(d.q, 0);
        @memset(d.k, 0);
        @memset(d.v, c.toBf16(1));
        if (poisoned) {
            const at = d.cacheAt(geo.poison_key orelse geo.first + 1, 0, 7);
            d.v[at] = if (poison == .inf) 0x7f80 else 0x7fc0;
            if (geo.latent) d.k[at] = d.v[at];
        }
        return d;
    }

    pub fn cacheAt(d: Data, key: usize, head: usize, col: usize) usize {
        const g = d.geo;
        const slot = if (g.ring == 0) key else key % g.ring;
        return if (g.cache == .token) (slot * g.kv + head) * g.dim + col else (head * g.cap + slot) * g.dim + col;
    }
    pub fn queryAt(d: Data, row: usize, head: usize, col: usize) usize {
        const g = d.geo;
        return if (g.query == .token) (row * g.heads + head) * g.dim + col else (head * d.rows + row) * g.dim + col;
    }

    pub fn scale(d: Data) f64 {
        return d.geo.scale orelse 1 / @sqrt(@as(f64, @floatFromInt(d.geo.dim)));
    }

    /// `.inf` ties finite maxima; `.nan` changes max after 32 keys then ties; both use signed-zero/subnormal V.
    pub fn plantIdentity(d: *Data) void {
        d.identity = true;
        const g = d.geo;
        for (0..g.cap) |key| for (0..g.kv) |head| {
            const zero: u16 = if (key % 2 == 0) 0 else 0x8000;
            d.v[d.cacheAt(key, head, 0)] = zero;
            d.v[d.cacheAt(key, head, 7)] = if ((key + head) % 2 == 0) 0x0001 else 0x8001;
            if (d.poison == .inf) {
                d.k[d.cacheAt(key, head, 0)] = zero;
                if (g.latent) d.v[d.cacheAt(key, head, 0)] = zero;
            } else {
                const target: f32 = if (key < 32) -1 else 1;
                const score_bits = c.toBf16(target / d.scale());
                d.k[d.cacheAt(key, head, 0)] = score_bits;
                if (g.latent) d.v[d.cacheAt(key, head, 0)] = score_bits;
            }
        };
        if (d.poison == .nan) {
            for (0..d.rows) |row| for (0..g.heads) |head| {
                d.q[d.queryAt(row, head, 0)] = c.toBf16(1);
            };
        } else if (d.q.len > 0) d.q[0] = 0x8000;
    }

    pub fn outputAt(d: Data, row: usize, head: usize, col: usize) usize {
        const g = d.geo;
        return if (g.output == .token) (row * g.heads + head) * g.dim + col else (head * d.rows + row) * g.dim + col;
    }

    pub fn count(d: Data) usize {
        return d.rows * d.geo.heads * d.geo.dim;
    }

    /// Binary64 softmax at real zero scores over the visible masked-value set, excluding unseen keys before K/V reads.
    fn referenceIdentity(d: Data, want: []f64, mag: []f64) void {
        const g = d.geo;
        @memset(want, 0);
        for (0..d.rows) |r| {
            const hi = g.first + r + 1;
            const skip = if (r == 0) @max(if (g.skip_zero_first) @as(usize, 1) else 0, g.selected_shift_first) else 0;
            const lo = @max(if (g.window > 0) hi -| g.window else 0, skip);
            for (0..g.heads) |h| {
                var total: f64 = if (g.sink) 1 else 0;
                const kh = h / (g.heads / g.kv);
                for (lo..hi) |key| {
                    if (g.identity_all_masked and r == 0) continue;
                    if (g.selected_prefix) |prefix| if (key >= prefix + (if (r == 0) g.selected_shift_first else 0) and key < (hi - 1) / 8 * 8) continue;
                    // plantIdentity leaves every other Q/K component zero.
                    const q = c.fromBf16(d.q[d.queryAt(r, h, 0)]);
                    const k = c.fromBf16(d.k[d.cacheAt(key, kh, 0)]);
                    const probability = @exp(q * k * d.scale());
                    total += probability;
                    for (0..g.dim) |col| {
                        const i = d.outputAt(r, h, col);
                        want[i] += probability * c.fromBf16(d.v[d.cacheAt(key, kh, col)]);
                    }
                }
                for (0..g.dim) |col| {
                    const i = d.outputAt(r, h, col);
                    const x = if (total == 0) 0 else want[i] / total;
                    want[i] = if (g.gated) c.fromBf16(c.toBf16(@floatCast(x))) * 0.5 else x;
                    mag[i] = @abs(want[i]);
                }
            }
        }
    }
    pub fn reference(d: Data, want: []f64, mag: []f64) void {
        if (d.identity) return d.referenceIdentity(want, mag);
        const g = d.geo;
        for (0..d.rows) |r| {
            const hi = g.first + r + 1;
            const skip = if (r == 0) @max(if (g.skip_zero_first) @as(usize, 1) else 0, g.selected_shift_first) else 0;
            const lo = @max(if (g.window > 0) hi -| g.window else 0, skip);
            for (0..g.heads) |h| {
                var ordinary: f64 = 0;
                var exceptional: f64 = 0;
                var total: f64 = 0;
                for (lo..hi) |key| {
                    if (g.selected_prefix) |prefix| if (key >= prefix + (if (r == 0) g.selected_shift_first else 0) and key < (hi - 1) / 8 * 8) continue;
                    const kh = h / (g.heads / g.kv);
                    const score: f64 = if (g.latent and kh == 0 and d.poisoned and key == (g.poison_key orelse g.first + 1))
                        0 * @as(f64, c.fromBf16(d.k[d.cacheAt(key, kh, 7)]))
                    else
                        0;
                    const p = @exp(score);
                    ordinary += p;
                    exceptional += p * @as(f64, c.fromBf16(d.v[d.cacheAt(key, kh, 7)]));
                    total += p;
                }
                if (g.sink) total += @exp(@as(f64, 0));
                for (0..g.dim) |col| {
                    const i = d.outputAt(r, h, col);
                    const x = (if (col == 7) exceptional else ordinary) / total;
                    want[i] = if (g.gated) @as(f64, c.fromBf16(c.toBf16(@floatCast(x)))) * 0.5 else x;
                    mag[i] = @abs(want[i]);
                }
            }
        }
    }
};

test "masked value: hidden Inf and NaN do not enter the binary64 formula" {
    for (std.enums.values(Poison)) |poison| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const d = try Data.init(a, .{ .heads = 4, .kv = 2, .dim = 32, .cap = 64 }, 5, true, poison);
        const want = try a.alloc(f64, d.count());
        const mag = try a.alloc(f64, want.len);
        d.reference(want, mag);
        try std.testing.expectEqual(@as(f64, 1), want[d.outputAt(0, 0, 7)]);
        try std.testing.expectEqual(if (poison == .inf) c.Class.pos_inf else c.Class.nan, c.classOf(want[d.outputAt(1, 0, 7)]));
        try std.testing.expectEqual(@as(f64, 1), want[d.outputAt(1, 2, 7)]);
    }
}

test "masked value: aliased latent keys retain the visible score exception" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try Data.init(a, .{ .heads = 4, .kv = 1, .dim = 32, .cap = 64, .latent = true }, 5, true, .inf);
    const want = try a.alloc(f64, d.count());
    const mag = try a.alloc(f64, want.len);
    d.reference(want, mag);
    try std.testing.expectEqual(@as(f64, 1), want[0]);
    try std.testing.expect(std.math.isNan(want[d.outputAt(1, 0, 0)]));
}

test "masked value: ring wrap uses absolute visibility and physical slots" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try Data.init(a, .{ .heads = 4, .kv = 2, .dim = 32, .cap = 32, .first = 31, .window = 8, .ring = 32 }, 5, true, .inf);
    const want = try a.alloc(f64, d.count());
    const mag = try a.alloc(f64, want.len);
    d.reference(want, mag);
    try std.testing.expectEqual(@as(f64, 1), want[d.outputAt(0, 0, 7)]);
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[d.outputAt(1, 0, 7)]));
    try std.testing.expectEqual(d.cacheAt(0, 0, 7), d.cacheAt(32, 0, 7));
}

test "masked value: sparse padding omits key zero only where the real list does" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try Data.init(a, .{ .heads = 4, .kv = 1, .dim = 32, .cap = 64, .latent = true, .poison_key = 0, .skip_zero_first = true }, 5, true, .nan);
    const want = try a.alloc(f64, d.count());
    const mag = try a.alloc(f64, want.len);
    d.reference(want, mag);
    try std.testing.expectEqual(@as(f64, 1), want[0]);
    try std.testing.expect(std.math.isNan(want[d.outputAt(1, 0, 0)]));
}

test "masked value: sparse own region stops exposing an unselected prior region" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try Data.init(a, .{ .heads = 4, .kv = 2, .dim = 32, .cap = 8192, .first = 4104, .selected_prefix = 4096 }, 16, true, .inf);
    const want = try a.alloc(f64, d.count());
    const mag = try a.alloc(f64, want.len);
    d.reference(want, mag);
    try std.testing.expectEqual(@as(f64, 1), want[d.outputAt(0, 0, 7)]);
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[d.outputAt(1, 0, 7)]));
    try std.testing.expectEqual(@as(f64, 1), want[d.outputAt(8, 0, 7)]);
}

test "masked value: a zero-score sink adds to the denominator but has no value" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try Data.init(a, .{ .heads = 4, .kv = 1, .dim = 32, .cap = 64, .sink = true }, 5, true, .inf);
    const want = try a.alloc(f64, d.count());
    const mag = try a.alloc(f64, want.len);
    d.reference(want, mag);
    try std.testing.expectEqual(@as(f64, 17.0 / 18.0), want[0]);
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[d.outputAt(1, 0, 7)]));
}

test "masked value: identity plants ties, changed maxima, signed zero and subnormals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const geo: Geometry = .{ .heads = 1, .kv = 1, .dim = 8, .cap = 64, .first = 16 };
    const tied = try Data.init(a, geo, 8, false, .inf);
    var planted = tied;
    planted.plantIdentity();
    try std.testing.expectEqual(@as(u16, 0), planted.k[planted.cacheAt(0, 0, 0)]);
    try std.testing.expectEqual(@as(u16, 0x8000), planted.k[planted.cacheAt(1, 0, 0)]);
    try std.testing.expectEqual(@as(u16, 0x8000), planted.v[planted.cacheAt(1, 0, 0)]);
    try std.testing.expectEqual(@as(u16, 0x0001), planted.v[planted.cacheAt(0, 0, 7)]);
    try std.testing.expectEqual(@as(u16, 0x8001), planted.v[planted.cacheAt(1, 0, 7)]);
    const changed = try Data.init(a, geo, 8, false, .nan);
    var maximum = changed;
    maximum.plantIdentity();
    try std.testing.expect(c.fromBf16(maximum.k[maximum.cacheAt(31, 0, 0)]) < 0);
    try std.testing.expect(c.fromBf16(maximum.k[maximum.cacheAt(32, 0, 0)]) > 0);
    try std.testing.expectEqual(@as(f32, 1), c.fromBf16(maximum.q[maximum.queryAt(0, 0, 0)]));
    const want = try a.alloc(f64, maximum.count());
    const mag = try a.alloc(f64, want.len);
    maximum.reference(want, mag);
    for (0..geo.dim) |col| try std.testing.expectEqual(@as(f64, 0), want[maximum.outputAt(0, 0, col)]);
}
