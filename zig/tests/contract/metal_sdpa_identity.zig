//! Raw Metal A/B probes for MLX-compatible decode SDPA, including both pass-one state and merge inputs.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const c = ct.spec;
const g = ct.gates;
const rig_ = @import("rig.zig");

const D = 128;
const H = 32;
const KVH = 2;
const GQA = H / KVH;
const floor = -std.math.floatMax(f32);
const scale: f32 = 1.0 / @sqrt(128.0);
const source = @embedFile("../../kernels/metal/ops/sdpa.metal");

pub const Kind = enum { vec, vec_cqt, part, part_cqt, part_gqa16, merge };

const vec_served = [_]c.Shape{ .{ .rows = 1, .n = H, .k = 1 }, .{ .rows = 1, .n = H, .k = 33 }, .{ .rows = 1, .n = H, .k = 300 }, .{ .rows = 1, .n = H, .k = 1023 } };
const vec_cqt_served = [_]c.Shape{ .{ .rows = 2, .n = H, .k = 33 }, .{ .rows = 2, .n = H, .k = 300 }, .{ .rows = 2, .n = H, .k = 1023 } };
const part_served = [_]c.Shape{ .{ .rows = 1, .n = H, .k = 1024 }, .{ .rows = 1, .n = H, .k = 1025 }, .{ .rows = 1, .n = H, .k = 1500 }, .{ .rows = 1, .n = H, .k = 4096 } };
const part_cqt_served = [_]c.Shape{ .{ .rows = 2, .n = H, .k = 1024 }, .{ .rows = 2, .n = H, .k = 1025 }, .{ .rows = 2, .n = H, .k = 1500 }, .{ .rows = 2, .n = H, .k = 4096 }, .{ .rows = 2, .n = H, .k = 8192 }, .{ .rows = 2, .n = H, .k = 8193 }, .{ .rows = 2, .n = H, .k = 20000 }, .{ .rows = 2, .n = H, .k = 40000 }, .{ .rows = 2, .n = H, .k = 70000 } };
const gqa_served = [_]c.Shape{ .{ .rows = 1, .n = H, .k = 8192 }, .{ .rows = 1, .n = H, .k = 8193 }, .{ .rows = 1, .n = H, .k = 20000 }, .{ .rows = 1, .n = H, .k = 40000 }, .{ .rows = 1, .n = H, .k = 70000 } };
const merge_served = [_]c.Shape{
    .{ .rows = 1, .n = H, .k = 1024 },  .{ .rows = 2, .n = H, .k = 1024 },  .{ .rows = 1, .n = H, .k = 1025 }, .{ .rows = 2, .n = H, .k = 1025 }, .{ .rows = 1, .n = H, .k = 1500 },  .{ .rows = 2, .n = H, .k = 1500 },  .{ .rows = 1, .n = H, .k = 4096 },  .{ .rows = 2, .n = H, .k = 4096 },
    .{ .rows = 1, .n = H, .k = 8192 },  .{ .rows = 2, .n = H, .k = 8192 },  .{ .rows = 1, .n = H, .k = 8193 }, .{ .rows = 2, .n = H, .k = 8193 }, .{ .rows = 1, .n = H, .k = 20000 }, .{ .rows = 2, .n = H, .k = 20000 }, .{ .rows = 1, .n = H, .k = 40000 }, .{ .rows = 2, .n = H, .k = 40000 },
    .{ .rows = 1, .n = H, .k = 70000 }, .{ .rows = 2, .n = H, .k = 70000 },
};

fn served(kind: Kind) []const c.Shape {
    return switch (kind) {
        .vec => &vec_served,
        .vec_cqt => &vec_cqt_served,
        .part => &part_served,
        .part_cqt => &part_cqt_served,
        .part_gqa16 => &gqa_served,
        .merge => &merge_served,
    };
}

fn entry(kind: Kind) []const u8 {
    return switch (kind) {
        .vec => "tf_sdpa_vec_d128",
        .vec_cqt => "tf_sdpa_vec_d128_cqt",
        .part => "tf_sdpa_2p1_d128",
        .part_cqt => "tf_sdpa_2p1_d128_cqt",
        .part_gqa16 => "tf_sdpa_2p1_gqa16_d128",
        .merge => "tf_sdpa_2p2_d128",
    };
}
fn firstStage(kind: Kind) bool {
    return kind == .part or kind == .part_cqt or kind == .part_gqa16;
}
fn causal(kind: Kind) bool {
    return kind == .vec_cqt or kind == .part_cqt;
}
fn transposed(kind: Kind) bool {
    return causal(kind);
}
fn blocksFor(keys: usize, rows: usize) usize {
    _ = rows; // Both raw-A/B targets are 'd' Ultras; Nemotron's gqa * rows is 16 or 32.
    if (keys >= 65536) return 1024;
    if (keys >= 16384) return 512;
    return 128;
}

fn cacheCapacity(keys: usize) usize {
    return (keys + 2047) / 2048 * 2048 + @as(usize, @intFromBool(keys % 2048 == 0)) * 2048;
}

fn depth(s: c.Shape) usize {
    return D + s.k + blocksFor(s.k, s.rows) + 32;
}

fn rngValue(rng: std.Random) u16 {
    return c.toBf16(rng.float(f32) - 0.5);
}

const Data = struct {
    rows: usize,
    keys: usize,
    cap: usize,
    blocks: usize,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    part: []u16 = &.{},
    sums: []f32 = &.{},
    maxs: []f32 = &.{},

    fn init(a: std.mem.Allocator, kind: Kind, shape: c.Shape, fill: g.Fill, seed: u64) !Data {
        var d: Data = .{
            .rows = shape.rows,
            .keys = shape.k,
            .cap = cacheCapacity(shape.k),
            .blocks = if (firstStage(kind) or kind == .merge) blocksFor(shape.k, shape.rows) else 0,
        };
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (kind == .merge) {
            const slots = d.rows * H * d.blocks;
            d.part = try a.alloc(u16, slots * D);
            d.sums = try a.alloc(f32, slots);
            d.maxs = try a.alloc(f32, slots);
            for (d.part, 0..) |*x, i| x.* = switch (fill) {
                .stress => c.toBf16(@floatFromInt(@as(i32, @intCast(i % 8)) - 4)),
                else => rngValue(rng),
            };
            for (d.sums, d.maxs) |*sum, *max| {
                sum.* = if (fill == .stress) 1 else 0.25 + rng.float(f32);
                max.* = if (fill == .stress) 0 else rng.float(f32) * 4 - 2;
            }
            if (fill == .nonfinite) {
                d.part[0] = 0x7fc0;
                d.part[1] = 0x7f80;
                d.part[2] = 0xff80;
                d.sums[0] = 1;
                d.maxs[0] = 0;
            }
            if (ct.identity.requested(seed)) d.plantMergeIdentity();
            return d;
        }
        d.q = try a.alloc(u16, d.rows * H * D);
        d.k = try a.alloc(u16, KVH * d.cap * D);
        d.v = try a.alloc(u16, KVH * d.cap * D);
        for (d.q) |*x| x.* = if (fill == .stress) 0 else rngValue(rng);
        for (d.k) |*x| x.* = if (fill == .stress) 0 else rngValue(rng);
        for (d.v, 0..) |*x, i| x.* = switch (fill) {
            .stress => c.toBf16(@floatFromInt(@as(i32, @intCast(i % 8)) - 4)),
            else => rngValue(rng),
        };
        // GQA16's reciprocal normalization is exercised by random cases; zero stress values keep its raw gate exact.
        if (kind == .part_gqa16 and fill == .stress) @memset(d.v, 0);
        if (fill == .nonfinite) {
            // Value-only exceptional inputs have unambiguous signs through every positive-weight path.
            d.v[0] = 0x7fc0;
            d.v[1] = 0x7f80;
            d.v[2] = 0xff80;
        }
        if (ct.identity.requested(seed)) d.plantAttentionIdentity(kind);
        return d;
    }

    fn plantAttentionIdentity(d: *Data, kind: Kind) void {
        // Even heads tie zero maxima; odd change max after key zero; head 2 has negative-zero Q and unread K/V padding.
        @memset(d.q, 0);
        @memset(d.k, 0);
        @memset(d.v, 0);
        for (0..d.rows) |row| for (0..H) |head| {
            const at = if (transposed(kind)) (row * H + head) * D else (head * d.rows + row) * D;
            if (head % 2 == 1) d.q[at] = c.toBf16(1);
            if (head % 4 == 2) d.q[at] = 0x8000;
        };
        for (0..KVH) |kv| for (0..d.keys) |key| {
            const at = (kv * d.cap + key) * D;
            d.k[at] = c.toBf16(@floatFromInt(key));
            d.v[at] = if (key % 2 == 0) 0x0000 else 0x8000;
            d.v[at + 1] = if (key % 2 == 0) 0x0001 else 0x8001;
            d.v[at + 2] = if (key % 3 == 0) 0x007f else 0x8001;
            d.v[at + 3] = if (key % 3 == 1) 0x0080 else 0x8080;
        };
    }

    fn plantMergeIdentity(d: *Data) void {
        for (0..d.rows) |row| for (0..H) |head| for (0..d.blocks) |block| {
            const at = (row * H + head) * d.blocks + block;
            var maximum: f32 = switch (block % 8) {
                0 => 0.0,
                1 => -0.0,
                2 => -1.0,
                3, 4 => 1.0,
                5, 6 => 2.0,
                else => -std.math.inf(f32),
            };
            if (head % 4 == 1) maximum = if (block % 2 == 0) 0.0 else -0.0;
            if (head % 4 == 2) maximum = @floatFromInt(block);
            if (head % 4 == 3) maximum = -std.math.inf(f32);
            d.maxs[at] = maximum;
            const empty = maximum == -std.math.inf(f32);
            d.sums[at] = if (empty) 0 else switch (block % 6) {
                0 => 0.0,
                1 => -0.0,
                2 => @bitCast(@as(u32, 1)),
                3 => @bitCast(@as(u32, 0x007fffff)),
                4 => 1.0,
                else => 3.0,
            };
            for (0..D) |e| {
                d.part[at * D + e] = if (empty) 0 else switch ((block + e) % 6) {
                    0 => 0x0000,
                    1 => 0x8000,
                    2 => 0x0001,
                    3 => 0x8001,
                    4 => 0x007f,
                    else => 0x0080,
                };
            }
        };
        // A complete all-empty merge checks the total == 0 path independently of masked blocks.
        if (H > 4) for (0..d.blocks) |block| {
            const at = 4 * d.blocks + block;
            d.maxs[at] = -std.math.inf(f32);
            d.sums[at] = 0;
            @memset(d.part[at * D ..][0..D], 0);
        };
    }

    fn qAt(d: *const Data, kind: Kind, head: usize, row: usize, e: usize) f64 {
        const at = if (transposed(kind)) (row * H + head) * D + e else (head * d.rows + row) * D + e;
        return c.fromBf16(d.q[at]);
    }

    fn vAt(d: *const Data, head: usize, key: usize, e: usize) f64 {
        return c.fromBf16(d.v[((head / GQA) * d.cap + key) * D + e]);
    }

    fn score(d: *const Data, kind: Kind, head: usize, row: usize, key: usize) f64 {
        var sum: f64 = 0;
        for (0..D) |e| {
            const kval = c.fromBf16(d.k[((head / GQA) * d.cap + key) * D + e]);
            sum += scale * d.qAt(kind, head, row, e) * kval;
        }
        return sum;
    }
};

const Stats = struct { top: f64, mass: f64 };

fn lastVisible(d: *const Data, kind: Kind, row: usize) isize {
    return if (causal(kind)) @as(isize, @intCast(d.keys - d.rows + row)) else @as(isize, @intCast(d.keys - 1));
}

/// Independent online-softmax reference over one ordinary strided part or one contiguous GQA16 chunk.
fn expectedPart(d: *const Data, kind: Kind, head: usize, row: usize, from: usize, stride: usize, scores: []f64, weights: []f64, out: []f64) Stats {
    const last = lastVisible(d, kind, row);
    var used: usize = 0;
    var top = -std.math.inf(f64);
    var key = from;
    while (key < d.keys) : (key += stride) {
        if (@as(isize, @intCast(key)) > last) continue;
        scores[used] = d.score(kind, head, row, key);
        top = @max(top, scores[used]);
        used += 1;
    }
    if (used == 0) {
        @memset(out, 0);
        return .{ .top = floor, .mass = 0 };
    }
    var mass: f64 = 0;
    for (scores[0..used], 0..) |score, i| {
        weights[i] = @exp(score - top);
        mass += weights[i];
    }
    for (0..D) |e| {
        var sum: f64 = 0;
        var i: usize = 0;
        key = from;
        while (key < d.keys) : (key += stride) {
            if (@as(isize, @intCast(key)) > last) continue;
            sum += weights[i] * d.vAt(head, key, e);
            i += 1;
        }
        out[e] = sum;
    }
    return .{ .top = top, .mass = mass };
}

fn expected(p: *Probe, want: []f64) !void {
    const d = &p.data;
    if (p.kind == .merge) return expectedMerge(p, want);
    const scores = try p.a.alloc(f64, d.keys);
    defer p.a.free(scores);
    const weights = try p.a.alloc(f64, d.keys);
    defer p.a.free(weights);
    const numerator = try p.a.alloc(f64, D);
    defer p.a.free(numerator);
    if (!firstStage(p.kind)) {
        for (0..d.rows) |row| for (0..H) |head| {
            const stat = expectedPart(d, p.kind, head, row, 0, 1, scores, weights, numerator);
            const out = want[(head * d.rows + row) * D ..][0..D];
            if (stat.mass == 0) @memset(out, 0) else for (out, numerator) |*x, n| x.* = n / stat.mass;
        };
        return;
    }
    const slots = H * d.rows * d.blocks;
    for (0..d.rows) |row| for (0..H) |head| for (0..d.blocks) |block| {
        const at = (head * d.rows + row) * d.blocks + block;
        const slot = at * D;
        const stat = if (p.kind == .part_gqa16) blk: {
            const chunk = (d.keys + d.blocks - 1) / d.blocks;
            const lo = @min(d.keys, block * chunk);
            const hi = @min(d.keys, lo + chunk);
            break :blk expectedChunk(d, head, lo, hi, scores, weights, numerator);
        } else expectedPart(d, p.kind, head, row, block, d.blocks, scores, weights, numerator);
        @memcpy(want[slot..][0..D], numerator);
        want[slots * D + at] = stat.mass;
        want[slots * D + slots + at] = stat.top;
    };
}

fn expectedChunk(d: *const Data, head: usize, lo: usize, hi: usize, scores: []f64, weights: []f64, out: []f64) Stats {
    var top = -std.math.inf(f64);
    for (lo..hi, 0..) |key, i| {
        scores[i] = d.score(.part_gqa16, head, 0, key);
        top = @max(top, scores[i]);
    }
    if (lo == hi) {
        @memset(out, 0);
        return .{ .top = floor, .mass = 0 };
    }
    var mass: f64 = 0;
    for (scores[0 .. hi - lo], 0..) |score, i| {
        weights[i] = @exp(score - top);
        mass += weights[i];
    }
    for (0..D) |e| {
        var sum: f64 = 0;
        for (lo..hi, 0..) |key, i| sum += weights[i] * d.vAt(head, key, e);
        out[e] = sum;
    }
    return .{ .top = top, .mass = mass };
}

fn expectedMerge(p: *Probe, want: []f64) void {
    const d = &p.data;
    for (0..d.rows) |row| for (0..H) |head| {
        const base = (row * H + head) * d.blocks;
        var top: f64 = floor;
        for (0..d.blocks) |block| top = @max(top, d.maxs[base + block]);
        var total: f64 = 0;
        for (0..d.blocks) |block| {
            const m = d.maxs[base + block];
            const weight = @exp(m - top);
            total += d.sums[base + block] * weight;
        }
        for (0..D) |e| {
            var numerator: f64 = 0;
            for (0..d.blocks) |block| {
                const m = d.maxs[base + block];
                const weight = @exp(m - top);
                numerator += c.fromBf16(d.part[(base + block) * D + e]) * weight;
            }
            want[(row * H + head) * D + e] = if (total == 0) numerator else numerator / total;
        }
    };
}

fn putBf16(out: []f64, at: usize, values: []const f64) void {
    for (values, 0..) |value, i| {
        const bits = c.toBf16(@floatCast(value));
        out[at + 2 * i] = bits & 0xff;
        out[at + 2 * i + 1] = bits >> 8;
    }
}

fn putF32(out: []f64, at: usize, value: f64) void {
    const bits: u32 = @bitCast(@as(f32, @floatCast(value)));
    for (0..4) |i| out[at + i] = (bits >> @intCast(8 * i)) & 0xff;
}

fn rawReference(p: *Probe, out: []f64) !void {
    const numeric = try p.a.alloc(f64, p.numericCount());
    defer p.a.free(numeric);
    try expected(p, numeric);
    if (firstStage(p.kind)) {
        const d = &p.data;
        const slots = H * d.rows * d.blocks;
        putBf16(out, 0, numeric[0 .. slots * D]);
        const sums_at = slots * D * 2;
        const maxs_at = sums_at + slots * 4;
        for (0..slots) |i| {
            putF32(out, sums_at + 4 * i, numeric[slots * D + i]);
            putF32(out, maxs_at + 4 * i, numeric[slots * D + slots + i]);
        }
    } else putBf16(out, 0, numeric);
}

fn words(bytes: []const u8) []const u16 {
    return @alignCast(std.mem.bytesAsSlice(u16, bytes));
}

fn floats(bytes: []const u8) []const f32 {
    return @alignCast(std.mem.bytesAsSlice(f32, bytes));
}

fn close(got: f64, want: f64, bf16: bool) bool {
    if (!std.math.isFinite(want) or !std.math.isFinite(got)) return c.classOf(got) == c.classOf(want);
    const relative = if (bf16) 0.03 else 0.0003;
    return @abs(got - want) <= relative * @abs(want) + 0.0001;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    kind: Kind,
    control: bool,
    label: [180]u8 = undefined,
    label_len: usize = 0,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    data: Data = undefined,
    input: [3]rig_.Ref = undefined,
    output: [3]rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    fn editedSource(a: std.mem.Allocator, kind: Kind, control: bool) ![]u8 {
        if (!control) return a.dupe(u8, source);
        const edit = if (kind == .merge) [2][]const u8{
            "for (int j = 0; j < 4; ++j) acc[j] += same ? static_cast<float>(p[j]) : f * static_cast<float>(p[j]);",
            "for (int j = 0; j < 4; ++j) acc[j] += (same ? static_cast<float>(p[j]) : f * static_cast<float>(p[j])) + 0.25f;",
        } else [2][]const u8{
            "static_cast<float>(scale) * q[",
            "q[",
        };
        const needed: usize = if (kind == .merge) 1 else 3;
        if (std.mem.count(u8, source, edit[0]) != needed) return error.SdpaControlEditMissing;
        return std.mem.replaceOwned(u8, a, source, edit[0], edit[1]);
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const text = try editedSource(a, kind, control);
        defer a.free(text);
        const pipe = try rig_.pipeline(device, text, entry(kind));
        if (pipe.simdWidth() != 32) return error.SdpaSimdWidth;
        const threads: usize = if (kind == .vec or kind == .vec_cqt or kind == .merge) 1024 else if (kind == .part_gqa16 or kind == .part_cqt) 512 else 32 * GQA;
        if (pipe.maxThreads() < threads) return error.SdpaThreadCapacity;
        var p: Probe = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .kind = kind, .control = control };
        const name = try std.fmt.bufPrint(&p.label, "{s}metal ops {s}{s}", .{ if (control) "control: " else "", entry(kind), if (control and kind == .merge) " biased merge parts" else if (control) " missing query scale" else "" });
        p.label_len = name.len;
        return p;
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        if (p.arena) |*arena| arena.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        return .{
            .ctx = p,
            .vt = &vt,
            .control = if (p.control) .shapes else null,
            .adversarial = !p.control,
            .spec = .{
                .name = p.label[0..p.label_len],
                .accum = .f32,
                .depth = depth,
                .out = .raw,
                .nan = .{ .stated = "Value NaN and infinities propagate through positive weights; finite outputs remain in the independent host reference's tolerance." },
                .served = served(p.kind),
                .sums = .{ .f16 = 0, .bf16 = 0, .exact = true, .why = "unit weights, key-independent integer values and power-of-two merge totals are exact in every order; no low-precision margin is claimed" },
            },
        };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != H or shape.k == 0 or shape.rows == 0) return false;
        if (shape.rows > 8 or (firstStage(p.kind) and shape.rows > 2 and fill != .nonfinite)) return false;
        p.rig.reset();
        if (p.arena) |*arena| arena.deinit();
        p.arena = .init(p.a);
        p.shape = shape;
        // Gate 1 clamps to five rows; pass one keeps one/two rows instead of a 32 x 16 x R group; GQA16 keeps one row.
        if (firstStage(p.kind)) p.shape.rows = if (p.kind == .part_cqt) 2 else 1;
        p.data = try Data.init(p.arena.?.allocator(), p.kind, p.shape, fill, seed);
        const d = &p.data;
        if (p.kind == .merge) {
            p.input[0] = try p.rig.input(std.mem.sliceAsBytes(d.part));
            p.input[1] = try p.rig.input(std.mem.sliceAsBytes(d.sums));
            p.input[2] = try p.rig.input(std.mem.sliceAsBytes(d.maxs));
            p.output[0] = try p.rig.output(d.rows * H * D * 2);
        } else {
            p.input[0] = try p.rig.input(std.mem.sliceAsBytes(d.q));
            p.input[1] = try p.rig.input(std.mem.sliceAsBytes(d.k));
            p.input[2] = try p.rig.input(std.mem.sliceAsBytes(d.v));
            if (firstStage(p.kind)) {
                const slots = d.rows * H * d.blocks;
                p.output[0] = try p.rig.output(slots * D * 2);
                p.output[1] = try p.rig.output(slots * 4);
                p.output[2] = try p.rig.output(slots * 4);
            } else p.output[0] = try p.rig.output(d.rows * H * D * 2);
        }
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const d = &p.data;
        e.setPipeline(p.pipe);
        for (p.input, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        if (p.kind == .merge) {
            const params = [_]i32{@intCast(d.blocks)};
            e.setBytes(std.mem.sliceAsBytes(&params), 3);
            e.setBuffer(p.output[0].buf, p.output[0].off, 4);
            e.dispatchGroups(mtl.Size.of(H, d.rows, 1), mtl.Size.of(1024, 1, 1));
        } else {
            const params = if (firstStage(p.kind)) [_]i32{ @intCast(d.keys), @intCast(d.blocks) } else [_]i32{ GQA, @intCast(d.keys) };
            const strides = [_]i64{ @intCast(d.cap * D), D, @intCast(d.cap * D), D };
            e.setBytes(std.mem.sliceAsBytes(&params), 3);
            e.setBytes(std.mem.sliceAsBytes(&strides), 4);
            e.setBytes(std.mem.asBytes(&scale), 5);
            const outputs = if (firstStage(p.kind)) 3 else 1;
            for (p.output[0..outputs], 6..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
            if (firstStage(p.kind)) {
                e.dispatchGroups(mtl.Size.of(KVH, 1, d.blocks), mtl.Size.of(32, GQA, if (p.kind == .part_gqa16) 1 else d.rows));
            } else e.dispatchGroups(mtl.Size.of(H, d.rows, 1), mtl.Size.of(1024, 1, 1));
        }
        try p.rig.finish(pass);
    }

    fn outputBytes(p: *Probe) usize {
        if (firstStage(p.kind)) return p.data.rows * H * p.data.blocks * (D * 2 + 8);
        return p.data.rows * H * D * 2;
    }

    fn numericCount(p: *Probe) usize {
        if (firstStage(p.kind)) return p.data.rows * H * p.data.blocks * (D + 2);
        return p.data.rows * H * D;
    }

    pub fn count(p: *Probe) usize {
        return p.outputBytes();
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try rawReference(p, want);
        @memset(mag, 0);
    }

    pub fn simulate(p: *Probe, _: c.Accum, _: g.Order, out: []f64) !void {
        try rawReference(p, out);
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const numeric = try p.a.alloc(f64, p.numericCount());
        defer p.a.free(numeric);
        try expected(p, numeric);
        const d = &p.data;
        if (firstStage(p.kind)) {
            const slots = d.rows * H * d.blocks;
            const part = words(p.rig.bytes(p.output[0], slots * D * 2));
            const sums = floats(p.rig.bytes(p.output[1], slots * 4));
            const maxs = floats(p.rig.bytes(p.output[2], slots * 4));
            var nonfinite: usize = 0;
            for (part, 0..) |got, i| {
                nonfinite += @intFromBool(!std.math.isFinite(c.fromBf16(got)));
                if (!close(c.fromBf16(got), numeric[i], true)) return std.fmt.bufPrint(&p.msg, "PART {d} is {e:.6}, host {e:.6}", .{ i, c.fromBf16(got), numeric[i] });
            }
            for (0..slots) |i| {
                nonfinite += @intFromBool(!std.math.isFinite(sums[i]) or !std.math.isFinite(maxs[i]));
                if (!close(sums[i], numeric[slots * D + i], false)) return std.fmt.bufPrint(&p.msg, "SUMS {d} is {e:.6}, host {e:.6}", .{ i, sums[i], numeric[slots * D + i] });
                if (!close(maxs[i], numeric[slots * D + slots + i], false)) return std.fmt.bufPrint(&p.msg, "MAXS {d} is {e:.6}, host {e:.6}", .{ i, maxs[i], numeric[slots * D + slots + i] });
            }
            if (nonfinite == numeric.len) return std.fmt.bufPrint(&p.msg, "every output is nonfinite", .{});
        } else {
            const out = words(p.rig.bytes(p.output[0], numeric.len * 2));
            var nonfinite: usize = 0;
            for (out, numeric, 0..) |got, want, i| {
                nonfinite += @intFromBool(!std.math.isFinite(c.fromBf16(got)));
                if (!close(c.fromBf16(got), want, true)) return std.fmt.bufPrint(&p.msg, "O {d} is {e:.6}, host {e:.6}", .{ i, c.fromBf16(got), want });
            }
            if (nonfinite == numeric.len) return std.fmt.bufPrint(&p.msg, "every output is nonfinite", .{});
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [12]Probe = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |control| {
            const probe = &s.probes[s.made];
            probe.* = try Probe.init(a, device, queue, kind, control);
            s.made += 1;
            const kernel = probe.kernel();
            if (only) |needle| if (std.mem.indexOf(u8, kernel.spec.name, needle) == null) continue;
            try out.append(a, kernel);
        };
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
        const a = s.a;
        a.destroy(s);
    }
};
