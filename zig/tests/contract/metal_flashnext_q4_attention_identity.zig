//! Raw Qwen4 parts, exact attn2 two-head transform and gated merge; PO/PM and OUT launches cover QSA-served shapes.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const c = ct.spec;
const g = ct.gates;
const rig_ = @import("rig.zig");
const attn2 = @import("../../src/families/flashnext/attn2.zig");
const patch = @import("../../src/families/flashnext/tree.zig").patch;

const D = 256;
const H = 24;
const KVH = 2;
const P = 16;
const PW = 13952;
const scale: f32 = 0.0625;
const parts_name = "custom_kernel_q4_attn_parts_f6b4ffe8e2609998_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_int32_t_int32_t_floatc_float_float";
const merge_name = "custom_kernel_q4_attn_merge_gate_7070840cd3c22e98_float_float_bfloat16_t_bfloat16_t";
const parts_source = @embedFile("../../kernels/metal/flashnext/q4_attn_parts_f6b4ffe8e2609998.metal");
const merge_source = @embedFile("../../kernels/metal/flashnext/q4_attn_merge_gate_7070840cd3c22e98.metal");

pub const Kind = enum { parts, parts_two_heads, merge };
pub const served = [_]c.Shape{
    .{ .rows = 1, .n = 1, .k = 1024 }, // dense, exactly partitioned
    .{ .rows = 1, .n = 2, .k = 17 }, // sparse, empty and one-key parts
    .{ .rows = 128, .n = 2, .k = 33 }, // the widest recorded batch
    .{ .rows = 1, .n = 2, .k = 1025 }, // sparse, uneven long parts
};

fn depth(s: c.Shape) usize {
    return D + P + s.k;
}

fn label(kind: Kind, control: bool) []const u8 {
    const what = switch (kind) {
        .parts => "recorded q4_attn_parts PO+PM",
        .parts_two_heads => "attn2 two-head q4_attn_parts PO+PM",
        .merge => "q4_attn_merge_gate OUT",
    };
    return (if (control) "control: metal flashnext Qwen4 " else "metal flashnext Qwen4 ") ++ what;
}

fn spec(kind: Kind, control: bool) c.Spec {
    return .{
        .name = label(kind, control),
        .accum = .f32,
        .depth = depth,
        .mids = if (kind == .merge) 1 else 0,
        .out = if (kind == .merge) .bf16 else .f32,
        .tiles = .{ .{}, .{}, .{ .size = P } },
        .served = &served,
        .nan = .{ .stated = "Finite Q/K and merge statistics retain each exceptional numerator class through the independent binary64 formula." },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "equal-score integer partials prove FP32 summation; low-precision margins are immaterial to this raw identity registration" },
    };
}

fn source(a: std.mem.Allocator, kind: Kind, control: bool) ![]u8 {
    const base = switch (kind) {
        .parts => try a.dupe(u8, parts_source),
        .parts_two_heads => try patch(a, parts_source, &attn2.edits),
        .merge => try a.dupe(u8, merge_source),
    };
    defer a.free(base);
    if (!control) return a.dupe(u8, base);
    const edit = switch (kind) {
        .parts => [2][]const u8{ "q[i] = SCALE[0] * float(qp[i])", "q[i] = float(qp[i])" },
        .parts_two_heads => [2][]const u8{ "q[hh][i] = SCALE[0] * float(qp[i])", "q[hh][i] = float(qp[i])" },
        .merge => [2][]const u8{
            "OUT[(size_t(r) * H + h) * D + d] = bfloat(float(bfloat(acc / total)) * bsig(g));",
            "if (d != D - 1) OUT[(size_t(r) * H + h) * D + d] = bfloat(float(bfloat(acc / total)) * bsig(g));",
        },
    };
    if (std.mem.count(u8, base, edit[0]) != 1) return error.FlashnextControlEditMissing;
    return std.mem.replaceOwned(u8, a, base, edit[0], edit[1]);
}

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

fn value(rng: std.Random) u16 {
    return c.toBf16(rng.float(f32) * 2 - 1);
}

fn stressValue(j: usize, n: usize) f32 {
    const half = n / 2;
    if (j >= 2 * half) return 0;
    const at = if (j < half) j else j - half;
    const positive: f32 = if (at % 2 == 0) 127 else 16256;
    const residual: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
    return if (j < half) positive + residual else residual - positive;
}

const Data = struct {
    rows: usize,
    keys: usize,
    cap: usize,
    sparse: bool,
    stress: bool,
    q: []u16 = &.{},
    key: []u16 = &.{},
    val: []u16 = &.{},
    ids: []i32 = &.{},
    nk: []i32 = &.{},
    spr: []i32 = &.{},
    po: []f32 = &.{},
    pm: []f32 = &.{},
    gp: []u16 = &.{},

    fn init(a: std.mem.Allocator, kind: Kind, shape: c.Shape, fill: g.Fill, seed: u64) !Data {
        var d: Data = .{
            .rows = shape.rows,
            .keys = shape.k,
            .cap = shape.k * (if (shape.n == 2) @as(usize, 2) else 1),
            .sparse = shape.n == 2,
            .stress = fill == .stress,
        };
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (kind == .merge) {
            d.po = try a.alloc(f32, shape.rows * H * P * D);
            d.pm = try a.alloc(f32, shape.rows * H * P * 2);
            d.gp = try a.alloc(u16, shape.rows * PW);
            for (d.po) |*x| x.* = rng.float(f32) * 4 - 2;
            for (0..shape.rows * H * P) |i| {
                d.pm[2 * i] = rng.float(f32) * 6 - 3;
                d.pm[2 * i + 1] = 0.125 + rng.float(f32);
            }
            for (d.gp) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
            switch (fill) {
                .random => {},
                .stress => {
                    for (0..shape.rows * H * P) |i| {
                        d.pm[2 * i] = 0;
                        d.pm[2 * i + 1] = 1;
                    }
                    for (0..shape.rows * H) |i| for (0..P) |part| for (0..D) |e| {
                        d.po[(i * P + part) * D + e] = stressValue(part, P);
                    };
                    @memset(d.gp, c.toBf16(20));
                },
                .nonfinite => {
                    for (0..shape.rows * H * P) |i| {
                        d.pm[2 * i] = 0;
                        d.pm[2 * i + 1] = 1;
                    }
                    d.po[0] = std.math.nan(f32);
                    d.po[1] = std.math.inf(f32);
                    d.po[2] = -std.math.inf(f32);
                },
            }
            if (ct.identity.requested(seed)) d.plantMergeIdentity();
            return d;
        }
        d.q = try a.alloc(u16, shape.rows * H * D);
        d.key = try a.alloc(u16, KVH * d.cap * D);
        d.val = try a.alloc(u16, KVH * d.cap * D);
        d.ids = try a.alloc(i32, shape.rows * shape.k);
        d.nk = try a.alloc(i32, shape.rows);
        d.spr = try a.alloc(i32, shape.rows);
        for (d.q) |*x| x.* = if (d.stress) 0 else value(rng);
        for (d.key) |*x| x.* = if (d.stress) 0 else value(rng);
        for (d.val) |*x| x.* = value(rng);
        @memset(d.nk, @intCast(shape.k));
        @memset(d.spr, @intFromBool(d.sparse));
        for (0..shape.rows) |r| for (0..shape.k) |j| {
            d.ids[r * shape.k + j] = @intCast(2 * (shape.k - 1 - j) + 1);
            if (d.stress) {
                const part = partOf(j, shape.k);
                const lo = part * shape.k / P;
                const hi = (part + 1) * shape.k / P;
                const source_key = d.keyAt(r, j);
                for (0..KVH) |kv| @memset(d.val[d.cacheAt(kv, source_key)..][0..D], c.toBf16(stressValue(j - lo, hi - lo)));
            }
        };
        if (fill == .nonfinite) {
            const source_key = d.keyAt(0, 0);
            for (0..KVH) |kv| for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |x, e| {
                d.val[d.cacheAt(kv, source_key) + e] = x;
            };
        }
        if (ct.identity.requested(seed)) d.plantPartsIdentity();
        return d;
    }

    fn cacheAt(d: *const Data, kv: usize, key: usize) usize {
        return (kv * d.cap + key) * D;
    }

    fn keyAt(d: *const Data, r: usize, j: usize) usize {
        return if (d.sparse) @intCast(d.ids[r * d.keys + j]) else j;
    }

    fn plantPartsIdentity(d: *Data) void {
        // Equal finite maxima and ties at heads 0/2/3; increasing finite maxima at head 1.
        for (0..H) |h| @memset(d.q[h * D ..][0..D], 0);
        d.q[D] = c.toBf16(1);
        for (0..d.keys) |j| {
            const key = d.keyAt(0, j);
            for (0..KVH) |kv| {
                const at = d.cacheAt(kv, key);
                @memset(d.key[at..][0..D], 0);
                d.key[at] = c.toBf16(@floatFromInt(j));
                @memset(d.val[at..][0..D], 0);
                // Signed zero, both BF16-subnormal signs, and a stable-score mask of empty parts.
                d.val[at] = if (j % 2 == 0) 0x0000 else 0x8000;
                d.val[at + 1] = if (j % 2 == 0) 0x0001 else 0x8001;
                d.val[at + 2] = if (j % 3 == 0) 0x0001 else 0x8001;
            }
        }
    }

    fn plantMergeIdentity(d: *Data) void {
        for (0..d.rows) |r| for (0..H) |h| {
            for (0..P) |part| {
                const at = (r * H + h) * P + part;
                // Signed-zero/tied maxima, changed maxima, and empty-part masks.
                const m: f32 = switch (part) {
                    0 => 0.0,
                    1 => -0.0,
                    2 => -1.0,
                    3 => 1.0,
                    4 => 1.0,
                    5 => 2.0,
                    6 => 2.0,
                    else => -std.math.inf(f32),
                };
                d.pm[2 * at] = if (h % 4 == 3 and part > 0) -std.math.inf(f32) else m;
                const mass: f32 = switch (part % 4) {
                    0 => 0.0,
                    1 => -0.0,
                    2 => @bitCast(@as(u32, 1)),
                    else => @bitCast(@as(u32, 0x007fffff)),
                };
                d.pm[2 * at + 1] = if (d.pm[2 * at] == -std.math.inf(f32)) 0 else mass;
                for (0..D) |e| {
                    const v: f32 = switch ((part + e) % 5) {
                        0 => 0.0,
                        1 => -0.0,
                        2 => @bitCast(@as(u32, 1)),
                        3 => @bitCast(@as(u32, 0x007fffff)),
                        else => @bitCast(@as(u32, 0x00800000)),
                    };
                    d.po[at * D + e] = if (d.pm[2 * at] == -std.math.inf(f32)) 0 else v;
                }
            }
            for (0..D) |e| {
                const at = r * PW + h * 2 * D + D + e;
                d.gp[at] = switch (e % 6) {
                    0 => 0x0000,
                    1 => 0x8000,
                    2 => 0x0001,
                    3 => 0x8001,
                    4 => 0x7f7f,
                    else => 0xff7f,
                };
            }
        };
        // All parts masked on one head: top and total exercise the all-empty merge path.
        if (d.rows != 0) for (0..P) |part| {
            const at = 4 * P + part;
            d.pm[2 * at] = -std.math.inf(f32);
            d.pm[2 * at + 1] = 0;
            @memset(d.po[at * D ..][0..D], 0);
        };
    }
};

fn partOf(j: usize, n: usize) usize {
    var part: usize = 0;
    while (part + 1 < P and j >= (part + 1) * n / P) : (part += 1) {}
    return part;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    kind: Kind,
    control: bool,
    arena: ?std.heap.ArenaAllocator = null,
    data: Data = undefined,
    input: [6]rig_.Ref = undefined,
    output: [2]rig_.Ref = undefined,
    out_bytes: usize = 0,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const text = try source(a, kind, control);
        defer a.free(text);
        const pipe = try rig_.pipeline(device, text, if (kind == .merge) merge_name else parts_name);
        if (pipe.maxThreads() < 256 or pipe.simdWidth() != 32) return error.FlashnextAttentionThreads;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .kind = kind, .control = control };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        if (p.arena) |*arena| arena.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (p.control) .shapes else null, .adversarial = !p.control, .spec = spec(p.kind, p.control) };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 128 or shape.n < 1 or shape.n > 2 or shape.k == 0 or shape.k > 2048) return false;
        p.rig.reset();
        if (p.arena) |*arena| arena.deinit();
        p.arena = .init(p.a);
        p.data = try Data.init(p.arena.?.allocator(), p.kind, shape, fill, seed);
        const d = &p.data;
        if (p.kind == .merge) {
            for ([_][]const u8{ std.mem.sliceAsBytes(d.po), std.mem.sliceAsBytes(d.pm), std.mem.sliceAsBytes(d.gp) }, 0..) |bytes, i| p.input[i] = try p.rig.input(bytes);
            p.out_bytes = d.rows * H * D * 2;
            p.output[0] = try p.rig.output(p.out_bytes);
        } else {
            for ([_][]const u8{ std.mem.sliceAsBytes(d.q), std.mem.sliceAsBytes(d.key), std.mem.sliceAsBytes(d.val), std.mem.sliceAsBytes(d.ids), std.mem.sliceAsBytes(d.nk), std.mem.sliceAsBytes(d.spr) }, 0..) |bytes, i| p.input[i] = try p.rig.input(bytes);
            p.output[0] = try p.rig.output(d.rows * H * P * D * 4);
            p.output[1] = try p.rig.output(d.rows * H * P * 2 * 4);
        }
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const d = &p.data;
        e.setPipeline(p.pipe);
        if (p.kind == .merge) {
            for (p.input[0..3], 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
            e.setBuffer(p.output[0].buf, p.output[0].off, 3);
            e.dispatchGroups(mtl.Size.of(1, H, d.rows), mtl.Size.of(256, 1, 1));
        } else {
            for (p.input, [_]usize{ 0, 1, 3, 4, 6, 7 }) |ref, i| e.setBuffer(ref.buf, ref.off, i);
            const ks = [_]i32{ 1, KVH, @intCast(d.cap), D };
            const ids = [_]i32{ @intCast(d.rows), @intCast(d.keys) };
            e.setBytes(std.mem.sliceAsBytes(&ks), 2);
            e.setBytes(std.mem.sliceAsBytes(&ids), 5);
            e.setBytes(std.mem.asBytes(&scale), 8);
            for (p.output, 9..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
            const heads_per_group: usize = if (p.kind == .parts_two_heads) attn2.heads else 1;
            // attn2.call halves exactly this recorded grid dimension.
            e.dispatchGroups(mtl.Size.of(H / heads_per_group, d.rows, P), mtl.Size.of(256, 1, 1));
        }
        e.barrier();
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return if (p.kind == .merge) p.data.rows * H * D else p.data.rows * H * P * (D + 2);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (p.kind == .merge) return p.referenceMerge(want, mag);
        try p.referenceParts(want, mag);
    }

    fn referenceParts(p: *Probe, want: []f64, mag: []f64) !void {
        const d = &p.data;
        const scores = try p.a.alloc(f64, d.keys);
        defer p.a.free(scores);
        const po_count = d.rows * H * P * D;
        for (0..d.rows) |r| for (0..H) |h| for (0..P) |part| {
            const at = (r * H + h) * P + part;
            const lo = part * d.keys / P;
            const hi = (part + 1) * d.keys / P;
            const kv = h / (H / KVH);
            var top = -std.math.inf(f64);
            for (lo..hi, 0..) |j, n| {
                var score: f64 = 0;
                for (0..D) |e| {
                    const q: f64 = c.fromBf16(d.q[(r * H + h) * D + e]);
                    const k: f64 = c.fromBf16(d.key[d.cacheAt(kv, d.keyAt(r, j)) + e]);
                    score += q * k;
                }
                scores[n] = score * scale;
                top = @max(top, scores[n]);
            }
            var mass: f64 = 0;
            var score_mag: f64 = 0;
            for (scores[0 .. hi - lo]) |score| {
                mass += @exp(score - top);
                score_mag = @max(score_mag, @abs(score));
            }
            for (0..D) |e| {
                var sum: f64 = 0;
                var terms: f64 = 0;
                for (lo..hi, 0..) |j, n| {
                    const weight = @exp(scores[n] - top);
                    const v: f64 = c.fromBf16(d.val[d.cacheAt(kv, d.keyAt(r, j)) + e]);
                    sum += weight * v;
                    terms += weight * @abs(v);
                }
                want[at * D + e] = sum;
                mag[at * D + e] = terms;
            }
            want[po_count + 2 * at] = top;
            mag[po_count + 2 * at] = score_mag + 1;
            want[po_count + 2 * at + 1] = mass;
            mag[po_count + 2 * at + 1] = mass;
        };
    }

    fn referenceMerge(p: *Probe, want: []f64, mag: []f64) !void {
        const d = &p.data;
        var weights: [P]f64 = undefined;
        for (0..d.rows) |r| for (0..H) |h| {
            const at = (r * H + h) * P;
            var top = -std.math.inf(f64);
            for (0..P) |part| top = @max(top, d.pm[2 * (at + part)]);
            var total: f64 = 0;
            for (0..P) |part| {
                const mk = d.pm[2 * (at + part)];
                weights[part] = if (mk == -std.math.inf(f32)) 0 else @exp(mk - top);
                total += d.pm[2 * (at + part) + 1] * weights[part];
            }
            for (0..D) |e| {
                var acc: f64 = 0;
                var terms: f64 = 0;
                for (0..P) |part| {
                    const v = d.po[(at + part) * D + e];
                    acc += v * weights[part];
                    terms += @abs(v) * weights[part];
                }
                const i = (r * H + h) * D + e;
                const gate: f64 = c.fromBf16(d.gp[r * PW + h * 2 * D + D + e]);
                want[i] = bf(bf(acc / total) * bf(1 / (1 + @exp(-gate))));
                mag[i] = terms / @abs(total);
            }
        };
    }

    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        const s = spec(p.kind, p.control);
        for (want, mag, allow) |w, m, *limit| {
            limit.* = if (p.data.stress) c.boundExact(s, w) else m * 0x1p-7 + @abs(w) * 0x1p-8 + 1e-6;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (p.kind == .merge) return p.simulateMerge(acc, order, out);
        try p.simulateParts(acc, order, out);
    }

    fn simulateParts(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const d = &p.data;
        const terms = try p.a.alloc(f64, d.keys);
        defer p.a.free(terms);
        const po_count = d.rows * H * P * D;
        for (0..d.rows) |r| for (0..H) |h| for (0..P) |part| {
            const at = (r * H + h) * P + part;
            const lo = part * d.keys / P;
            const hi = (part + 1) * d.keys / P;
            const kv = h / (H / KVH);
            const ones = terms[0 .. hi - lo];
            @memset(ones, 1);
            out[po_count + 2 * at] = if (hi == lo) -std.math.inf(f64) else 0;
            out[po_count + 2 * at + 1] = ct.rows.sumIn(acc, order, ones);
            for (0..D) |e| {
                for (lo..hi, 0..) |j, n| terms[n] = c.fromBf16(d.val[d.cacheAt(kv, d.keyAt(r, j)) + e]);
                out[at * D + e] = ct.rows.sumIn(acc, order, terms[0 .. hi - lo]);
            }
        };
    }

    fn simulateMerge(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) void {
        const d = &p.data;
        var terms: [P]f64 = undefined;
        for (0..d.rows) |r| for (0..H) |h| for (0..D) |e| {
            const at = (r * H + h) * P;
            for (0..P) |part| terms[part] = d.po[(at + part) * D + e];
            const numerator = ct.rows.sumIn(acc, order, &terms);
            for (0..P) |part| terms[part] = d.pm[2 * (at + part) + 1];
            const total = ct.rows.sumIn(acc, order, &terms);
            out[(r * H + h) * D + e] = bf(numerator / total);
        };
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [6]Probe = undefined,
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
