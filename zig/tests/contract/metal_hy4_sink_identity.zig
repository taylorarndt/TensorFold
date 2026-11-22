//! HY4's learned-sink attention finish on Metal: raw FP32 outputs and the stable-max identity.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;

const kernel_name = "tf_hy4_finish_attention";
const plain_name = "metal hy4 sink tf_hy4_finish_attention raw output";
const control_name = "control: metal hy4 sink tf_hy4_finish_attention raw output (sink omitted)";
const served = [_]c.Shape{
    .{ .rows = 1, .n = 1, .k = 64 },
    .{ .rows = 2, .n = 3, .k = 128 },
    .{ .rows = 4, .n = 2, .k = 512 }, // HY4's full latent width
};

fn noDepth(_: c.Shape) usize {
    return 0;
}

fn spec(control: bool) c.Spec {
    return .{
        .name = if (control) control_name else plain_name,
        .accum = .none,
        .depth = noDepth,
        .ops = 8,
        .out = .f32,
        .served = &served,
    };
}

fn source(a: std.mem.Allocator, control: bool) ![]u8 {
    if (!control) return a.dupe(u8, sources.hy4_flash_attention_sink);
    const from = "precise::exp(sink - top)";
    if (std.mem.count(u8, sources.hy4_flash_attention_sink, from) != 1) return error.Hy4SinkControlEditMissing;
    return std.mem.replaceOwned(u8, a, sources.hy4_flash_attention_sink, from, "0.0f");
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: bool,
    shape: c.Shape = undefined,
    maxima: []f32 = &.{},
    mass: []f32 = &.{},
    numerator: []f32 = &.{},
    sinks: []f32 = &.{},
    input: [4]rig_.Ref = undefined,
    output: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Probe {
        const text = try source(a, control);
        defer a.free(text);
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try rig_.pipeline(device, text, kernel_name),
            .control = control,
        };
    }

    fn free(p: *Probe) void {
        for ([_][]f32{ p.maxima, p.mass, p.numerator, p.sinks }) |x| p.a.free(x);
        p.maxima = &.{};
        p.mass = &.{};
        p.numerator = &.{};
        p.sinks = &.{};
    }

    pub fn deinit(p: *Probe) void {
        p.free();
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        return .{
            .ctx = p,
            .vt = &vt,
            .control = if (p.control) .shapes else null,
            .adversarial = !p.control,
            .spec = spec(p.control),
        };
    }

    fn plane(p: *const Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn count(p: *const Probe) usize {
        return p.plane() * p.shape.k;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        p.free();
        p.rig.reset();
        p.shape = shape;
        p.maxima = try p.a.alloc(f32, p.plane());
        p.mass = try p.a.alloc(f32, p.plane());
        p.numerator = try p.a.alloc(f32, p.count());
        p.sinks = try p.a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.maxima) |*x| x.* = @floatCast(rng.floatNorm(f64));
        for (p.mass) |*x| x.* = @floatCast(0.5 + rng.float(f64));
        for (p.numerator) |*x| x.* = @floatCast(rng.floatNorm(f64));
        for (p.sinks) |*x| x.* = @floatCast(rng.floatNorm(f64));
        switch (fill) {
            .random => {},
            .stress => {
                @memset(p.maxima, 0);
                @memset(p.mass, 1);
                @memset(p.sinks, 0);
            },
            .nonfinite => p.plantNonfinite(),
        }
        if (fill == .random and contract.identity.requested(seed)) p.plantAdversarial();
        const bytes = [_][]const u8{
            std.mem.sliceAsBytes(p.maxima),
            std.mem.sliceAsBytes(p.mass),
            std.mem.sliceAsBytes(p.numerator),
            std.mem.sliceAsBytes(p.sinks),
        };
        for (bytes, &p.input) |b, *r| r.* = try p.rig.input(b);
        p.output = try p.rig.output(p.count() * 4);
        return true;
    }

    fn plantNonfinite(p: *Probe) void {
        p.numerator[5] = std.math.nan(f32);
        p.numerator[9] = std.math.inf(f32);
        if (p.plane() > 1) @memset(p.numerator[p.shape.k..][0..p.shape.k], std.math.nan(f32));
        if (p.plane() > 2) {
            p.maxima[2] = std.math.inf(f32);
            p.mass[2] = std.math.inf(f32);
        }
        if (p.plane() > 3) p.maxima[3] = -std.math.inf(f32);
        if (p.shape.n > 1) p.sinks[1] = -std.math.inf(f32);
    }

    fn plantAdversarial(p: *Probe) void {
        for (0..p.plane()) |rh| {
            const head = rh % p.shape.n;
            const row = rh * p.shape.k;
            // Ties include opposite signed zeros; the other rows exercise both changed-max directions.
            const zero: f32 = if (rh % 2 == 0) 0.0 else -0.0;
            p.sinks[head] = zero;
            p.maxima[rh] = switch (rh % 3) {
                0 => -zero,
                1 => -3.0, // sink-dominant: the old maximum must be rescaled
                else => 3.0, // old-state dominant
            };
            p.mass[rh] = switch (rh % 4) {
                0 => -0.0,
                1 => @bitCast(@as(u32, 1)),
                2 => @bitCast(@as(u32, 0x007fffff)),
                else => 1.0,
            };
            p.numerator[row] = -0.0;
            p.numerator[row + 1] = @bitCast(@as(u32, 1));
            p.numerator[row + 2] = @bitCast(@as(u32, 0x80000001));
            if (p.shape.k > 3) p.numerator[row + 3] = @bitCast(@as(u32, 0x007fffff));
        }
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.input, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setBuffer(p.output.buf, p.output.off, 4);
        const args = extern struct { rows: u32, heads: u32, width: u32 }{
            .rows = @intCast(p.shape.rows),
            .heads = @intCast(p.shape.n),
            .width = @intCast(p.shape.k),
        };
        e.setValue(args, 5);
        e.dispatchThreads(mtl.Size.of(p.shape.k, p.plane(), 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const rh = i / p.shape.k;
            const head = rh % p.shape.n;
            const top = @max(@as(f64, p.maxima[rh]), @as(f64, p.sinks[head]));
            const prior = @exp(@as(f64, p.maxima[rh]) - top);
            const denominator = @as(f64, p.mass[rh]) * prior + @exp(@as(f64, p.sinks[head]) - top);
            const value = @as(f64, p.numerator[i]) * prior;
            w.* = value / denominator;
            m.* = @abs(value / denominator);
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [2]Probe = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for ([_]bool{ false, true }) |control| {
            const p = &s.probes[s.made];
            p.* = try Probe.init(a, device, queue, control);
            s.made += 1;
            const k = p.kernel();
            if (only) |needle| if (std.mem.indexOf(u8, k.spec.name, needle) == null) continue;
            try out.append(a, k);
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes[0..s.made]) |*p| p.deinit();
        const a = s.a;
        a.destroy(s);
    }
};
