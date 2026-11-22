//! Actual split-softmax bodies face independent binary64 and four planted faults through the maintained gates.
const std = @import("std");
const ct = @import("core").contract;
const common = @import("../../src/families/qwen4/qsa_attn_contract.zig");
const a = std.testing.allocator;
extern fn tf_q4_qsa_attn_cpu(c_int, [*]const [*]const u8, [*]f32, [*]f32, [*]u16, c_int, c_int, c_uint) void;
extern fn tf_q4_qsa_attn_legacy_cpu([*]const [*]const u8, [*]f32, [*]f32, [*]u16, c_int, c_int, c_uint) void;
const Backend = struct {
    pub const name = "host actual MSL";
    pub const Rig = @import("contract_rig.zig").HostRig;
    pub const served: []const ct.spec.Shape = &.{ .{ .rows = 1, .n = 1, .k = 1024 }, .{ .rows = 1, .n = 2, .k = 17 }, .{ .rows = 2, .n = 2, .k = 33 } };
    input: [6][*]const u8 = undefined,
    po: []f32 = undefined,
    pm: []f32 = undefined,
    output: []u16 = undefined,
    control: common.Control = .production,
    legacy: bool = false,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, control: common.Control) !void {
        b.control = control;
        for ([_][]const u8{ std.mem.sliceAsBytes(f.query), std.mem.sliceAsBytes(f.key), std.mem.sliceAsBytes(f.value), std.mem.sliceAsBytes(f.ids), std.mem.sliceAsBytes(f.nk), std.mem.sliceAsBytes(f.spr) }, 0..) |bytes, i| b.input[i] = (try r.add(bytes, bytes.len)).ptr;
        b.po = @alignCast(std.mem.bytesAsSlice(f32, try r.add(null, f.rows * 24 * 16 * 256 * 4)));
        b.pm = @alignCast(std.mem.bytesAsSlice(f32, try r.add(null, f.rows * 24 * 16 * 2 * 4)));
        b.output = @alignCast(std.mem.bytesAsSlice(u16, try r.add(null, f.rows * 24 * 256 * 2)));
    }
    pub fn launch(b: *@This(), _: *Rig, f: common.Fixture, repeat: u32) !void {
        if (b.legacy) tf_q4_qsa_attn_legacy_cpu(&b.input, b.po.ptr, b.pm.ptr, b.output.ptr, @intCast(f.rows), @intCast(f.keys), repeat) else tf_q4_qsa_attn_cpu(@intCast(@backingInt(b.control)), &b.input, b.po.ptr, b.pm.ptr, b.output.ptr, @intCast(f.rows), @intCast(f.keys), repeat);
    }
};
const NativeCase = common.Case(ct, Backend);

const Pair = struct {
    before: NativeCase,
    after: NativeCase,

    fn deinit(p: *Pair) void {
        p.before.deinit();
        p.after.deinit();
    }
    fn prepare(p: *Pair, shape: ct.spec.Shape, fill: ct.gates.Fill) !void {
        for ([_]*NativeCase{ &p.before, &p.after }) |s| try std.testing.expect(try s.prepare(shape, fill, 0x91e7));
    }
    fn bytes(label: []const u8, part: []const u8, old: []const u8, new: []const u8) !void {
        if (std.mem.eql(u8, old, new)) return;
        std.debug.print("QSA_ATTN_LEGACY {s}: {s} bytes differ\n", .{ label, part });
        return error.QsaLegacyBitsDiffer;
    }
    fn run(p: *Pair, label: []const u8) !void {
        try bytes(label, "input", try p.before.rig.inputBytes(), try p.after.rig.inputBytes());
        try p.before.run();
        try p.after.run();
        if (!try p.before.rig.intact() or !try p.after.rig.intact()) return error.QsaLegacyGuardWrite;
        try bytes(label, "partial output", std.mem.sliceAsBytes(p.before.backend.po), std.mem.sliceAsBytes(p.after.backend.po));
        try bytes(label, "partial maximum/sum", std.mem.sliceAsBytes(p.before.backend.pm), std.mem.sliceAsBytes(p.after.backend.pm));
        try bytes(label, "BF16 output", std.mem.sliceAsBytes(p.before.backend.output), std.mem.sliceAsBytes(p.after.backend.output));
    }
};

const Directed = enum { equal_finite, changed_maximum, signed_zero, subnormal, nonfinite_values, positive_inf, negative_inf, quiet_nan, all_masked };

fn directed(p: *Pair, sparse: bool, mode: Directed) !void {
    try p.prepare(.{ .rows = 1, .n = if (sparse) 2 else 1, .k = 256 }, .stress);
    for ([_]*NativeCase{ &p.before, &p.after }) |s| {
        const f = &s.fixture;
        switch (mode) {
            .equal_finite => {}, // Two keys per simdgroup in every part force local, tile and split equalities.
            .all_masked => @memset(f.nk, 0),
            .signed_zero => {
                for (f.query, 0..) |*v, i| v.* = if (i % 2 == 0) 0 else 0x8000;
                for (f.key, 0..) |*v, i| v.* = if (i % 2 == 0) 0x8000 else 0;
                for (f.value, 0..) |*v, i| v.* = if (i % 2 == 0) 0 else 0x8000;
            },
            .subnormal => @memset(f.value, 0x0001),
            .nonfinite_values => for (f.value, 0..) |*v, i| switch (i % common.width) {
                0 => v.* = 0x7fd5, // Quiet NaN with a noncanonical payload.
                1 => v.* = 0x7f80,
                2 => v.* = 0xff80,
                3 => v.* = 0x7fa5, // Signaling NaN: host arithmetic may quiet or alter its payload.
                else => {},
            },
            .positive_inf, .negative_inf, .quiet_nan => {
                for (0..common.heads) |h| f.query[h * common.width] = if (mode == .quiet_nan) 0x7fd5 else 0x7f80;
                for (0..f.capacity) |j| for (0..common.kv) |h| {
                    f.key[(j * common.kv + h) * common.width] = if (mode == .negative_inf) 0xbf80 else 0x3f80;
                };
            },
            .changed_maximum => {
                for (0..common.heads) |h| f.query[h * common.width] = 0x3f80;
                for (0..f.capacity) |j| for (0..common.kv) |h| {
                    // The second key of each simdgroup and each later part has a strictly larger finite score.
                    f.key[(j * common.kv + h) * common.width] = ct.spec.toBf16(@floatFromInt(j));
                };
            },
        }
        s.rig.reset();
        try s.backend.bind(&s.rig, s.fixture, s.control);
    }
    try p.run(@tagName(mode));
    switch (mode) {
        .subnormal => {
            var found = false;
            for (p.before.backend.po) |v| {
                const bits: u32 = @bitCast(v);
                found = found or ((bits & 0x7f800000) == 0 and (bits & 0x007fffff) != 0);
            }
            try std.testing.expect(found); // Prove the old-state multiplies really saw subnormal accumulators.
        },
        .changed_maximum => {
            const pm = p.before.backend.pm;
            try std.testing.expect(pm[0] < pm[30]); // Distinct part maxima force nonidentity split correction.
        },
        else => {},
    }
}
test "Qwen4 QSA native contract: attention actual parts and merge catch nonfinite repeat tail and reduced sums" {
    const Case = common.Case(ct, Backend);
    const g = ct.gates;
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    var good: Case = .{ .a = a, .rig = .{}, .backend = .{} };
    defer good.deinit();
    for ([_]g.Gate{ .nonfinite, .repeat, .shapes, .sums }) |gate| {
        const got = try g.one(a, gate, good.kernel(), 0x367);
        if (got != .pass) {
            g.quiet = false;
            _ = try g.one(a, gate, good.kernel(), 0x367);
            g.quiet = true;
        }
        try std.testing.expectEqual(g.Verdict.pass, got);
    }
    for ([_]common.Control{ .nan, .repeat, .tail, .sums }) |control| {
        var bad: Case = .{ .a = a, .rig = .{}, .backend = .{}, .control = control };
        defer bad.deinit();
        const got = try g.one(a, bad.kernel().control.?, bad.kernel(), 0x367);
        if (got != .fail) std.debug.print("QSA_ATTN_CONTROL {t}: {t}\n", .{ control, got });
        try std.testing.expectEqual(g.Verdict.fail, got);
    }
}

test "Qwen4 QSA native contract: actual pre-KP10 and guarded parts+merge bytes match" {
    var pair: Pair = .{
        .before = .{ .a = a, .rig = .{}, .backend = .{ .legacy = true } },
        .after = .{ .a = a, .rig = .{}, .backend = .{} },
    };
    defer pair.deinit();
    for (Backend.served) |shape| for ([_]ct.gates.Fill{ .random, .stress }) |fill| {
        try pair.prepare(shape, fill);
        try pair.run(@tagName(fill));
    };
    try directed(&pair, false, .equal_finite);
    try directed(&pair, true, .equal_finite);
    // The translated signature supports SPARSE+IDS, but has no separate tree-mask argument to exercise.
    for ([_]Directed{ .changed_maximum, .signed_zero, .subnormal, .nonfinite_values, .positive_inf, .negative_inf, .quiet_nan, .all_masked }) |mode|
        try directed(&pair, false, mode);
}
