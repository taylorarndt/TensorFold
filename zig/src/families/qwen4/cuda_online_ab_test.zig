//! CPU byte A/B compares pre-KP10-restored and guarded CUDA online updates using scalar QK/PV, not GPU execution.
const std = @import("std");
const common = @import("qsa_attn_contract.zig");
const Shape = @import("core").contract.spec.Shape;
const t = std.testing;
extern fn tf_q4_cuda_online_ab([*]const f32, [*]const u16, c_int, c_int, c_int, [*]f32, [*]f32, [*]u16, [*]u16) void;
extern fn tf_q4_cuda_served_ab([*]const u16, [*]const u16, [*]const u16, [*]const i32, [*]const i32, [*]const i32, c_int, c_int, c_int, [*]f32, [*]f32, [*]u16, [*]u16) void;
fn equalBytes(before: anytype, after: anytype) !void {
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(before), std.mem.sliceAsBytes(after));
}
fn served(f: common.Fixture, fused: bool) !void {
    const a = t.allocator;
    const nch = (f.keys + 511) / 512;
    const parts = f.rows * common.heads * nch * 258;
    const count = f.rows * common.heads * common.width;
    const old_parts = try a.alloc(f32, parts);
    defer a.free(old_parts);
    const new_parts = try a.alloc(f32, parts);
    defer a.free(new_parts);
    const old_out = try a.alloc(u16, count);
    defer a.free(old_out);
    const new_out = try a.alloc(u16, count);
    defer a.free(new_out);
    tf_q4_cuda_served_ab(f.query.ptr, f.key.ptr, f.value.ptr, f.ids.ptr, f.nk.ptr, f.spr.ptr, @intCast(f.rows), @intCast(f.keys), @intFromBool(fused), old_parts.ptr, new_parts.ptr, old_out.ptr, new_out.ptr);
    try equalBytes(old_parts, new_parts);
    try equalBytes(old_out, new_out);
}
fn directedOne(scores: []const f32, values: []const u16, chunk: usize, fused: bool) !void {
    const a = t.allocator;
    const nch = (scores.len + chunk - 1) / chunk;
    const old_parts = try a.alloc(f32, nch * 258);
    defer a.free(old_parts);
    const new_parts = try a.alloc(f32, nch * 258);
    defer a.free(new_parts);
    var old_out: [256]u16 = undefined;
    var new_out: [256]u16 = undefined;
    tf_q4_cuda_online_ab(scores.ptr, values.ptr, @intCast(scores.len), @intCast(chunk), @intFromBool(fused), old_parts.ptr, new_parts.ptr, &old_out, &new_out);
    try equalBytes(old_parts, new_parts);
    try equalBytes(&old_out, &new_out);
}
fn directed(scores: []const f32, values: []const u16, chunk: usize) !void {
    if (values.len != scores.len * common.width) return error.BadCudaOnlineFixture;
    try directedOne(scores, values, chunk, false);
    try directedOne(scores, values, chunk, true);
}
fn bits(word: u32) f32 {
    return @bitCast(word);
}
test "Qwen4 CUDA online A/B: every served shape is legacy/guarded byte-equal in split and fused paths" {
    // CUDA's four served shapes plus the three existing host attention shapes; row2x2x33 is not a duplicate.
    const shapes = [_]Shape{
        .{ .rows = 1, .n = 1, .k = 1024 },
        .{ .rows = 128, .n = 2, .k = 33 },
        .{ .rows = 1, .n = 2, .k = 513 },
        .{ .rows = 1, .n = 2, .k = 1025 },
        .{ .rows = 1, .n = 2, .k = 17 },
        .{ .rows = 2, .n = 2, .k = 33 },
    };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for (shapes, 0..) |shape, i| {
        const f = try common.Fixture.make(arena.allocator(), shape, .random, 0x91e2 + i);
        try served(f, false);
        try served(f, true);
    }
}
test "Qwen4 CUDA online A/B: finite ties, changed maxima, tile boundaries and split merges preserve every state byte" {
    var scores: [2][96]f32 = undefined;
    var values: [96 * 256]u16 = undefined;
    const pattern = [_]u16{ 0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x0000, 0x8000, 0x3f40, 0xbf40 };
    for (0..96) |i| {
        scores[0][i] = if (i % 9 == 0) -0.0 else 0.25;
        scores[1][i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 48)) * 0.03125;
        for (0..256) |d| values[i * 256 + d] = pattern[(i + d) % pattern.len];
    }
    // CH=512 repeats or raises a finite old m on the second and third tiles.
    for (scores) |row| try directed(&row, &values, 512);
    // CH=32 puts exactly the same recurrences into chunk partials and the actual final merge body.
    for (scores) |row| try directed(&row, &values, 32);
}
test "Qwen4 CUDA online A/B: signed zeros, subnormals, +/-inf, quiet NaNs and all-masked tiles compare raw bits" {
    var scores: [66]f32 = @splat(0);
    scores[0] = -0.0;
    scores[1] = 0.0;
    scores[2] = bits(0x00000001);
    scores[3] = bits(0x80000001);
    scores[32] = bits(0x00000001);
    scores[33] = bits(0x80000001);
    var values: [66 * 256]u16 = undefined;
    const classes = [_]u16{ 0x0000, 0x8000, 0x0001, 0x8001, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1 };
    for (0..scores.len) |i| {
        for (0..256) |d| {
            values[i * 256 + d] = if (i < 12 or i == 32 or i == 33) classes[(i + d) % classes.len] else 0x3f80;
        }
    }
    try directed(&scores, &values, 512);
    try directed(&scores, &values, 32);

    var masked: [33]f32 = @splat(-std.math.inf(f32));
    var masked_values: [33 * 256]u16 = @splat(0x3f80);
    try directed(&masked, &masked_values, 512);
    try directed(&masked, &masked_values, 16);

    var exceptional: [65]f32 = @splat(0);
    exceptional[4] = std.math.inf(f32);
    exceptional[37] = -std.math.inf(f32);
    exceptional[63] = bits(0x7fc00023);
    exceptional[64] = bits(0xffc00029);
    var exceptional_values: [65 * 256]u16 = @splat(0x3f80);
    exceptional_values[2] = 0x7fc1;
    exceptional_values[3] = 0x7f80;
    exceptional_values[4] = 0xff80;
    try directed(&exceptional, &exceptional_values, 512);
    try directed(&exceptional, &exceptional_values, 16);
}
