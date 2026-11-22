//! Host bit references for finite stable-max guards and the Metal source inventory they cover.
const std = @import("std");

const Sample = struct {
    old_max: f32,
    new_max: f32,
    score: f32,
    old_sum: f32,
    old_out: f32,
    value: f32,
};
const State = struct { sum: f32, out: f32 };

fn sameMax(old_max: f32, new_max: f32) bool {
    return new_max == old_max and std.math.isFinite(old_max);
}

fn legacy(s: Sample) State {
    const f = @exp(s.old_max - s.new_max);
    const e = @exp(s.score - s.new_max);
    return .{ .sum = @mulAdd(f32, s.old_sum, f, e), .out = @mulAdd(f32, e, s.value, s.old_out * f) };
}

fn serial(s: Sample) State {
    const same = sameMax(s.old_max, s.new_max);
    const f = if (same) 1.0 else @exp(s.old_max - s.new_max);
    const e = @exp(s.score - s.new_max);
    const sum_fast = s.old_sum + e;
    return .{
        .sum = if (same and std.math.isFinite(sum_fast)) sum_fast else @mulAdd(f32, s.old_sum, f, e),
        .out = @mulAdd(f32, e, s.value, if (same and std.math.isFinite(s.old_out)) s.old_out else s.old_out * f),
    };
}

fn legacyOldFirst(s: Sample) State {
    const f = @exp(s.old_max - s.new_max);
    const e = @exp(s.score - s.new_max);
    return .{ .sum = @mulAdd(f32, s.old_sum, f, e), .out = @mulAdd(f32, s.old_out, f, e * s.value) };
}

fn oldFirst(s: Sample) State {
    const same = sameMax(s.old_max, s.new_max);
    const f = if (same) 1.0 else @exp(s.old_max - s.new_max);
    const e = @exp(s.score - s.new_max);
    const sum_fast = s.old_sum + e;
    const out_fast = s.old_out + e * s.value;
    return .{
        .sum = if (same and std.math.isFinite(sum_fast)) sum_fast else @mulAdd(f32, s.old_sum, f, e),
        .out = if (same and std.math.isFinite(out_fast)) out_fast else @mulAdd(f32, s.old_out, f, e * s.value),
    };
}

fn legacyScale(s: Sample) f32 {
    return s.old_out * @exp(s.old_max - s.new_max);
}

fn scaled(s: Sample) f32 {
    return if (sameMax(s.old_max, s.new_max) and std.math.isFinite(s.old_out)) s.old_out else s.old_out * @exp(s.old_max - s.new_max);
}

fn legacyMerge(part: f32, part_max: f32, top: f32, acc: f32) f32 {
    return @mulAdd(f32, part, @exp(part_max - top), acc);
}

fn merged(part: f32, part_max: f32, top: f32, acc: f32) f32 {
    const fast = part + acc;
    return if (sameMax(part_max, top) and std.math.isFinite(fast)) fast else @mulAdd(f32, part, @exp(part_max - top), acc);
}

fn plantedControl(s: Sample) State {
    const same = s.new_max == s.old_max;
    const f = if (same) 1.0 else @exp(s.old_max - s.new_max);
    const e = @exp(s.score - s.new_max);
    return .{
        .sum = if (same) s.old_sum + e else @mulAdd(f32, s.old_sum, f, e),
        .out = @mulAdd(f32, e, s.value, if (same) s.old_out else s.old_out * f),
    };
}

fn bits(x: f32) u32 {
    return @bitCast(x);
}

fn expectBits(want: f32, got: f32) !void {
    try std.testing.expectEqual(bits(want), bits(got));
}

fn expectState(want: State, got: State) !void {
    try expectBits(want.sum, got.sum);
    try expectBits(want.out, got.out);
}

fn expectSample(s: Sample) !void {
    try expectState(legacy(s), serial(s));
    try expectState(legacyOldFirst(s), oldFirst(s));
    try expectBits(legacyScale(s), scaled(s));
}

const stable_maxima = [_]f32{
    0.0,      -0.0,      1.25,     -7.5,    0x1p-149, -0x1p-149,
    0x1p-126, -0x1p-126, 0x1p-127, 65504.0,
};
const states = [_]f32{
    0.0,      -0.0,              1.0,                -3.25,                                    0x1p-149, -0x1p-149,
    0x1p-126, std.math.inf(f32), -std.math.inf(f32), @as(f32, @bitCast(@as(u32, 0x7fc01234))),
};
const scores = [_]f32{ 0.0, -std.math.inf(f32), 0x1p-149, -0.0, 2.0 };

test "stable finite maxima preserve serial, prompt, NAX and separate-scale bits" {
    for (stable_maxima) |m| for (states) |old_sum| for (states) |old_out| for (scores) |score| {
        try expectSample(.{ .old_max = m, .new_max = m, .score = score, .old_sum = old_sum, .old_out = old_out, .value = 0x1p-149 });
        try expectSample(.{ .old_max = m, .new_max = m, .score = score, .old_sum = old_sum, .old_out = old_out, .value = -2.0 });
    };
}

test "signed zero equality and subnormal old state are exact" {
    const zeros = [_][2]f32{ .{ 0.0, -0.0 }, .{ -0.0, 0.0 } };
    for (zeros) |pair| for (states) |old_sum| for (states) |old_out| {
        try expectSample(.{ .old_max = pair[0], .new_max = pair[1], .score = -std.math.inf(f32), .old_sum = old_sum, .old_out = old_out, .value = 0x1p-149 });
        try expectBits(legacyMerge(0x1p-149, pair[0], pair[1], -0.0), merged(0x1p-149, pair[0], pair[1], -0.0));
    };
}

test "NaN and infinities never take the identity guard" {
    const nan = @as(f32, @bitCast(@as(u32, 0x7fc055aa)));
    const inf = std.math.inf(f32);
    const cases = [_][2]f32{
        .{ nan, nan },  .{ inf, inf }, .{ -inf, -inf },
        .{ -inf, 0.0 }, .{ 0.0, inf }, .{ nan, 1.0 },
        .{ 1.0, nan },
    };
    for (cases) |pair| for (states) |old_sum| for (states) |old_out| for (scores) |score| {
        try expectSample(.{ .old_max = pair[0], .new_max = pair[1], .score = score, .old_sum = old_sum, .old_out = old_out, .value = -2.0 });
    };
}

test "split and merge stable identities preserve FMA operand order" {
    const vals = [_]f32{ 0.0, -0.0, 0x1p-149, -0x1p-149, 1.0, -3.0, @as(f32, @bitCast(@as(u32, 0x7fc01234))) };
    for (stable_maxima) |m| for (vals) |part| for (vals) |acc| {
        try expectBits(legacyMerge(part, m, m, acc), merged(part, m, m, acc));
    };
    try expectBits(legacyMerge(2.0, 0.0, 3.0, 5.0), merged(2.0, 0.0, 3.0, 5.0));
}

test "row and tree masks retain old state and mask tile exponentials only" {
    const masks = [_]bool{ false, true, false, true, true, false, true, true };
    for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |rows| for ([_]usize{ 1, 4, 8, 32, 256 }) |width| {
        var maximum: f32 = 1.0;
        var state = State{ .sum = 0x1p-149, .out = -0.0 };
        for (0..rows) |row| for (0..width) |col| {
            const live = masks[(row * width + col) % masks.len];
            if (!live) continue;
            const score: f32 = if ((row + col) % 3 == 0) -std.math.inf(f32) else 0.0;
            const next = if (maximum < score) score else maximum;
            const sample = Sample{ .old_max = maximum, .new_max = next, .score = score, .old_sum = state.sum, .old_out = state.out, .value = 0x1p-149 };
            const want = legacy(sample);
            const got = serial(sample);
            try expectState(want, got);
            state = got;
            maximum = next;
        };
    };
}

test "planted missing-finite control is detected and never a default failure" {
    const ninf = -std.math.inf(f32);
    const sample = Sample{ .old_max = ninf, .new_max = ninf, .score = 0.0, .old_sum = 3.0, .old_out = 5.0, .value = 1.0 };
    const want = legacy(sample);
    const bad = plantedControl(sample);
    try std.testing.expect(bits(want.sum) != bits(bad.sum) or bits(want.out) != bits(bad.out));
}

const sources = [_][]const u8{
    @embedFile("../../kernels/metal/core/prompt_attention_nax.metal"),
    @embedFile("../../kernels/metal/core/shared_attention.metal"),
    @embedFile("../../kernels/metal/deepseek_v41/csa2_attn.metal"),
    @embedFile("../../kernels/metal/deepseek_v41/vision.metal"),
    @embedFile("../../kernels/metal/drama3/attn.metal"),
    @embedFile("../../kernels/metal/drama3/codec.metal"),
    @embedFile("../../kernels/metal/flashnext/q4_attn_merge_gate_7070840cd3c22e98.metal"),
    @embedFile("../../kernels/metal/flashnext/q4_attn_parts_f6b4ffe8e2609998.metal"),
    @embedFile("../../kernels/metal/flint/attention_nax.metal"),
    @embedFile("../../kernels/metal/flint/glue.metal"),
    @embedFile("../../kernels/metal/gemma4/glue.metal"),
    @embedFile("../../kernels/metal/gemma4/prompt_attention.metal"),
    @embedFile("../../kernels/metal/glm/sparse_attention.metal"),
    @embedFile("../../kernels/metal/glm_draft_attention.metal"),
    @embedFile("../../kernels/metal/glm_dsa/mla.metal"),
    @embedFile("../../kernels/metal/glm_sparse_nax.metal"),
    @embedFile("../../kernels/metal/h3/h3_attention.metal"),
    @embedFile("../../kernels/metal/hy4_flash/attention_sink.metal"),
    @embedFile("../../kernels/metal/hy4_flash/mla_attention.metal"),
    @embedFile("../../kernels/metal/hy4_flash/mla_nax.metal"),
    @embedFile("../../kernels/metal/idol_attention.metal"),
    @embedFile("../../kernels/metal/idol_decoder.metal"),
    @embedFile("../../kernels/metal/idol_dino.metal"),
    @embedFile("../../kernels/metal/idol_vae.metal"),
    @embedFile("../../kernels/metal/kimi/mla.metal"),
    @embedFile("../../kernels/metal/lfm2/lfm2.metal"),
    @embedFile("../../kernels/metal/lfm2/lfm2_attn_nax.metal"),
    @embedFile("../../kernels/metal/lfm3/attention.metal"),
    @embedFile("../../kernels/metal/llama/glue.metal"),
    @embedFile("../../kernels/metal/minimax_m3/attention_nax.metal"),
    @embedFile("../../kernels/metal/minimax_m3/dspark.metal"),
    @embedFile("../../kernels/metal/minimax_m3/glue.metal"),
    @embedFile("../../kernels/metal/moss_image_decoder.metal"),
    @embedFile("../../kernels/metal/moss_image_encoder.metal"),
    @embedFile("../../kernels/metal/nemotron/attn_merge.metal"),
    @embedFile("../../kernels/metal/nemotron/attn_partial.metal"),
    @embedFile("../../kernels/metal/nemotron_tree.metal"),
    @embedFile("../../kernels/metal/ops/sdpa.metal"),
    @embedFile("../../kernels/metal/prefill/attention_nax.metal"),
    @embedFile("../../kernels/metal/prefill/fn_attn.metal"),
    @embedFile("../../kernels/metal/prefill/qmm6_nax_b.metal"),
    @embedFile("../../kernels/metal/qwen27/attention_mpp.metal"),
    @embedFile("../../kernels/metal/qwen27/attention_prompt.metal"),
    @embedFile("../../kernels/metal/qwen27/attention_prompt_sg.metal"),
    @embedFile("../../kernels/metal/qwen3_5/attn_merge.metal"),
    @embedFile("../../kernels/metal/qwen3_5/attn_partial.metal"),
    @embedFile("../../kernels/metal/qwen3_5/paged.metal"),
    @embedFile("../../kernels/metal/spark25/glue.metal"),
    @embedFile("../../kernels/metal/step5/attend_nax.metal"),
    @embedFile("../../kernels/metal/step5/sparse.metal"),
    @embedFile("../../kernels/metal/uocr/vision.metal"),
    @embedFile("../../src/families/flashnext/attn2.zig"),
};

test "inventoried Metal identity paths have default-on compile-time controls" {
    const flag = "TF_METAL_SOFTMAX_IDENTITY_GUARD";
    const default = "#ifndef " ++ flag ++ "\n#define " ++ flag ++ " 1\n#endif\n";
    for (sources, 0..) |source, i| {
        try std.testing.expect(std.mem.indexOf(u8, source, flag ++ " &&") != null);
        if (i + 1 == sources.len) {
            // The generated FlashNext shader inherits this default from its base q4_attn_parts source.
            try std.testing.expect(std.mem.indexOf(u8, source, "pub const edits") != null);
        } else {
            try std.testing.expect(std.mem.startsWith(u8, source, default));
        }
    }
}
