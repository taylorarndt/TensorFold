//! Contract A/B launches: raw old/new bytes over every served shape, independent of gate tolerances.
const std = @import("std");
const contract = @import("core").contract;
const dump = @import("byte_dump.zig");
const identity = @import("identity_dump.zig");
const adversarial = @import("identity_cases.zig");

pub fn run(gpu: anytype, kernels: []const contract.gates.Kernel) !void {
    if (identity.active()) return raw(gpu, kernels);
    if (!dump.active()) return;
    const a = gpu.gpa;
    for (kernels) |kernel| {
        if (kernel.control != null) continue;
        for (kernel.spec.served, 0..) |shape, form| {
            if (!try kernel.vt.prepare(kernel.ctx, shape, .random, 0x6b63)) return error.DumpShapeRefused;
            try kernel.vt.run(kernel.ctx);
            if (!try kernel.vt.intact(kernel.ctx)) return error.DumpGuard;
            if (kernel.vt.overlap(kernel.ctx) and !kernel.spec.in_place) return error.DumpOverlap;
            const bytes = try kernel.vt.output(kernel.ctx);
            const count = if (kernel.spec.out == .raw) bytes.len else kernel.vt.count(kernel.ctx);
            const dtype: dump.DType = switch (kernel.spec.out) {
                .bf16 => .bf16,
                .f16 => .f16,
                .f32 => .f32,
                .raw => .raw,
            };
            try dump.output(a, "kernel-contract", kernel.spec.name, "outputs", &.{count}, dtype, .{ .form = form, .served = shape, .seed = @as(u64, 0x6b63) }, bytes);
        }
    }
}

/// Fixtures cover random, non-finite, stress and finite A/B; distinctly keyed controls must stay red.
fn raw(gpu: anytype, kernels: []const contract.gates.Kernel) !void {
    const a = gpu.gpa;
    const Case = struct { name: []const u8, fill: contract.gates.Fill };
    const cases = [_]Case{ .{ .name = "random", .fill = .random }, .{ .name = "nonfinite", .fill = .nonfinite }, .{ .name = "stress", .fill = .stress }, .{ .name = "identity", .fill = .random } };
    for (kernels, 0..) |kernel, ki| {
        for (kernel.spec.served, 0..) |shape, form| {
            for (cases) |case| {
                if (kernel.control != null and case.fill != .random) continue;
                if (std.mem.eql(u8, case.name, "identity") and !adversarial.supported(kernel)) continue;
                const key = try std.fmt.allocPrint(a, "kernel-{d:0>4}/shape-{d:0>3}/{s}", .{ ki, form, case.name });
                defer a.free(key);
                const seed = if (std.mem.eql(u8, case.name, "identity")) adversarial.seed else std.hash.Wyhash.hash(0x6b63, key);
                if (!try kernel.vt.prepare(kernel.ctx, shape, case.fill, seed)) return error.IdentityDumpShapeRefused;
                try kernel.vt.run(kernel.ctx);
                if (!try kernel.vt.intact(kernel.ctx)) return error.IdentityDumpGuard;
                if (kernel.vt.overlap(kernel.ctx) and !kernel.spec.in_place) return error.IdentityDumpOverlap;
                const bytes = try kernel.vt.output(kernel.ctx);
                if (bytes.len != try std.math.mul(usize, kernel.vt.count(kernel.ctx), contract.spec.size(kernel.spec.out))) return error.IdentityDumpLength;
                const control: ?[]const u8 = if (kernel.control) |gate| @tagName(gate) else null;
                try identity.output(gpu, a, key, .{
                    .kernel_index = ki,
                    .kernel = kernel.spec.name,
                    .control = control,
                    .served_index = form,
                    .served = shape,
                    .fill = case.name,
                    .seed = seed,
                    .output_elements = kernel.vt.count(kernel.ctx),
                    .output_type = @tagName(kernel.spec.out),
                }, bytes);
                if (kernel.spec.replaces) |what| {
                    const baseline = kernel.vt.replaced orelse return error.IdentityDumpNoReplacement;
                    const replaced = try baseline(kernel.ctx);
                    if (replaced.len != bytes.len) return error.IdentityDumpReplacementLength;
                    const replaced_key = try std.fmt.allocPrint(a, "{s}/replacement", .{key});
                    defer a.free(replaced_key);
                    try identity.output(gpu, a, replaced_key, .{
                        .kernel_index = ki,
                        .kernel = kernel.spec.name,
                        .control = control,
                        .served_index = form,
                        .served = shape,
                        .fill = case.name,
                        .seed = seed,
                        .output_elements = kernel.vt.count(kernel.ctx),
                        .output_type = @tagName(kernel.spec.out),
                        .role = "replacement",
                        .replacement = what,
                    }, replaced);
                }
            }
        }
    }
}
