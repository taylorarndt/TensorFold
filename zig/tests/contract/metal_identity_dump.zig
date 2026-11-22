//! Raw old/new Metal output-byte receipts; schema and output(gpu, a, key, meta, bytes) match CUDA identity_dump.
const std = @import("std");
const contract = @import("contract");

var root: ?[]const u8 = null;
var records: usize = 0;

pub fn directory() ?[]const u8 {
    return root;
}

pub fn active() bool {
    return root != null;
}

/// One payload and one stable receipt; no guards, conversions or candidate fallback.
pub fn output(gpu: anytype, a: std.mem.Allocator, key: []const u8, meta: anytype, bytes: []const u8) !void {
    const dir = root orelse return;
    if (@import("builtin").cpu.arch.endian() != .little) return error.IdentityDumpEndian;
    const relative = try std.fmt.allocPrint(a, "{d:0>8}.bin", .{records});
    defer a.free(relative);
    const path = try std.fs.path.join(a, &.{ dir, relative });
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = bytes });
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    const hex = std.fmt.bytesToHex(hash, .lower);
    const line = try std.json.Stringify.valueAlloc(a, .{
        .version = 1,
        .key = key,
        .file = relative,
        .bytes = bytes.len,
        .sha256 = @as([]const u8, &hex),
        .meta = meta,
    }, .{});
    defer a.free(line);
    std.debug.print("IDENTITY {s}\n", .{line});
    records += 1;
}

/// Every shape/input class is prepared independently; gates still require each planted control's red verdict.
pub fn run(gpu: anytype, a: std.mem.Allocator, kernels: []const contract.gates.Kernel, dump_root: ?[]const u8, gates: bool) !void {
    if (dump_root != null and !gates) return error.DumpRequiresGates;
    root = dump_root;
    records = 0;
    if (root == null) return;
    const Case = struct { name: []const u8, fill: contract.gates.Fill };
    const cases = [_]Case{
        .{ .name = "random", .fill = .random },
        .{ .name = "nonfinite", .fill = .nonfinite },
        .{ .name = "stress", .fill = .stress },
        .{ .name = "identity", .fill = .random },
    };
    for (kernels, 0..) |kernel, ki| {
        for (kernel.spec.served, 0..) |shape, form| {
            for (cases) |case| {
                if (kernel.control != null and case.fill != .random) continue;
                if (std.mem.eql(u8, case.name, "identity") and !contract.identity.supported(kernel)) continue;
                const key = try std.fmt.allocPrint(a, "kernel-{d:0>4}/shape-{d:0>3}/{s}", .{ ki, form, case.name });
                defer a.free(key);
                const seed = if (std.mem.eql(u8, case.name, "identity")) contract.identity.seed else std.hash.Wyhash.hash(0x6b63, key);
                if (!try kernel.vt.prepare(kernel.ctx, shape, case.fill, seed)) return error.IdentityDumpShapeRefused;
                try kernel.vt.run(kernel.ctx);
                if (!try kernel.vt.intact(kernel.ctx)) return error.IdentityDumpGuard;
                if (kernel.vt.overlap(kernel.ctx) and !kernel.spec.in_place) return error.IdentityDumpOverlap;
                const bytes = try kernel.vt.output(kernel.ctx);
                const elements = kernel.vt.count(kernel.ctx);
                if (bytes.len != elements * contract.size(kernel.spec.out)) return error.IdentityDumpOutputSize;
                const control: ?[]const u8 = if (kernel.control) |gate| @tagName(gate) else null;
                try output(gpu, a, key, .{
                    .kernel_index = ki,
                    .kernel = kernel.spec.name,
                    .control = control,
                    .served_index = form,
                    .served = shape,
                    .fill = case.name,
                    .seed = seed,
                    .output_elements = elements,
                    .output_type = @tagName(kernel.spec.out),
                }, bytes);
            }
        }
    }
}
