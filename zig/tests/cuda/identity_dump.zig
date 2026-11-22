//! Raw, test-only old/new contract receipts. The payload files contain output bytes only.
const std = @import("std");
const options = @import("cuda_dump_options");

var records: usize = 0;

/// The directory is selected explicitly per build; no candidate result is ever a default baseline.
pub fn directory() ?[]const u8 {
    if (comptime !options.enabled) return null;
    const value = std.c.getenv("TF_CUDA_IDENTITY_DUMP") orelse return null;
    const path = std.mem.span(value);
    return if (path.len == 0) null else path;
}

pub fn active() bool {
    return directory() != null;
}

/// Write synchronized payload/key records; scratch allocations and guard bytes never enter a receipt.
pub fn output(gpu: anytype, a: std.mem.Allocator, key: []const u8, meta: anytype, bytes: []const u8) !void {
    const root = directory() orelse return;
    if (@import("builtin").cpu.arch.endian() != .little) return error.IdentityDumpEndian;
    const relative = try std.fmt.allocPrint(a, "{d:0>8}.bin", .{records});
    defer a.free(relative);
    const path = try std.fs.path.join(a, &.{ root, relative });
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
