//! DeepSeek replay's test-only raw old/new receipts, enabled without CUDA-runner build options.
const std = @import("std");

pub fn directory() ?[]const u8 {
    const value = std.c.getenv("TF_CUDA_IDENTITY_DUMP") orelse return null;
    const path = std.mem.span(value);
    return if (path.len == 0) null else path;
}

pub fn active() bool {
    return directory() != null;
}

/// A key-hashed filename remains unique across separate `tf-dsv41-test replay` processes.
pub fn output(gpu: anytype, a: std.mem.Allocator, key: []const u8, meta: anytype, bytes: []const u8) !void {
    const root = directory() orelse return;
    if (@import("builtin").cpu.arch.endian() != .little) return error.IdentityDumpEndian;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(key, &digest, .{});
    const name = std.fmt.bytesToHex(digest, .lower);
    const relative = try std.fmt.allocPrint(a, "dsv41-{s}.bin", .{name});
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
}
