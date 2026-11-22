//! Run the identity arithmetic/source suite with zig/ as its package root.
const std = @import("std");
const suite = @import("tests/contract/metal_softmax_identity.zig");

test {
    std.testing.refAllDecls(suite);
}
