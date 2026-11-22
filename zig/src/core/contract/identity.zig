//! Backend-neutral old/new identity fixtures: a reserved seed and an explicit fixture capability.
const gates = @import("gates.zig");

/// Tests use `.random` fill, then plant backend-specific stable/changed-max, tie, signed-zero and subnormal cases.
pub const seed: u64 = 0x9e7a_4f31_c58d_20b6;

pub fn requested(actual: u64) bool {
    return actual == seed;
}

/// Only a fixture that implements the reserved plant may advertise `Kernel.adversarial`.
pub fn supported(kernel: gates.Kernel) bool {
    return kernel.adversarial;
}
