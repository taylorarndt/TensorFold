//! Keep kernel contract imports and embedded Metal sources inside the same zig package root.
pub const main = @import("tests/contract/metal_main.zig").main;
