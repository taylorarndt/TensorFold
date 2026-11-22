//! KP10's direct raw-output identity suites under one owner.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const g = ct.gates;
const Flash = @import("metal_flashnext_q4_attention_identity.zig");
const Hy4 = @import("metal_hy4_sink_identity.zig");
const Sdpa = @import("metal_sdpa_identity.zig");

pub const Suite = struct {
    a: std.mem.Allocator,
    flash: *Flash.Suite,
    hy4: *Hy4.Suite,
    sdpa: *Sdpa.Suite,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const flash = try Flash.Suite.create(a, device, queue, out, only);
        errdefer flash.destroy();
        const hy4 = try Hy4.Suite.create(a, device, queue, out, only);
        errdefer hy4.destroy();
        const sdpa = try Sdpa.Suite.create(a, device, queue, out, only);
        errdefer sdpa.destroy();
        const self = try a.create(Suite);
        self.* = .{ .a = a, .flash = flash, .hy4 = hy4, .sdpa = sdpa };
        return self;
    }

    pub fn destroy(self: *Suite) void {
        self.sdpa.destroy();
        self.hy4.destroy();
        self.flash.destroy();
        self.a.destroy(self);
    }
};
