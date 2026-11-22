//! CUDA's import of the backend-neutral identity fixture convention.
const identity = @import("core").contract.identity;
pub const seed = identity.seed;
pub const requested = identity.requested;
pub const supported = identity.supported;
