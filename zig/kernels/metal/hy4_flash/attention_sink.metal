#ifndef TF_METAL_SOFTMAX_IDENTITY_GUARD
#define TF_METAL_SOFTMAX_IDENTITY_GUARD 1
#endif
// Finish a head's online softmax with its learned, zero-value sink.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Hy4SinkArgs {
    uint rows;
    uint heads;
    uint width;
};

kernel void tf_hy4_finish_attention(
    const device float *maxima [[buffer(0)]],
    const device float *mass [[buffer(1)]],
    const device float *numerator [[buffer(2)]],
    const device float *sinks [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant Hy4SinkArgs &args [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint column = tid.x;
    const uint row_head = tid.y;
    if (column >= args.width || row_head >= ulong(args.rows) * args.heads) return;

    const float sink = sinks[row_head % args.heads];
    const float top = max(maxima[row_head], sink);
    const bool same = TF_METAL_SOFTMAX_IDENTITY_GUARD && top == maxima[row_head] && isfinite(maxima[row_head]);
    const float prior = same ? 1.0f : precise::exp(maxima[row_head] - top);
    const float denominator = ((same && isfinite(mass[row_head])) ? mass[row_head] : mass[row_head] * prior) + precise::exp(sink - top);
    const ulong offset = ulong(row_head) * args.width + column;
    output[offset] = ((same && isfinite(numerator[offset])) ? numerator[offset] : numerator[offset] * prior) / denominator;
}
