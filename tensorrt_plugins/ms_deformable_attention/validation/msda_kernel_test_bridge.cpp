#include "ms_deformable_attention_kernel.h"

extern "C" int msda_forward_fp32(cudaStream_t stream, float const* value,
    int32_t const* shapes, int32_t const* starts, float const* locations,
    float const* weights, int32_t batch, int32_t spatialSize,
    int32_t heads, int32_t channels, int32_t levels, int32_t queries,
    int32_t points, float* output)
{
    return launchMSDeformableAttentionFP32(stream, value, shapes, starts,
        locations, weights, batch, spatialSize, heads, channels, levels,
        queries, points, output);
}

extern "C" int msda_forward_fp16(cudaStream_t stream, __half const* value,
    int32_t const* shapes, int32_t const* starts, __half const* locations,
    __half const* weights, int32_t batch, int32_t spatialSize,
    int32_t heads, int32_t channels, int32_t levels, int32_t queries,
    int32_t points, __half* output)
{
    return launchMSDeformableAttentionFP16(stream, value, shapes, starts,
        locations, weights, batch, spatialSize, heads, channels, levels,
        queries, points, output);
}
