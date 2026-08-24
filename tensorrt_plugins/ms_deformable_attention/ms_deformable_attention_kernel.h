#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

int32_t launchMSDeformableAttentionFP32(
    cudaStream_t stream, float const* value, int32_t const* spatialShapes,
    int32_t const* levelStartIndex, float const* samplingLocations,
    float const* attentionWeights, int32_t batch, int32_t spatialSize,
    int32_t numHeads, int32_t channelsPerHead, int32_t numLevels,
    int32_t numQuery, int32_t numPoints, float* output) noexcept;
