/*
 * FP32 forward-only adaptation of MMCV 1.4.0 MSDeformableAttention.
 *
 * The interpolation and accumulation order follow:
 * mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh
 * Copyright (c) 2020 SenseTime. Licensed under Apache-2.0.
 */

#include "ms_deformable_attention_kernel.h"

#include <cuda_runtime.h>

namespace
{
constexpr int32_t kCudaThreads = 1024;

template <typename T>
__device__ float loadAsFloat(T const* value, int32_t offset)
{
    return static_cast<float>(value[offset]);
}

template <>
__device__ float loadAsFloat(__half const* value, int32_t offset)
{
    return __half2float(value[offset]);
}

template <typename T>
__device__ float bilinearSample(T const* value, int32_t height,
    int32_t width, int32_t numHeads, int32_t channels, float h, float w,
    int32_t head, int32_t channel)
{
    int32_t const hLow = static_cast<int32_t>(floorf(h));
    int32_t const wLow = static_cast<int32_t>(floorf(w));
    int32_t const hHigh = hLow + 1;
    int32_t const wHigh = wLow + 1;
    float const lh = h - static_cast<float>(hLow);
    float const lw = w - static_cast<float>(wLow);
    float const hh = 1.0F - lh;
    float const hw = 1.0F - lw;
    int32_t const widthStride = numHeads * channels;
    int32_t const heightStride = width * widthStride;
    int32_t const base = head * channels + channel;

    float v1 = 0.0F;
    float v2 = 0.0F;
    float v3 = 0.0F;
    float v4 = 0.0F;
    if (hLow >= 0 && wLow >= 0)
    {
        v1 = loadAsFloat(value,
            hLow * heightStride + wLow * widthStride + base);
    }
    if (hLow >= 0 && wHigh <= width - 1)
    {
        v2 = loadAsFloat(value,
            hLow * heightStride + wHigh * widthStride + base);
    }
    if (hHigh <= height - 1 && wLow >= 0)
    {
        v3 = loadAsFloat(value,
            hHigh * heightStride + wLow * widthStride + base);
    }
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        v4 = loadAsFloat(value,
            hHigh * heightStride + wHigh * widthStride + base);
    }
    float const w1 = hh * hw;
    float const w2 = hh * lw;
    float const w3 = lh * hw;
    float const w4 = lh * lw;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T>
__global__ void msDeformableAttentionKernel(int32_t count,
    T const* value, int32_t const* spatialShapes,
    int32_t const* levelStartIndex, T const* samplingLocations,
    T const* attentionWeights, int32_t spatialSize, int32_t numHeads,
    int32_t channels, int32_t numLevels, int32_t numQuery,
    int32_t numPoints, T* output)
{
    for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += blockDim.x * gridDim.x)
    {
        int32_t cursor = index;
        int32_t const channel = cursor % channels;
        cursor /= channels;
        int32_t const samplingIndex = cursor;
        int32_t const head = cursor % numHeads;
        cursor /= numHeads;
        cursor /= numQuery;
        int32_t const batch = cursor;

        int32_t weightOffset = samplingIndex * numLevels * numPoints;
        int32_t locationOffset = weightOffset * 2;
        int32_t const valueStride = numHeads * channels;
        int32_t const batchValueOffset = batch * spatialSize * valueStride;
        float result = 0.0F;
        for (int32_t level = 0; level < numLevels; ++level)
        {
            int32_t const height = spatialShapes[level * 2];
            int32_t const width = spatialShapes[level * 2 + 1];
            T const* levelValue = value + batchValueOffset
                + levelStartIndex[level] * valueStride;
            for (int32_t point = 0; point < numPoints; ++point)
            {
                float const x = loadAsFloat(samplingLocations, locationOffset);
                float const y = loadAsFloat(samplingLocations, locationOffset + 1);
                float const weight = loadAsFloat(attentionWeights, weightOffset);
                float const h = y * static_cast<float>(height) - 0.5F;
                float const w = x * static_cast<float>(width) - 0.5F;
                if (h > -1.0F && w > -1.0F
                    && h < static_cast<float>(height)
                    && w < static_cast<float>(width))
                {
                    result += bilinearSample(levelValue, height, width,
                        numHeads, channels, h, w, head, channel) * weight;
                }
                ++weightOffset;
                locationOffset += 2;
            }
        }
        output[index] = static_cast<T>(result);
    }
}
} // namespace

int32_t launchMSDeformableAttentionFP32(
    cudaStream_t stream, float const* value, int32_t const* spatialShapes,
    int32_t const* levelStartIndex, float const* samplingLocations,
    float const* attentionWeights, int32_t batch, int32_t spatialSize,
    int32_t numHeads, int32_t channelsPerHead, int32_t numLevels,
    int32_t numQuery, int32_t numPoints, float* output) noexcept
{
    int64_t const count64 = static_cast<int64_t>(batch) * numQuery
        * numHeads * channelsPerHead;
    if (count64 <= 0 || count64 > INT32_MAX)
    {
        return 1;
    }
    int32_t const count = static_cast<int32_t>(count64);
    int32_t const blocks = (count + kCudaThreads - 1) / kCudaThreads;
    msDeformableAttentionKernel<float><<<blocks, kCudaThreads, 0, stream>>>(count,
        value, spatialShapes, levelStartIndex, samplingLocations,
        attentionWeights, spatialSize, numHeads, channelsPerHead, numLevels,
        numQuery, numPoints, output);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

int32_t launchMSDeformableAttentionFP16(
    cudaStream_t stream, __half const* value, int32_t const* spatialShapes,
    int32_t const* levelStartIndex, __half const* samplingLocations,
    __half const* attentionWeights, int32_t batch, int32_t spatialSize,
    int32_t numHeads, int32_t channelsPerHead, int32_t numLevels,
    int32_t numQuery, int32_t numPoints, __half* output) noexcept
{
    int64_t const count64 = static_cast<int64_t>(batch) * numQuery
        * numHeads * channelsPerHead;
    if (count64 <= 0 || count64 > INT32_MAX)
    {
        return 1;
    }
    int32_t const count = static_cast<int32_t>(count64);
    int32_t const blocks = (count + kCudaThreads - 1) / kCudaThreads;
    msDeformableAttentionKernel<__half><<<blocks, kCudaThreads, 0, stream>>>(
        count, value, spatialShapes, levelStartIndex, samplingLocations,
        attentionWeights, spatialSize, numHeads, channelsPerHead, numLevels,
        numQuery, numPoints, output);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}
