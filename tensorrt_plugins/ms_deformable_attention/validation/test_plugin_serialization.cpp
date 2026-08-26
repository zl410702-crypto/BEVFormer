#include "ms_deformable_attention_plugin.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace
{
template <typename T>
T* copyToDevice(std::vector<T> const& host)
{
    T* device = nullptr;
    cudaMalloc(&device, host.size() * sizeof(T));
    cudaMemcpy(device, host.data(), host.size() * sizeof(T),
        cudaMemcpyHostToDevice);
    return device;
}

nvinfer1::Dims dims(std::initializer_list<int32_t> values)
{
    nvinfer1::Dims result{};
    result.nbDims = static_cast<int32_t>(values.size());
    std::copy(values.begin(), values.end(), result.d);
    return result;
}
} // namespace

int main()
{
    // Small deterministic L=1 contract: B=1,S=4,H=1,D=2,Q=3,P=2.
    std::vector<float> value{1, 10, 2, 20, 3, 30, 4, 40};
    std::vector<int32_t> shapes{2, 2};
    std::vector<int32_t> starts{0};
    std::vector<float> locations{
        .25F, .25F, .75F, .75F,
        .50F, .50F, .25F, .75F,
        .75F, .25F, .50F, .50F};
    std::vector<float> weights{.5F, .5F, .25F, .75F, .6F, .4F};
    float* dValue = copyToDevice(value);
    int32_t* dShapes = copyToDevice(shapes);
    int32_t* dStarts = copyToDevice(starts);
    float* dLocations = copyToDevice(locations);
    float* dWeights = copyToDevice(weights);
    float* outputA = nullptr;
    float* outputB = nullptr;
    cudaMalloc(&outputA, 6 * sizeof(float));
    cudaMalloc(&outputB, 6 * sizeof(float));

    nvinfer1::PluginTensorDesc input[5]{};
    input[0].dims = dims({1, 4, 1, 2});
    input[1].dims = dims({1, 2});
    input[2].dims = dims({1});
    input[3].dims = dims({1, 3, 1, 1, 2, 2});
    input[4].dims = dims({1, 3, 1, 1, 2});
    for (int32_t i = 0; i < 5; ++i)
    {
        input[i].format = nvinfer1::TensorFormat::kLINEAR;
        input[i].type = (i == 1 || i == 2) ? nvinfer1::DataType::kINT32
                                           : nvinfer1::DataType::kFLOAT;
    }
    nvinfer1::PluginTensorDesc output[1]{};
    output[0].dims = dims({1, 3, 2});
    output[0].format = nvinfer1::TensorFormat::kLINEAR;
    output[0].type = nvinfer1::DataType::kFLOAT;
    void const* inputs[]{dValue, dShapes, dStarts, dLocations, dWeights};
    void* outputsA[]{outputA};
    void* outputsB[]{outputB};

    MSDeformableAttentionPlugin created;
    if (created.getSerializationSize() != 0
        || created.enqueue(input, output, inputs, outputsA, nullptr, 0) != 0)
    {
        std::cerr << "created plugin enqueue failed\n";
        return 1;
    }
    MSDeformableAttentionPlugin restored(nullptr, 0);
    if (restored.enqueue(input, output, inputs, outputsB, nullptr, 0) != 0)
    {
        std::cerr << "deserialized plugin enqueue failed\n";
        return 1;
    }
    std::vector<float> hostA(6);
    std::vector<float> hostB(6);
    cudaMemcpy(hostA.data(), outputA, hostA.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(hostB.data(), outputB, hostB.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (hostA != hostB)
    {
        std::cerr << "serialization output mismatch\n";
        return 1;
    }
    std::cout << "Serialization bytes: 0\n"
              << "Create -> deserialize -> enqueue: BITWISE IDENTICAL\n";
    cudaFree(dValue);
    cudaFree(dShapes);
    cudaFree(dStarts);
    cudaFree(dLocations);
    cudaFree(dWeights);
    cudaFree(outputA);
    cudaFree(outputB);
    return 0;
}
