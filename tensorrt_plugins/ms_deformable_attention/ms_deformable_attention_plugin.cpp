#include "ms_deformable_attention_plugin.h"
#include "ms_deformable_attention_kernel.h"

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <new>

namespace
{
constexpr char const* kPluginName = "MSDeformableAttention";
constexpr char const* kPluginVersion = "1";

bool positiveDims(nvinfer1::Dims const& dims) noexcept
{
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            return false;
        }
    }
    return true;
}

bool validRuntimeContract(nvinfer1::PluginTensorDesc const* input,
    nvinfer1::PluginTensorDesc const* output) noexcept
{
    auto const& value = input[0].dims;
    auto const& shapes = input[1].dims;
    auto const& starts = input[2].dims;
    auto const& locations = input[3].dims;
    auto const& weights = input[4].dims;
    auto const& result = output[0].dims;
    return value.nbDims == 4 && shapes.nbDims == 2 && starts.nbDims == 1
        && locations.nbDims == 6 && weights.nbDims == 5
        && result.nbDims == 3 && positiveDims(value)
        && positiveDims(shapes) && positiveDims(starts)
        && positiveDims(locations) && positiveDims(weights)
        && shapes.d[1] == 2 && locations.d[5] == 2
        && value.d[0] == locations.d[0]
        && value.d[0] == weights.d[0]
        && value.d[2] == locations.d[2]
        && value.d[2] == weights.d[2]
        && shapes.d[0] == starts.d[0]
        && shapes.d[0] == locations.d[3]
        && shapes.d[0] == weights.d[3]
        && locations.d[1] == weights.d[1]
        && locations.d[4] == weights.d[4]
        && result.d[0] == value.d[0]
        && result.d[1] == locations.d[1]
        && result.d[2] == value.d[2] * value.d[3];
}
} // namespace

MSDeformableAttentionPlugin::MSDeformableAttentionPlugin(
    void const*, size_t length)
{
    // The plugin is parameter-free. Reject incompatible serialized payloads
    // by leaving the object valid; the creator checks length before creation.
    (void) length;
}

char const* MSDeformableAttentionPlugin::getPluginType() const noexcept
{
    return kPluginName;
}

char const* MSDeformableAttentionPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

int32_t MSDeformableAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs MSDeformableAttentionPlugin::getOutputDimensions(
    int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output{};
    if (outputIndex != 0 || nbInputs != 5 || inputs[0].nbDims != 4
        || inputs[3].nbDims != 6)
    {
        output.nbDims = -1;
        return output;
    }
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[3].d[1];
    output.d[2] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
        *inputs[0].d[2], *inputs[0].d[3]);
    return output;
}

bool MSDeformableAttentionPlugin::supportsFormatCombination(int32_t pos,
    nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != 5 || nbOutputs != 1
        || pos < 0 || pos >= nbInputs + nbOutputs
        || inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 1 || pos == 2)
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    if (pos == 0)
    {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT
            || inOut[pos].type == nvinfer1::DataType::kHALF;
    }
    return inOut[pos].type == inOut[0].type;
}

void MSDeformableAttentionPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept
{
#ifdef BEVFORMER_MSDA_DEBUG
    if (inputs != nullptr && nbInputs == 5)
    {
        auto const& value = inputs[0].desc.dims;
        auto const& locations = inputs[3].desc.dims;
        std::cerr << "[MSDA Plugin] B=" << value.d[0]
                  << " S=" << value.d[1] << " Q=" << locations.d[1]
                  << " H=" << value.d[2] << " D=" << value.d[3]
                  << " L=" << inputs[1].desc.dims.d[0]
                  << " P=" << locations.d[4] << " dtype="
                  << (inputs[0].desc.type == nvinfer1::DataType::kFLOAT
                          ? "FP32" : "FP16") << '\n';
    }
#else
    (void) inputs;
    (void) nbInputs;
#endif
}

size_t MSDeformableAttentionPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const*, int32_t,
    nvinfer1::PluginTensorDesc const*, int32_t) const noexcept
{
    return 0;
}

int32_t MSDeformableAttentionPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void*,
    cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr
        || outputs == nullptr || !validRuntimeContract(inputDesc, outputDesc))
    {
        return 1;
    }
    auto const& value = inputDesc[0].dims;
    auto const& locations = inputDesc[3].dims;
    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
    {
        return launchMSDeformableAttentionFP32(stream,
            static_cast<float const*>(inputs[0]),
            static_cast<int32_t const*>(inputs[1]),
            static_cast<int32_t const*>(inputs[2]),
            static_cast<float const*>(inputs[3]),
            static_cast<float const*>(inputs[4]), value.d[0], value.d[1],
            value.d[2], value.d[3], inputDesc[1].dims.d[0], locations.d[1],
            locations.d[4], static_cast<float*>(outputs[0]));
    }
    if (inputDesc[0].type == nvinfer1::DataType::kHALF)
    {
        return launchMSDeformableAttentionFP16(stream,
            static_cast<__half const*>(inputs[0]),
            static_cast<int32_t const*>(inputs[1]),
            static_cast<int32_t const*>(inputs[2]),
            static_cast<__half const*>(inputs[3]),
            static_cast<__half const*>(inputs[4]), value.d[0], value.d[1],
            value.d[2], value.d[3], inputDesc[1].dims.d[0], locations.d[1],
            locations.d[4], static_cast<__half*>(outputs[0]));
    }
    return 1;
}

nvinfer1::DataType MSDeformableAttentionPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept
{
    return index == 0 && inputTypes != nullptr && nbInputs == 5
        ? inputTypes[0] : nvinfer1::DataType::kFLOAT;
}

int32_t MSDeformableAttentionPlugin::initialize() noexcept
{
    return 0;
}

void MSDeformableAttentionPlugin::terminate() noexcept {}

size_t MSDeformableAttentionPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void MSDeformableAttentionPlugin::serialize(void*) const noexcept {}

void MSDeformableAttentionPlugin::destroy() noexcept
{
    delete this;
}

MSDeformableAttentionPlugin* MSDeformableAttentionPlugin::clone() const noexcept
{
    auto* plugin = new (std::nothrow) MSDeformableAttentionPlugin();
    if (plugin != nullptr)
    {
        plugin->setPluginNamespace(mNamespace.c_str());
    }
    return plugin;
}

void MSDeformableAttentionPlugin::setPluginNamespace(
    char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

char const* MSDeformableAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

MSDeformableAttentionPluginCreator::MSDeformableAttentionPluginCreator()
{
    mFields.nbFields = 0;
    mFields.fields = nullptr;
}

char const* MSDeformableAttentionPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* MSDeformableAttentionPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

nvinfer1::PluginFieldCollection const*
MSDeformableAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

nvinfer1::IPluginV2* MSDeformableAttentionPluginCreator::createPlugin(
    char const*, nvinfer1::PluginFieldCollection const*) noexcept
{
    auto* plugin = new (std::nothrow) MSDeformableAttentionPlugin();
    if (plugin != nullptr)
    {
        plugin->setPluginNamespace(mNamespace.c_str());
    }
    return plugin;
}

nvinfer1::IPluginV2* MSDeformableAttentionPluginCreator::deserializePlugin(
    char const*, void const* serialData, size_t serialLength) noexcept
{
    if (serialLength != 0)
    {
        return nullptr;
    }
    auto* plugin = new (std::nothrow)
        MSDeformableAttentionPlugin(serialData, serialLength);
    if (plugin != nullptr)
    {
        plugin->setPluginNamespace(mNamespace.c_str());
    }
    return plugin;
}

void MSDeformableAttentionPluginCreator::setPluginNamespace(
    char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

char const*
MSDeformableAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(MSDeformableAttentionPluginCreator);
