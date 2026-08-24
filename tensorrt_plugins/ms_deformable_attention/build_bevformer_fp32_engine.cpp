#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <dlfcn.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace
{
class Logger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            std::cerr << "[TensorRT] " << static_cast<int32_t>(severity)
                      << ": " << message << '\n';
        }
    }
};

template <typename T>
struct TrtDestroy
{
    void operator()(T* value) const noexcept
    {
        if (value != nullptr)
        {
            value->destroy();
        }
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDestroy<T>>;

bool save(std::string const& path, nvinfer1::IHostMemory const& engine)
{
    std::ofstream output(path, std::ios::binary);
    output.write(static_cast<char const*>(engine.data()), engine.size());
    return output.good();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " model.onnx plugin.so model_fp32.engine\n";
        return 2;
    }
    Logger logger;
    void* plugin = dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL);
    if (plugin == nullptr)
    {
        std::cerr << "dlopen failed: " << dlerror() << '\n';
        return 1;
    }

    TrtPtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    uint32_t const flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtPtr<nvinfer1::INetworkDefinition> network{
        builder == nullptr ? nullptr : builder->createNetworkV2(flags)};
    TrtPtr<nvonnxparser::IParser> parser{
        network == nullptr ? nullptr : nvonnxparser::createParser(*network, logger)};
    if (builder == nullptr || network == nullptr || parser == nullptr)
    {
        std::cerr << "failed to create TensorRT builder/network/parser\n";
        return 1;
    }
    if (!parser->parseFromFile(argv[1],
            static_cast<int32_t>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ONNX parse failed with " << parser->getNbErrors()
                  << " errors\n";
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cerr << "  " << parser->getError(i)->desc() << '\n';
        }
        return 1;
    }

    int32_t pluginLayers = 0;
    for (int32_t i = 0; i < network->getNbLayers(); ++i)
    {
        if (network->getLayer(i)->getType() == nvinfer1::LayerType::kPLUGIN_V2)
        {
            ++pluginLayers;
        }
    }
    std::cout << "Parser: PASS\nPluginV2 layers: " << pluginLayers << '\n';
    if (pluginLayers != 12)
    {
        std::cerr << "expected exactly 12 MSDeformableAttention plugin layers\n";
        return 1;
    }

    TrtPtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (config == nullptr)
    {
        std::cerr << "failed to create builder config\n";
        return 1;
    }
    // FP32 only. No kFP16/kINT8/TF32 policy is enabled by this tool.
    config->setMaxWorkspaceSize(4ULL << 30);
    TrtPtr<nvinfer1::IHostMemory> serialized{
        builder->buildSerializedNetwork(*network, *config)};
    if (serialized == nullptr)
    {
        std::cerr << "FP32 engine build failed\n";
        return 1;
    }
    if (!save(argv[3], *serialized))
    {
        std::cerr << "failed to save engine: " << argv[3] << '\n';
        return 1;
    }
    std::cout << "FP32 engine: PASS\nSize: " << serialized->size() << '\n';
    // Keep the shared object loaded until all TensorRT objects are destroyed.
    return 0;
}
