#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
  }
};

template <typename T>
struct TrtDestroy {
  void operator()(T* value) const { if (value) value->destroy(); }
};

std::vector<char> readFile(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open " + path);
  const auto size = stream.tellg();
  stream.seekg(0);
  std::vector<char> data(static_cast<std::size_t>(size));
  if (!stream.read(data.data(), size)) throw std::runtime_error("cannot read " + path);
  return data;
}

void writeFile(const std::string& path, const void* data, std::size_t size) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream || !stream.write(static_cast<const char*>(data), size))
    throw std::runtime_error("cannot write " + path);
}

std::int64_t volume(const nvinfer1::Dims& dims) {
  std::int64_t result = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) throw std::runtime_error("runtime binding is not static");
    result *= dims.d[i];
  }
  return result;
}

std::size_t elementSize(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kINT32: return sizeof(std::int32_t);
    case nvinfer1::DataType::kINT8: return sizeof(std::int8_t);
    default: throw std::runtime_error("unsupported runtime binding dtype");
  }
}

void validateAffineGridBinding(const std::string& name,
                               nvinfer1::DataType dtype,
                               const nvinfer1::Dims& dims) {
  if (name != "affine_grid") return;
  if (dtype != nvinfer1::DataType::kFLOAT || dims.nbDims != 3 ||
      dims.d[0] != 1 || dims.d[1] != 2500 || dims.d[2] != 2) {
    throw std::runtime_error(
        "affine_grid must be static FP32 [1,2500,2]");
  }
}

const char* dataTypeName(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return "FP32";
    case nvinfer1::DataType::kINT32: return "INT32";
    case nvinfer1::DataType::kINT8: return "INT8";
    default: return "unsupported";
  }
}

void checkCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: run_bevformer_engine ENGINE PLUGIN FRAME_DIR OUTPUT_DIR FRAME1_PREV\n";
    return 2;
  }
  try {
    void* plugin = dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL);
    if (!plugin) throw std::runtime_error(dlerror());
    Logger logger;
    auto engineBytes = readFile(argv[1]);
    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime(
        nvinfer1::createInferRuntime(logger));
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>> engine(
        runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size()));
    if (!engine) throw std::runtime_error("engine deserialization failed");
    std::unique_ptr<nvinfer1::IExecutionContext,
                    TrtDestroy<nvinfer1::IExecutionContext>> context(
        engine->createExecutionContext());
    if (!context) throw std::runtime_error("execution context creation failed");

    std::vector<void*> bindings(engine->getNbBindings(), nullptr);
    std::vector<std::vector<char>> host(engine->getNbBindings());
    for (int i = 0; i < engine->getNbBindings(); ++i) {
      const std::string name = engine->getBindingName(i);
      const auto dtype = engine->getBindingDataType(i);
      validateAffineGridBinding(name, dtype, context->getBindingDimensions(i));
      const std::size_t bytes = static_cast<std::size_t>(volume(
          context->getBindingDimensions(i))) * elementSize(dtype);
      host[i].resize(bytes);
      if (engine->bindingIsInput(i)) {
        const std::string input = (name == "prev_bev" && std::string(argv[5]) != "-")
            ? argv[5] : std::string(argv[3]) + "/" + name + ".bin";
        host[i] = readFile(input);
        if (host[i].size() != bytes) throw std::runtime_error(name + " byte-size mismatch");
      }
      checkCuda(cudaMalloc(&bindings[i], bytes), "cudaMalloc");
      if (engine->bindingIsInput(i))
        checkCuda(cudaMemcpy(bindings[i], host[i].data(), bytes,
                             cudaMemcpyHostToDevice), "input cudaMemcpy");
      std::cout << (engine->bindingIsInput(i) ? "input " : "output ")
                << name << " dtype=" << dataTypeName(dtype)
                << " bytes=" << bytes << '\n';
    }
    if (!context->executeV2(bindings.data())) throw std::runtime_error("executeV2 failed");
    for (int i = 0; i < engine->getNbBindings(); ++i) {
      if (!engine->bindingIsInput(i)) {
        checkCuda(cudaMemcpy(host[i].data(), bindings[i], host[i].size(),
                             cudaMemcpyDeviceToHost), "output cudaMemcpy");
        writeFile(std::string(argv[4]) + "/" + engine->getBindingName(i) + ".bin",
                  host[i].data(), host[i].size());
      }
      cudaFree(bindings[i]);
    }
    // Keep the plugin library loaded until process exit; engine/context
    // destructors may still dispatch into its virtual methods.
    (void)plugin;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
