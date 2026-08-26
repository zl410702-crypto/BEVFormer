cmake_minimum_required(VERSION 3.18)
project(bevformer_msda_plugin LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

option(BEVFORMER_MSDA_DEBUG "Print configured MSDA tensor contracts" OFF)
option(BEVFORMER_MSDA_BUILD_TESTS "Build plugin serialization test" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT 8.4.12.5 root (PONY_ASW Orin target)")
if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInferVersion.h")
  message(FATAL_ERROR "Set TENSORRT_ROOT to the PONY_ASW TensorRT 8.4.12.5 tree")
endif()

file(READ "${TENSORRT_ROOT}/include/NvInferVersion.h" TRT_VERSION_HEADER)
if(NOT TRT_VERSION_HEADER MATCHES "NV_TENSORRT_MAJOR 8" OR
   NOT TRT_VERSION_HEADER MATCHES "NV_TENSORRT_MINOR 4" OR
   NOT TRT_VERSION_HEADER MATCHES "NV_TENSORRT_PATCH 12" OR
   NOT TRT_VERSION_HEADER MATCHES "NV_TENSORRT_BUILD 5")
  message(FATAL_ERROR "PONY_ASW Orin requires exactly TensorRT 8.4.12.5")
endif()

find_library(NVINFER_LIBRARY nvinfer HINTS "${TENSORRT_ROOT}/lib" REQUIRED)
find_library(NVONNXPARSER_LIBRARY nvonnxparser
  HINTS "${TENSORRT_ROOT}/lib" REQUIRED)

add_library(bevformer_msda_plugin SHARED
  ms_deformable_attention_plugin.cpp
  ms_deformable_attention_kernel.cu)
target_include_directories(bevformer_msda_plugin PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${TENSORRT_ROOT}/include")
target_link_libraries(bevformer_msda_plugin PRIVATE "${NVINFER_LIBRARY}")
if(BEVFORMER_MSDA_DEBUG)
  target_compile_definitions(bevformer_msda_plugin PRIVATE
    BEVFORMER_MSDA_DEBUG=1)
endif()
set_target_properties(bevformer_msda_plugin PROPERTIES
  OUTPUT_NAME ms_deformable_attention_plugin
  CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS bevformer_msda_plugin LIBRARY DESTINATION lib)

add_executable(build_bevformer_fp32_engine build_bevformer_fp32_engine.cpp)
target_include_directories(build_bevformer_fp32_engine PRIVATE
  "${TENSORRT_ROOT}/include")
target_link_libraries(build_bevformer_fp32_engine PRIVATE
  "${NVINFER_LIBRARY}" "${NVONNXPARSER_LIBRARY}" dl)
install(TARGETS build_bevformer_fp32_engine RUNTIME DESTINATION bin)

if(BEVFORMER_MSDA_BUILD_TESTS)
  add_executable(test_msda_plugin_serialization
    validation/test_plugin_serialization.cpp)
  target_link_libraries(test_msda_plugin_serialization PRIVATE
    bevformer_msda_plugin CUDA::cudart)
endif()
