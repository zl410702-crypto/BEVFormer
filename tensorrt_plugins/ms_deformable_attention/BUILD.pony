load("//common/third_party:cuda.bzl", "cuda_library")

# Copy/rename this file to BUILD when integrating the plugin directory into
# PONY_ASW. The Orin platform selects TensorRT 8.4.12.5 through
# @tensorrt//:nvinfer; the x86 platform selects its host-only TensorRT package.
package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "msda_kernel",
    srcs = ["ms_deformable_attention_kernel.cu"],
    hdrs = ["ms_deformable_attention_kernel.h"],
    deps = [
        "@cuda//:cudart",
    ],
)

cc_binary(
    name = "libbevformer_msda_plugin.so",
    srcs = [
        "ms_deformable_attention_plugin.cpp",
        "ms_deformable_attention_plugin.h",
    ],
    linkshared = True,
    deps = [
        ":msda_kernel",
        "@cuda//:cudart",
        "@tensorrt//:nvinfer",
        "@tensorrt//:nvinfer_plugin",
    ],
    linkopts = ["-ldl"],
)
