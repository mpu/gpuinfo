load("@rules_cuda//cuda:defs.bzl", "cuda_library")

ABSL_GCC_FLAGS = [
    "-Wall",
    "-Wextra",
    "-Wcast-qual",
    "-Wconversion-null",
    "-Wformat-security",
    "-Wmissing-declarations",
    "-Woverlength-strings",
    "-Wpointer-arith",
    "-Wundef",
    "-Wunused-local-typedefs",
    "-Wunused-result",
    "-Wvarargs",
    "-Wvla",
    "-Wwrite-strings",
    # "-H",  # display included headers
]

cc_binary(
  name = "gpuinfo",
  deps = [
    "@com_google_absl//absl/flags:flag",
    "@com_google_absl//absl/flags:parse",
    "@com_google_absl//absl/flags:usage",
    "@com_google_absl//absl/log:check",
    "@com_google_absl//absl/log:flags",
    "@com_google_absl//absl/log:initialize",
    "@com_google_absl//absl/log:log",
    "@com_google_absl//absl/strings:str_format",
    "@com_google_absl//absl/strings:string_view",
    "@com_google_absl//absl/time:time",
    "@local_cuda//:cuda_runtime",
    ":cuda_helpers",
    ":memory_kernels",
  ],
  copts = ABSL_GCC_FLAGS,
  srcs = ["gpuinfo.cc"],
)

cc_library(
  name = "cuda_helpers",
  deps = [
    "@com_google_absl//absl/log:check",
  ],
  hdrs = ["cuda_helpers.h"],
)

cuda_library(
  name = "memory_kernels",
  deps = [
    ":cuda_helpers",
  ],
  hdrs = ["memory.h"],
  srcs = ["memory.cu"],
)

# vim: sw=2 et
