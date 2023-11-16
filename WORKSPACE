load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
  name = "bazel_skylib",
  urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz"],
  sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
)

http_archive(
  name = "com_google_absl",
  urls = ["https://github.com/abseil/abseil-cpp/archive/716fa00789b60ff52473eabc3ac201eb61744392.zip"],
  strip_prefix = "abseil-cpp-716fa00789b60ff52473eabc3ac201eb61744392",
  sha256 = "7536e637ae7e873c5a60fc55f8c4329e473060088d7b32e39adaa50f106281fc",
)

# http_archive(
#   name = "com_google_benchmark",
#   urls = ["https://github.com/google/benchmark/archive/bf585a2789e30585b4e3ce6baf11ef2750b54677.zip"],
#   strip_prefix = "benchmark-bf585a2789e30585b4e3ce6baf11ef2750b54677",
#   sha256 = "2a778d821997df7d8646c9c59b8edb9a573a6e04c534c01892a40aa524a7b68c",
# )

http_archive(
  name = "rules_cuda",
  urls = ["https://github.com/bazel-contrib/rules_cuda/archive/1a2ec3d1ffacf3c462b69c2bbac91111d1752d21.tar.gz"],
  strip_prefix = "rules_cuda-1a2ec3d1ffacf3c462b69c2bbac91111d1752d21",
  sha256 = "63ba2219104f73bca2ad1d4886df9afa404a14a7339395e629d0ef115c770c56",
)
load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
rules_cuda_dependencies()
register_detected_cuda_toolchains()
