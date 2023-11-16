#include <atomic>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "cuda_runtime.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/initialize.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "cuda_helpers.h"
#include "memory.h"

using absl::PrintF;
using absl::StrFormat;
using absl::string_view;

ABSL_FLAG(int, device, 0, "CUDA device");
ABSL_FLAG(int, detail, 1, "Detail level for device attributes (up to 3)");
ABSL_FLAG(bool, mem_bench, false, "Run memory benchmarking");

namespace {

struct DeviceAttr {
  cudaDeviceAttr attr;
  string_view descr;
  int detail = 2;
};

std::vector<DeviceAttr> GetAttributes()
{
  return {
    { cudaDevAttrMaxThreadsPerBlock, "Max threads per block", 0 },
    { cudaDevAttrMaxBlockDimX, "Max block dimension X", 0 },
    { cudaDevAttrMaxBlockDimY, "Max block dimension Y", 0 },
    { cudaDevAttrMaxBlockDimZ, "Max block dimension Z", 0 },
    { cudaDevAttrMaxGridDimX, "Max grid dimension X", 0 },
    { cudaDevAttrMaxGridDimY, "Max grid dimension Y", 0 },
    { cudaDevAttrMaxGridDimZ, "Max grid dimension Z", 0 },
    { cudaDevAttrMaxSharedMemoryPerBlock, "Max shared memory per block", 0 },
    { cudaDevAttrTotalConstantMemory, "Constant memory size", 0 },
    { cudaDevAttrWarpSize, "Warp size", 0 },
    { cudaDevAttrMaxPitch, "Max memory copy pitch" },
    { cudaDevAttrMaxRegistersPerBlock, "Max registers per block", 0 },
    { cudaDevAttrClockRate, "Peak clock frequency (kHz)", 0 },
    // cudaDevAttrTextureAlignment
    { cudaDevAttrGpuOverlap, "Device can overlap memory copy and compute", 0 },
    { cudaDevAttrMultiProcessorCount, "Number of multiprocessors", 0 },
    { cudaDevAttrKernelExecTimeout, "Kernel run time limit" },
    // cudaDevAttrIntegrated
    { cudaDevAttrCanMapHostMemory, "Device can map host memory", 0 },
    { cudaDevAttrComputeMode, "Device compute mode" },
    // cudaDevAttrMaxTexture1DWidth
    // cudaDevAttrMaxTexture2DWidth
    // cudaDevAttrMaxTexture2DHeight
    // cudaDevAttrMaxTexture3DWidth
    // cudaDevAttrMaxTexture3DHeight
    // cudaDevAttrMaxTexture3DDepth
    // cudaDevAttrMaxTexture2DLayeredWidth
    // cudaDevAttrMaxTexture2DLayeredHeight
    // cudaDevAttrMaxTexture2DLayeredLayers
    // cudaDevAttrSurfaceAlignment
    { cudaDevAttrConcurrentKernels, "Device can execute kernels concurrently", 0 },
    { cudaDevAttrEccEnabled, "ECC support enabled", 0 },
    { cudaDevAttrPciBusId, "Device PCI Bus ID" },
    { cudaDevAttrPciDeviceId, "Device PCI Device ID" },
    { cudaDevAttrTccDriver, "Device is using TCC driver model" },
    { cudaDevAttrMemoryClockRate, "Peak memory clock frequency (kHz)", 0 },
    { cudaDevAttrGlobalMemoryBusWidth, "Global memory bus width (bits)", 0 },
    { cudaDevAttrL2CacheSize, "L2 cache size", 0 },
    { cudaDevAttrMaxThreadsPerMultiProcessor, "Max resident threads per multiprocessor", 0 },
    { cudaDevAttrAsyncEngineCount, "Number of asynchronous engines", 0 },
    { cudaDevAttrUnifiedAddressing, "Device shares unified addressing with host", 0 },
    // cudaDevAttrMaxTexture1DLayeredWidth
    // cudaDevAttrMaxTexture1DLayeredLayers
    // cudaDevAttrMaxTexture2DGatherWidth
    // cudaDevAttrMaxTexture2DGatherHeight
    // cudaDevAttrMaxTexture3DWidthAlt
    // cudaDevAttrMaxTexture3DHeightAlt
    // cudaDevAttrMaxTexture3DDepthAlt
    { cudaDevAttrMaxTexture3DDepthAlt, "Device PCI domain ID" },
    // cudaDevAttrTexturePitchAlignment
    // cudaDevAttrMaxTextureCubemapWidth
    // cudaDevAttrMaxTextureCubemapLayeredWidth
    // cudaDevAttrMaxTextureCubemapLayeredLayers
    // cudaDevAttrMaxSurface1DWidth
    // cudaDevAttrMaxSurface2DWidth
    // cudaDevAttrMaxSurface2DHeight
    // cudaDevAttrMaxSurface3DWidth
    // cudaDevAttrMaxSurface3DHeight
    // cudaDevAttrMaxSurface3DDepth
    // cudaDevAttrMaxSurface1DLayeredWidth
    // cudaDevAttrMaxSurface1DLayeredLayers
    // cudaDevAttrMaxSurface2DLayeredWidth
    // cudaDevAttrMaxSurface2DLayeredHeight
    // cudaDevAttrMaxSurface2DLayeredLayers
    // cudaDevAttrMaxSurfaceCubemapWidth
    // cudaDevAttrMaxSurfaceCubemapLayeredWidth
    // cudaDevAttrMaxSurfaceCubemapLayeredLayers
    // cudaDevAttrMaxTexture1DLinearWidth
    // cudaDevAttrMaxTexture2DLinearWidth
    // cudaDevAttrMaxTexture2DLinearHeight
    // cudaDevAttrMaxTexture2DLinearPitch
    // cudaDevAttrMaxTexture2DMipmappedWidth
    // cudaDevAttrMaxTexture2DMipmappedHeight
    { cudaDevAttrComputeCapabilityMajor, "Major compute capability version number", 0 },
    { cudaDevAttrComputeCapabilityMinor, "Minor compute capability version number", 0 },
    // cudaDevAttrMaxTexture1DMipmappedWidth
    { cudaDevAttrStreamPrioritiesSupported, "Device supports stream priorities", 0 },
    { cudaDevAttrGlobalL1CacheSupported, "Device supports caching globals in L1", 0 },
    { cudaDevAttrLocalL1CacheSupported, "Device supports caching locals in L1", 0 },
    { cudaDevAttrMaxSharedMemoryPerMultiprocessor, "Max shared memory per multiprocessor", 0 },
    { cudaDevAttrMaxRegistersPerMultiprocessor, "Max 32-bit registers per multiprocessor", 0 },
    { cudaDevAttrManagedMemory, "Device can allocate managed memory", 0 },
    // cudaDevAttrIsMultiGpuBoard
    // cudaDevAttrMultiGpuBoardGroupID
    { cudaDevAttrHostNativeAtomicSupported, "Device host link supports native atomics" },
    { cudaDevAttrSingleToDoublePrecisionPerfRatio, "Single to double precision perf ratio" },
    // cudaDevAttrPageableMemoryAccess
    // cudaDevAttrConcurrentManagedAccess
    // cudaDevAttrComputePreemptionSupported
    { cudaDevAttrCanUseHostPointerForRegisteredMem, "Device can use registered host pointer" },
    { cudaDevAttrCooperativeLaunch, "Device can launch cooperative kernels" },
    // cudaDevAttrCooperativeMultiDeviceLaunch (deprecated)
    // cudaDevAttrMaxSharedMemoryPerBlockOptin
    // cudaDevAttrCanFlushRemoteWrites
    { cudaDevAttrHostRegisterSupported, "Device can use cudaHostRegister" },
    // cudaDevAttrPageableMemoryAccessUsesHostPageTables
    // cudaDevAttrDirectManagedMemAccessFromHost
    { cudaDevAttrMaxBlocksPerMultiprocessor, "Max number of blocks per multiprocessor", 0 },
    { cudaDevAttrMaxPersistingL2CacheSize, "Max L2 persisting lines (bytes)" },
    // cudaDevAttrMaxAccessPolicyWindowSize
    { cudaDevAttrReservedSharedMemoryPerBlock, "Shared mem reserved by CUDA driver per block" },
    { cudaDevAttrSparseCudaArraySupported, "Device supports sparse arrays" },
    { cudaDevAttrHostRegisterReadOnlySupported, "Device supports cudaHostRegisterReadOnly flag" },
    // cudaDevAttrTimelineSemaphoreInteropSupported
    // cudaDevAttrMaxTimelineSemaphoreInteropSupported (deprecated)
    { cudaDevAttrMemoryPoolsSupported, "Device supports cudaMallocAsync and cudaMemPool" },
    { cudaDevAttrGPUDirectRDMASupported, "Device supports RDMA APIs" },
    // cudaDevAttrGPUDirectRDMAFlushWritesOptions
    // cudaDevAttrGPUDirectRDMAWritesOrdering
    // cudaDevAttrMemoryPoolSupportedHandleTypes
    // cudaDevAttrClusterLaunch
    // cudaDevAttrDeferredMappingCudaArraySupported
    { cudaDevAttrIpcEventSupport, "Device supports IPC events" },
    // cudaDevAttrMemSyncDomainCount
    // cudaDevAttrNumaConfig
    // cudaDevAttrNumaId
    // cudaDevAttrMpsEnabled
    // cudaDevAttrHostNumaId
  };
}

constexpr size_t kBenchIter = 4;
constexpr size_t kHostThreads = 3;
constexpr size_t k1Gb = 1024 * 1024 * 1024;

struct Timings {
  long mean_us;
  long stddev_us;
};

std::string HumanThroughput(size_t bytes, Timings t)
{
  auto tpt = static_cast<double>(bytes) / t.mean_us * 1e6 / k1Gb;
  return absl::StrFormat("%.2fGb/s (stddev ±%ldus)", tpt, t.stddev_us);
}

template<typename Fn>
Timings Bench(Fn f)
{
  std::vector<long> times(kBenchIter);

  for (size_t n = 0; n < kBenchIter; n++) {
    auto const start = absl::Now();
    f();
    times[n] = (absl::Now() - start) / absl::Microseconds(1);
  }

  long mean =
    std::accumulate(times.begin(), times.end(), 0) / kBenchIter;

  long sqsum = 0;
  for (auto t : times)
    sqsum += (t - mean) * (t - mean);

  return {mean, static_cast<long>(std::sqrt(sqsum / kBenchIter))};
}

void PrintResult(std::string descr, std::string result)
{
  size_t constexpr kWidth = 35;

  std::string dots(std::max(kWidth - descr.size(), 0ul), '.');
  PrintF("%s:%s %s\n", descr, dots, result);
}

void HostBenchmark()
{
  auto constexpr kVecBytes = 2 * k1Gb;

  {
    std::vector<uint> v(kVecBytes / sizeof(uint));

    auto hst = Bench([&] {
      for (size_t i = 0; i < v.size(); i++) {
        v[i] = i * i;
      }
    });
    PrintResult(
        "Host sequential write",
        HumanThroughput(kVecBytes, hst));

    auto hpt = Bench([&] {
      std::atomic<size_t> index{0};
      size_t step = (v.size() + kHostThreads - 1) / kHostThreads;
      std::vector<std::thread> threads;
      for (size_t tid = 0; tid < kHostThreads; tid++) {
        threads.emplace_back([&] {
          auto startIdx = index.fetch_add(step);
          auto endIdx = std::min(v.size(), startIdx + step);
          for (auto i = startIdx; i < endIdx; i++) {
            v[i] = i * i;
          }
        });
      }
      for (auto& t : threads) t.join();
    });
    PrintResult(
        StrFormat("Host parallel write (%d threads)", kHostThreads),
        HumanThroughput(kVecBytes, hpt));
  }

  {
    /* huge difference with size_t instead of int */
    std::vector<size_t> v(kVecBytes / sizeof(size_t));
    for (size_t i = 0; i < v.size(); i++) v[i] = i * i;

    size_t seq_sum;

    auto const hst = Bench([&] {
      size_t sum = 0;
      for (size_t i = 0; i < v.size(); i++) {
        sum += v[i];
      }
      seq_sum = sum;
    });
    PrintResult(
        "Host sequential read",
        HumanThroughput(kVecBytes, hst));

    auto const hpt = Bench([&] {
      size_t step =
        (v.size() + kHostThreads-1) / kHostThreads;
      std::atomic<size_t> chunk{0};
      std::atomic<size_t> par_sum{0};

      std::vector<std::thread> threads;
      for (size_t tid = 0; tid < kHostThreads; tid++) {
        threads.emplace_back([&, tid] {
          auto starti = chunk.fetch_add(step);
          auto endi = std::min(v.size(), starti + step);
          size_t psum = 0;
          for (auto i = starti; i < endi; i++) {
            psum += v[i];
          }
          par_sum += psum;
        });
      }
      for (auto& t : threads) t.join();

      CHECK_EQ(par_sum, seq_sum);
    });
    PrintResult(
        StrFormat("Host parallel read (%d threads)", kHostThreads),
        HumanThroughput(kVecBytes, hpt));
  }
}

void TransferBenchmark()
{
  auto constexpr kVecBytes = 2 * k1Gb;

  std::vector<uint, CudaAlloc<uint>>
    hostv(kVecBytes / sizeof(uint));
  auto hostp = hostv.data();
  for (size_t i = 0; i < hostv.size(); i++)
    hostv[i] = i * i;

  void* devp;
  CUDA_CHECK(cudaMalloc(&devp, kVecBytes));

  auto h2dt = Bench([&] {
    CUDA_CHECK(cudaMemcpy(devp, hostp, kVecBytes, cudaMemcpyDefault));
  });
  PrintResult(
      "Pinned host memory to device",
       HumanThroughput(kVecBytes, h2dt));

  auto d2ht = Bench([&] {
    CUDA_CHECK(cudaMemcpy(hostp, devp, kVecBytes, cudaMemcpyDefault));
  });
  PrintResult(
      "Device to pinned host memory",
      HumanThroughput(kVecBytes, d2ht));

  CUDA_CHECK(cudaFree(devp));
}

void DeviceBenchmark()
{
  constexpr auto kVecBytes = 2 * k1Gb;

  std::vector<uint, CudaAlloc<uint>>
    hostv(kVecBytes / sizeof(uint));
  auto hostp = hostv.data();

  uint cpu_sum = 0;
  for (size_t i = 0; i < hostv.size(); i++) {
    hostv[i] = i * i;
    cpu_sum += i * i;
  }

  void* devp;
  CUDA_CHECK(cudaMalloc(&devp, kVecBytes));
  CUDA_CHECK(cudaMemcpy(devp, hostp, kVecBytes, cudaMemcpyDefault));

  auto buf = allocWorkBuffer(hostv.size());
  auto drt = Bench([&] {
    uint cuda_sum = launchSumKernel(
        reinterpret_cast<uint*>(devp), hostv.size(), buf);
    CHECK_EQ(cuda_sum, cpu_sum);
    // LOG(INFO) << "cuda_sum=" << cuda_sum;
  });
  CUDA_CHECK(cudaFree(buf));
  PrintResult(
      "Device read throughput",
       HumanThroughput(kVecBytes, drt));

  CUDA_CHECK(cudaMemset(devp, 0, kVecBytes));
  CUDA_CHECK(cudaDeviceSynchronize());

  auto dwt = Bench([&] {
    launchFillKernel(
        reinterpret_cast<uint*>(devp), hostv.size());
  });

  CUDA_CHECK(cudaMemcpy(hostp, devp, kVecBytes, cudaMemcpyDefault));
  CHECK_EQ(std::accumulate(hostv.begin(), hostv.end(), 0u), cpu_sum);

  PrintResult(
      "Device write throughput",
       HumanThroughput(kVecBytes, dwt));

  CUDA_CHECK(cudaFree(devp));
}

} // namespace

int main(int argc, char** argv)
{
  absl::InitializeLog();
  absl::SetProgramUsageMessage("");
  absl::ParseCommandLine(argc, argv);

  int dev_cnt;
  CUDA_CHECK(cudaGetDeviceCount(&dev_cnt));
  if (dev_cnt == 0) {
    PrintF("Error: No CUDA device available\n");
    exit(1);
  }

  auto dev_idx{absl::GetFlag(FLAGS_device)};
  if (dev_idx >= dev_cnt) {
    PrintF(
        "Error: Device index %d is invalid, "
        "should be less than %d\n", dev_idx, dev_cnt);
    exit(1);
  }

  CUDA_CHECK(cudaSetDevice(dev_idx));

  if (auto detail = absl::GetFlag(FLAGS_detail)) {
    cudaDeviceProp dev_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, dev_idx));
    PrintF("Attributes for device %d (%s):\n", dev_idx, dev_prop.name);

    auto dev_attrs{GetAttributes()};
    size_t max_desc_len = 0;
    for (const auto& da : dev_attrs) {
      if (da.detail >= detail) continue;
      max_desc_len = std::max(max_desc_len, da.descr.size());
    }
    for (const auto& da : dev_attrs) {
      if (da.detail >= detail) continue;
      int value;
      CUDA_CHECK(cudaDeviceGetAttribute(&value, da.attr, dev_idx));
      std::string dots(max_desc_len - da.descr.size(), '.');
      PrintF("%s:%s %d\n", da.descr, dots, value);
    }

  }
  if (absl::GetFlag(FLAGS_mem_bench)) {
    PrintF("\nMemory benchmarking:\n");
    HostBenchmark();
    TransferBenchmark();
    DeviceBenchmark();
  }
}

// vim: sw=2 et
