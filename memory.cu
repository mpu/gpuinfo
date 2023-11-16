#include "memory.h"
#include "cuda_helpers.h"

// Kernels to measure device memory throughput

// For read throughput we use a reduction kernel that should
// be bandwidth-limited; this lets us sanity check that the
// computation is seeing all the memory.

__inline__ __device__
uint warpReduceSum(uint val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

// Sum reduction for at most 1024 values (one block).
// The correct reduction result is returned by in the
// first thread only.
__inline__ __device__
uint blockReduceSum(uint val)
{
  static __shared__ uint shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;  // warp id

  val = warpReduceSum(val);
  if (lane == 0)
    shared[wid] = val;

  __syncthreads();  // wait for partial reductions in shared[]

  if (wid == 0) {
    // final reduce in the first warp
    if (threadIdx.x * warpSize < blockDim.x)
      // in warp N, read the result of the partial summation
      // of threads [lane * warpSize, (lane+1) * warpSize[
      val = shared[lane];
    else
      // the block is smaller than 1024
      val = 0;

    val = warpReduceSum(val);
  }

  return val;
}

/*
 * This kernel uses grid-stride loops which has two advantages: it
 * allows us to reuse the kernel for both passes, and it computes
 * per-thread sums in registers before calling blockReduceSum().
 * Doing the sum in registers significantly reduces the communication
 * between threads for large array sizes. Note that the output array
 * must have a size equal to or larger than the number of thread
 * blocks in the grid because each block writes to a unique location
 * within the array.
 */
__global__
void gridReduceSum(uint *in, uint *out, uint N)
{
  uint sum = 0;

  // reduce multiple elements per thread
  for (uint n = blockIdx.x * blockDim.x + threadIdx.x;
       n < N;
       n += blockDim.x * gridDim.x) {
    sum += in[n];
  }
  sum = blockReduceSum(sum);

  if (threadIdx.x == 0)
    out[blockIdx.x] = sum;
}

__global__
void gridFill(uint *out, uint N)
{
  uint step = blockDim.x * gridDim.x;
  uint n0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint n = n0; n < N; n += step)
    out[n] = n * n;
}


namespace {

std::tuple<int, int> getParams(size_t N)
{
  int threads = 256;
  int blocks = min((N + threads - 1) / threads, 1024ul);
  return {threads, blocks};
}

} // namespace

void *allocWorkBuffer(size_t N)
{

  auto [threads, blocks] = getParams(N);

  void *ptr;
  CUDA_CHECK(cudaMalloc(&ptr, blocks * sizeof(uint)));
  return ptr;
}

uint launchSumKernel(uint *in, size_t N, void *buf)
{
  auto [threads, blocks] = getParams(N);
  auto out = reinterpret_cast<uint*>(buf);

  gridReduceSum<<<blocks, threads>>>(in, out, N);
  gridReduceSum<<<1, 1024>>>(out, out, blocks);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  uint sum;
  CUDA_CHECK(cudaMemcpy(&sum, out, sizeof sum, cudaMemcpyDefault));
  return sum;
}

void launchFillKernel(uint *out, size_t N)
{
  int threads = 256;
  int blocks = min(1024ul, (N + threads-1) / threads);

  if (blocks == 0) return;

  gridFill<<<blocks, threads>>>(out, N);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}


// vim: sw=2 et
