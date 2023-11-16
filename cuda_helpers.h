#pragma once

#include "absl/log/check.h"

#define CUDA_CHECK(expr)                                                \
  do {                                                                  \
    cudaError_t err = (expr);                                           \
    CHECK_EQ(err, cudaSuccess) << "cuda: " << cudaGetErrorString(err);  \
  } while (0)

template<class T>
struct CudaAlloc
{
  typedef T value_type;
 
  CudaAlloc() = default;

  template<class U>
  constexpr CudaAlloc(const CudaAlloc<U>&) noexcept {}

  [[nodiscard]]
  T* allocate(std::size_t n)
  {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
	throw std::bad_array_new_length();

    void *ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, n * sizeof(T)));
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t) noexcept
  {
    CUDA_CHECK(cudaFreeHost(p));
  }
};

template<class T, class U>
bool operator==(const CudaAlloc <T>&, const CudaAlloc <U>&)
{
  return true;
}

template<class T, class U>
bool operator!=(const CudaAlloc <T>&, const CudaAlloc <U>&)
{
  return false;
}
