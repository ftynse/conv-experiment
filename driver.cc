#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

#include "mlir/ExecutionEngine/CRunnerUtils.h"

constexpr const static int N = 5;
constexpr const static int H = 80;
constexpr const static int W = 100;
constexpr const static int KH = 3;
constexpr const static int KW = 3;
constexpr const static int C = 128;
constexpr const static int F = 128;

template <typename... Args>
inline float *alloc_float(Args... size) {
  return static_cast<float *>(std::malloc((size * ... * sizeof(float))));
}

template <typename... Args>
StridedMemRefType<float, sizeof...(Args)> alloc_float_memref(Args... size) {
  StridedMemRefType<float, sizeof...(Args)> descriptor;
  descriptor.basePtr = alloc_float(std::forward<Args>(size)...);
  descriptor.data = descriptor.basePtr;
  descriptor.offset = 0;
  std::vector<int64_t> sizes({size...});
  std::copy(std::begin(sizes), std::end(sizes), std::begin(descriptor.sizes));
  if constexpr (sizeof...(Args) > 0) {
    descriptor.strides[0] = 1;
  }
  std::partial_sum(std::begin(sizes), std::prev(std::end(sizes)),
                   std::next(std::begin(descriptor.strides)), std::multiplies<int64_t>());

  size_t total_size = std::accumulate(std::begin(sizes), std::end(sizes), 1,
                                      std::multiplies<int64_t>());
  for (int64_t i = 0; i < total_size; ++i)
    descriptor.data[i] = 1.0f;

  return descriptor;
}

template <typename T>
void free_memref(T &&memref) {
  std::free(memref.basePtr);
}

extern "C" {
void _mlir_ciface_conv(StridedMemRefType<float, 4> *result,
                       StridedMemRefType<float, 4> *input,
                       StridedMemRefType<float, 4> *fitler,
                       StridedMemRefType<float, 1> *bias,
                       StridedMemRefType<float, 4> *bias_init,
                       StridedMemRefType<float, 4> *output);
}

int main() {
  StridedMemRefType<float, 4> input = alloc_float_memref(N, (H + KH - 1), (W + KW - 1), C);
  StridedMemRefType<float, 4> filter = alloc_float_memref(F, KH, KW, C);
  StridedMemRefType<float, 1> bias = alloc_float_memref(C);
  StridedMemRefType<float, 4> output = alloc_float_memref(N, H, W, F);

  StridedMemRefType<float, 4> bias_init = alloc_float_memref(N, H, W, F);

  StridedMemRefType<float, 4> result;

  auto start = std::chrono::high_resolution_clock::now();
  _mlir_ciface_conv(&result, &input, &filter, &bias, &bias_init, &output);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << ms << "ms" << std::endl;

  free_memref(std::move(input));
  free_memref(std::move(filter));
  free_memref(std::move(bias));
  free_memref(std::move(output));

  free_memref(std::move(bias_init));
  return EXIT_SUCCESS;
}
