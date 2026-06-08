#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err__));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st__ = (call);                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__,   \
              (int)st__);                                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

inline void randomize_matrix(float* m, int n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < n; ++i)
    m[i] = dist(gen);
}

inline bool verify(const float* ref, const float* test, int n,
                   float rtol = 1e-2f, float atol = 1e-3f) {
  int bad = 0;
  double max_abs = 0.0, max_rel = 0.0;
  for (int i = 0; i < n; ++i) {
    double d = std::fabs((double)ref[i] - (double)test[i]);
    double rel = d / (std::fabs((double)ref[i]) + 1e-6);
    if (d > max_abs) max_abs = d;
    if (rel > max_rel) max_rel = rel;
    if (d > atol + rtol * std::fabs((double)ref[i])) ++bad;
  }
  printf("  verify: %s  (max_abs=%.3e, max_rel=%.3e, mismatches=%d/%d)\n",
         bad == 0 ? "PASS" : "FAIL", max_abs, max_rel, bad, n);
  return bad == 0;
}

template <typename F>
float time_ms(F fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i)
    fn();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    fn();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

inline double gflops(int M, int N, int K, float ms) {
  return (2.0 * M * N * K) / (ms / 1e3) / 1e9;
}
