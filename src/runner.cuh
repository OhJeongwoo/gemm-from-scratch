#pragma once

#include <cublas_v2.h>

#include "kernels/00_cublas.cuh"
#include "kernels/01_naive.cuh"
#include "kernels/02_coalesced.cuh"
#include "kernels/03_shared_mem.cuh"
#include "kernels/04_blocktiling_1d.cuh"
#include "kernels/05_blocktiling_2d.cuh"
#include "utils.cuh"

inline const char* kernel_name(int id) {
  switch (id) {
    case 0: return "cuBLAS (reference)";
    case 1: return "naive";
    case 2: return "global coalescing";
    case 3: return "shared-mem blocking";
    case 4: return "1D blocktiling";
    case 5: return "2D blocktiling";
    default: return "unknown";
  }
}

inline void run_kernel(int id, int M, int N, int K, float alpha, float beta,
                       const float* A, const float* B, float* C,
                       cublasHandle_t handle) {
  switch (id) {
    case 0: run_cublas(handle, M, N, K, alpha, beta, A, B, C); break;
    case 1: run_naive(M, N, K, alpha, beta, A, B, C); break;
    case 2: run_coalesced(M, N, K, alpha, beta, A, B, C); break;
    case 3: run_shared_mem(M, N, K, alpha, beta, A, B, C); break;
    case 4: run_blocktiling_1d(M, N, K, alpha, beta, A, B, C); break;
    case 5: run_blocktiling_2d(M, N, K, alpha, beta, A, B, C); break;
    default:
      fprintf(stderr, "unknown kernel id %d\n", id);
      exit(EXIT_FAILURE);
  }
}
