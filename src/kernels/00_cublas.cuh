#pragma once

#include <cublas_v2.h>

#include "../utils.cuh"

// Row-major C = A*B via cuBLAS (column-major) by computing C^T = B^T * A^T.
inline void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                       float beta, const float* A, const float* B, float* C) {
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                           N, A, K, &beta, C, N));
}
