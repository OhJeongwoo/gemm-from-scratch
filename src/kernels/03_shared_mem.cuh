#pragma once

#include "../utils.cuh"

// Requires M, N, K to be multiples of BS.
template <int BS>
__global__ void sgemm_shared(int M, int N, int K, float alpha, float beta,
                             const float* A, const float* B, float* C) {
  __shared__ float As[BS * BS];
  __shared__ float Bs[BS * BS];

  const uint cRow = blockIdx.y;  // block row in C
  const uint cCol = blockIdx.x;  // block col in C
  const uint tRow = threadIdx.x / BS;
  const uint tCol = threadIdx.x % BS;

  // Move pointers to this block's tile origin.
  A += cRow * BS * K;
  B += cCol * BS;
  C += cRow * BS * N + cCol * BS;

  float acc = 0.0f;
  for (int bk = 0; bk < K; bk += BS) {
    As[tRow * BS + tCol] = A[tRow * K + tCol];
    Bs[tRow * BS + tCol] = B[tRow * N + tCol];
    __syncthreads();

    A += BS;
    B += BS * N;

    for (int k = 0; k < BS; ++k)
      acc += As[tRow * BS + k] * Bs[k * BS + tCol];
    __syncthreads();
  }

  C[tRow * N + tCol] = alpha * acc + beta * C[tRow * N + tCol];
}

inline void run_shared_mem(int M, int N, int K, float alpha, float beta,
                           const float* A, const float* B, float* C) {
  const int BS = 32;
  dim3 block(BS * BS);
  dim3 grid(CEIL_DIV(N, BS), CEIL_DIV(M, BS));
  sgemm_shared<BS><<<grid, block>>>(M, N, K, alpha, beta, A, B, C);
}
