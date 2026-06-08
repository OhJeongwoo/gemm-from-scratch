#pragma once

#include "../utils.cuh"

template <int BLOCKSIZE>
__global__ void sgemm_coalesced(int M, int N, int K, float alpha, float beta,
                                const float* A, const float* B, float* C) {
  const uint row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

inline void run_coalesced(int M, int N, int K, float alpha, float beta,
                          const float* A, const float* B, float* C) {
  const int BS = 32;
  dim3 block(BS * BS);
  dim3 grid(CEIL_DIV(M, BS), CEIL_DIV(N, BS));
  sgemm_coalesced<BS><<<grid, block>>>(M, N, K, alpha, beta, A, B, C);
}
