#pragma once

#include "../utils.cuh"

__global__ void sgemm_naive(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
  const uint row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

inline void run_naive(int M, int N, int K, float alpha, float beta,
                      const float* A, const float* B, float* C) {
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_naive<<<grid, block>>>(M, N, K, alpha, beta, A, B, C);
}
