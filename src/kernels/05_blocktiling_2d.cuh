#pragma once

#include "../utils.cuh"

// Threads = (BM * BN) / (TM * TN). Each thread owns a TM x TN micro-tile of C.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_blocktiling_2d(int M, int N, int K, float alpha,
                                     float beta, const float* A, const float* B,
                                     float* C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  constexpr int numThreads = (BM * BN) / (TM * TN);

  const uint threadCol = threadIdx.x % (BN / TN);
  const uint threadRow = threadIdx.x / (BN / TN);

  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  constexpr int strideA = numThreads / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;
  constexpr int strideB = numThreads / BN;

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int bk = 0; bk < K; bk += BK) {
    for (int o = 0; o < BM; o += strideA)
      As[(innerRowA + o) * BK + innerColA] = A[(innerRowA + o) * K + innerColA];
    for (int o = 0; o < BK; o += strideB)
      Bs[(innerRowB + o) * BN + innerColB] = B[(innerRowB + o) * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dk = 0; dk < BK; ++dk) {
      for (int i = 0; i < TM; ++i)
        regM[i] = As[(threadRow * TM + i) * BK + dk];
      for (int i = 0; i < TN; ++i)
        regN[i] = Bs[dk * BN + threadCol * TN + i];
      for (int rm = 0; rm < TM; ++rm)
        for (int rn = 0; rn < TN; ++rn)
          threadResults[rm * TN + rn] += regM[rm] * regN[rn];
    }
    __syncthreads();
  }

  for (int rm = 0; rm < TM; ++rm) {
    for (int rn = 0; rn < TN; ++rn) {
      const uint row = threadRow * TM + rm;
      const uint col = threadCol * TN + rn;
      C[row * N + col] =
          alpha * threadResults[rm * TN + rn] + beta * C[row * N + col];
    }
  }
}

inline void run_blocktiling_2d(int M, int N, int K, float alpha, float beta,
                               const float* A, const float* B, float* C) {
  const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  dim3 block((BM * BN) / (TM * TN));
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_blocktiling_2d<BM, BN, BK, TM, TN>
      <<<grid, block>>>(M, N, K, alpha, beta, A, B, C);
}
