#pragma once

#include "../utils.cuh"

// Threads = (BM * BN) / TM. Each thread owns a TM-tall column of C.
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_blocktiling_1d(int M, int N, int K, float alpha,
                                     float beta, const float* A, const float* B,
                                     float* C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint threadRow = threadIdx.x / BN;
  const uint threadCol = threadIdx.x % BN;

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[TM] = {0.0f};

  for (int bk = 0; bk < K; bk += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dk = 0; dk < BK; ++dk) {
      float tmpB = Bs[dk * BN + threadCol];
      for (int res = 0; res < TM; ++res)
        threadResults[res] += As[(threadRow * TM + res) * BK + dk] * tmpB;
    }
    __syncthreads();
  }

  for (int res = 0; res < TM; ++res) {
    const uint row = threadRow * TM + res;
    C[row * N + threadCol] =
        alpha * threadResults[res] + beta * C[row * N + threadCol];
  }
}

inline void run_blocktiling_1d(int M, int N, int K, float alpha, float beta,
                               const float* A, const float* B, float* C) {
  const int BM = 64, BN = 64, BK = 8, TM = 8;
  dim3 block((BM * BN) / TM);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_blocktiling_1d<BM, BN, BK, TM>
      <<<grid, block>>>(M, N, K, alpha, beta, A, B, C);
}
