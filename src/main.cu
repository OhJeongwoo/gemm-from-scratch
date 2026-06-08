// ./gemm <kernel_id> [M] [N] [K] [iters]

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "runner.cuh"
#include "utils.cuh"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: %s <kernel_id> [M] [N] [K] [iters]\n"
            "  kernel_id: 0 cuBLAS | 1 naive | 2 coalesced | 3 shared | "
            "4 1D-tile | 5 2D-tile\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  const int kernel_id = atoi(argv[1]);
  const int M = argc > 2 ? atoi(argv[2]) : 4096;
  const int N = argc > 3 ? atoi(argv[3]) : 4096;
  const int K = argc > 4 ? atoi(argv[4]) : 4096;
  const int iters = argc > 5 ? atoi(argv[5]) : 50;
  const int warmup = 5;
  const float alpha = 1.0f, beta = 0.0f;

  printf("kernel %d (%s) | M=%d N=%d K=%d | iters=%d\n", kernel_id,
         kernel_name(kernel_id), M, N, K, iters);

  // Host inputs (reproducible).
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
  randomize_matrix(hA.data(), (int)hA.size(), /*seed=*/1234);
  randomize_matrix(hB.data(), (int)hB.size(), /*seed=*/5678);

  // Device buffers.
  float *dA, *dB, *dC, *dC_ref;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC_ref, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dC_ref, 0, (size_t)M * N * sizeof(float)));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  run_cublas(handle, M, N, K, alpha, beta, dA, dB, dC_ref);
  CUDA_CHECK(cudaDeviceSynchronize());
  float cublas_ms = time_ms(
      [&] { run_cublas(handle, M, N, K, alpha, beta, dA, dB, dC_ref); }, warmup,
      iters);
  double cublas_gflops = gflops(M, N, K, cublas_ms);

  run_kernel(kernel_id, M, N, K, alpha, beta, dA, dB, dC, handle);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hC((size_t)M * N), hC_ref((size_t)M * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hC_ref.data(), dC_ref, hC_ref.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  bool ok = verify(hC_ref.data(), hC.data(), (int)hC.size());

  float ms = time_ms(
      [&] { run_kernel(kernel_id, M, N, K, alpha, beta, dA, dB, dC, handle); },
      warmup, iters);
  double gf = gflops(M, N, K, ms);
  double pct = 100.0 * gf / cublas_gflops;

  printf("  time   : %.3f ms\n", ms);
  printf("  perf   : %.1f GFLOP/s  (%.1f%% of cuBLAS @ %.1f GFLOP/s)\n", gf, pct,
         cublas_gflops);

  printf("CSV,%d,%s,%d,%d,%d,%.4f,%.2f,%.2f,%s\n", kernel_id,
         kernel_name(kernel_id), M, N, K, ms, gf, pct, ok ? "PASS" : "FAIL");

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dC_ref));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
