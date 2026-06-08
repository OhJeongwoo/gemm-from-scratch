# gemm-from-scratch

CUDA FP32 GEMM kernels from naive to register-tiled, benchmarked against cuBLAS.

Computes `C = A · B` (row-major) where `A` is `M×K`, `B` is `K×N`, `C` is `M×N`.

## Hardware

| | |
|---|---|
| GPU              | NVIDIA RTX 4080 (Ada Lovelace, sm_89), 16 GB GDDR6X |
| Memory bandwidth | 716.8 GB/s |
| FP32 peak        | 48.7 TFLOP/s |
| CPU              | Intel Core i7-13700K |
| OS               | Ubuntu 22.04 LTS (kernel 6.8) |
| NVIDIA driver    | 560.35.03 |
| CUDA toolkit     | 12.6 |

## Benchmark — 4096³ FP32

| # | Kernel              | GFLOP/s | % cuBLAS |
|---|---------------------|--------:|---------:|
| 1 | naive               |     396 |     1.2% |
| 2 | global coalescing   |   2,608 |     7.8% |
| 3 | shared-mem blocking |   4,014 |    12.4% |
| 4 | 1D blocktiling      |  12,038 |    38.1% |
| 5 | 2D blocktiling      |  22,709 |    65.3% |
| 0 | cuBLAS              | ~34,000 |     100% |

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Default arch is `sm_89` (Ada). Change `CMAKE_CUDA_ARCHITECTURES` in [CMakeLists.txt](CMakeLists.txt) for other GPUs.

## Run

```bash
./build/gemm <kernel_id> [M] [N] [K] [iters]
./build/gemm 5 4096 4096 4096
```

Kernel ids: `0` cuBLAS · `1` naive · `2` coalesced · `3` shared · `4` 1D tile · `5` 2D tile.

## Sweep / profile

```bash
./scripts/benchmark.sh                       # all kernels × shapes -> benchmarks/results.csv
sudo ./scripts/profile.sh 5 4096 4096 4096   # ncu, 7 metrics (sudo needed on GeForce)
```

## Requirements

CUDA 12.x · CMake ≥ 3.20 · NVIDIA GPU.
