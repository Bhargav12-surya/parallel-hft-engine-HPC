# Backend setup

## CPU

Serial and native threading are always available. OpenMP is built when CMake finds its C runtime. Linux workers respect the allowed CPU affinity mask. Windows affinity currently uses one processor-group mask; very large multi-group machines need additional affinity work. Leave enough cores for the UI while demonstrating.

## MPI

Install a C MPI SDK/runtime (for example Open MPI on Linux or Microsoft MPI on Windows), then reconfigure. The optional worker links only the MPI C target. Put the worker beside the GUI or launch from the build directory. `mpiexec` must be on PATH.

```sh
mpiexec -n 2 ./hft_mpi_worker --symbols 16 --events 2000 --out mpi-two
mpiexec -n 4 ./hft_mpi_worker --symbols 16 --events 2000 --threads 2 --out hybrid
mpiexec -n 4 ./hft_mpi_worker --symbols 16 --events 2000 --topology central --out central
mpiexec -n 2 ./hft_mpi_worker --cg
mpiexec -n 2 ./hft_mpi_worker --train --config ../configs/S6.json --out trained
```

`partition=0` uses cyclic pair assignment; `partition=1` uses contiguous pair blocks. The GUI launcher uses argument arrays on POSIX and a quoted command line on Windows, reads stdout without blocking the UI, and terminates its owned process group/job on cancellation or a 300-second timeout. It does not shell-expand user paths. Do not use shell metacharacters as worker paths.

MPI config broadcast currently uses a binary POD record, so all ranks must use the same build, architecture layout and byte order. The derived event/order/fill datatypes describe their fields explicitly. Integer decisions remain identical because global symbol IDs and seeds are retained. Multi-node launch, firewall and transport setup is outside the desktop's responsibility.

## OpenCL C

Install an OpenCL SDK/ICD loader and device driver. Configure with `HFT_OPENCL=ON` (default). Devices must support fp64. The default selector requires a GPU. For **CPU-only OpenCL validation**, explicitly set `HFT_OPENCL_CPU=1`; results identify the device and must not be described as GPU performance.

Kernel source is embedded from the actual shared C headers and `kernels/compute.c` during CMake configuration. Workgroup size is bounded by the kernel limit. `streams=2` partitions pairs across two profiling queues. `pinned=1` creates mapped host-allocated staging storage, unmaps it before buffer copies, and respects buffer lifetimes. Actual overlap and pinned-memory benefits depend on the driver/device and must be measured.

Host/device struct sizes are checked. An allocation or build failure reports the real error instead of falling back under a GPU label. Device tests exercise all 13 strategies, matrix output, inference, gradients and Q training. They return CTest skip code 77 when no eligible device is present.

## NVIDIA C APIs

Configure `-DHFT_NVIDIA=ON` with a CUDA Toolkit that provides C-compatible cuBLAS, cuRAND and CUDA runtime headers/libraries. This does not enable the CUDA CMake language. `hft_nvidia_risk` centers observations on the host, computes covariance with SGEMM and generates seeded normals with cuRAND. C CPU routines complete factorization and risk aggregation.

NVIDIA-library and physical-GPU behavior was not validated in this environment. The bundled Windows executables exclude these optional libraries; no missing vendor DLLs are silently downloaded. cuRAND streams are not claimed to match the project's custom Philox stream bit-for-bit.

No physical GPU performance, occupancy, CUDA Graph or cuDNN results are included.
