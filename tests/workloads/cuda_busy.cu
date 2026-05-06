// cuda_busy.cu - small self-contained CUDA stresser for the e2e harness.
//
// Loops a busy kernel until SIGINT or the requested duration. Prints
// nothing to stdout in steady state so callers can pipe its stderr to
// a log file. Works on any sm_70+ device; nvcc compiles it without
// any external dependency beyond CUDA Runtime.
//
// Build:
//   nvcc -O2 -arch=native cuda_busy.cu -o cuda_busy
//
// Run:
//   ./cuda_busy --duration 30
//   ./cuda_busy --duration 0   # run forever, kill with Ctrl-C
//
// Used by tests/e2e/throttle-induced.sh as a PyTorch-free GPU stress
// path so the test runs on bare cloud GPU VMs that do not ship
// PyTorch by default.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>

static volatile sig_atomic_t stop_flag = 0;
static void on_signal(int sig) { (void)sig; stop_flag = 1; }

#define CUDA_CHECK(x) do {                                                  \
    cudaError_t err__ = (x);                                                \
    if (err__ != cudaSuccess) {                                             \
        fprintf(stderr, "cuda_busy: %s -> %s\n", #x, cudaGetErrorString(err__)); \
        return 2;                                                           \
    }                                                                       \
} while (0)

// Compute-bound kernel that does ~thousands of FMAs per thread per
// launch. The exact arithmetic is uninteresting; the goal is to keep
// every SM warp resident so power draw and clock-throttle reasons
// surface in nvidia-smi / NVML.
__global__ void busy_kernel(float *out, int iters, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float a = (float)tid * 1.000173f;
    float b = 0.999827f;
    for (int i = 0; i < iters; i++) {
        a = a * b + 1.000003f;
        b = b * a + 0.999991f;
    }
    out[tid] = a + b;
}

int main(int argc, char **argv) {
    int duration = 30;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--duration") && i + 1 < argc) {
            duration = atoi(argv[++i]);
        }
    }

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    const int n = 1 << 22;            // 4 Mi threads
    const int block = 256;
    const int grid  = (n + block - 1) / block;
    const int iters = 4096;

    float *d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    time_t deadline = time(NULL) + duration;
    int launches = 0;
    while (!stop_flag && (duration == 0 || time(NULL) < deadline)) {
        busy_kernel<<<grid, block>>>(d_out, iters, n);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            fprintf(stderr, "cuda_busy: kernel launch -> %s\n", cudaGetErrorString(e));
            cudaFree(d_out);
            return 3;
        }
        launches++;
        // No cudaDeviceSynchronize: keep the queue saturated so the
        // device runs hot. cudaFree at exit will drain the work.
    }

    cudaDeviceSynchronize();
    fprintf(stderr, "cuda_busy: %d launches, %ds elapsed\n",
            launches, duration);
    cudaFree(d_out);
    return 0;
}
