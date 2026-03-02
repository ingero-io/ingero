// managed_alloc.cu — Exercises cudaMallocManaged (Unified Memory).
//
// Validates that Ingero traces cudaMallocManaged uprobes. Unified Memory
// enables transparent page migration between host and device — first-touch
// page faults are an invisible latency source that Ingero can now detect.
//
// Compile: nvcc managed_alloc.cu -o managed_alloc
// Run:     ./managed_alloc
//
// Traced APIs:
//   cudaMallocManaged    — Unified Memory allocation (new in v0.8)
//   cudaLaunchKernel     — GPU kernel launch (triggers first-touch migration)
//   cudaDeviceSynchronize — wait for GPU completion
//   cudaFree             — free unified memory

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1024 * 1024)
#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Kernel that touches managed memory — triggers first-touch page migration.
__global__ void initAndScale(float *data, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = (float)i * scale;
    }
}

// Kernel that reads managed memory — pages already resident on device.
__global__ void reduce(float *data, float *result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

int main() {
    printf("managed_alloc: exercising cudaMallocManaged (Unified Memory)\n");

    float *data, *result;

    // API: cudaMallocManaged — allocate Unified Memory.
    // This is the key API Ingero v0.8 now traces.
    printf("  cudaMallocManaged: %d floats (%.1f MB)...\n", N, (float)N * sizeof(float) / 1e6);
    CHECK(cudaMallocManaged(&data, N * sizeof(float)));
    CHECK(cudaMallocManaged(&result, sizeof(float)));

    // Touch from host first — pages allocated on host.
    printf("  Host touch: initializing data on CPU...\n");
    for (int i = 0; i < 1024; i++) {
        data[i] = (float)i;
    }
    *result = 0.0f;

    // API: cudaLaunchKernel — kernel touches managed memory.
    // First-touch page fault: pages migrate from host to device.
    printf("  cudaLaunchKernel: initAndScale (triggers page migration)...\n");
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    initAndScale<<<blocks, threads>>>(data, 2.0f, N);
    CHECK(cudaDeviceSynchronize());

    // Second kernel — pages already on device, no migration.
    printf("  cudaLaunchKernel: reduce (pages already on device)...\n");
    reduce<<<blocks, threads>>>(data, result, N);
    CHECK(cudaDeviceSynchronize());

    // Read result from host — triggers page migration back.
    printf("  Host read: result = %.0f (triggers device→host migration)\n", *result);

    // Cleanup.
    CHECK(cudaFree(data));
    CHECK(cudaFree(result));

    printf("managed_alloc: done (Ingero should have traced cudaMallocManaged calls)\n");
    return 0;
}
