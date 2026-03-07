// cuda_c_test.cu — Minimal C CUDA workload exercising all 4 traced APIs.
//
// Proves Ingero traces ANY language that calls libcudart.so, not just Python.
// This is a pure C program — no PyTorch, no Python, just direct CUDA calls.
//
// Compile: nvcc cuda_c_test.cu -o cuda_c_test
// Run:     ./cuda_c_test
//
// Traced APIs:
//   cudaMalloc           — GPU memory allocation
//   cudaMemcpy           — Host↔GPU data transfer
//   cudaLaunchKernel     — GPU compute kernel launch (via <<<>>>)
//   cudaStreamSynchronize / cudaDeviceSynchronize — wait for completion

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Simple vector add kernel — exercises cudaLaunchKernel.
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    printf("cuda_c_test: exercising all 4 traced CUDA APIs\n");

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Host allocation.
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // API 1: cudaMalloc — allocate GPU memory.
    printf("  cudaMalloc x3...\n");
    CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // API 2: cudaMemcpy — copy data Host→Device.
    printf("  cudaMemcpy H->D x2...\n");
    CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // API 3: cudaLaunchKernel — launch compute kernel.
    printf("  cudaLaunchKernel (vectorAdd)...\n");
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // API 4: cudaDeviceSynchronize — wait for GPU to finish.
    printf("  cudaDeviceSynchronize...\n");
    CHECK(cudaDeviceSynchronize());

    // API 2 again: cudaMemcpy — copy result Device→Host.
    printf("  cudaMemcpy D->H...\n");
    CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result.
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            errors++;
        }
    }

    if (errors == 0) {
        printf("  PASS: all %d elements correct\n", N);
    } else {
        printf("  FAIL: %d errors\n", errors);
    }

    // Cleanup.
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    printf("cuda_c_test: done (Ingero should have traced all CUDA calls)\n");
    return errors > 0 ? 1 : 0;
}
