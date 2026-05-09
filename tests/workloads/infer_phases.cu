// infer_phases.cu — synthetic prefill/decode CUDA workload for v0.16.x
// AWS validation. Generates an alternating pattern that the agent's
// internal/infer phase classifier should bin into PhasePrefill (many
// launches between syncs, fat avg-kernel) and PhaseDecode (few
// launches between syncs, thin avg-kernel).
//
// Build:
//   nvcc -O2 -arch=native -cudart=shared infer_phases.cu -o infer_phases
//
// Run:
//   ./infer_phases --steps 200 --slow-decode-at 150
//
// Each "prefill" step launches many kernels then syncs; each "decode"
// step launches few then syncs. Optionally inject a 10× slow decode
// at a chosen step number to verify the phase-aware outlier detector.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>

static volatile sig_atomic_t stop_flag = 0;
static void on_signal(int sig) { (void)sig; stop_flag = 1; }

#define CUDA_CHECK(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "infer_phases: %s -> %s\n", #x, cudaGetErrorString(err__)); \
        return 2; \
    } \
} while (0)

// Two kernels with different shapes:
//   prefill_kernel: longer per-launch (heavy FMA loop)
//   decode_kernel:  short per-launch (single FMA per thread)
__global__ void prefill_kernel(float *out, int iters, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float a = (float)tid * 1.000173f, b = 0.999827f;
    for (int i = 0; i < iters; i++) {
        a = a * b + 1.000003f;
        b = b * a + 0.999991f;
    }
    out[tid] = a + b;
}

__global__ void decode_kernel(float *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = (float)tid * 1.000003f + 0.999991f;
}

int main(int argc, char **argv) {
    int total_steps = 200;
    int slow_decode_at = -1;     // step idx to inject slow decode (decode steps only)
    int prefill_launches = 250;  // > PrefillMinLaunches=200 → prefill
    int decode_launches = 10;    // < DecodeMaxLaunches=50 → decode
    int prefill_kernel_iters = 500;
    int prefill_period = 5;      // every 5th step is prefill, rest decode
    int step_delay_ms = 100;     // pause between steps so trace can attach + steps are slow enough to be meaningful

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            total_steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--slow-decode-at") && i + 1 < argc) {
            slow_decode_at = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--prefill-period") && i + 1 < argc) {
            prefill_period = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--step-delay-ms") && i + 1 < argc) {
            step_delay_ms = atoi(argv[++i]);
        }
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    const int N = 1024;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    int prefill_count = 0, decode_count = 0, slow_count = 0;
    fprintf(stderr, "infer_phases: starting %d steps "
                    "(prefill_period=%d, slow_decode_at=%d)\n",
            total_steps, prefill_period, slow_decode_at);

    for (int s = 0; s < total_steps && !stop_flag; s++) {
        // is_slow checked first so it overrides the prefill cadence
        // when the operator picks an injection step that happens to
        // align with `prefill_period`.
        int is_slow = (s == slow_decode_at);
        int is_prefill = !is_slow && (s % prefill_period) == 0;

        if (is_prefill) {
            // Prefill: many heavy launches, then sync.
            for (int k = 0; k < prefill_launches; k++) {
                prefill_kernel<<<8, 128>>>(d_out, prefill_kernel_iters, N);
            }
            prefill_count++;
        } else if (is_slow) {
            // Slow decode: same decode_kernel shape (few launches, thin
            // kernel) but sleep BEFORE the sync so the step duration is
            // dramatically longer than the decode-phase baseline. This
            // mimics a real slowdown (GPU contention, scheduler stall,
            // bus saturation) that the agent must catch — the phase
            // classifier sees decode-shape signals (low launch count,
            // tiny avg kernel, no NCCL) so the slow step lands in the
            // decode bucket and fires as 3x outlier against the decode
            // p95.
            for (int k = 0; k < decode_launches; k++) {
                decode_kernel<<<8, 128>>>(d_out, N);
            }
            usleep(150 * 1000); // 150ms — well over decode-phase p95
            slow_count++;
            fprintf(stderr, "infer_phases: step %d injected slow decode\n", s);
        } else {
            // Normal decode: few light launches, then sync.
            for (int k = 0; k < decode_launches; k++) {
                decode_kernel<<<8, 128>>>(d_out, N);
            }
            decode_count++;
        }
        CUDA_CHECK(cudaStreamSynchronize(0));
        if (step_delay_ms > 0) {
            usleep((useconds_t)step_delay_ms * 1000);
        }
    }

    cudaFree(d_out);
    fprintf(stderr, "infer_phases: prefill=%d decode=%d slow_decode=%d\n",
            prefill_count, decode_count, slow_count);
    return 0;
}
