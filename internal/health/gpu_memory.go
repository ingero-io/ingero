// Package health - GPU memory reader via nvidia-smi subprocess.
//
// The health score's "memory" signal is headroom: free/total. Host RAM
// (/proc/meminfo) correlates with GPU-memory for training workloads that
// mirror tensors to system RAM, but diverges for inference and for
// workloads that pin memory only on-device. gpuMemReader reads actual
// GPU memory via `nvidia-smi`, summed across all visible GPUs.
//
// Design: the subprocess is invoked through an injectable run() closure
// so tests don't need a real nvidia-smi binary. When nvidia-smi is not
// on PATH at construction time, Read() returns an error without running
// anything and the caller falls back to the host-RAM proxy.
//
// v1.0 deliberately uses the subprocess path (zero new dependencies).
// v1.1 target is github.com/NVIDIA/go-nvml.
package health

import (
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// gpuMemReader reads aggregate GPU memory via nvidia-smi.
// Safe for concurrent use; each Read spawns its own subprocess.
type gpuMemReader struct {
	// run returns raw nvidia-smi stdout. Injected so tests can stub.
	// nil => reader is not available (nvidia-smi missing at construction).
	run     func(ctx context.Context) ([]byte, error)
	timeout time.Duration
	log     *slog.Logger
}

// newGPUMemReader looks up nvidia-smi once at construction. If absent,
// the returned reader's Read always errors with ErrGPUMemUnavailable.
func newGPUMemReader(log *slog.Logger) *gpuMemReader {
	if log == nil {
		log = slog.Default()
	}
	r := &gpuMemReader{
		timeout: 2 * time.Second,
		log:     log,
	}
	path, err := exec.LookPath("nvidia-smi")
	if err != nil || path == "" {
		return r
	}
	r.run = func(ctx context.Context) ([]byte, error) {
		return exec.CommandContext(ctx, path,
			"--query-gpu=memory.used,memory.total",
			"--format=csv,noheader,nounits",
		).Output()
	}
	return r
}

// Available reports whether Read can succeed on this host.
func (r *gpuMemReader) Available() bool { return r != nil && r.run != nil }

// Read returns aggregate used and total MB across all visible GPUs.
// Errors if nvidia-smi is missing, the subprocess fails, the output is
// malformed, or no GPUs are reported. Callers should fall back to a
// different memory signal on any error.
func (r *gpuMemReader) Read(ctx context.Context) (usedMB, totalMB int64, err error) {
	if r == nil || r.run == nil {
		return 0, 0, fmt.Errorf("nvidia-smi not available")
	}
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()

	out, err := r.run(ctx)
	if err != nil {
		return 0, 0, fmt.Errorf("nvidia-smi: %w", err)
	}
	return parseNvidiaSMI(out)
}

// maxNvidiaSMIOutput caps the size of stdout we will parse. One GPU row is
// roughly 20 bytes; 4 KiB is enough headroom for ~200 GPUs of valid data.
// A hostile or buggy nvidia-smi on a compromised node could otherwise
// return gigabytes of output, which strings.Split would fully materialize
// in RAM.
const maxNvidiaSMIOutput = 4 * 1024

// parseNvidiaSMI parses the CSV output of
//
//	nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
//
// which looks like:
//
//	4096, 16384
//	8192, 16384
//
// one line per GPU, values in MiB. Returns summed used/total across
// all GPUs. Output longer than maxNvidiaSMIOutput is rejected.
func parseNvidiaSMI(out []byte) (usedMB, totalMB int64, err error) {
	if len(out) > maxNvidiaSMIOutput {
		return 0, 0, fmt.Errorf("nvidia-smi output exceeds %d bytes", maxNvidiaSMIOutput)
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return 0, 0, fmt.Errorf("nvidia-smi returned empty output")
	}
	var seen int
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != 2 {
			return 0, 0, fmt.Errorf("unexpected nvidia-smi line: %q", line)
		}
		u, err := strconv.ParseInt(strings.TrimSpace(parts[0]), 10, 64)
		if err != nil {
			return 0, 0, fmt.Errorf("parse used MB %q: %w", parts[0], err)
		}
		t, err := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
		if err != nil {
			return 0, 0, fmt.Errorf("parse total MB %q: %w", parts[1], err)
		}
		if u < 0 || t <= 0 {
			return 0, 0, fmt.Errorf("nvidia-smi reported non-positive values: used=%d total=%d", u, t)
		}
		usedMB += u
		totalMB += t
		seen++
	}
	if seen == 0 {
		return 0, 0, fmt.Errorf("nvidia-smi reported no GPUs")
	}
	return usedMB, totalMB, nil
}
