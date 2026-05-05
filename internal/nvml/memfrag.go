package nvml

// NVML memory poll wrapper for the v0.14 item D memfrag heuristic.
//
// Mirrors the throttle.go pattern: a Runner closure invokes
// nvidia-smi with the memory.used/free/total query; tests inject a
// fake runner so no real GPU is required.
//
// CAVEAT: this is a polling-based heuristic, NOT the IOCTL-level
// memfrag tracking that v0.15 W1 brings via a kprobe on
// nvidia_unlocked_ioctl. Burst-allocations shorter than the poll
// interval are missed by design; the OTLP metric description carries
// the same warning.

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// MemoryReading is one polled NVML memory snapshot for one GPU.
// Bytes are converted from MiB by the parser (nvidia-smi reports
// memory in MiB by default with units stripped via nounits).
type MemoryReading struct {
	UUID       string
	UsedBytes  int64
	FreeBytes  int64
	TotalBytes int64
}

// ComputeAppReading is one row from `nvidia-smi --query-compute-apps`,
// covering one (GPU, PID) pair currently executing GPU work.
type ComputeAppReading struct {
	UUID      string
	PID       uint32
	UsedBytes int64
}

// GetMemoryUsage returns one MemoryReading per visible GPU.
//
// The caller wires NewMemoryRunner() in production; tests pass a
// closure that returns a canned CSV byte slice.
func GetMemoryUsage(ctx context.Context, run Runner) ([]MemoryReading, error) {
	if run == nil {
		return nil, fmt.Errorf("nvml: nvidia-smi not available")
	}
	out, err := run(ctx)
	if err != nil {
		return nil, fmt.Errorf("nvml: nvidia-smi: %w", err)
	}
	return parseMemoryCSV(out)
}

// GetComputeApps returns one ComputeAppReading per (GPU, PID) pair.
//
// Mirrors GetMemoryUsage; uses a separate Runner because the query
// shape is different (5-column output rather than 4-column).
func GetComputeApps(ctx context.Context, run Runner) ([]ComputeAppReading, error) {
	if run == nil {
		return nil, fmt.Errorf("nvml: nvidia-smi not available")
	}
	out, err := run(ctx)
	if err != nil {
		return nil, fmt.Errorf("nvml: nvidia-smi: %w", err)
	}
	return parseComputeAppsCSV(out)
}

// maxMemoryOutput caps both parsers. ~80 bytes per GPU row plus
// generous headroom for compute-apps output (one row per running
// GPU job per GPU).
const maxMemoryOutput = 32 * 1024

// parseMemoryCSV parses
//
//	nvidia-smi --query-gpu=uuid,memory.used,memory.free,memory.total \
//	  --format=csv,noheader,nounits
//
// Each line is `GPU-<uuid>, <used_mib>, <free_mib>, <total_mib>`.
func parseMemoryCSV(out []byte) ([]MemoryReading, error) {
	if len(out) > maxMemoryOutput {
		return nil, fmt.Errorf("nvml: nvidia-smi memory output exceeds %d bytes", maxMemoryOutput)
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return nil, fmt.Errorf("nvml: nvidia-smi memory query returned empty output")
	}
	var readings []MemoryReading
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != 4 {
			return nil, fmt.Errorf("nvml: unexpected memory line %q", line)
		}
		uuid := strings.TrimSpace(parts[0])
		if uuid == "" {
			return nil, fmt.Errorf("nvml: empty uuid in line %q", line)
		}
		used, err := parseMiBField(parts[1])
		if err != nil {
			return nil, fmt.Errorf("nvml: parse used %q: %w", parts[1], err)
		}
		free, err := parseMiBField(parts[2])
		if err != nil {
			return nil, fmt.Errorf("nvml: parse free %q: %w", parts[2], err)
		}
		total, err := parseMiBField(parts[3])
		if err != nil {
			return nil, fmt.Errorf("nvml: parse total %q: %w", parts[3], err)
		}
		readings = append(readings, MemoryReading{
			UUID:       uuid,
			UsedBytes:  used,
			FreeBytes:  free,
			TotalBytes: total,
		})
	}
	if len(readings) == 0 {
		return nil, fmt.Errorf("nvml: no GPU rows parsed from memory query")
	}
	return readings, nil
}

// parseComputeAppsCSV parses
//
//	nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
//	  --format=csv,noheader,nounits
//
// One row per (GPU, PID) currently executing compute work. Empty
// output is valid (no GPU jobs running) - returns a nil slice rather
// than an error in that case.
func parseComputeAppsCSV(out []byte) ([]ComputeAppReading, error) {
	if len(out) > maxMemoryOutput {
		return nil, fmt.Errorf("nvml: nvidia-smi compute-apps output exceeds %d bytes", maxMemoryOutput)
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return nil, nil // no jobs running - not an error
	}
	var readings []ComputeAppReading
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != 3 {
			return nil, fmt.Errorf("nvml: unexpected compute-apps line %q", line)
		}
		uuid := strings.TrimSpace(parts[0])
		if uuid == "" {
			return nil, fmt.Errorf("nvml: empty uuid in line %q", line)
		}
		pidStr := strings.TrimSpace(parts[1])
		pid64, err := strconv.ParseUint(pidStr, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("nvml: parse pid %q: %w", pidStr, err)
		}
		used, err := parseMiBField(parts[2])
		if err != nil {
			return nil, fmt.Errorf("nvml: parse compute-apps used %q: %w", parts[2], err)
		}
		readings = append(readings, ComputeAppReading{
			UUID:      uuid,
			PID:       uint32(pid64),
			UsedBytes: used,
		})
	}
	return readings, nil
}

// parseMiBField parses a single MiB-unit field from nvidia-smi output.
// Accepts plain numbers (with --format=...nounits) and the
// "[Not Supported]" / "[N/A]" sentinels (returned as 0 with no error
// so the rest of the row still parses; absence is conveyed by the
// caller's later interpretation).
func parseMiBField(s string) (int64, error) {
	v := strings.TrimSpace(s)
	switch v {
	case "[Not Supported]", "[N/A]", "N/A":
		return 0, nil
	}
	// Strip an optional " MiB" suffix in case nounits is not honoured
	// by some nvidia-smi version.
	v = strings.TrimSuffix(v, " MiB")
	v = strings.TrimSuffix(v, "MiB")
	v = strings.TrimSpace(v)
	mib, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parse %q as MiB int: %w", s, err)
	}
	return mib * 1024 * 1024, nil
}

// NewMemoryRunner returns a Runner backed by `nvidia-smi` for the
// memory.used/free/total query. nil when nvidia-smi is not on PATH.
func NewMemoryRunner() Runner {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil || path == "" {
		return nil
	}
	return func(ctx context.Context) ([]byte, error) {
		ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		return exec.CommandContext(ctx, path,
			"--query-gpu=uuid,memory.used,memory.free,memory.total",
			"--format=csv,noheader,nounits",
		).Output()
	}
}

// NewComputeAppsRunner returns a Runner backed by `nvidia-smi` for the
// compute-apps query. nil when nvidia-smi is not on PATH.
func NewComputeAppsRunner() Runner {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil || path == "" {
		return nil
	}
	return func(ctx context.Context) ([]byte, error) {
		ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		return exec.CommandContext(ctx, path,
			"--query-compute-apps=gpu_uuid,pid,used_memory",
			"--format=csv,noheader,nounits",
		).Output()
	}
}

// FragmentationEstimate is a coarse heuristic that converts a
// (used, free, total) tuple into a number in [0, 1]. The intent is
// to flag "I have lots of free memory but allocations are failing"
// situations that an allocator-level fragmenting workload tends to
// produce.
//
// The current heuristic is intentionally simple: when used+free is
// less than total (i.e. NVML reports a larger total than allocator
// accounting), the gap is interpreted as fragmented overhead and
// scaled by total. When used+free covers total, we return 0.
//
// This will be REPLACED by event-driven IOCTL-level tracking in
// v0.15 W1; the v0.14 surface exists so dashboards have a baseline
// metric to pin to before v0.15 lands.
func FragmentationEstimate(used, free, total int64) float64 {
	if total <= 0 {
		return 0
	}
	accounted := used + free
	if accounted >= total {
		return 0
	}
	gap := total - accounted
	frac := float64(gap) / float64(total)
	if frac < 0 {
		return 0
	}
	if frac > 1 {
		return 1
	}
	return frac
}
