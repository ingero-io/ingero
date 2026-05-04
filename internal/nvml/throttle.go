// Package nvml exposes a minimal, pure-Go wrapper around NVML clock-throttle
// reasons via a `nvidia-smi` subprocess.
//
// The agent has no go-nvml dependency today (see internal/health/gpu_memory.go
// for the same justification). The throttle bitmask is a stable nvidia-smi
// query field, so the subprocess form is contract-equivalent to the cgo
// `nvmlDeviceGetCurrentClocksThrottleReasons` call:
//
//	nvidia-smi --query-gpu=uuid,clocks_throttle_reasons.active \
//	  --format=csv,noheader
package nvml

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// ErrNotSupported is returned for a single device when nvidia-smi reports
// "[Not Supported]" or "[N/A]" for the throttle field. Consumer GPUs return
// this; callers log once and skip the device without panicking.
var ErrNotSupported = errors.New("clocks_throttle_reasons not supported on this device")

// Reading is one decoded throttle sample for one GPU. Bitmask carries the
// unaltered NVML clock-throttle reason mask; Buckets is the bit-to-bucket
// decode produced by DecodeReasons (see decoder.go for the mapping table).
type Reading struct {
	UUID    string
	Bitmask uint64
	Buckets ThrottleBuckets
	// Err is non-nil only for "skip this device" cases (ErrNotSupported).
	// A wholly-failed nvidia-smi invocation is reported via the wrapper's
	// returned error, not via per-Reading Err.
	Err error
}

// Runner is the injectable subprocess closure. nil means nvidia-smi is not
// on PATH; callers treat that as "no NVML readings".
type Runner func(ctx context.Context) ([]byte, error)

// GetCurrentClocksThrottleReasons returns one Reading per visible GPU.
//
// Mirrors the cgo signature `nvmlDeviceGetCurrentClocksThrottleReasons` for
// every GPU in one call. Implementation uses `nvidia-smi`; tests stub the
// invocation via the runner closure so no real GPU is required.
func GetCurrentClocksThrottleReasons(ctx context.Context, run Runner) ([]Reading, error) {
	if run == nil {
		return nil, fmt.Errorf("nvml: nvidia-smi not available")
	}
	out, err := run(ctx)
	if err != nil {
		return nil, fmt.Errorf("nvml: nvidia-smi: %w", err)
	}
	return parseThrottleCSV(out)
}

// maxThrottleOutput caps the parser. ~80 bytes per GPU row (UUID + bitmask);
// 8 KiB allows ~100 GPUs of valid data with headroom.
const maxThrottleOutput = 8 * 1024

// parseThrottleCSV parses the CSV output of:
//
//	nvidia-smi --query-gpu=uuid,clocks_throttle_reasons.active \
//	  --format=csv,noheader
//
// Each line is `GPU-<uuid>, 0xN` (hex bitmask) or `GPU-<uuid>, [Not Supported]`.
// Returns one Reading per line; unsupported devices set Err=ErrNotSupported.
func parseThrottleCSV(out []byte) ([]Reading, error) {
	if len(out) > maxThrottleOutput {
		return nil, fmt.Errorf("nvml: nvidia-smi output exceeds %d bytes", maxThrottleOutput)
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return nil, fmt.Errorf("nvml: nvidia-smi returned empty output")
	}
	var readings []Reading
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != 2 {
			return nil, fmt.Errorf("nvml: unexpected line %q", line)
		}
		uuid := strings.TrimSpace(parts[0])
		raw := strings.TrimSpace(parts[1])
		if uuid == "" {
			return nil, fmt.Errorf("nvml: empty uuid in line %q", line)
		}
		switch raw {
		case "[Not Supported]", "[N/A]", "N/A":
			readings = append(readings, Reading{UUID: uuid, Err: ErrNotSupported})
			continue
		}
		// nvidia-smi can emit "0x..." or plain decimal depending on driver.
		// Strip a leading 0x/0X to normalise.
		base := 10
		num := raw
		if strings.HasPrefix(num, "0x") || strings.HasPrefix(num, "0X") {
			num = num[2:]
			base = 16
		}
		v, err := strconv.ParseUint(num, base, 64)
		if err != nil {
			return nil, fmt.Errorf("nvml: parse bitmask %q: %w", raw, err)
		}
		readings = append(readings, Reading{
			UUID:    uuid,
			Bitmask: v,
			Buckets: DecodeReasons(v),
		})
	}
	if len(readings) == 0 {
		return nil, fmt.Errorf("nvml: no GPU rows parsed")
	}
	return readings, nil
}

// NewSubprocessRunner returns a runner backed by `nvidia-smi`. Returns nil
// when nvidia-smi is not on PATH; callers should treat that as "no NVML".
// The 2 s timeout matches gpuMemReader (internal/health/gpu_memory.go).
func NewSubprocessRunner() Runner {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil || path == "" {
		return nil
	}
	return func(ctx context.Context) ([]byte, error) {
		ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		return exec.CommandContext(ctx, path,
			"--query-gpu=uuid,clocks_throttle_reasons.active",
			"--format=csv,noheader",
		).Output()
	}
}
