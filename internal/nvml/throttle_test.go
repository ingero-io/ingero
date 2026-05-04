package nvml

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
)

// TestParseThrottleCSV_TwoGPUs asserts per-GPU labels: a fake 2-GPU device
// list must produce two readings with distinct UUIDs and the raw bitmask
// preserved. (Bucket decoding is covered in decoder tests.)
func TestParseThrottleCSV_TwoGPUs(t *testing.T) {
	out := []byte("GPU-aaaaaaaa-1111-2222-3333-444444444444, 0x4\n" +
		"GPU-bbbbbbbb-5555-6666-7777-888888888888, 0x40\n")
	rs, err := parseThrottleCSV(out)
	if err != nil {
		t.Fatalf("parseThrottleCSV: %v", err)
	}
	if len(rs) != 2 {
		t.Fatalf("want 2 readings, got %d", len(rs))
	}
	uuids := map[string]bool{}
	for _, r := range rs {
		if r.Err != nil {
			t.Fatalf("unexpected err on %q: %v", r.UUID, r.Err)
		}
		uuids[r.UUID] = true
	}
	if len(uuids) != 2 {
		t.Fatalf("expected 2 distinct UUIDs, got %v", uuids)
	}
	if rs[0].Bitmask != 0x4 || rs[1].Bitmask != 0x40 {
		t.Fatalf("bitmasks wrong: %+v", rs)
	}
	// SwPowerCap on GPU 1 -> Power+SW true.
	if !rs[0].Buckets.Power || !rs[0].Buckets.SW {
		t.Fatalf("GPU1 expected Power+SW, got %+v", rs[0].Buckets)
	}
	// HwThermalSlowdown on GPU 2 -> Thermal+HW true.
	if !rs[1].Buckets.Thermal || !rs[1].Buckets.HW {
		t.Fatalf("GPU2 expected Thermal+HW, got %+v", rs[1].Buckets)
	}
}

// TestParseThrottleCSV_NotSupported covers the consumer-GPU error path
// (★3 H6). The reading carries ErrNotSupported and the poller is expected
// to skip metric emission for that device without panicking.
func TestParseThrottleCSV_NotSupported(t *testing.T) {
	out := []byte("GPU-aaaa, [Not Supported]\nGPU-bbbb, 0x4\n")
	rs, err := parseThrottleCSV(out)
	if err != nil {
		t.Fatalf("parseThrottleCSV: %v", err)
	}
	if len(rs) != 2 {
		t.Fatalf("want 2 readings, got %d", len(rs))
	}
	if !errors.Is(rs[0].Err, ErrNotSupported) {
		t.Fatalf("GPU1 expected ErrNotSupported, got %v", rs[0].Err)
	}
	if rs[1].Err != nil {
		t.Fatalf("GPU2 expected no error, got %v", rs[1].Err)
	}
	if rs[1].Bitmask != 0x4 {
		t.Fatalf("GPU2 bitmask wrong: %+v", rs[1])
	}
}

// TestParseThrottleCSV_DecimalBitmask covers older drivers that emit a
// decimal bitmask instead of "0x...".
func TestParseThrottleCSV_DecimalBitmask(t *testing.T) {
	rs, err := parseThrottleCSV([]byte("GPU-aaaa, 64\n"))
	if err != nil {
		t.Fatalf("parseThrottleCSV: %v", err)
	}
	if len(rs) != 1 || rs[0].Bitmask != 64 {
		t.Fatalf("decimal 64 expected bitmask 64, got %+v", rs)
	}
}

// TestParseThrottleCSV_Empty rejects empty output as a programming error
// rather than silently emitting no metrics.
func TestParseThrottleCSV_Empty(t *testing.T) {
	if _, err := parseThrottleCSV([]byte("")); err == nil {
		t.Fatalf("expected error on empty output")
	}
	if _, err := parseThrottleCSV([]byte("   \n")); err == nil {
		t.Fatalf("expected error on whitespace-only output")
	}
}

// TestParseThrottleCSV_OversizedOutput protects against a hostile or
// runaway nvidia-smi binary returning megabytes of data.
func TestParseThrottleCSV_OversizedOutput(t *testing.T) {
	big := strings.Repeat("GPU-aa, 0x4\n", 1024) // ~12 KiB > 8 KiB cap
	if _, err := parseThrottleCSV([]byte(big)); err == nil {
		t.Fatalf("expected error on oversized output")
	}
}

// TestGetCurrentClocksThrottleReasons_NilRunner covers the no-nvidia-smi
// case: the API returns a clear error rather than panicking.
func TestGetCurrentClocksThrottleReasons_NilRunner(t *testing.T) {
	if _, err := GetCurrentClocksThrottleReasons(context.Background(), nil); err == nil {
		t.Fatalf("expected error when runner is nil")
	}
}

// TestGetCurrentClocksThrottleReasons_RunError surfaces subprocess errors
// as a wrapped error from the wrapper.
func TestGetCurrentClocksThrottleReasons_RunError(t *testing.T) {
	wantErr := fmt.Errorf("synthetic")
	r := func(ctx context.Context) ([]byte, error) { return nil, wantErr }
	_, err := GetCurrentClocksThrottleReasons(context.Background(), r)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !errors.Is(err, wantErr) {
		t.Fatalf("error did not wrap synthetic: %v", err)
	}
}

// TestGetCurrentClocksThrottleReasons_OK end-to-end through the wrapper
// with a stubbed runner. Confirms two GPUs decoded correctly and the
// label dimension carries through.
func TestGetCurrentClocksThrottleReasons_OK(t *testing.T) {
	r := func(ctx context.Context) ([]byte, error) {
		return []byte("GPU-aaaa, 0x4\nGPU-bbbb, 0x40\n"), nil
	}
	rs, err := GetCurrentClocksThrottleReasons(context.Background(), r)
	if err != nil {
		t.Fatalf("GetCurrentClocksThrottleReasons: %v", err)
	}
	if len(rs) != 2 {
		t.Fatalf("want 2 readings, got %d", len(rs))
	}
	if rs[0].UUID != "GPU-aaaa" || rs[1].UUID != "GPU-bbbb" {
		t.Fatalf("UUIDs not preserved: %+v", rs)
	}
}
