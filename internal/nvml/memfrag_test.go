package nvml

import (
	"context"
	"errors"
	"testing"
)

func TestParseMemoryCSVTwoGPUs(t *testing.T) {
	out := []byte("GPU-aaaa, 4096, 12288, 16384\nGPU-bbbb, 2048, 6144, 8192\n")
	rs, err := parseMemoryCSV(out)
	if err != nil {
		t.Fatalf("parseMemoryCSV: %v", err)
	}
	if len(rs) != 2 {
		t.Fatalf("len = %d, want 2", len(rs))
	}
	if rs[0].UUID != "GPU-aaaa" || rs[0].UsedBytes != 4096*1024*1024 {
		t.Errorf("rs[0] = %+v", rs[0])
	}
	if rs[1].UUID != "GPU-bbbb" || rs[1].FreeBytes != 6144*1024*1024 {
		t.Errorf("rs[1] = %+v", rs[1])
	}
}

func TestParseMemoryCSVTrimsMiBSuffix(t *testing.T) {
	out := []byte("GPU-aaaa, 4096 MiB, 12288 MiB, 16384 MiB\n")
	rs, err := parseMemoryCSV(out)
	if err != nil {
		t.Fatalf("parseMemoryCSV: %v", err)
	}
	if len(rs) != 1 || rs[0].UsedBytes != 4096*1024*1024 {
		t.Fatalf("rs = %+v", rs)
	}
}

func TestParseMemoryCSVEmpty(t *testing.T) {
	if _, err := parseMemoryCSV([]byte("")); err == nil {
		t.Fatalf("empty input must return error, got nil")
	}
}

func TestParseComputeAppsCSVRows(t *testing.T) {
	out := []byte("GPU-aaaa, 12345, 1024\nGPU-aaaa, 12346, 2048\nGPU-bbbb, 12345, 4096\n")
	rs, err := parseComputeAppsCSV(out)
	if err != nil {
		t.Fatalf("parseComputeAppsCSV: %v", err)
	}
	if len(rs) != 3 {
		t.Fatalf("len = %d, want 3", len(rs))
	}
	if rs[0].PID != 12345 || rs[0].UUID != "GPU-aaaa" || rs[0].UsedBytes != 1024*1024*1024 {
		t.Errorf("rs[0] = %+v", rs[0])
	}
	// Same PID using two GPUs (multi-GPU job) - this is real and the
	// parser must not deduplicate.
	if rs[1].PID != 12346 || rs[2].PID != 12345 {
		t.Errorf("expected two rows for PID 12345 + one for 12346")
	}
}

func TestParseComputeAppsCSVNoJobs(t *testing.T) {
	rs, err := parseComputeAppsCSV([]byte(""))
	if err != nil {
		t.Fatalf("empty compute-apps must NOT error (no GPU jobs running is valid): %v", err)
	}
	if rs != nil {
		t.Fatalf("expected nil readings, got %+v", rs)
	}
}

func TestGetMemoryUsageRunnerError(t *testing.T) {
	want := errors.New("boom")
	run := func(context.Context) ([]byte, error) { return nil, want }
	if _, err := GetMemoryUsage(context.Background(), run); err == nil {
		t.Fatal("want runner error, got nil")
	}
}

func TestGetMemoryUsageNilRunner(t *testing.T) {
	if _, err := GetMemoryUsage(context.Background(), nil); err == nil {
		t.Fatal("want error for nil runner, got nil")
	}
}

func TestFragmentationEstimate(t *testing.T) {
	cases := []struct {
		name              string
		used, free, total int64
		want              float64
	}{
		{"unfragmented", 4 << 30, 12 << 30, 16 << 30, 0},
		{"fully accounted", 8 << 30, 8 << 30, 16 << 30, 0},
		{"half-accounted", 4 << 30, 4 << 30, 16 << 30, 0.5},
		{"zero total", 0, 0, 0, 0},
		{"over-accounted", 16 << 30, 8 << 30, 16 << 30, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := FragmentationEstimate(c.used, c.free, c.total)
			if got != c.want {
				t.Errorf("FragmentationEstimate(%d,%d,%d) = %v, want %v",
					c.used, c.free, c.total, got, c.want)
			}
		})
	}
}
