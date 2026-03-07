package sysinfo

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// writeTempFile creates a temp file with the given content and returns its path.
func writeTempFile(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return path
}

func TestParseCPULine(t *testing.T) {
	tests := []struct {
		name    string
		line    string
		wantErr bool
	}{
		{
			name: "normal",
			line: "cpu  10132153 290696 3084719 46828483 16683 0 25195 0 0 0",
		},
		{
			name: "minimal 8 fields",
			line: "cpu  100 200 300 400 500 600 700 0",
		},
		{
			name:    "too few fields",
			line:    "cpu  100 200",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			j, err := parseCPULine(tt.line)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if j.total() == 0 {
				t.Error("total should be > 0")
			}
		})
	}
}

func TestReadCPU(t *testing.T) {
	content := `cpu  10132153 290696 3084719 46828483 16683 0 25195 0 0 0
cpu0 1393280 32966 572056 13343292 6130 0 17875 0 0 0
`
	c := New()
	c.procStat = writeTempFile(t, "stat", content)

	j, err := c.readCPU()
	if err != nil {
		t.Fatalf("readCPU: %v", err)
	}
	if j.user != 10132153 {
		t.Errorf("user = %d, want 10132153", j.user)
	}
	if j.idle != 46828483 {
		t.Errorf("idle = %d, want 46828483", j.idle)
	}
}

func TestReadMeminfo(t *testing.T) {
	content := `MemTotal:       16384000 kB
MemFree:         1234567 kB
MemAvailable:   11468000 kB
Buffers:          234567 kB
Cached:          3456789 kB
SwapTotal:       2097152 kB
SwapFree:         524288 kB
`
	c := New()
	c.procMeminfo = writeTempFile(t, "meminfo", content)

	m, err := c.readMeminfo()
	if err != nil {
		t.Fatalf("readMeminfo: %v", err)
	}
	if m.totalKB != 16384000 {
		t.Errorf("totalKB = %d, want 16384000", m.totalKB)
	}
	if m.availKB != 11468000 {
		t.Errorf("availKB = %d, want 11468000", m.availKB)
	}
	if m.swapTotalKB != 2097152 {
		t.Errorf("swapTotalKB = %d, want 2097152", m.swapTotalKB)
	}
	if m.swapFreeKB != 524288 {
		t.Errorf("swapFreeKB = %d, want 524288", m.swapFreeKB)
	}
}

func TestReadLoadavg(t *testing.T) {
	content := "3.21 2.45 1.89 4/1234 56789\n"
	c := New()
	c.procLoadavg = writeTempFile(t, "loadavg", content)

	l1, l5, err := c.readLoadavg()
	if err != nil {
		t.Fatalf("readLoadavg: %v", err)
	}
	if l1 != 3.21 {
		t.Errorf("load1 = %f, want 3.21", l1)
	}
	if l5 != 2.45 {
		t.Errorf("load5 = %f, want 2.45", l5)
	}
}

func TestReadPageFaults(t *testing.T) {
	content := `pgpgin 12345678
pgpgout 23456789
pgfault 345678901
pgmajfault 42000
`
	c := New()
	c.procVmstat = writeTempFile(t, "vmstat", content)

	pf, err := c.readPageFaults()
	if err != nil {
		t.Fatalf("readPageFaults: %v", err)
	}
	if pf != 42000 {
		t.Errorf("pgmajfault = %d, want 42000", pf)
	}
}

func TestCPUDelta(t *testing.T) {
	// First read: 100 user, 900 idle (10% CPU).
	stat1 := "cpu  100 0 0 900 0 0 0 0 0 0\n"
	// Second read: 200 user, 1800 idle → delta = 100 user, 900 idle → 10% CPU.
	stat2 := "cpu  200 0 0 1800 0 0 0 0 0 0\n"

	dir := t.TempDir()
	statPath := filepath.Join(dir, "stat")

	c := New()
	c.procStat = statPath
	c.procLoadavg = writeTempFile(t, "loadavg", "0 0 0 0/0 0\n")
	c.procMeminfo = writeTempFile(t, "meminfo", "MemTotal: 16384000 kB\nMemAvailable: 8192000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n")
	c.procVmstat = writeTempFile(t, "vmstat", "pgmajfault 100\n")

	// First poll (baseline).
	os.WriteFile(statPath, []byte(stat1), 0644)
	c.poll()
	if c.snapshot.CPUPercent != 0 {
		t.Errorf("first poll CPU should be 0 (no delta), got %f", c.snapshot.CPUPercent)
	}

	// Second poll (delta).
	os.WriteFile(statPath, []byte(stat2), 0644)
	c.poll()
	// Delta: total went from 1000 to 2000 (+1000). Idle went from 900 to 1800 (+900).
	// CPU% = (1000-900)/1000 * 100 = 10%.
	if c.snapshot.CPUPercent != 10.0 {
		t.Errorf("CPU = %.1f%%, want 10.0%%", c.snapshot.CPUPercent)
	}
}

func TestMemoryPercent(t *testing.T) {
	c := New()
	c.procStat = writeTempFile(t, "stat", "cpu  0 0 0 0 0 0 0 0 0 0\n")
	c.procLoadavg = writeTempFile(t, "loadavg", "0 0 0 0/0 0\n")
	c.procMeminfo = writeTempFile(t, "meminfo",
		"MemTotal: 16000 kB\nMemAvailable: 4000 kB\nSwapTotal: 1024 kB\nSwapFree: 0 kB\n")
	c.procVmstat = writeTempFile(t, "vmstat", "pgmajfault 0\n")

	c.poll()
	c.poll() // Need 2 polls for initialized=true.

	snap := c.Snapshot()
	// 12000 used / 16000 total = 75%.
	if snap.MemUsedPct != 75.0 {
		t.Errorf("MemUsedPct = %.1f%%, want 75.0%%", snap.MemUsedPct)
	}
	if snap.MemAvailMB != 3 { // 4000 KB / 1024 = 3 MB.
		t.Errorf("MemAvailMB = %d, want 3", snap.MemAvailMB)
	}
	if snap.SwapUsedMB != 1 { // 1024 KB / 1024 = 1 MB.
		t.Errorf("SwapUsedMB = %d, want 1", snap.SwapUsedMB)
	}
}

func TestPageFaultDelta(t *testing.T) {
	dir := t.TempDir()
	vmstatPath := filepath.Join(dir, "vmstat")

	c := New()
	c.procStat = writeTempFile(t, "stat", "cpu  0 0 0 0 0 0 0 0 0 0\n")
	c.procLoadavg = writeTempFile(t, "loadavg", "0 0 0 0/0 0\n")
	c.procMeminfo = writeTempFile(t, "meminfo", "MemTotal: 16000 kB\nMemAvailable: 8000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n")
	c.procVmstat = vmstatPath

	os.WriteFile(vmstatPath, []byte("pgmajfault 1000\n"), 0644)
	c.poll()

	os.WriteFile(vmstatPath, []byte("pgmajfault 1042\n"), 0644)
	c.poll()

	snap := c.Snapshot()
	if snap.PageFaults != 42 {
		t.Errorf("PageFaults = %d, want 42", snap.PageFaults)
	}
}

func TestIsAbnormal(t *testing.T) {
	tests := []struct {
		name string
		snap SystemSnapshot
		want bool
	}{
		{"normal", SystemSnapshot{CPUPercent: 50, MemUsedPct: 70}, false},
		{"high cpu", SystemSnapshot{CPUPercent: 95}, true},
		{"high mem", SystemSnapshot{MemUsedPct: 97}, true},
		{"swap", SystemSnapshot{SwapUsedMB: 100}, true},
		{"high load", SystemSnapshot{LoadAvg1: 12}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.snap.IsAbnormal(); got != tt.want {
				t.Errorf("IsAbnormal() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestStartStop(t *testing.T) {
	c := New()
	c.procStat = writeTempFile(t, "stat", "cpu  100 0 0 900 0 0 0 0 0 0\n")
	c.procLoadavg = writeTempFile(t, "loadavg", "1.5 1.2 0.9 2/100 1234\n")
	c.procMeminfo = writeTempFile(t, "meminfo", "MemTotal: 16000 kB\nMemAvailable: 8000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n")
	c.procVmstat = writeTempFile(t, "vmstat", "pgmajfault 0\n")

	c.Start()
	// Give the goroutine time to do at least one poll.
	time.Sleep(100 * time.Millisecond)

	snap := c.Snapshot()
	if snap.LoadAvg1 != 1.5 {
		t.Errorf("LoadAvg1 = %f, want 1.5", snap.LoadAvg1)
	}

	c.Stop()
}

func TestParseKBValue(t *testing.T) {
	tests := []struct {
		line string
		want int64
	}{
		{"MemTotal:       16384000 kB", 16384000},
		{"MemAvailable:   8192000 kB", 8192000},
		{"SwapTotal:             0 kB", 0},
		{"Bad:", 0},
	}
	for _, tt := range tests {
		if got := parseKBValue(tt.line); got != tt.want {
			t.Errorf("parseKBValue(%q) = %d, want %d", tt.line, got, tt.want)
		}
	}
}
