package mcp

import (
	"testing"
)

func TestTSCKey(t *testing.T) {
	tests := []struct {
		key  string
		tsc  bool
		want string
	}{
		{"timestamp", true, "t"},
		{"timestamp", false, "timestamp"},
		{"pid", true, "p"},
		{"duration_us", true, "d_us"},
		{"unknown_key", true, "unknown_key"},
		{"severity", true, "sev"},
		{"root_cause", true, "rc"},
	}
	for _, tt := range tests {
		got := TSCKey(tt.key, tt.tsc)
		if got != tt.want {
			t.Errorf("TSCKey(%q, %v) = %q, want %q", tt.key, tt.tsc, got, tt.want)
		}
	}
}

func TestTSCMap(t *testing.T) {
	// TSC on — abbreviated keys.
	m := TSCMap(true, "timestamp", "15:41:22", "pid", 4821, "operation", "cudaStreamSync")
	if m["t"] != "15:41:22" {
		t.Errorf("expected t=15:41:22, got %v", m["t"])
	}
	if m["p"] != 4821 {
		t.Errorf("expected p=4821, got %v", m["p"])
	}
	if m["op"] != "cudaStreamSync" {
		t.Errorf("expected op=cudaStreamSync, got %v", m["op"])
	}

	// TSC off — verbose keys.
	m2 := TSCMap(false, "timestamp", "15:41:22", "pid", 4821)
	if m2["timestamp"] != "15:41:22" {
		t.Errorf("expected timestamp=15:41:22, got %v", m2["timestamp"])
	}
	if m2["pid"] != 4821 {
		t.Errorf("expected pid=4821, got %v", m2["pid"])
	}
}

func TestTSCReverseMap(t *testing.T) {
	// Verify every forward mapping has a reverse.
	for k, v := range tscFieldMap {
		if tscReverseMap[v] != k {
			t.Errorf("reverse map: tscReverseMap[%q] = %q, want %q", v, tscReverseMap[v], k)
		}
	}
}
