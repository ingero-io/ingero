package mcp

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/store"
)

func TestFormatSessionsEmpty(t *testing.T) {
	text := formatSessions(nil, true)
	if !strings.Contains(text, "No trace sessions found") {
		t.Errorf("expected empty message, got: %s", text)
	}
}

func TestFormatSessionsTSC(t *testing.T) {
	start := time.Date(2026, 2, 25, 14, 0, 0, 0, time.UTC)
	stop := start.Add(60 * time.Second)
	sessions := []store.Session{{
		ID:        1,
		StartedAt: start,
		StoppedAt: stop,
		GPUModel:  "NVIDIA A100-SXM4-40GB",
		GPUDriver: "550.127.05",
		CPUModel:  "AMD EPYC 7713",
		CPUCores:  30,
		MemTotal:  200000,
		Kernel:    "5.15.0-100-generic",
		OSRelease: "Ubuntu 22.04.5 LTS",
		CUDAVer:   "12.4",
		PythonVer: "3.10.12",
		IngeroVer: "v0.6 (commit: abc1234)",
		PIDFilter: "32574",
		Flags:     "stack,record",
	}}

	text := formatSessions(sessions, true)
	var parsed []map[string]interface{}
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatalf("TSC output not valid JSON: %v\ntext: %s", err, text)
	}
	if len(parsed) != 1 {
		t.Fatalf("expected 1 session, got %d", len(parsed))
	}
	m := parsed[0]
	if m["gpu"] != "NVIDIA A100-SXM4-40GB" {
		t.Errorf("gpu = %v", m["gpu"])
	}
	if m["dur"] != "60s" {
		t.Errorf("dur = %v", m["dur"])
	}
	if m["pids"] != "32574" {
		t.Errorf("pids = %v", m["pids"])
	}
}

func TestFormatSessionsVerbose(t *testing.T) {
	start := time.Date(2026, 2, 25, 14, 0, 0, 0, time.UTC)
	sessions := []store.Session{{
		ID:        1,
		StartedAt: start,
		GPUModel:  "NVIDIA A100-SXM4-40GB",
		CPUModel:  "AMD EPYC 7713",
		CPUCores:  30,
		IngeroVer: "v0.6",
	}}

	text := formatSessions(sessions, false)
	if !strings.Contains(text, "1 trace session(s)") {
		t.Errorf("expected session count header, got: %s", text)
	}
	if !strings.Contains(text, "NVIDIA A100-SXM4-40GB") {
		t.Errorf("expected GPU model in output")
	}
	if !strings.Contains(text, "(still running)") {
		t.Errorf("expected 'still running' for zero StoppedAt")
	}
}
