package cli

import (
	"strings"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/correlate"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

// ---------------------------------------------------------------------------
// formatDuration tests
// ---------------------------------------------------------------------------

// TestFormatDuration verifies human-readable duration formatting.
func TestFormatDuration(t *testing.T) {
	tests := []struct {
		name string
		dur  time.Duration
		want string
	}{
		// Zero
		{"zero", 0, "0"},

		// Nanoseconds (< 1µs)
		{"1ns", 1 * time.Nanosecond, "1ns"},
		{"500ns", 500 * time.Nanosecond, "500ns"},
		{"999ns", 999 * time.Nanosecond, "999ns"},

		// Microseconds (< 1ms)
		{"1us", 1 * time.Microsecond, "1.0us"},
		{"1.5us", 1500 * time.Nanosecond, "1.5us"},
		{"9.9us", 9900 * time.Nanosecond, "9.9us"},
		{"10us", 10 * time.Microsecond, "10us"},
		{"456us", 456 * time.Microsecond, "456us"},
		{"999us", 999 * time.Microsecond, "999us"},

		// Milliseconds (< 1s)
		{"1ms", 1 * time.Millisecond, "1.0ms"},
		{"1.5ms", 1500 * time.Microsecond, "1.5ms"},
		{"9.9ms", 9900 * time.Microsecond, "9.9ms"},
		{"10ms", 10 * time.Millisecond, "10ms"},
		{"456ms", 456 * time.Millisecond, "456ms"},

		// Seconds (< 1min)
		{"1s", 1 * time.Second, "1.0s"},
		{"1.5s", 1500 * time.Millisecond, "1.5s"},
		{"9.9s", 9900 * time.Millisecond, "9.9s"},
		{"10s", 10 * time.Second, "10s"},
		{"45s", 45 * time.Second, "45s"},

		// Minutes (>= 1min)
		{"1m0s", 1 * time.Minute, "1m0s"},
		{"1m30s", 90 * time.Second, "1m30s"},
		{"5m0s", 5 * time.Minute, "5m0s"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatDuration(tt.dur)
			if got != tt.want {
				t.Errorf("formatDuration(%v) = %q, want %q", tt.dur, got, tt.want)
			}
		})
	}
}

// TestDebugf verifies the debug helper doesn't panic and respects debugMode.
func TestDebugf(t *testing.T) {
	// debugMode=false: no output, no panic.
	debugMode = false
	debugf("should not appear: %d", 42)

	// debugMode=true: writes to stderr, no panic.
	debugMode = true
	debugf("test message: %s %d", "hello", 42)
	debugMode = false
}

// ---------------------------------------------------------------------------
// shouldStore tests — validates the selective storage decision hierarchy
// ---------------------------------------------------------------------------

// TestShouldStoreHierarchy tests all 7 tiers of the shouldStore() decision
// hierarchy using a table-driven approach. Each tier is tested independently.
//
// Teaching note: Go's table-driven test pattern ([]struct{...} + t.Run) is
// the idiomatic way to test functions with many input combinations. Each
// sub-test gets its own t.Run name, so failures pinpoint the exact case.
func TestShouldStoreHierarchy(t *testing.T) {
	// Build a collector with baseline data so IsAnomaly() works.
	// We need ≥10 samples per op + a Snapshot() to update cachedP50.
	collector := stats.New()
	for i := 0; i < 100; i++ {
		collector.Record(events.Event{
			Timestamp: time.Now(),
			PID:       1000,
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDALaunchKernel),
			Duration:  10 * time.Microsecond,
		})
		collector.Record(events.Event{
			Timestamp: time.Now(),
			PID:       1000,
			Source:    events.SourceHost,
			Op:        uint8(events.HostPageAlloc),
			Duration:  5 * time.Microsecond,
		})
		collector.Record(events.Event{
			Timestamp: time.Now(),
			PID:       1000,
			Source:    events.SourceHost,
			Op:        uint8(events.HostSchedWakeup),
			Duration:  2 * time.Microsecond,
		})
		collector.Record(events.Event{
			Timestamp: time.Now(),
			PID:       1000,
			Source:    events.SourceDriver,
			Op:        uint8(events.DriverLaunchKernel),
			Duration:  8 * time.Microsecond,
		})
	}
	collector.Snapshot() // updates cachedP50 for all ops

	pastBootstrap := time.Now().Add(-1 * time.Minute) // well past 10s

	tests := []struct {
		name     string
		evt      events.Event
		start    time.Time
		recAll   bool
		wantStore bool
	}{
		// Tier 1: --record-all bypasses all filtering.
		{
			name:      "recordAll/cuLaunchKernel",
			evt:       events.Event{Source: events.SourceCUDA, Op: uint8(events.CUDALaunchKernel), Duration: 10 * time.Microsecond, PID: 1000},
			start:     pastBootstrap,
			recAll:    true,
			wantStore: true,
		},

		// Tier 2: Bootstrap window (first 10s) stores everything.
		{
			name:      "bootstrap/cuLaunchKernel",
			evt:       events.Event{Source: events.SourceCUDA, Op: uint8(events.CUDALaunchKernel), Duration: 10 * time.Microsecond, PID: 1000},
			start:     time.Now(), // just started
			recAll:    false,
			wantStore: true,
		},
		{
			name:      "bootstrap/sched_wakeup",
			evt:       events.Event{Source: events.SourceHost, Op: uint8(events.HostSchedWakeup), Duration: 2 * time.Microsecond, PID: 1000},
			start:     time.Now(),
			recAll:    false,
			wantStore: true,
		},

		// Tier 3: Process lifecycle always stored.
		{
			name:      "lifecycle/process_exec",
			evt:       events.Event{Source: events.SourceHost, Op: uint8(events.HostProcessExec), PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},
		{
			name:      "lifecycle/process_exit",
			evt:       events.Event{Source: events.SourceHost, Op: uint8(events.HostProcessExit), PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},
		{
			name:      "lifecycle/process_fork",
			evt:       events.Event{Source: events.SourceHost, Op: uint8(events.HostProcessFork), PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},
		{
			name:      "lifecycle/oom_kill",
			evt:       events.Event{Source: events.SourceHost, Op: uint8(events.HostOOMKill), PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},

		// Tier 4: sched_switch always stored (causal chain critical).
		{
			name:      "sched_switch",
			evt:       events.Event{Source: events.SourceHost, Op: uint8(events.HostSchedSwitch), Duration: 50 * time.Microsecond, PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},

		// Tier 4b: mm_page_alloc always stored (causal chain critical for
		// memory pressure detection — chain engine needs Args[0] bytes).
		// Real mm_page_alloc events have Duration=0; alloc bytes in Args[0].
		{
			name:      "mm_page_alloc",
			evt:       events.Event{Source: events.SourceHost, Op: uint8(events.HostPageAlloc), Duration: 0, PID: 1000, Args: [2]uint64{16 * 1024 * 1024, 0}},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},

		// Tier 5: Sync ops always stored (latency symptoms).
		{
			name:      "sync/cudaStreamSync",
			evt:       events.Event{Source: events.SourceCUDA, Op: uint8(events.CUDAStreamSync), Duration: 1 * time.Millisecond, PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},
		{
			name:      "sync/cudaDeviceSync",
			evt:       events.Event{Source: events.SourceCUDA, Op: uint8(events.CUDADeviceSync), Duration: 1 * time.Millisecond, PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},
		{
			name:      "sync/cuCtxSynchronize",
			evt:       events.Event{Source: events.SourceDriver, Op: uint8(events.DriverCtxSync), Duration: 1 * time.Millisecond, PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},

		// Tier 6: Anomalous events (duration > 3x p50) stored.
		{
			name:      "anomaly/cuLaunchKernel_100x",
			evt:       events.Event{Source: events.SourceCUDA, Op: uint8(events.CUDALaunchKernel), Duration: 1 * time.Millisecond, PID: 1000}, // 100x the 10us baseline
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},
		{
			name:      "anomaly/driverLaunchKernel_100x",
			evt:       events.Event{Source: events.SourceDriver, Op: uint8(events.DriverLaunchKernel), Duration: 800 * time.Microsecond, PID: 1000}, // 100x the 8us baseline
			start:     pastBootstrap,
			recAll:    false,
			wantStore: true,
		},

		// Tier 7: Normal events → aggregate only (NOT stored).
		{
			name:      "aggregate/cuLaunchKernel_normal",
			evt:       events.Event{Source: events.SourceCUDA, Op: uint8(events.CUDALaunchKernel), Duration: 10 * time.Microsecond, PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: false,
		},
		{
			name:      "aggregate/sched_wakeup_normal",
			evt:       events.Event{Source: events.SourceHost, Op: uint8(events.HostSchedWakeup), Duration: 2 * time.Microsecond, PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: false,
		},
		{
			name:      "aggregate/driverLaunchKernel_normal",
			evt:       events.Event{Source: events.SourceDriver, Op: uint8(events.DriverLaunchKernel), Duration: 8 * time.Microsecond, PID: 1000},
			start:     pastBootstrap,
			recAll:    false,
			wantStore: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldStore(tt.evt, tt.start, tt.recAll, collector, 0, nil)
			if got != tt.wantStore {
				t.Errorf("shouldStore() = %v, want %v", got, tt.wantStore)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Selective storage → causal chain integration tests
// ---------------------------------------------------------------------------

// TestSelectiveStoragePreservesSchedSwitchChain verifies that events preserved
// by selective storage are sufficient for sched_switch → cudaStreamSync causal
// chain detection during REPLAY (ingero explain / MCP get_causal_chains).
//
// Teaching note: This is a cross-package integration test. During live tracing,
// the correlator sees ALL events and chains work fine. But during replay
// (explain/MCP), only SQLite-stored events are available. This test verifies
// the stored subset is sufficient for the chain engine.
//
// The synthetic event stream has two phases:
//   Phase 1 (baseline): normal cudaStreamSync at 1ms + bulk cuLaunchKernel
//   Phase 2 (anomaly):  sched_switch burst → cudaStreamSync tail spike at 500ms
//
// shouldStore keeps: all sched_switch + all cudaStreamSync (always-store rules)
// shouldStore drops: cuLaunchKernel (normal, aggregated)
// Replay of stored events should still detect the chain.
func TestSelectiveStoragePreservesSchedSwitchChain(t *testing.T) {
	pid := uint32(1000)
	now := time.Now()

	// Collector for shouldStore's anomaly detection (simulates live mode).
	liveCollector := stats.New()

	var allEvents []events.Event

	// Phase 1: 50 normal cudaStreamSync (establish 1ms baseline).
	// Spread over 0-500ms so the timeline spans >1s total, ensuring
	// ReplayEventsForChains' 1-second windowed snapshot fires.
	for i := 0; i < 50; i++ {
		allEvents = append(allEvents, events.Event{
			Timestamp: now.Add(time.Duration(i) * 10 * time.Millisecond),
			PID:       pid,
			TID:       pid,
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDAStreamSync),
			Duration:  1 * time.Millisecond,
		})
	}

	// Phase 1: 200 normal cuLaunchKernel (bulk, should be aggregated).
	for i := 0; i < 200; i++ {
		allEvents = append(allEvents, events.Event{
			Timestamp: now.Add(time.Duration(500+i) * time.Millisecond),
			PID:       pid,
			TID:       pid,
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDALaunchKernel),
			Duration:  10 * time.Microsecond,
		})
	}

	// Phase 2: sched_switch burst (10 events, CPU preemption).
	// Placed at t=1200ms so the full timeline >1s, triggering the windowed
	// snapshot in ReplayEventsForChains (not just the final/global fallback).
	// Args[1] = target_pid (matches real eBPF host_trace.bpf.c output).
	for i := 0; i < 10; i++ {
		allEvents = append(allEvents, events.Event{
			Timestamp: now.Add(time.Duration(1200+i) * time.Millisecond),
			PID:       pid,
			TID:       pid,
			Source:    events.SourceHost,
			Op:        uint8(events.HostSchedSwitch),
			Duration:  5 * time.Millisecond,
			Args:      [2]uint64{0, uint64(pid)}, // target_pid in Args[1]
		})
	}

	// Phase 2: inflated cudaStreamSync (tail spike from CPU contention).
	for i := 0; i < 20; i++ {
		allEvents = append(allEvents, events.Event{
			Timestamp: now.Add(time.Duration(1210+i) * time.Millisecond),
			PID:       pid,
			TID:       pid,
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDAStreamSync),
			Duration:  500 * time.Millisecond, // 500x baseline
		})
	}

	// Simulate live mode: feed ALL events to collector + apply shouldStore.
	sessionStart := now.Add(-1 * time.Minute) // past bootstrap window
	var storedEvents []events.Event

	for i, evt := range allEvents {
		liveCollector.Record(evt)
		// Snapshot periodically to update cachedP50 (like real event loop).
		if i%50 == 49 {
			liveCollector.Snapshot()
		}
		if shouldStore(evt, sessionStart, false, liveCollector, 0, nil) {
			storedEvents = append(storedEvents, evt)
		}
	}

	// Verify filtering happened: some cuLaunchKernel events were dropped.
	if len(storedEvents) >= len(allEvents) {
		t.Fatalf("selective storage should filter events: stored=%d total=%d",
			len(storedEvents), len(allEvents))
	}

	// Count stored events by type.
	counts := countEventsByType(storedEvents)
	t.Logf("stored: %d/%d total, sched_switch=%d, cudaStreamSync=%d, cudaLaunchKernel=%d",
		len(storedEvents), len(allEvents),
		counts["sched_switch"], counts["cudaStreamSync"], counts["cudaLaunchKernel"])

	// All sched_switch must be preserved (always-store rule).
	if counts["sched_switch"] != 10 {
		t.Errorf("sched_switch: stored=%d, want 10 (always-store rule)", counts["sched_switch"])
	}
	// All cudaStreamSync must be preserved (sync op always-store rule).
	if counts["cudaStreamSync"] != 70 { // 50 baseline + 20 anomalous
		t.Errorf("cudaStreamSync: stored=%d, want 70 (sync op always-store rule)", counts["cudaStreamSync"])
	}

	// Replay stored events through chain engine (simulates ingero explain).
	// Use default stats.New() (window=1000) to match real callers in
	// explain.go and mcp/server.go. ReplayEventsForChains also creates an
	// internal globalCollector with window=max(len(evts), 1000).
	replayCollector := stats.New()
	replayCorr := correlate.New(correlate.WithMaxAge(0)) // no pruning for replay

	// Set system context (CPU >90% triggers SYSTEM layer → HIGH severity).
	replayCorr.SetSystemSnapshot(&correlate.SystemContext{
		CPUPercent: 92.0,
		MemUsedPct: 60.0,
		LoadAvg1:   8.0,
	})

	chains := correlate.ReplayEventsForChains(storedEvents, replayCollector, replayCorr, pid)

	if len(chains) == 0 {
		t.Fatal("replay of selectively stored events should detect causal chain")
	}

	// Verify a sched_switch chain was found with correct severity.
	foundSchedChain := false
	for _, ch := range chains {
		for _, te := range ch.Timeline {
			if te.Layer == "HOST" && strings.Contains(te.Op, "sched_switch") {
				foundSchedChain = true
				// CPU 92% > 90% threshold → severity must be HIGH.
				if ch.Severity != "HIGH" {
					t.Errorf("sched_switch chain severity = %q, want HIGH (CPU 92%%)", ch.Severity)
				}
			}
		}
	}
	if !foundSchedChain {
		t.Errorf("expected sched_switch chain in replay, got %d chains: %v",
			len(chains), chainSummaries(chains))
	}
}

// TestSelectiveStoragePreservesPageAllocChain verifies that mm_page_alloc
// events are preserved for memory pressure causal chain detection during replay.
//
// Teaching note: The causal chain engine counts mm_page_alloc events and sums
// their Args[0] (bytes). If totalAllocBytes > 1GB, it triggers a memory
// pressure chain. The aggregate table does NOT preserve Args, so individual
// events must be stored. Without the shouldStore fix for mm_page_alloc,
// replay would miss memory pressure chains entirely — a silent correctness gap.
//
// The symptom op is cudaStreamSync (always-stored as a sync op), not cudaMalloc,
// because both baseline AND anomalous events must survive selective storage for
// the chain engine to see a meaningful p99/p50 ratio. Sync ops are always
// stored, so they naturally preserve the full latency distribution.
func TestSelectiveStoragePreservesPageAllocChain(t *testing.T) {
	pid := uint32(1000)
	now := time.Now()

	liveCollector := stats.New()

	var allEvents []events.Event

	// 100 mm_page_alloc events, 16MB each (total = 1.6GB > 1GB threshold).
	// Real mm_page_alloc events have Duration=0; alloc bytes in Args[0].
	for i := 0; i < 100; i++ {
		allEvents = append(allEvents, events.Event{
			Timestamp: now.Add(time.Duration(i) * 10 * time.Millisecond),
			PID:       pid,
			TID:       pid,
			Source:    events.SourceHost,
			Op:        uint8(events.HostPageAlloc),
			Duration:  0,
			Args:      [2]uint64{16 * 1024 * 1024, 0}, // 16MB per alloc
		})
	}

	// 100 normal cuLaunchKernel (bulk, should be aggregated — verifies that
	// selective storage actually filters something in this test).
	for i := 0; i < 100; i++ {
		allEvents = append(allEvents, events.Event{
			Timestamp: now.Add(time.Duration(1000+i) * time.Millisecond),
			PID:       pid,
			TID:       pid,
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDALaunchKernel),
			Duration:  10 * time.Microsecond,
		})
	}

	// 50 normal cudaStreamSync (baseline at 1ms — always stored as sync op).
	for i := 0; i < 50; i++ {
		allEvents = append(allEvents, events.Event{
			Timestamp: now.Add(time.Duration(1100+i*10) * time.Millisecond),
			PID:       pid,
			TID:       pid,
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDAStreamSync),
			Duration:  1 * time.Millisecond,
		})
	}

	// 20 anomalous cudaStreamSync (inflated by memory pressure, 500x baseline).
	for i := 0; i < 20; i++ {
		allEvents = append(allEvents, events.Event{
			Timestamp: now.Add(time.Duration(1600+i*10) * time.Millisecond),
			PID:       pid,
			TID:       pid,
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDAStreamSync),
			Duration:  500 * time.Millisecond,
		})
	}

	// Filter through shouldStore.
	sessionStart := now.Add(-1 * time.Minute) // past bootstrap
	var storedEvents []events.Event

	for i, evt := range allEvents {
		liveCollector.Record(evt)
		if i%30 == 29 {
			liveCollector.Snapshot()
		}
		if shouldStore(evt, sessionStart, false, liveCollector, 0, nil) {
			storedEvents = append(storedEvents, evt)
		}
	}

	// Verify mm_page_alloc events are preserved and filtering happened.
	counts := countEventsByType(storedEvents)
	t.Logf("stored: %d/%d total, mm_page_alloc=%d, cudaStreamSync=%d, cudaLaunchKernel=%d",
		len(storedEvents), len(allEvents),
		counts["mm_page_alloc"], counts["cudaStreamSync"], counts["cudaLaunchKernel"])

	if counts["mm_page_alloc"] == 0 {
		t.Fatal("mm_page_alloc events must be stored for memory pressure chain detection; " +
			"shouldStore() must always-store mm_page_alloc (chain-critical)")
	}
	if counts["mm_page_alloc"] != 100 {
		t.Errorf("mm_page_alloc: stored=%d, want 100 (always-store rule)", counts["mm_page_alloc"])
	}
	// All cudaStreamSync stored (sync op always-store rule).
	if counts["cudaStreamSync"] != 70 {
		t.Errorf("cudaStreamSync: stored=%d, want 70", counts["cudaStreamSync"])
	}
	// Verify filtering actually happened (cuLaunchKernel should be aggregated).
	if len(storedEvents) >= len(allEvents) {
		t.Errorf("selective storage should filter events: stored=%d total=%d",
			len(storedEvents), len(allEvents))
	}

	// Replay stored events through chain engine.
	// Use default stats.New() (window=1000) to match real callers.
	replayCollector := stats.New()
	replayCorr := correlate.New(correlate.WithMaxAge(0))

	// Set system context with memory pressure (mem >95%, swap >0 → HIGH severity).
	replayCorr.SetSystemSnapshot(&correlate.SystemContext{
		CPUPercent: 40.0,
		MemUsedPct: 96.0,
		MemAvailMB: 200,
		SwapUsedMB: 500,
		LoadAvg1:   4.0,
	})

	chains := correlate.ReplayEventsForChains(storedEvents, replayCollector, replayCorr, pid)

	if len(chains) == 0 {
		t.Fatal("replay should detect memory pressure causal chain from stored mm_page_alloc events")
	}

	// Verify a mm_page_alloc chain was found with correct severity.
	foundPageAllocChain := false
	for _, ch := range chains {
		for _, te := range ch.Timeline {
			if te.Layer == "HOST" && strings.Contains(te.Op, "mm_page_alloc") {
				foundPageAllocChain = true
				// mem 96% > 95% + swap > 0 → severity must be HIGH.
				if ch.Severity != "HIGH" {
					t.Errorf("page_alloc chain severity = %q, want HIGH (mem 96%%)", ch.Severity)
				}
			}
		}
	}
	if !foundPageAllocChain {
		t.Errorf("expected mm_page_alloc chain in replay, got %d chains: %v",
			len(chains), chainSummaries(chains))
	}
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// countEventsByType counts stored events by their human-readable op name.
func countEventsByType(evts []events.Event) map[string]int {
	counts := make(map[string]int)
	for _, evt := range evts {
		counts[evt.OpName()]++
	}
	return counts
}

// chainSummaries returns a compact string representation of chains for test output.
func chainSummaries(chains []correlate.CausalChain) []string {
	var ss []string
	for _, ch := range chains {
		ops := make([]string, len(ch.Timeline))
		for i, te := range ch.Timeline {
			ops[i] = te.Layer + ":" + te.Op
		}
		ss = append(ss, ch.Severity+"["+strings.Join(ops, " → ")+"]")
	}
	return ss
}

// ---------------------------------------------------------------------------
// Stack sampling + garbage stack tests (v0.7)
// ---------------------------------------------------------------------------

func TestIsStackResolved(t *testing.T) {
	// Resolved: at least one frame has symbol info.
	resolved := []events.StackFrame{
		{IP: 0xdead},
		{IP: 0x1234, SymbolName: "cudaMalloc", File: "libcudart.so"},
	}
	if !isStackResolved(resolved) {
		t.Error("expected resolved stack to be detected")
	}

	// Python frame counts as resolved.
	pyResolved := []events.StackFrame{
		{IP: 0xdead},
		{IP: 0x5678, PyFile: "train.py", PyFunc: "forward", PyLine: 42},
	}
	if !isStackResolved(pyResolved) {
		t.Error("expected Python-resolved stack to be detected")
	}

	// PyFile-only frame counts as resolved (CPython walker edge case).
	pyFileOnly := []events.StackFrame{
		{IP: 0xdead},
		{IP: 0x9abc, PyFile: "train.py"},
	}
	if !isStackResolved(pyFileOnly) {
		t.Error("expected PyFile-only stack to be detected as resolved")
	}

	// All garbage: no symbol, no file, no Python info.
	garbage := []events.StackFrame{
		{IP: 0xdead},
		{IP: 0xbeef},
	}
	if isStackResolved(garbage) {
		t.Error("expected garbage stack to NOT be resolved")
	}

	// Empty stack.
	if isStackResolved(nil) {
		t.Error("expected nil stack to NOT be resolved")
	}
}

func TestShouldStoreStackSampling(t *testing.T) {
	collector := stats.New()
	sessionStart := time.Now().Add(-1 * time.Minute) // past bootstrap

	stack := []events.StackFrame{
		{IP: 0x1234, SymbolName: "cudaMalloc"},
	}
	stackHash := events.HashStackIPs(stack)

	// Create an event with a stack.
	evt := events.Event{
		Timestamp: time.Now(),
		PID:       1234,
		Source:    events.SourceCUDA,
		Op:        uint8(events.CUDAMalloc),
		Duration:  100 * time.Microsecond,
		Stack:     stack,
	}

	// With maxStackSamples=3, first 3 should store.
	samples := make(map[uint64]int)
	for i := 0; i < 3; i++ {
		if !shouldStore(evt, sessionStart, true, collector, 3, samples) {
			t.Errorf("event %d should store (under limit)", i)
		}
		samples[stackHash]++
	}

	// 4th should be rejected (limit reached, not an anomaly).
	if shouldStore(evt, sessionStart, true, collector, 3, samples) {
		t.Error("event 4 should NOT store (over stack sample limit)")
	}

	// With maxStackSamples=0 (unlimited), always stores.
	if !shouldStore(evt, sessionStart, true, collector, 0, samples) {
		t.Error("should store with unlimited stack samples")
	}

	// Events without stacks are unaffected by sampling.
	noStackEvt := evt
	noStackEvt.Stack = nil
	if !shouldStore(noStackEvt, sessionStart, true, collector, 3, samples) {
		t.Error("no-stack events should be unaffected by sampling")
	}
}

func TestShouldStoreStackSamplingAnomalyBypass(t *testing.T) {
	// Feed enough events to establish a p50 baseline, then verify that
	// an anomaly bypasses the stack sample limit.
	collector := stats.New()
	sessionStart := time.Now().Add(-1 * time.Minute)

	stack := []events.StackFrame{
		{IP: 0x1234, SymbolName: "cudaMalloc"},
	}
	stackHash := events.HashStackIPs(stack)

	baseEvt := events.Event{
		Timestamp: time.Now(),
		PID:       1234,
		Source:    events.SourceCUDA,
		Op:        uint8(events.CUDAMalloc),
		Duration:  100 * time.Microsecond,
		Stack:     stack,
	}

	// Record 20 events to establish a p50 baseline in the collector.
	for i := 0; i < 20; i++ {
		collector.Record(baseEvt)
	}
	collector.Snapshot() // updates cachedP50

	// Fill up the stack sample limit.
	samples := make(map[uint64]int)
	samples[stackHash] = 5 // already at limit

	// A normal event should be rejected (over limit, not anomaly).
	if shouldStore(baseEvt, sessionStart, true, collector, 5, samples) {
		t.Error("normal event should be rejected when over stack sample limit")
	}

	// An anomaly (100x duration) should bypass the limit.
	anomaly := baseEvt
	anomaly.Duration = 100 * time.Millisecond // 1000x the p50 of 100µs
	if !shouldStore(anomaly, sessionStart, true, collector, 5, samples) {
		t.Error("anomaly should bypass stack sample limit")
	}
}
