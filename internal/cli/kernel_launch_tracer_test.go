package cli

import (
	"testing"

	"github.com/ingero-io/ingero/internal/ebpf/kernellaunch"
)

// v0.15 item M: per-PID kernel-launch counters tests.

func TestRecordKernelLaunchEvent_PerPIDIsolation(t *testing.T) {
	resetKernelLaunchCounters()
	defer resetKernelLaunchCounters()

	recordKernelLaunchEvent(kernellaunch.Event{PID: 100, GridX: 16, GridY: 4, BlockX: 32, BlockY: 8})
	recordKernelLaunchEvent(kernellaunch.Event{PID: 100, GridX: 32, GridY: 4, BlockX: 64, BlockY: 4})
	recordKernelLaunchEvent(kernellaunch.Event{PID: 200, GridX: 8, BlockX: 256})

	got := snapshotKernelLaunchCounters()
	if len(got) != 2 {
		t.Fatalf("expected 2 distinct PIDs, got %d", len(got))
	}
	// Sorted ascending by PID.
	if got[0].PID != 100 || got[0].Count != 2 {
		t.Errorf("PID 100 count = %d, want 2", got[0].Count)
	}
	if got[1].PID != 200 || got[1].Count != 1 {
		t.Errorf("PID 200 count = %d, want 1", got[1].Count)
	}
}

func TestRecordKernelLaunchEvent_PopulatesHistograms(t *testing.T) {
	resetKernelLaunchCounters()
	defer resetKernelLaunchCounters()

	// Two launches with known dims. ThreadsPerBlock = BlockX*BlockY.
	// 32*8 = 256, 64*4 = 256
	recordKernelLaunchEvent(kernellaunch.Event{PID: 100, GridX: 16, GridY: 4, BlockX: 32, BlockY: 8})
	recordKernelLaunchEvent(kernellaunch.Event{PID: 100, GridX: 32, GridY: 4, BlockX: 64, BlockY: 4})

	got := snapshotKernelLaunchCounters()
	if len(got) != 1 || got[0].PID != 100 {
		t.Fatalf("got %+v, want one row for PID 100", got)
	}
	tpb := got[0].ThreadsPerBlockHist
	if tpb.Count != 2 {
		t.Errorf("ThreadsPerBlock Count=%d, want 2", tpb.Count)
	}
	if tpb.Sum != 256+256 {
		t.Errorf("ThreadsPerBlock Sum=%v, want 512", tpb.Sum)
	}
	gb := got[0].GridBlocksHist
	if gb.Count != 2 {
		t.Errorf("GridBlocks Count=%d, want 2", gb.Count)
	}
	// 16*4 + 32*4 = 64 + 128 = 192
	if gb.Sum != 192 {
		t.Errorf("GridBlocks Sum=%v, want 192", gb.Sum)
	}
}

func TestRecordKernelLaunchEvent_ZeroDimsIgnored(t *testing.T) {
	// An event with BlockX=0 (parser glitch) must NOT poison the
	// histogram with a zero observation. ThreadsPerBlock returns 0
	// and the recorder skips Observe.
	resetKernelLaunchCounters()
	defer resetKernelLaunchCounters()

	recordKernelLaunchEvent(kernellaunch.Event{PID: 100, BlockX: 0})

	got := snapshotKernelLaunchCounters()
	if len(got) != 1 {
		t.Fatalf("event still counts as a launch (BlockX=0 is event present); got %d", len(got))
	}
	if got[0].ThreadsPerBlockHist.Count != 0 {
		t.Errorf("zero-dim launch should NOT register in threads_per_block histogram; got Count=%d", got[0].ThreadsPerBlockHist.Count)
	}
}

func TestSnapshotKernelLaunchCounters_NilOnEmpty(t *testing.T) {
	resetKernelLaunchCounters()
	if got := snapshotKernelLaunchCounters(); got != nil {
		t.Errorf("expected nil on empty state, got %+v", got)
	}
}

func TestResetKernelLaunchCounters_Clears(t *testing.T) {
	resetKernelLaunchCounters()
	defer resetKernelLaunchCounters()
	recordKernelLaunchEvent(kernellaunch.Event{PID: 100, BlockX: 256})
	resetKernelLaunchCounters()
	if got := snapshotKernelLaunchCounters(); got != nil {
		t.Errorf("post-reset snapshot should be nil; got %+v", got)
	}
}
