package cli

import (
	"testing"

	"github.com/ingero-io/ingero/internal/ebpf/memfrag"
)

func TestRecordMemfragEvent_Tallies(t *testing.T) {
	resetMemfragCounters()
	defer resetMemfragCounters()

	recordMemfragEvent(memfrag.Event{Cmd: 0xC0184601})
	recordMemfragEvent(memfrag.Event{Cmd: 0xC0184601})
	recordMemfragEvent(memfrag.Event{Cmd: 0xC0184602})

	got := snapshotMemfragCounters()
	if len(got) != 2 {
		t.Fatalf("got %d distinct cmds, want 2", len(got))
	}
	// Sorted ascending.
	if got[0].Cmd != 0xC0184601 || got[0].Count != 2 {
		t.Errorf("cmd 0xC0184601 = %d, want 2 (rows=%+v)", got[0].Count, got)
	}
	if got[1].Cmd != 0xC0184602 || got[1].Count != 1 {
		t.Errorf("cmd 0xC0184602 = %d, want 1 (rows=%+v)", got[1].Count, got)
	}
}

func TestSnapshotMemfragCounters_NilOnEmpty(t *testing.T) {
	resetMemfragCounters()
	if got := snapshotMemfragCounters(); got != nil {
		t.Errorf("expected nil snapshot on empty state, got %+v", got)
	}
}

func TestResetMemfragCounters_Clears(t *testing.T) {
	resetMemfragCounters()
	defer resetMemfragCounters()
	recordMemfragEvent(memfrag.Event{Cmd: 1})
	resetMemfragCounters()
	if got := snapshotMemfragCounters(); got != nil {
		t.Errorf("post-reset snapshot should be nil; got %+v", got)
	}
}
