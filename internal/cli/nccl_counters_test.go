package cli

import (
	"testing"

	"github.com/ingero-io/ingero/internal/stats"
)

func TestRecordNCCLCollective_BasicTallies(t *testing.T) {
	resetNCCLCollectiveCounters()
	defer resetNCCLCollectiveCounters()

	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 1024})
	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 2048})
	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclBcast", CountBytes: 512})

	got := snapshotNCCLCollectiveCounters()
	if len(got) != 2 {
		t.Fatalf("expected 2 op-types, got %d (%+v)", len(got), got)
	}
	byOp := map[string]stats.NCCLCollectiveCounter{}
	for _, c := range got {
		byOp[c.OpType] = c
	}
	if byOp["ncclAllReduce"].Count != 2 || byOp["ncclAllReduce"].BytesTotal != 1024+2048 {
		t.Errorf("ncclAllReduce got %+v", byOp["ncclAllReduce"])
	}
	if byOp["ncclBcast"].Count != 1 || byOp["ncclBcast"].BytesTotal != 512 {
		t.Errorf("ncclBcast got %+v", byOp["ncclBcast"])
	}
}

func TestRecordNCCLCollective_BarrierSeparate(t *testing.T) {
	resetNCCLCollectiveCounters()
	defer resetNCCLCollectiveCounters()

	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 1024})
	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", IsBarrier: true})
	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", IsBarrier: true})

	got := snapshotNCCLCollectiveCounters()
	if len(got) != 2 {
		t.Fatalf("expected separate collective + barrier rows, got %d (%+v)", len(got), got)
	}
	var collective, barrier stats.NCCLCollectiveCounter
	for _, c := range got {
		if c.BarrierEvents > 0 {
			barrier = c
		} else {
			collective = c
		}
	}
	if collective.Count != 1 || collective.BytesTotal != 1024 {
		t.Errorf("collective row wrong: %+v", collective)
	}
	if barrier.BarrierEvents != 2 {
		t.Errorf("barrier row wrong: %+v", barrier)
	}
}

func TestRecordNCCLCollective_EmptyOpTypeIgnored(t *testing.T) {
	resetNCCLCollectiveCounters()
	defer resetNCCLCollectiveCounters()

	recordNCCLCollective(stats.NCCLDataPoint{OpType: "", CountBytes: 999})
	if got := snapshotNCCLCollectiveCounters(); got != nil {
		t.Errorf("expected nil snapshot after empty-op-type record, got %+v", got)
	}
}

func TestSnapshotNCCLCollectiveCounters_NilOnEmpty(t *testing.T) {
	resetNCCLCollectiveCounters()
	if got := snapshotNCCLCollectiveCounters(); got != nil {
		t.Errorf("expected nil on empty state, got %+v", got)
	}
}
