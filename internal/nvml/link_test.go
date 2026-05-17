package nvml

import (
	"context"
	"errors"
	"testing"
)

// Real `nvidia-smi nvlink -e` output collected from an H100 SXM5
// host. The format has been stable across 525, 535, 550, 560 drivers;
// pinning the corpus here turns a future driver-format change into a
// test failure rather than a silent emission gap.
const h100NVLinkOutput = `GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee)
         Link 0: Replay Errors: 1
         Link 0: Recovery Errors: 0
         Link 0: CRC Flit Errors: 0
         Link 0: CRC Data Errors: 0
         Link 0: ECC Errors: 0
         Link 0: Data Tx Errors: 0
         Link 0: Data Rx Errors: 0
         Link 1: Replay Errors: 2
         Link 1: Recovery Errors: 0
         Link 1: CRC Flit Errors: 0
         Link 1: CRC Data Errors: 0
         Link 1: ECC Errors: 0
         Link 1: Data Tx Errors: 0
         Link 1: Data Rx Errors: 0
GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-ffffffff-1111-2222-3333-444444444444)
         Link 0: Replay Errors: 0
         Link 0: Recovery Errors: 5
         Link 0: CRC Flit Errors: 0
         Link 0: CRC Data Errors: 0
         Link 0: ECC Errors: 0
         Link 0: Data Tx Errors: 0
         Link 0: Data Rx Errors: 0
`

// Older A100 driver format that uses the same Link/Field/Value
// shape; included to lock the parser against the format superset.
const a100NVLinkOutput = `GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-12345678-1234-1234-1234-123456789abc)
         Link 0: Replay Errors: 0
         Link 0: Recovery Errors: 0
         Link 0: CRC Flit Errors: 0
         Link 0: CRC Data Errors: 0
`

// Consumer GPU with no NVLink connections returns an empty body;
// the parser must not panic and must not invent a phantom GPU.
const noNVLinkOutput = ``

func TestParseNVLinkErrors_RealOutput(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want map[uint32]uint64
	}{
		{
			name: "H100 with errors on two GPUs",
			raw:  h100NVLinkOutput,
			want: map[uint32]uint64{0: 3, 1: 5},
		},
		{
			name: "A100 with all zeros",
			raw:  a100NVLinkOutput,
			want: map[uint32]uint64{0: 0},
		},
		{
			name: "no nvlink at all",
			raw:  noNVLinkOutput,
			want: map[uint32]uint64{},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := parseNVLinkErrors([]byte(tc.raw))
			if len(got) != len(tc.want) {
				t.Fatalf("len=%d want %d (got=%v)", len(got), len(tc.want), got)
			}
			for k, v := range tc.want {
				if got[k] != v {
					t.Errorf("gpu %d total=%d want %d (got=%v)", k, got[k], v, got)
				}
			}
		})
	}
}

// Throughput counters must not contaminate the error total. A busy
// GPU pushing terabytes through NVLink would otherwise look like a
// faulty one.
func TestParseNVLinkErrors_IgnoresThroughputFields(t *testing.T) {
	raw := `GPU 0: ignored
         Link 0: Tx Throughput: 999999999
         Link 0: Rx Throughput: 888888888
         Link 0: Replay Errors: 1
`
	got := parseNVLinkErrors([]byte(raw))
	if got[0] != 1 {
		t.Fatalf("gpu 0 total=%d want 1 (throughput leaked in?)", got[0])
	}
}

func TestParsePCIeCSV_HappyPath(t *testing.T) {
	raw := "0, GPU-aaa, 4, 4, 16, 16\n1, GPU-bbb, 3, 4, 8, 16\n"
	rows, err := parsePCIeCSV([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows=%d want 2", len(rows))
	}
	if rows[0].IsPCIeDowntrained() {
		t.Errorf("GPU 0 should not be downtrained (4/4 16/16)")
	}
	if !rows[1].IsPCIeDowntrained() {
		t.Errorf("GPU 1 SHOULD be downtrained (gen 3 of 4, width 8 of 16)")
	}
}

func TestParsePCIeCSV_NotSupportedZeroesField(t *testing.T) {
	raw := "0, GPU-aaa, 3, [Not Supported], 16, 16\n"
	rows, err := parsePCIeCSV([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].PCIeGenMax != 0 {
		t.Errorf("PCIeGenMax=%d want 0 for [Not Supported]", rows[0].PCIeGenMax)
	}
	if rows[0].IsPCIeDowntrained() {
		t.Errorf("downtrain should not fire when max unknown (Max guard)")
	}
}

func TestParsePCIeCSV_EmptyAndMalformed(t *testing.T) {
	if _, err := parsePCIeCSV([]byte("")); err == nil {
		t.Error("empty input should error")
	}
	if _, err := parsePCIeCSV([]byte("0, only-three-fields, 4\n")); err == nil {
		t.Error("short row should error")
	}
}

func TestGetLinkState_BothRunners(t *testing.T) {
	pcie := Runner(func(ctx context.Context) ([]byte, error) {
		return []byte("0, GPU-a, 4, 4, 16, 16\n1, GPU-b, 3, 4, 8, 16\n"), nil
	})
	nvlink := Runner(func(ctx context.Context) ([]byte, error) {
		return []byte(h100NVLinkOutput), nil
	})
	rows, err := GetLinkState(context.Background(), nvlink, pcie)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows=%d want 2", len(rows))
	}
	// Find GPU 0 and confirm both PCIe and NVLink fields populated.
	var gpu0 LinkReading
	for _, r := range rows {
		if r.Index == 0 {
			gpu0 = r
		}
	}
	if gpu0.PCIeGenCurrent != 4 || gpu0.NVLinkErrors != 3 {
		t.Errorf("gpu 0 = %+v want gen=4 nvlink=3", gpu0)
	}
}

func TestGetLinkState_BothNilReturnsEmpty(t *testing.T) {
	rows, err := GetLinkState(context.Background(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 0 {
		t.Errorf("rows=%d want 0 when both runners nil", len(rows))
	}
}

func TestGetLinkState_NVLinkErrorPropagated(t *testing.T) {
	sentinel := errors.New("nvidia-smi died")
	nvlink := Runner(func(ctx context.Context) ([]byte, error) { return nil, sentinel })
	pcie := Runner(func(ctx context.Context) ([]byte, error) {
		return []byte("0, GPU-a, 4, 4, 16, 16\n"), nil
	})
	_, err := GetLinkState(context.Background(), nvlink, pcie)
	if !errors.Is(err, sentinel) {
		t.Fatalf("err=%v want wrap of %v", err, sentinel)
	}
}

func TestNVLinkErrorTracker_FirstObservationSeedsNoEmit(t *testing.T) {
	tr := NewNVLinkErrorTracker(2)
	// The first poll observing 100 errors is the baseline -- maybe
	// the operator has been ignoring it for a week; the agent must
	// not page on that.
	got := tr.Observe(0, 100)
	if got.Kind != "" {
		t.Fatalf("first observation emitted: %+v", got)
	}
}

func TestNVLinkErrorTracker_EmitsOnSustainedDelta(t *testing.T) {
	tr := NewNVLinkErrorTracker(2)
	tr.Observe(0, 100) // seed
	tr.Observe(0, 101) // consecutive=1
	got := tr.Observe(0, 105) // consecutive=2 -> EMIT
	if got.Kind != FaultKindNVLink {
		t.Fatalf("Kind=%q want %q", got.Kind, FaultKindNVLink)
	}
	if got.Severity != HardwareFaultCritical {
		t.Fatalf("Severity=%q want critical", got.Severity)
	}
	if got.GPUID != 0 {
		t.Fatalf("GPUID=%d want 0", got.GPUID)
	}
	// Suppressed afterwards.
	again := tr.Observe(0, 110)
	if again.Kind != "" {
		t.Fatalf("re-emit while still sustained: %+v", again)
	}
}

func TestNVLinkErrorTracker_ZeroDeltaResets(t *testing.T) {
	tr := NewNVLinkErrorTracker(2)
	tr.Observe(0, 100) // seed
	tr.Observe(0, 101) // consecutive=1
	tr.Observe(0, 101) // zero delta -> reset
	tr.Observe(0, 102) // consecutive=1 (fresh run)
	got := tr.Observe(0, 103) // consecutive=2 -> EMIT (re-armed)
	if got.Kind == "" {
		t.Fatal("post-reset run did not emit")
	}
}

func TestPCIeDowntrainTracker_NoEmitWhenAtMax(t *testing.T) {
	tr := NewPCIeDowntrainTracker(3)
	r := LinkReading{Index: 0, PCIeGenCurrent: 4, PCIeGenMax: 4, PCIeWidthCurrent: 16, PCIeWidthMax: 16}
	for i := 0; i < 10; i++ {
		if got := tr.Observe(r); got.Kind != "" {
			t.Fatalf("emit at max: %+v", got)
		}
	}
}

func TestPCIeDowntrainTracker_EmitsAtSustain(t *testing.T) {
	tr := NewPCIeDowntrainTracker(3)
	r := LinkReading{Index: 0, PCIeGenCurrent: 3, PCIeGenMax: 4, PCIeWidthCurrent: 16, PCIeWidthMax: 16}
	tr.Observe(r) // 1
	tr.Observe(r) // 2
	got := tr.Observe(r) // 3 -> EMIT
	if got.Kind != FaultKindPCIeDowntrain {
		t.Fatalf("Kind=%q want %q", got.Kind, FaultKindPCIeDowntrain)
	}
	if got.GPUID != 0 || got.Severity != HardwareFaultCritical {
		t.Fatalf("unexpected emission shape: %+v", got)
	}
	// Suppressed afterwards.
	if again := tr.Observe(r); again.Kind != "" {
		t.Fatalf("re-emit while sustained: %+v", again)
	}
}

func TestPCIeDowntrainTracker_TransientLaneDoesNotEmit(t *testing.T) {
	tr := NewPCIeDowntrainTracker(3)
	down := LinkReading{Index: 0, PCIeGenCurrent: 3, PCIeGenMax: 4, PCIeWidthCurrent: 16, PCIeWidthMax: 16}
	up := LinkReading{Index: 0, PCIeGenCurrent: 4, PCIeGenMax: 4, PCIeWidthCurrent: 16, PCIeWidthMax: 16}
	tr.Observe(down) // 1
	tr.Observe(up) // reset
	tr.Observe(down) // 1
	tr.Observe(down) // 2
	if got := tr.Observe(up); got.Kind != "" {
		t.Fatalf("emit on re-uptrain: %+v", got)
	}
}

func TestPCIeDowntrainTracker_PerGPUIndependence(t *testing.T) {
	tr := NewPCIeDowntrainTracker(2)
	a := LinkReading{Index: 0, PCIeGenCurrent: 3, PCIeGenMax: 4, PCIeWidthCurrent: 16, PCIeWidthMax: 16}
	b := LinkReading{Index: 1, PCIeGenCurrent: 3, PCIeGenMax: 4, PCIeWidthCurrent: 16, PCIeWidthMax: 16}
	tr.Observe(a)
	if got := tr.Observe(b); got.Kind != "" {
		t.Fatalf("GPU B emitted on its first observation: %+v", got)
	}
}
