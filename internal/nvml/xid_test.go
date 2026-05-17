package nvml

import (
	"context"
	"errors"
	"testing"
)

// Real-world kmsg lines collected from production GPU hosts and
// driver release-note examples. Format varies across nvidia-driver
// versions (450 through 560 verified). Pinning the regression
// corpus here means a parser refactor breaks the test before it
// breaks a live wire emission.
func TestParseXidLine_RealCorpus(t *testing.T) {
	cases := []struct {
		name    string
		line    string
		want    XidEvent
		matches bool
	}{
		{
			name:    "kmsg prefix, GPU off the bus",
			line:    `4,1234,5678901,-;NVRM: Xid (PCI:0000:1a:00): 79, pid='<unknown>', name=<unknown>, GPU has fallen off the bus.`,
			want:    XidEvent{XidNumber: 79, PciBusID: "00000000:1a:00.0"},
			matches: true,
		},
		{
			name:    "dmesg prefix, MMU fault with pid",
			line:    `[12345.678901] NVRM: Xid (PCI:0000:3b:00): 31, pid=12345, name=python3, Ch 00000010, intr 10000000.`,
			want:    XidEvent{XidNumber: 31, PciBusID: "00000000:3b:00.0", PID: 12345},
			matches: true,
		},
		{
			name:    "no prefix, full PCI domain",
			line:    `NVRM: Xid (PCI:00000000:65:00): 13, pid=4242, Graphics SM Warp Exception on (GPC 0, TPC 0, SM 0): ESR 0x504648=0x1`,
			want:    XidEvent{XidNumber: 13, PciBusID: "00000000:65:00.0", PID: 4242},
			matches: true,
		},
		{
			name:    "uppercase bus id, no pid field",
			line:    `[ 100.0] NVRM: Xid (PCI:0000:1A:00): 48`,
			want:    XidEvent{XidNumber: 48, PciBusID: "00000000:1a:00.0"},
			matches: true,
		},
		{
			name:    "pid=<unknown> tolerated as no-pid",
			line:    `NVRM: Xid (PCI:0000:1a:00): 79, pid='<unknown>', GPU has fallen off the bus.`,
			want:    XidEvent{XidNumber: 79, PciBusID: "00000000:1a:00.0"},
			matches: true,
		},
		{
			name:    "non-Xid kernel line",
			line:    `4,1,2,-;Linux version 5.15.0-foo`,
			matches: false,
		},
		{
			name:    "NVRM line that is not an Xid",
			line:    `NVRM: loading NVIDIA UNIX x86_64 Kernel Module  560.35.03`,
			matches: false,
		},
		{
			name:    "Xid mention without PCI prefix is dropped",
			line:    `NVRM: Xid 79`,
			matches: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ParseXidLine(tc.line)
			if ok != tc.matches {
				t.Fatalf("matches=%v want %v (got=%+v)", ok, tc.matches, got)
			}
			if !tc.matches {
				return
			}
			if got != tc.want {
				t.Fatalf("got %+v want %+v", got, tc.want)
			}
		})
	}
}

// The critical-xid list is the seam between FOSS detection and the
// EE-side node_cordon + pod_drain dispatch. Locking the membership
// here protects the operator playbook from a silent rotation in
// either direction.
func TestIsCriticalXid_Membership(t *testing.T) {
	criticals := []uint32{13, 31, 43, 45, 48, 56, 57, 58, 62, 63, 64, 65, 68, 69, 73, 74, 79}
	for _, n := range criticals {
		if !IsCriticalXid(n) {
			t.Errorf("Xid %d should be critical", n)
		}
	}
	for _, n := range []uint32{1, 8, 12, 32, 44, 100, 0} {
		if IsCriticalXid(n) {
			t.Errorf("Xid %d should NOT be critical", n)
		}
	}
}

// XidToHardwareFault maps the parsed event onto the wire-facing
// HardwareFault struct. Severity must follow IsCriticalXid; PID must
// pass through; Timestamp must be stamped.
func TestXidToHardwareFault_SeverityAndShape(t *testing.T) {
	ev := XidEvent{XidNumber: 79, PciBusID: "00000000:1a:00.0", PID: 4321}
	fault := XidToHardwareFault(ev, 2)
	if fault.Kind != FaultKindXid {
		t.Fatalf("Kind=%q want %q", fault.Kind, FaultKindXid)
	}
	if fault.Severity != HardwareFaultCritical {
		t.Fatalf("Severity=%q want critical for Xid 79", fault.Severity)
	}
	if fault.GPUID != 2 {
		t.Fatalf("GPUID=%d want 2", fault.GPUID)
	}
	if fault.XidNumber != 79 {
		t.Fatalf("XidNumber=%d want 79", fault.XidNumber)
	}
	if fault.PID != 4321 {
		t.Fatalf("PID=%d want 4321", fault.PID)
	}
	if fault.Timestamp.IsZero() {
		t.Fatal("Timestamp not stamped")
	}

	warning := XidToHardwareFault(XidEvent{XidNumber: 8}, 0)
	if warning.Severity != HardwareFaultWarning {
		t.Fatalf("Xid 8 Severity=%q want warning", warning.Severity)
	}
}

func TestNormalizePCIBusID_AllForms(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"0000:1a:00", "00000000:1a:00.0"},
		{"0000:1A:00.0", "00000000:1a:00.0"},
		{"00000000:1A:00.0", "00000000:1a:00.0"},
		{"00000000:1a:00", "00000000:1a:00.0"},
		{" 0000:1a:00 ", "00000000:1a:00.0"},
	}
	for _, tc := range cases {
		if got := normalizePCIBusID(tc.in); got != tc.want {
			t.Errorf("normalizePCIBusID(%q)=%q want %q", tc.in, got, tc.want)
		}
	}
}

func TestResolvePciIndex_HappyPath(t *testing.T) {
	stub := Runner(func(ctx context.Context) ([]byte, error) {
		return []byte("0, 00000000:1A:00.0\n1, 00000000:3B:00.0\n"), nil
	})
	got, err := ResolvePciIndex(context.Background(), stub)
	if err != nil {
		t.Fatal(err)
	}
	if got["00000000:1a:00.0"] != 0 {
		t.Errorf("00000000:1a:00.0 -> %d want 0", got["00000000:1a:00.0"])
	}
	if got["00000000:3b:00.0"] != 1 {
		t.Errorf("00000000:3b:00.0 -> %d want 1", got["00000000:3b:00.0"])
	}
}

func TestResolvePciIndex_NilRunnerReturnsError(t *testing.T) {
	_, err := ResolvePciIndex(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error when nvidia-smi unavailable")
	}
}

func TestResolvePciIndex_BadOutput(t *testing.T) {
	stub := Runner(func(ctx context.Context) ([]byte, error) {
		return []byte("garbage line without comma\n"), nil
	})
	if _, err := ResolvePciIndex(context.Background(), stub); err == nil {
		t.Fatal("expected parse error for garbage output")
	}
}

func TestResolvePciIndex_PropagatesRunnerError(t *testing.T) {
	sentinel := errors.New("nvidia-smi crashed")
	stub := Runner(func(ctx context.Context) ([]byte, error) {
		return nil, sentinel
	})
	_, err := ResolvePciIndex(context.Background(), stub)
	if !errors.Is(err, sentinel) {
		t.Fatalf("err=%v want wrapping %v", err, sentinel)
	}
}
