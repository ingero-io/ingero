package cli

import (
	"bytes"
	"strings"
	"testing"
)

// v0.12.2 (QA audit ★2 #10): unit-cover the early-exit paths of
// setupNCCLTracer. We can't drive the attach success path without real
// BPF, but the no-libnccl + unprivileged paths are exactly the ones a
// user is most likely to hit on a misconfigured box, and were the
// audit's specific request.

func TestSetupNCCLTracer_NoLibNCCL(t *testing.T) {
	var buf bytes.Buffer
	tracer, count := setupNCCLTracer(ncclSetupParams{
		explicitLib:       "",
		targetPIDs:        []int{1234},
		explicitPIDs:      true,
		geteuid:           func() int { return 0 }, // pretend root, isolate the libnccl path
		hasCapBPF:         func() bool { return true },
		findLibForPID:     func(pid int) string { return "" },
		findLibSystemwide: func() string { return "" },
		debugf:            func(string, ...any) {},
		stderr:            &buf,
	})
	if tracer != nil {
		t.Fatalf("tracer should be nil when no libnccl found, got %T", tracer)
	}
	if count != 0 {
		t.Fatalf("probe count should be 0, got %d", count)
	}
	// v0.15 F1: the no-eager-libnccl path now arms a lazy tracer
	// instead of returning a hard "no libnccl" error. On unprivileged
	// runners that cannot load BPF the lazy Prepare itself fails, the
	// function logs the failure and returns (nil, 0). Either of those
	// markers in stderr proves the F1 path was entered correctly.
	out := buf.String()
	if !strings.Contains(out, "lazy-attach armed") &&
		!strings.Contains(out, "NCCL lazy-attach unavailable") {
		t.Fatalf("expected lazy-attach marker in stderr, got: %q", out)
	}
}

func TestSetupNCCLTracer_UnprivilegedNoLib(t *testing.T) {
	var buf bytes.Buffer
	tracer, count := setupNCCLTracer(ncclSetupParams{
		explicitLib:       "",
		targetPIDs:        nil,
		explicitPIDs:      false,
		geteuid:           func() int { return 1000 }, // non-root
		hasCapBPF:         func() bool { return false },
		findLibForPID:     func(pid int) string { return "" },
		findLibSystemwide: func() string { return "" },
		debugf:            func(string, ...any) {},
		stderr:            &buf,
	})
	if tracer != nil || count != 0 {
		t.Fatalf("expected (nil, 0), got (%v, %d)", tracer, count)
	}
	out := buf.String()
	if !strings.Contains(out, "CAP_BPF") {
		t.Fatalf("expected CAP_BPF warning, got: %q", out)
	}
	if !strings.Contains(out, "lazy-attach armed") &&
		!strings.Contains(out, "NCCL lazy-attach unavailable") {
		t.Fatalf("expected lazy-attach marker in stderr, got: %q", out)
	}
}

func TestSetupNCCLTracer_PrivilegedNoWarning(t *testing.T) {
	var buf bytes.Buffer
	setupNCCLTracer(ncclSetupParams{
		explicitLib:       "",
		geteuid:           func() int { return 0 },
		hasCapBPF:         func() bool { return false }, // doesn't matter when euid==0
		findLibForPID:     func(pid int) string { return "" },
		findLibSystemwide: func() string { return "" },
		debugf:            func(string, ...any) {},
		stderr:            &buf,
	})
	if strings.Contains(buf.String(), "CAP_BPF") {
		t.Fatalf("euid=0 should suppress CAP_BPF warning, got: %q", buf.String())
	}
}

func TestSetupNCCLTracer_PIDLookupOrder(t *testing.T) {
	// findLibForPID hits first, then findLibSystemwide. PID lookup
	// short-circuits on first match.
	var perPIDCalls []int
	sysCalled := false
	var buf bytes.Buffer
	setupNCCLTracer(ncclSetupParams{
		explicitLib:  "",
		targetPIDs:   []int{0, 100, 200, 300}, // 0 is skipped
		explicitPIDs: true,
		geteuid:      func() int { return 0 },
		hasCapBPF:    func() bool { return true },
		findLibForPID: func(pid int) string {
			perPIDCalls = append(perPIDCalls, pid)
			if pid == 200 {
				return "/fake/libnccl.so"
			}
			return ""
		},
		findLibSystemwide: func() string {
			sysCalled = true
			return ""
		},
		debugf: func(string, ...any) {},
		stderr: &buf,
	})
	wantCalls := []int{100, 200} // 0 skipped, 200 returns first hit
	if len(perPIDCalls) != len(wantCalls) {
		t.Fatalf("findLibForPID called for %v, want %v", perPIDCalls, wantCalls)
	}
	for i, p := range wantCalls {
		if perPIDCalls[i] != p {
			t.Fatalf("findLibForPID[%d]=%d, want %d", i, perPIDCalls[i], p)
		}
	}
	if sysCalled {
		t.Fatalf("findLibSystemwide should not be called once a per-PID match wins")
	}
	// We hit a fake path so Attach will fail. That's expected — the
	// warning lands in stderr but the test passes (returns nil, 0).
	if !strings.Contains(buf.String(), "NCCL tracing unavailable") {
		t.Fatalf("expected attach-failure warning, got: %q", buf.String())
	}
}

func TestSetupNCCLTracer_ExplicitLibSkipsLookup(t *testing.T) {
	perPIDCalled := false
	sysCalled := false
	var buf bytes.Buffer
	setupNCCLTracer(ncclSetupParams{
		explicitLib:  "/explicit/libnccl.so",
		targetPIDs:   []int{100},
		explicitPIDs: true,
		geteuid:      func() int { return 0 },
		hasCapBPF:    func() bool { return true },
		findLibForPID: func(pid int) string {
			perPIDCalled = true
			return ""
		},
		findLibSystemwide: func() string {
			sysCalled = true
			return ""
		},
		debugf: func(string, ...any) {},
		stderr: &buf,
	})
	if perPIDCalled || sysCalled {
		t.Fatalf("explicit libpath must short-circuit lookup; per-PID=%v sys=%v", perPIDCalled, sysCalled)
	}
}
