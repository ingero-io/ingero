package discover

import (
	"runtime"
	"testing"
)

func TestCheckPtraceScopeResult(t *testing.T) {
	// On Linux, reads the actual ptrace_scope.
	// On non-Linux, returns OK with "Yama LSM not present".
	result := CheckPtraceScopeResult()
	if result.Name != "ptrace_scope" {
		t.Errorf("Name = %q, want ptrace_scope", result.Name)
	}
	// On any system, should not panic and should return a valid result.
	t.Logf("ptrace_scope check: OK=%v, Value=%s, Detail=%s", result.OK, result.Value, result.Detail)
}

func TestCheckPtraceScope(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("ptrace_scope requires /proc (Linux only)")
	}

	level, err := CheckPtraceScope()
	if err != nil {
		// Some containers don't have Yama LSM — not a test failure.
		t.Skipf("ptrace_scope not available: %v", err)
	}
	if level < 0 || level > 3 {
		t.Errorf("CheckPtraceScope() = %d, want 0-3", level)
	}
	t.Logf("ptrace_scope level: %d", level)
}
