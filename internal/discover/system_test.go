package discover

import (
	"os"
	"runtime"
	"strings"
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

func TestBTFRecommendationForPackageManager(t *testing.T) {
	tests := []struct {
		name     string
		pm       string
		contains string
	}{
		{
			name:     "apt uses linux-tools with uname",
			pm:       "apt",
			contains: "linux-tools-$(uname -r)",
		},
		{
			name:     "dnf uses kernel debuginfo",
			pm:       "dnf",
			contains: "dnf install kernel-debuginfo",
		},
		{
			name:     "unknown uses generic guidance",
			pm:       "unknown",
			contains: "package name varies by distro",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reco := btfRecommendationForPackageManager(tt.pm)
			if !strings.Contains(reco, tt.contains) {
				t.Fatalf("recommendation %q missing %q", reco, tt.contains)
			}
		})
	}
}

func TestCheckNVIDIAVersionRecommendation(t *testing.T) {
	old := checkNVIDIAVersion("545.23")
	if old.OK {
		t.Fatal("expected old driver to fail 550+ check")
	}
	if !strings.Contains(old.Recommendation, "nvidia.com") {
		t.Fatalf("expected recommendation to include driver download link, got %q", old.Recommendation)
	}

	newer := checkNVIDIAVersion("550.40")
	if !newer.OK {
		t.Fatal("expected 550+ driver to pass")
	}
	if newer.Recommendation != "" {
		t.Fatalf("expected no recommendation for passing driver check, got %q", newer.Recommendation)
	}
}

func TestCheckPrivileges(t *testing.T) {
	res := CheckPrivileges()
	if res.Name != "Privileges" {
		t.Fatalf("unexpected check name: %q", res.Name)
	}

	if os.Geteuid() == 0 {
		if !res.OK || res.Optional {
			t.Fatalf("root run should be OK and non-optional, got OK=%v Optional=%v", res.OK, res.Optional)
		}
		return
	}

	if res.OK {
		t.Fatal("non-root run should not fully pass privileges check")
	}
	if !res.Optional {
		t.Fatal("non-root privileges check should be optional")
	}
	if !strings.Contains(strings.ToLower(res.Detail), "sudo") {
		t.Fatalf("expected non-root detail to mention sudo, got %q", res.Detail)
	}
	if res.Recommendation == "" {
		t.Fatal("expected recommendation for non-root privileges check")
	}
}

func TestRunAllChecksIncludesPrivileges(t *testing.T) {
	results := RunAllChecks()
	found := false
	for _, r := range results {
		if r.Name == "Privileges" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("RunAllChecks() missing Privileges check")
	}
}
