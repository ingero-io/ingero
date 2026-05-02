package orchestrator

import (
	"testing"
)

func TestParseContainerID_DockerV1(t *testing.T) {
	cgroup := `12:devices:/docker/abcd1234ef5678901234567890abcdef1234567890abcdef1234567890abcdef
11:cpu,cpuacct:/docker/abcd1234ef5678901234567890abcdef1234567890abcdef1234567890abcdef
`
	got := parseContainerID(cgroup)
	want := "abcd1234ef5678901234567890abcdef1234567890abcdef1234567890abcdef"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestParseContainerID_DockerV2(t *testing.T) {
	cgroup := "0::/system.slice/docker-7d8e9f0a1b2c3d4e5f6789abcdef0123456789abcdef0123456789abcdef0123.scope\n"
	got := parseContainerID(cgroup)
	want := "7d8e9f0a1b2c3d4e5f6789abcdef0123456789abcdef0123456789abcdef0123"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestParseContainerID_K8sContainerd(t *testing.T) {
	cgroup := "0::/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod1234abcd_5678_efff_aaaa_bbbbccccdddd.slice/cri-containerd-deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef.scope\n"
	got := parseContainerID(cgroup)
	want := "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestParseContainerID_PodNameSpoof_BlockedByPrefix(t *testing.T) {
	// v0.12.4 (Sec audit ★3): a pod name with a 12-hex prefix that's
	// also valid hex must NOT be returned as the container ID. The
	// prefix-anchored regex requires "docker|containerd|cri-containerd|crio"
	// before the hex. Real container ID below should still win.
	realID := "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
	cgroup := "0::/kubepods.slice/kubepods-besteffort.slice/" +
		"kubepods-besteffort-pod1234abcd_5678_efff_aaaa_bbbbccccdddd.slice/" +
		"cri-containerd-" + realID + ".scope\n"
	got := parseContainerID(cgroup)
	if got != realID {
		t.Fatalf("got %q want %q", got, realID)
	}
}

func TestParseContainerID_ShortHexUnanchored_Refused(t *testing.T) {
	// A pod name like "abc123def4567890" (12+ hex chars) on a
	// no-runtime-prefix line must not be picked up. Without anchoring
	// on the runtime token, the v0.12.3 fallback would have returned
	// it; v0.12.4 refuses.
	cgroup := "12:devices:/kubepods/pod-abc123def4567890\n"
	if got := parseContainerID(cgroup); got != "" {
		t.Fatalf("expected empty (no runtime prefix); got %q", got)
	}
}

func TestParseContainerID_HostNoMatch(t *testing.T) {
	// Bare host cgroup paths don't carry a container ID.
	cgroup := `12:devices:/user.slice
11:cpu,cpuacct:/user.slice
0::/user.slice/user-1000.slice
`
	if got := parseContainerID(cgroup); got != "" {
		t.Fatalf("expected empty on host cgroups, got %q", got)
	}
}

func TestParseContainerID_Lowercased(t *testing.T) {
	cgroup := "12:devices:/docker/ABCD1234EF5678901234567890ABCDEF1234567890ABCDEF1234567890ABCDEF\n"
	got := parseContainerID(cgroup)
	if got == "" {
		t.Fatal("expected non-empty match")
	}
	for _, r := range got {
		if r >= 'A' && r <= 'Z' {
			t.Fatalf("expected lowercase ID, got %q", got)
		}
	}
}

func TestDetectSlurm(t *testing.T) {
	t.Setenv("SLURM_JOB_ID", "12345")
	id := detectSlurm()
	if id.Orchestrator != OrchestratorSlurm {
		t.Fatalf("orchestrator = %q want %q", id.Orchestrator, OrchestratorSlurm)
	}
	if id.JobID != "12345" {
		t.Fatalf("JobID = %q want 12345", id.JobID)
	}
}

func TestDetectSlurm_Absent(t *testing.T) {
	t.Setenv("SLURM_JOB_ID", "")
	if id := detectSlurm(); id.Orchestrator != "" {
		t.Fatalf("expected empty without SLURM_JOB_ID, got %+v", id)
	}
}

func TestDetectECS_V4(t *testing.T) {
	t.Setenv("ECS_CONTAINER_METADATA_URI_V4", "http://169.254.170.2/v4/abc")
	t.Setenv("ECS_CONTAINER_METADATA_URI", "")
	if id := detectECS(); id.Orchestrator != OrchestratorECS {
		t.Fatalf("expected ECS, got %+v", id)
	}
}

func TestDetectECS_V3Fallback(t *testing.T) {
	t.Setenv("ECS_CONTAINER_METADATA_URI_V4", "")
	t.Setenv("ECS_CONTAINER_METADATA_URI", "http://169.254.170.2/v3/xyz")
	if id := detectECS(); id.Orchestrator != OrchestratorECS {
		t.Fatalf("expected ECS via V3 env, got %+v", id)
	}
}

func TestDetectECS_Absent(t *testing.T) {
	t.Setenv("ECS_CONTAINER_METADATA_URI_V4", "")
	t.Setenv("ECS_CONTAINER_METADATA_URI", "")
	if id := detectECS(); id.Orchestrator != "" {
		t.Fatalf("expected empty without ECS env, got %+v", id)
	}
}

func TestDetect_PriorityOrder(t *testing.T) {
	// SLURM env present but the test process isn't in K8s/Docker/ECS.
	// detect() should pick Slurm.
	t.Setenv("SLURM_JOB_ID", "999")
	t.Setenv("ECS_CONTAINER_METADATA_URI_V4", "")
	t.Setenv("ECS_CONTAINER_METADATA_URI", "")
	id := Detect()
	// The test harness might be in a real container; if so, K8s/Docker
	// could win. Accept any non-empty match but assert the function
	// returned a value rather than panicking.
	switch id.Orchestrator {
	case OrchestratorSlurm:
		if id.JobID != "999" {
			t.Fatalf("Slurm match but JobID = %q want 999", id.JobID)
		}
	case OrchestratorK8s, OrchestratorDocker, OrchestratorECS:
		// Test harness happens to run in a container; that's fine,
		// the priority order picked the higher-precedence detector.
	case OrchestratorNone:
		t.Fatal("expected SOME orchestrator (SLURM_JOB_ID was set)")
	default:
		t.Fatalf("unexpected orchestrator: %q", id.Orchestrator)
	}
}
