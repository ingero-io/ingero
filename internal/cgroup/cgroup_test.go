package cgroup

import (
	"testing"
)

func TestParseContainerID(t *testing.T) {
	tests := []struct {
		name     string
		path     string
		wantID   string
	}{
		{
			name:   "containerd v2",
			path:   "/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-podabc123def456.slice/cri-containerd-a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2.scope",
			wantID: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
		},
		{
			name:   "containerd v1",
			path:   "/kubepods/burstable/pod12345678-1234-1234-1234-123456789abc/a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
			wantID: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
		},
		{
			name:   "CRI-O v2",
			path:   "/kubepods.slice/crio-a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2.scope",
			wantID: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
		},
		{
			name:   "CRI-O v1",
			path:   "/kubepods.slice/crio-a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2.scope",
			wantID: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
		},
		{
			name:   "Docker v1",
			path:   "/docker/a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
			wantID: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
		},
		{
			name:   "Docker K8s v1",
			path:   "/kubepods/pod12345678-1234-1234-1234-123456789abc/a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
			wantID: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
		},
		{
			name:   "host process (root cgroup)",
			path:   "/",
			wantID: "",
		},
		{
			name:   "host process (init.scope)",
			path:   "/init.scope",
			wantID: "",
		},
		{
			name:   "systemd slice (no container)",
			path:   "/system.slice/sshd.service",
			wantID: "",
		},
		{
			name:   "empty path",
			path:   "",
			wantID: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseContainerID(tt.path)
			if got != tt.wantID {
				t.Errorf("ParseContainerID(%q) = %q, want %q", tt.path, got, tt.wantID)
			}
		})
	}
}

func TestParseCGroupFile_V2(t *testing.T) {
	// cgroup v2: single "0::/<path>" line
	content := "0::/kubepods.slice/kubepods-burstable.slice/cri-containerd-abc123.scope\n"
	got := parseCGroupFile(content)
	want := "/kubepods.slice/kubepods-burstable.slice/cri-containerd-abc123.scope"
	if got != want {
		t.Errorf("parseCGroupFile() = %q, want %q", got, want)
	}
}

func TestParseCGroupFile_V1(t *testing.T) {
	// cgroup v1: multiple lines, pick longest path
	content := `12:memory:/kubepods/burstable/podUID/containerID
11:cpuset:/kubepods/burstable/podUID/containerID
10:pids:/kubepods/burstable/podUID/containerID
9:devices:/kubepods/burstable/podUID
`
	got := parseCGroupFile(content)
	// All memory/cpuset/pids lines have the same length, any is fine.
	// The important thing is we get the deepest (longest) path.
	want := "/kubepods/burstable/podUID/containerID"
	if got != want {
		t.Errorf("parseCGroupFile() = %q, want %q", got, want)
	}
}

func TestParseCGroupFile_V1Root(t *testing.T) {
	// Host process on cgroup v1: all paths are "/"
	content := `12:memory:/
11:cpuset:/
`
	got := parseCGroupFile(content)
	if got != "/" {
		t.Errorf("parseCGroupFile() = %q, want /", got)
	}
}

func TestParseCGroupFile_Empty(t *testing.T) {
	got := parseCGroupFile("")
	if got != "" {
		t.Errorf("parseCGroupFile() = %q, want empty", got)
	}
}

func TestParseCGroupFile_MixedV1V2(t *testing.T) {
	// Some systems show both v1 and v2 lines. v2 line (hierarchy 0) wins.
	content := `0::/kubepods.slice/cri-containerd-abc.scope
1:name=systemd:/kubepods/burstable/podUID
`
	got := parseCGroupFile(content)
	want := "/kubepods.slice/cri-containerd-abc.scope"
	if got != want {
		t.Errorf("parseCGroupFile() = %q, want %q", got, want)
	}
}
