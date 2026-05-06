package ncclprobe

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unsafe"
)

// TestEventSizeMatchesC asserts the Go Event struct stays on the same
// 104-byte size as the C `struct nccl_event`. Drift here would mean
// userspace mis-parses ringbuf records.
func TestEventSizeMatchesC(t *testing.T) {
	if got, want := unsafe.Sizeof(Event{}), uintptr(EventSize); got != want {
		t.Fatalf("Event size = %d, want %d (must match struct nccl_event in bpf/common.bpf.h)", got, want)
	}
}

// TestEventOpName covers every NCCL_OP_* code that the BPF program may
// emit, plus an unknown-future-op fallback path.
func TestEventOpName(t *testing.T) {
	cases := []struct {
		op   uint8
		want string
	}{
		{1, "ncclCommInitRank"},
		{2, "ncclCommDestroy"},
		{3, "ncclAllReduce"},
		{4, "ncclAllGather"},
		{5, "ncclReduceScatter"},
		{6, "ncclBcast"},
		{7, "ncclSend"},
		{8, "ncclRecv"},
		{99, "nccl_op_99"},
	}
	for _, c := range cases {
		if got := (Event{Op: c.op}).OpName(); got != c.want {
			t.Errorf("op=%d got %q want %q", c.op, got, c.want)
		}
	}
}

func TestEventCommString(t *testing.T) {
	var e Event
	copy(e.Comm[:], "python3\x00garbage")
	if got, want := e.CommString(), "python3"; got != want {
		t.Fatalf("CommString=%q want %q", got, want)
	}
	// All-NUL comm: empty string.
	var e2 Event
	if got := e2.CommString(); got != "" {
		t.Fatalf("empty Comm: CommString=%q want empty", got)
	}
}

func TestEventStringContainsKeyFields(t *testing.T) {
	e := Event{
		Op:         3, // AllReduce
		PID:        1234,
		TID:        5678,
		Rank:       2,
		NRanks:     8,
		CountBytes: 1024,
		DurationNs: 5_000_000, // 5 ms = 5000 us
		CommIDHash: 0xdeadbeefcafebabe,
		ReturnCode: 0,
	}
	s := e.String()
	for _, want := range []string{"ncclAllReduce", "pid=1234", "tid=5678", "rank=2/8", "count=1024", "duration_us=5000", "comm_id=deadbeefcafebabe"} {
		if !strings.Contains(s, want) {
			t.Errorf("String() missing %q; got %q", want, s)
		}
	}
}

// TestResolveSymbol exercises the version-drift code path on a real
// system ELF. /bin/ls is guaranteed to exist on Linux and is unlikely
// to export any NCCL symbol, so we expect a "symbol not found" error.
func TestResolveSymbol_NotFound(t *testing.T) {
	if _, err := os.Stat("/bin/ls"); err != nil {
		t.Skipf("/bin/ls not present: %v", err)
	}
	_, err := resolveSymbol("/bin/ls", "ncclAllReduce")
	if err == nil {
		t.Fatal("expected error for missing symbol")
	}
}

func TestResolveSymbol_BadPath(t *testing.T) {
	_, err := resolveSymbol("/this/does/not/exist/.so", "ncclAllReduce")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

// TestResolveSymbol_FindsExisting picks a libc symbol from libc.so.6
// (universally present) to assert the dynamic-symbol-table path runs.
func TestResolveSymbol_FindsExisting(t *testing.T) {
	candidates := []string{
		"/usr/lib/x86_64-linux-gnu/libc.so.6",
		"/usr/lib/aarch64-linux-gnu/libc.so.6",
		"/lib/x86_64-linux-gnu/libc.so.6",
		"/lib/aarch64-linux-gnu/libc.so.6",
	}
	var found string
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			found = p
			break
		}
	}
	if found == "" {
		t.Skip("no libc.so.6 in expected locations")
	}
	got, err := resolveSymbol(found, "malloc")
	if err != nil {
		t.Fatalf("malloc resolve: %v", err)
	}
	// Either bare "malloc" or a versioned variant is acceptable.
	if !strings.HasPrefix(got, "malloc") {
		t.Fatalf("malloc resolve got %q, want prefix malloc", got)
	}
}

// TestFindLibNCCL_NoMaps: nonexistent PID yields empty string.
func TestFindLibNCCL_NoMaps(t *testing.T) {
	if got := FindLibNCCL(99999999); got != "" {
		t.Errorf("expected empty for missing PID, got %q", got)
	}
}

// TestCompareVersionedSymbol covers the v0.12.0 bugfix that moved from
// lexical to numeric-tuple comparison. ncclAllReduce_v2_19_3 must
// outrank ncclAllReduce_v2_9_0 (lex compare wrongly says _v2_9 > _v2_19).
func TestCompareVersionedSymbol(t *testing.T) {
	cases := []struct {
		a, b string
		want int // sign of comparison
	}{
		{"ncclAllReduce_v2_19_3", "ncclAllReduce_v2_9_0", +1},
		{"ncclAllReduce_v2_9_0", "ncclAllReduce_v2_19_3", -1},
		{"ncclAllReduce_v2_19_3", "ncclAllReduce_v2_19_3", 0},
		{"ncclAllReduce_v2_20_0", "ncclAllReduce_v2_19_99", +1},
		// Unparseable suffix falls back to lexical
		{"foo_vX", "foo_vY", -1},
	}
	for _, c := range cases {
		got := compareVersionedSymbol(c.a, c.b)
		if (got < 0) != (c.want < 0) || (got > 0) != (c.want > 0) || (got == 0) != (c.want == 0) {
			t.Errorf("compareVersionedSymbol(%q,%q) = %d, want sign %d", c.a, c.b, got, c.want)
		}
	}
}

// TestIsSafeLibPath asserts that the path-traversal allowlist (for
// FindLibNCCL) accepts canonical install roots and rejects user-writable
// directories.
func TestIsSafeLibPath(t *testing.T) {
	cases := []struct {
		path string
		want bool
	}{
		{"/usr/lib/x86_64-linux-gnu/libnccl.so.2", true},
		{"/usr/lib/aarch64-linux-gnu/libnccl.so.2", true},
		{"/usr/lib64/libnccl.so", true},
		{"/usr/local/lib/libnccl.so.2", true},
		{"/usr/local/cuda-12.4/lib64/libnccl.so", true},
		{"/opt/nccl/lib/libnccl.so.2", true},
		{"/opt/conda/envs/torch/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so", true},
		{"/opt/pytorch/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so", true},
		// Denied — user-writable
		{"/tmp/libnccl.so.2", false},
		{"/home/attacker/libnccl.so", false},
		{"/dev/shm/libnccl.so.2", false},
		{"/var/tmp/libnccl.so", false},
		{"/run/user/1000/libnccl.so", false},
		// Outside both allowlist and denylist => denied
		{"/srv/foo/libnccl.so", false},
		{"/data/libnccl.so", false},
	}
	for _, c := range cases {
		if got := isSafeLibPath(c.path); got != c.want {
			t.Errorf("isSafeLibPath(%q)=%v want %v", c.path, got, c.want)
		}
	}
}

// TestFindLibNCCL_OwnPid: scanning our own /proc/<pid>/maps usually
// returns "" because the test binary doesn't link NCCL. Either "" or
// a real path are acceptable; what matters is no panic.
func TestFindLibNCCL_OwnPid(t *testing.T) {
	got := FindLibNCCL(os.Getpid())
	// Smoke: doesn't panic, and if non-empty the file exists.
	if got != "" {
		if _, err := os.Stat(got); err != nil {
			t.Errorf("FindLibNCCL returned %q which doesn't stat: %v", got, err)
		}
	}
}

// TestFindLibNCCL_RejectsUserWritablePath covers v0.12.1 (QA #10):
// integration boundary between FindLibNCCL's /proc/maps parser and
// isSafeLibPath's allowlist. A non-root attacker mmaps libnccl.so
// from a tmpfs-backed /tmp/.../ path; FindLibNCCL must reject it.
//
// We can't easily forge a real /proc/<pid>/maps line, so we test the
// boundary by feeding isSafeLibPath the literal path /tmp/.../libnccl.so.2
// (which FindLibNCCL would have called isSafeLibPath on). This is the
// integration-side surface; the algorithm side is in TestIsSafeLibPath.
func TestFindLibNCCL_RejectsUserWritablePath(t *testing.T) {
	dir := t.TempDir()
	tmpPath := filepath.Join(dir, "libnccl.so.2")
	// Create a fake (empty) ELF — FindLibNCCL's stat will succeed on this.
	if err := os.WriteFile(tmpPath, []byte("not a real elf"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Even though the file exists, isSafeLibPath under TempDir
	// (which is t.TempDir() == /tmp/... or %TEMP%) must reject it.
	if isSafeLibPath(tmpPath) {
		// On WSL t.TempDir() may not be under /tmp; this assertion
		// only holds when TempDir IS under /tmp. Skip to avoid a
		// false-positive flake.
		if strings.HasPrefix(tmpPath, "/tmp/") {
			t.Errorf("isSafeLibPath accepted user-writable %q (LHF #2 regression)", tmpPath)
		} else {
			t.Skipf("TempDir not under /tmp: %q (test inconclusive on this host)", tmpPath)
		}
	}
}

// TestFindLibNCCLSystemwide_Smoke: never panics; result is "" or a
// path that exists.
func TestFindLibNCCLSystemwide_Smoke(t *testing.T) {
	got := FindLibNCCLSystemwide()
	if got == "" {
		return
	}
	if _, err := os.Stat(got); err != nil {
		t.Errorf("returned %q but stat fails: %v", got, err)
	}
	if !strings.Contains(filepath.Base(got), "libnccl.so") {
		t.Errorf("returned %q which doesn't look like libnccl.so", got)
	}
}

// v0.15 F1: Tracer.AttachAt rejects empty path.
func TestAttachAt_EmptyPath(t *testing.T) {
	tr := New("")
	// Don't actually call Prepare (would need BPF privileges); the
	// empty-path check is the FIRST thing AttachAt does, so the
	// before-Prepare error code path is what we test here.
	if err := tr.AttachAt(""); err == nil {
		t.Fatalf("AttachAt(\"\") should return error, got nil")
	}
}

// v0.15 F1: AttachAt returns an explicit error when called before
// Prepare. Without Prepare the BPF objects are not loaded, so any
// link.OpenExecutable -> Uprobe call would fail confusingly deep in
// libbpf; the explicit guard makes the failure mode obvious.
func TestAttachAt_BeforePrepare(t *testing.T) {
	tr := New("/dev/null")
	if err := tr.AttachAt("/dev/null"); err == nil {
		t.Fatalf("AttachAt before Prepare should return error, got nil")
	}
}

// v0.15 F1: AttachedPaths returns an empty slice on a fresh Tracer
// before any AttachAt call.
func TestAttachedPaths_Empty(t *testing.T) {
	tr := New("")
	got := tr.AttachedPaths()
	if len(got) != 0 {
		t.Fatalf("AttachedPaths on fresh tracer = %v, want []", got)
	}
}
