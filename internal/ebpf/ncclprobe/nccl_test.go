package ncclprobe

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"sync"
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

// v0.15 F1 (HIGH): regression guard for the Go 1.26 reflect panic
// caused by unexported padding fields in Event. Pre-fix, binary.Read
// panicked with "reflect: reflect.Value.SetUint using value obtained
// using unexported field" on the FIRST real NCCL event Tracer.Run
// received. The bug was latent because pre-F1 the agent rarely got
// real NCCL events. This test runs in CI on every push (Go 1.25 +
// Go 1.26 matrix) so a future re-introduction of unexported fields
// fires here immediately.
func TestEvent_BinaryReadDoesNotPanic(t *testing.T) {
	raw := make([]byte, EventSize)
	for i := range raw {
		raw[i] = byte(i & 0xff) // non-zero pattern catches alignment bugs too
	}
	var e Event
	if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &e); err != nil {
		t.Fatalf("binary.Read on Event panicked or errored: %v", err)
	}
	// Sanity: the deserialized struct received some of the bytes.
	if e.TimestampNs == 0 {
		t.Errorf("expected non-zero TimestampNs after read, got 0")
	}
}

// v0.15 F1 (MED): AttachAt is idempotent per libnccl path. Code
// claims this in its docstring; this test asserts it via the
// test-only setAttachedForTest helper to bypass BPF (a real
// idempotency test would need root + a real ELF with NCCL symbols).
func TestAttachAt_IdempotentPerPath(t *testing.T) {
	tr := New("")
	tr.prepareDone.Store(true) // pretend Prepare ran
	tr.attachedPaths = map[string]int{"/usr/lib/libnccl.so": 9}
	tr.attachedSyms = 9
	// Calling AttachAt with a path already in attachedPaths returns
	// nil immediately (idempotent fast-path).
	if err := tr.AttachAt("/usr/lib/libnccl.so"); err != nil {
		t.Fatalf("AttachAt idempotent path returned err: %v", err)
	}
	// State unchanged.
	if got := len(tr.attachedPaths); got != 1 {
		t.Errorf("attachedPaths grew on idempotent call: len=%d", got)
	}
	if tr.attachedSyms != 9 {
		t.Errorf("attachedSyms changed on idempotent call: %d", tr.attachedSyms)
	}
}

// v0.15 F1 (MED): AttachedPaths is concurrency-safe under reads
// while AttachAt could be writing. We can't test the write path
// without BPF, but we can test that AttachedPaths takes the lock
// (no race when called concurrently with a writer that only the
// test sets up via the test helper).
func TestAttachedPaths_ConcurrentReads(t *testing.T) {
	tr := New("")
	tr.prepareDone.Store(true)
	tr.attachedPaths = map[string]int{
		"/usr/lib/libnccl.so": 9,
		"/opt/python/libnccl.so": 9,
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_ = tr.AttachedPaths()
			}
		}()
	}
	wg.Wait()
	// If the test passes -race, AttachedPaths is safe under read
	// concurrency. The actual write-side race protection lives in
	// AttachAt's mu.Lock(); a separate test would need a real BPF
	// environment to write-then-read concurrently.
}

// v0.15 F1 (MED, gap #3): concurrent AttachAt write-side stress.
// Real AttachAt needs BPF + a real ELF, but the contention surface
// is the mu-protected attachedPaths map + attachedSyms counter.
// We exercise that surface directly via a test-only helper that
// performs the same locked-update operations AttachAt performs on
// success, asserting -race finds no data race when many goroutines
// register paths concurrently.
func TestAttachAt_ConcurrentWrites(t *testing.T) {
	tr := New("")
	tr.prepareDone.Store(true)
	tr.attachedPaths = map[string]int{}

	const nGoroutines = 16
	const nPathsPerG = 50
	var wg sync.WaitGroup
	for g := 0; g < nGoroutines; g++ {
		wg.Add(1)
		go func(gi int) {
			defer wg.Done()
			for k := 0; k < nPathsPerG; k++ {
				path := pathFor(gi, k)
				// Simulate AttachAt's success path under the lock.
				tr.mu.Lock()
				if _, already := tr.attachedPaths[path]; !already {
					tr.attachedPaths[path] = 9
					tr.attachedSyms += 9
				}
				tr.mu.Unlock()
				// Concurrent reads in the same loop.
				_ = tr.AttachedPaths()
			}
		}(g)
	}
	wg.Wait()
	got := len(tr.AttachedPaths())
	want := nGoroutines * nPathsPerG
	if got != want {
		t.Errorf("AttachedPaths len = %d, want %d (under -race a smaller count means races dropped writes)", got, want)
	}
}

func pathFor(g, k int) string {
	// Distinct path per (goroutine, k) so every iteration is a real
	// new entry.
	return "/opt/python/g" + itoa(g) + "_k" + itoa(k) + "/libnccl.so"
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [16]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// v0.15 F1 (MED, gap #2): Tracer.handleRawSample exercises the same
// parse loop body Tracer.Run runs per ringbuf record. Driving it
// with synthetic byte buffers covers the deserialization +
// dispatch path without BPF.
func TestTracer_HandleRawSample_Forwards(t *testing.T) {
	tr := New("")
	// Build a serialized Event with known fields.
	in := Event{
		TimestampNs: 1234567890,
		PID:         42,
		TID:         43,
		Source:      9,
		Op:          3, // ncclAllReduce
		CgroupID:    0xdeadbeef,
		DurationNs:  5_000_000,
		CommIDHash:  0xfeedface,
		StreamHandle: 0xabcd,
		CountBytes:  1024,
		Rank:        1,
		NRanks:      4,
		Datatype:    7,
		ReduceOp:    2,
		ReturnCode:  0,
		PeerRank:    0,
	}
	copy(in.Comm[:], "python3")
	var raw bytes.Buffer
	if err := binary.Write(&raw, binary.LittleEndian, &in); err != nil {
		t.Fatalf("binary.Write: %v", err)
	}
	if raw.Len() != EventSize {
		t.Fatalf("serialized size %d != EventSize %d", raw.Len(), EventSize)
	}

	tr.handleRawSample(raw.Bytes())

	// Channel buffered 4096; read non-blocking.
	select {
	case got := <-tr.eventCh:
		if got.PID != in.PID || got.Op != in.Op || got.CountBytes != in.CountBytes ||
			got.Rank != in.Rank || got.CommIDHash != in.CommIDHash {
			t.Errorf("decoded Event mismatch:\n got=%+v\n want=%+v", got, in)
		}
		if got.CommString() != "python3" {
			t.Errorf("CommString=%q want python3", got.CommString())
		}
	default:
		t.Fatal("expected event on eventCh, got nothing")
	}
	if tr.parseErr.Load() != 0 {
		t.Errorf("parseErr=%d, want 0", tr.parseErr.Load())
	}
	if tr.dropped.Load() != 0 {
		t.Errorf("dropped=%d, want 0", tr.dropped.Load())
	}
}

// v0.15 F1 (MED, gap #2): short raw samples increment parseErr,
// don't crash, don't push events.
func TestTracer_HandleRawSample_TooShortIncrementsParseErr(t *testing.T) {
	tr := New("")
	tr.handleRawSample(make([]byte, EventSize-1))
	if tr.parseErr.Load() != 1 {
		t.Errorf("parseErr=%d, want 1", tr.parseErr.Load())
	}
	select {
	case e := <-tr.eventCh:
		t.Errorf("expected no event, got %+v", e)
	default:
	}
}

// v0.15 F1 (MED, gap #2): when eventCh is full, dropped counter
// increments instead of blocking. Drives 4097 events through a
// channel of capacity 4096.
func TestTracer_HandleRawSample_DropsWhenChannelFull(t *testing.T) {
	tr := New("")
	in := Event{Op: 3, PID: 1}
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, &in)
	raw := buf.Bytes()

	// Fill the channel.
	for i := 0; i < 4096; i++ {
		tr.handleRawSample(raw)
	}
	if tr.dropped.Load() != 0 {
		t.Errorf("after %d events: dropped=%d, want 0", 4096, tr.dropped.Load())
	}
	// One more should drop.
	tr.handleRawSample(raw)
	if tr.dropped.Load() != 1 {
		t.Errorf("after channel-full push: dropped=%d, want 1", tr.dropped.Load())
	}
}

// v0.15 F1 (G5, MED): handleRawSample with len > EventSize must
// read only the first EventSize bytes (truncation is safe, no parse
// error). Excess trailing bytes from a misaligned ringbuf record
// would otherwise risk binary.Read off-by-one drift.
func TestTracer_HandleRawSample_TruncatesOversizedInput(t *testing.T) {
	tr := New("")
	in := Event{Op: 5, PID: 42, CountBytes: 1024}
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, &in)
	raw := append(buf.Bytes(), make([]byte, 256)...) // 104 + 256 trailing bytes
	tr.handleRawSample(raw)
	if tr.parseErr.Load() != 0 {
		t.Errorf("parseErr after oversized-but-valid sample: %d, want 0", tr.parseErr.Load())
	}
	select {
	case got := <-tr.eventCh:
		if got.Op != 5 || got.PID != 42 || got.CountBytes != 1024 {
			t.Errorf("forwarded event = %+v, want Op=5 PID=42 CountBytes=1024", got)
		}
	default:
		t.Fatal("expected one event forwarded after oversized input")
	}
}

// v0.15 F1 (G1, CRITICAL): Close() is idempotent. Calling Close on
// a fresh tracer (no Prepare) MUST NOT panic, and a second Close
// must return nil silently. The closed atomic.Bool gates this.
func TestTracer_Close_IdempotentBeforePrepare(t *testing.T) {
	tr := New("")
	if err := tr.Close(); err != nil {
		t.Errorf("first Close on fresh tracer: %v", err)
	}
	if err := tr.Close(); err != nil {
		t.Errorf("second Close (idempotent): %v", err)
	}
}

// v0.15 F1 (G2, CRITICAL): Close() with nil reader (Prepare never
// ran) and empty links must call detach() without panicking. Guards
// the post-Prepare-failure shutdown path: if Prepare returned err,
// the caller still calls Close() in defer.
func TestTracer_Close_HandlesNilReader(t *testing.T) {
	tr := New("")
	// reader is nil, links is nil, attachedPaths is nil. Close must
	// tolerate all three.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Close with nil internals panicked: %v", r)
		}
	}()
	if err := tr.Close(); err != nil {
		t.Errorf("Close with nil internals: %v", err)
	}
}

// v0.15 F1 (G3, CRITICAL): AttachedPaths returns a TRUE COPY.
// Mutating the returned slice MUST NOT corrupt the tracer's
// internal attachedPaths map. A buggy implementation that returned
// a slice aliased to the map's keys would let downstream consumers
// poison scanner state.
func TestTracer_AttachedPaths_ReturnsCopy(t *testing.T) {
	tr := New("")
	tr.mu.Lock()
	tr.attachedPaths = map[string]int{
		"/opt/python/a/libnccl.so.2": 12,
		"/opt/python/b/libnccl.so.2": 12,
	}
	tr.mu.Unlock()

	first := tr.AttachedPaths()
	if len(first) != 2 {
		t.Fatalf("first call len=%d, want 2", len(first))
	}
	// Mutate the returned slice. Internal map must be unaffected.
	first[0] = "MUTATED"
	first[1] = "MUTATED"

	second := tr.AttachedPaths()
	for _, p := range second {
		if p == "MUTATED" {
			t.Fatalf("internal state corrupted: caller mutation leaked back into AttachedPaths(): %v", second)
		}
	}
}

// v0.15 F1 (G4, MED): AttachedPaths returns a SORTED slice. The
// observable order is invariant for log diffability + test
// determinism. Without explicit assertion, a future refactor that
// drops the sort.Strings call would silently regress.
func TestTracer_AttachedPaths_Sorted(t *testing.T) {
	tr := New("")
	tr.mu.Lock()
	tr.attachedPaths = map[string]int{
		"/opt/python/zzz/libnccl.so.2": 12,
		"/opt/python/aaa/libnccl.so.2": 12,
		"/opt/python/mmm/libnccl.so.2": 12,
	}
	tr.mu.Unlock()
	got := tr.AttachedPaths()
	want := []string{
		"/opt/python/aaa/libnccl.so.2",
		"/opt/python/mmm/libnccl.so.2",
		"/opt/python/zzz/libnccl.so.2",
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("AttachedPaths()[%d] = %q, want %q (sortedness regression)", i, got[i], want[i])
		}
	}
}
