package cli

import (
	"errors"
	"log/slog"
	"sync"
	"testing"

	"github.com/ingero-io/ingero/internal/ebpf/ncclprobe"
)

// fakeAttacher is an in-memory implementation of ncclAttacher for
// unit tests. Records AttachAt calls + return errors per path.
type fakeAttacher struct {
	mu       sync.Mutex
	calls    []string         // libPath of every AttachAt call (in order)
	attached map[string]bool  // current "attached" set (the bool == success)
	errOn    map[string]error // optional per-path error
}

func newFakeAttacher() *fakeAttacher {
	return &fakeAttacher{
		attached: map[string]bool{},
		errOn:    map[string]error{},
	}
}

func (f *fakeAttacher) AttachAt(libPath string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, libPath)
	if err, ok := f.errOn[libPath]; ok {
		return err
	}
	f.attached[libPath] = true
	return nil
}

func (f *fakeAttacher) AttachedPaths() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]string, 0, len(f.attached))
	for k := range f.attached {
		out = append(out, k)
	}
	return out
}

// v0.15 F1 (HIGH): dispatchNCCLAttach returns 0 + does no calls when
// the attacher is nil (no --nccl set, scanner runs only for the
// gauge metrics).
func TestDispatchNCCLAttach_NilAttacherIsNoop(t *testing.T) {
	got := dispatchNCCLAttach([]ncclprobe.NCCLProcess{
		{PID: 1, LibPath: "/usr/lib/libnccl.so"},
	}, nil, slog.Default())
	if got != 0 {
		t.Errorf("nil attacher should return 0, got %d", got)
	}
}

// v0.15 F1 (HIGH): de-duplicates by LibPath within a single batch.
// Five processes sharing the same libnccl path produce one AttachAt
// call, not five.
func TestDispatchNCCLAttach_DedupesPathsWithinBatch(t *testing.T) {
	att := newFakeAttacher()
	batch := []ncclprobe.NCCLProcess{
		{PID: 100, LibPath: "/usr/lib/libnccl.so"},
		{PID: 101, LibPath: "/usr/lib/libnccl.so"},
		{PID: 102, LibPath: "/usr/lib/libnccl.so"},
	}
	dispatchNCCLAttach(batch, att, slog.Default())
	if len(att.calls) != 1 {
		t.Errorf("expected 1 AttachAt call, got %d (%v)", len(att.calls), att.calls)
	}
}

// v0.15 F1 (HIGH): different paths each get one AttachAt call.
func TestDispatchNCCLAttach_AdditiveAcrossPaths(t *testing.T) {
	att := newFakeAttacher()
	batch := []ncclprobe.NCCLProcess{
		{PID: 100, LibPath: "/usr/lib/libnccl.so"},
		{PID: 101, LibPath: "/opt/python/.../libnccl.so.2"},
		{PID: 102, LibPath: "/opt/conda/.../libnccl.so"},
	}
	got := dispatchNCCLAttach(batch, att, slog.Default())
	if len(att.calls) != 3 {
		t.Errorf("expected 3 AttachAt calls, got %d (%v)", len(att.calls), att.calls)
	}
	if got != 3 {
		t.Errorf("expected new_paths=3, got %d", got)
	}
}

// v0.15 F1 (HIGH): empty LibPath skips the AttachAt call (some
// processes have python loaded but not libnccl yet; we should not
// drive an AttachAt with "").
func TestDispatchNCCLAttach_SkipsEmptyLibPath(t *testing.T) {
	att := newFakeAttacher()
	batch := []ncclprobe.NCCLProcess{
		{PID: 100, LibPath: ""},
		{PID: 101, LibPath: "/opt/python/.../libnccl.so.2"},
	}
	dispatchNCCLAttach(batch, att, slog.Default())
	if len(att.calls) != 1 || att.calls[0] != "/opt/python/.../libnccl.so.2" {
		t.Errorf("expected single non-empty call, got %v", att.calls)
	}
}

// v0.15 F1 (MED): AttachAt errors are caught + logged (debug); they
// must NOT crash the loop or skip the rest of the batch. Verify
// remaining paths still get tried.
func TestDispatchNCCLAttach_ErrorOnOnePathDoesNotSkipOthers(t *testing.T) {
	att := newFakeAttacher()
	att.errOn["/bad/libnccl.so"] = errors.New("attach failed")
	batch := []ncclprobe.NCCLProcess{
		{PID: 100, LibPath: "/bad/libnccl.so"},
		{PID: 101, LibPath: "/good/libnccl.so"},
	}
	dispatchNCCLAttach(batch, att, slog.Default())
	if len(att.calls) != 2 {
		t.Errorf("expected both AttachAt attempted, got %v", att.calls)
	}
	// /good should be in attached set; /bad should not.
	gotPaths := att.AttachedPaths()
	if len(gotPaths) != 1 || gotPaths[0] != "/good/libnccl.so" {
		t.Errorf("expected only /good to attach, got %v", gotPaths)
	}
}

// v0.15 F1 (HIGH): empty batch returns 0, makes no calls.
func TestDispatchNCCLAttach_EmptyBatch(t *testing.T) {
	att := newFakeAttacher()
	got := dispatchNCCLAttach(nil, att, slog.Default())
	if got != 0 || len(att.calls) != 0 {
		t.Errorf("empty batch should produce no work, got new=%d calls=%v", got, att.calls)
	}
}

// v0.15 F1 (MED): subsequent batches that re-discover the same
// process should not re-call AttachAt (idempotency at the dispatcher
// level + at the underlying tracer). Test the dispatcher level: if
// AttachedPaths already lists the path, no growth is reported. Note
// that dispatchNCCLAttach calls AttachAt unconditionally (Tracer's
// own idempotency-per-path makes this cheap) but the new_paths
// counter relies on AttachedPaths having grown.
func TestDispatchNCCLAttach_NoGrowthOnRepeatedBatch(t *testing.T) {
	att := newFakeAttacher()
	att.attached["/usr/lib/libnccl.so"] = true // already attached
	batch := []ncclprobe.NCCLProcess{
		{PID: 100, LibPath: "/usr/lib/libnccl.so"},
	}
	got := dispatchNCCLAttach(batch, att, slog.Default())
	if got != 0 {
		t.Errorf("expected 0 new paths on repeat, got %d", got)
	}
}

// v0.15 F1 regression (2026-05-07 Lambda A10 finding):
// dispatchNCCLAttach must NOT panic when called with a typed-nil
// *ncclprobe.Tracer. The agent passes (*Tracer)(nil) returned by
// setupNCCLTracer when --nccl is off, and Go widens that to a
// non-nil ncclAttacher interface; a naive `att == nil` check is
// FALSE and the next method call segfaults. Found running with
// --prometheus on a host without --nccl.
func TestDispatchNCCLAttach_TypedNilTracerNoPanic(t *testing.T) {
	var nt *ncclprobe.Tracer // typed nil widened to interface
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("typed-nil tracer panicked: %v", r)
		}
	}()
	got := dispatchNCCLAttach(
		[]ncclprobe.NCCLProcess{{PID: 1, LibPath: "/some/libnccl.so"}},
		nt,
		slog.Default(),
	)
	if got != 0 {
		t.Errorf("typed-nil tracer should report 0 newly-attached; got %d", got)
	}
}

func TestIsNilNCCLAttacher(t *testing.T) {
	var nt *ncclprobe.Tracer
	if !isNilNCCLAttacher(nt) {
		t.Errorf("typed-nil *Tracer should be reported as nil")
	}
	live := newFakeAttacher()
	if isNilNCCLAttacher(live) {
		t.Errorf("live fakeAttacher should NOT be reported as nil")
	}
}
