package annotate

import (
	"fmt"
	"net"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/annotate"
	"github.com/ingero-io/ingero/pkg/contract"
)

// memSink is an in-memory Sink for tests.
type memSink struct {
	mu   sync.Mutex
	rows []annotate.Annotation
	fail bool // when true, RecordAnnotation returns an error
}

func (m *memSink) RecordAnnotation(a annotate.Annotation) error {
	if m.fail {
		return fmt.Errorf("memSink: forced failure")
	}
	m.mu.Lock()
	m.rows = append(m.rows, a)
	m.mu.Unlock()
	return nil
}

func (m *memSink) count() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.rows)
}

func (m *memSink) snapshot() []annotate.Annotation {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]annotate.Annotation, len(m.rows))
	copy(out, m.rows)
	return out
}

// startServer builds a server bound to a per-test socket directory so
// tests do not collide on the shared contract path.
func startServer(t *testing.T, sink Sink) *Server {
	t.Helper()
	dir := t.TempDir()
	s := &Server{
		socketDir:  dir,
		socketPath: dir + "/" + contract.AnnotationSocketName,
		socketGid:  -1,
		sink:       sink,
		resolver:   annotate.ResolveIncarnation,
		conns:      make(map[net.Conn]struct{}),
	}
	if err := s.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// waitFor polls cond until it is true or the deadline elapses.
func waitFor(t *testing.T, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for: %s", msg)
}

func dial(t *testing.T, s *Server) net.Conn {
	t.Helper()
	c, err := net.Dial("unix", s.socketPath)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	return c
}

func TestServer_HappyPathRoundTrip(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)

	c := dial(t, s)
	line := `{"labels":{"step":"7"}}` + "\n"
	if _, err := c.Write([]byte(line)); err != nil {
		t.Fatalf("write: %v", err)
	}
	c.Close()

	waitFor(t, func() bool { return sink.count() == 1 }, "one annotation ingested")
	got := sink.snapshot()[0]
	if got.Labels["step"] != "7" {
		t.Errorf("labels = %v", got.Labels)
	}
	if got.TimestampNs == 0 {
		t.Error("expected receive-time stamp when ts omitted")
	}
}

func TestServer_ConcurrentWriters(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)

	const writers = 8
	const perWriter = 25
	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			c, err := net.Dial("unix", s.socketPath)
			if err != nil {
				t.Errorf("writer %d dial: %v", id, err)
				return
			}
			defer c.Close()
			for i := 0; i < perWriter; i++ {
				line := fmt.Sprintf(`{"labels":{"w":"%d","i":"%d"}}`+"\n", id, i)
				if _, err := c.Write([]byte(line)); err != nil {
					t.Errorf("writer %d write: %v", id, err)
					return
				}
			}
		}(w)
	}
	wg.Wait()
	waitFor(t, func() bool { return sink.count() == writers*perWriter },
		"all concurrent annotations ingested")
}

func TestServer_PartialNDJSONSplit(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)

	c := dial(t, s)
	defer c.Close()
	// Write one logical line in three chunks, with a delay so the
	// server's scanner sees a partial buffer.
	chunks := []string{`{"labels":{"st`, `ep":"3"`, "}}\n"}
	chunks[2] = `}}` + "\n"
	for _, ch := range chunks {
		if _, err := c.Write([]byte(ch)); err != nil {
			t.Fatalf("write chunk: %v", err)
		}
		time.Sleep(20 * time.Millisecond)
	}
	waitFor(t, func() bool { return sink.count() == 1 },
		"split line reassembled and ingested")
}

func TestServer_MalformedLineRejectedListenerSurvives(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)

	c := dial(t, s)
	// A non-JSON line, then a valid one on the same connection.
	if _, err := c.Write([]byte("not json at all\n")); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := c.Write([]byte(`{"labels":{"ok":"1"}}` + "\n")); err != nil {
		t.Fatalf("write: %v", err)
	}
	c.Close()

	waitFor(t, func() bool { return sink.count() == 1 },
		"valid line after a malformed one still ingested")
	if r := s.ReadStats().Rejected; r < 1 {
		t.Errorf("expected at least one rejected line, got %d", r)
	}

	// A fresh connection still works - the listener survived.
	c2 := dial(t, s)
	c2.Write([]byte(`{"labels":{"after":"1"}}` + "\n"))
	c2.Close()
	waitFor(t, func() bool { return sink.count() == 2 },
		"listener survived the malformed line")
}

func TestServer_OversizedLineRejected(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)

	c := dial(t, s)
	huge := strings.Repeat("x", contract.AnnotationMaxLineBytes+1024)
	line := `{"labels":{"k":"` + huge + `"}}` + "\n"
	c.Write([]byte(line))
	c.Close()

	waitFor(t, func() bool { return s.ReadStats().Rejected >= 1 },
		"oversized line rejected")
	if sink.count() != 0 {
		t.Errorf("oversized line should not be ingested, got %d rows", sink.count())
	}
}

func TestServer_ConnectNeverSend(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)

	c := dial(t, s)
	waitFor(t, func() bool { return s.ReadStats().Accepted >= 1 },
		"silent connection accepted")
	// Hold the connection open briefly, then close without sending.
	time.Sleep(50 * time.Millisecond)
	c.Close()
	// A subsequent writer still works.
	c2 := dial(t, s)
	c2.Write([]byte(`{"labels":{"k":"v"}}` + "\n"))
	c2.Close()
	waitFor(t, func() bool { return sink.count() == 1 },
		"server survives an idle connection")
}

func TestServer_MidLineDisconnect(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)

	c := dial(t, s)
	// Write a partial line (no newline) then abruptly close.
	c.Write([]byte(`{"labels":{"partial":`))
	c.Close()

	// The partial line is never completed, so nothing is ingested, and
	// the server does not crash. A later writer still works.
	c2 := dial(t, s)
	c2.Write([]byte(`{"labels":{"k":"v"}}` + "\n"))
	c2.Close()
	waitFor(t, func() bool { return sink.count() == 1 },
		"server survives a mid-line disconnect")
}

func TestServer_PeerCredCaptured(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)

	c := dial(t, s)
	c.Write([]byte(`{"labels":{"k":"v"}}` + "\n"))
	c.Close()
	waitFor(t, func() bool { return sink.count() == 1 }, "ingested")

	got := sink.snapshot()[0]
	if got.Provenance.PeerUID != uint32(os.Getuid()) {
		t.Errorf("PeerUID = %d, want %d", got.Provenance.PeerUID, os.Getuid())
	}
	if got.Provenance.PeerPID != uint32(os.Getpid()) {
		t.Errorf("PeerPID = %d, want %d", got.Provenance.PeerPID, os.Getpid())
	}
}

func TestServer_DefaultSocketMode0700(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)
	info, err := os.Stat(s.socketPath)
	if err != nil {
		t.Fatalf("stat socket: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o700 {
		t.Errorf("default socket mode = %#o, want 0700", perm)
	}
}

func TestServer_SocketGidOptIn(t *testing.T) {
	sink := &memSink{}
	dir := t.TempDir()
	s := &Server{
		socketDir:  dir,
		socketPath: dir + "/" + contract.AnnotationSocketName,
		socketGid:  os.Getgid(), // chown to our own gid - always permitted
		sink:       sink,
		resolver:   annotate.ResolveIncarnation,
		conns:      make(map[net.Conn]struct{}),
	}
	if err := s.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer s.Close()

	info, err := os.Stat(s.socketPath)
	if err != nil {
		t.Fatalf("stat socket: %v", err)
	}
	// chown to our own gid succeeds, so the mode is widened to 0770.
	if perm := info.Mode().Perm(); perm != 0o770 {
		t.Errorf("gid-opt-in socket mode = %#o, want 0770", perm)
	}
}

func TestServer_ValidationRejections(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)

	bad := []string{
		`{"labels":{}}`,                                    // no labels
		`{"labels":{"bad key":"v"}}`,                       // invalid key charset
		`{"labels":{"k":"v"},"span_start":9,"span_end":2}`, // reversed span
		`{}`, // no labels field at all
	}
	c := dial(t, s)
	for _, b := range bad {
		c.Write([]byte(b + "\n"))
	}
	// One valid line so we can wait for the connection to be drained.
	c.Write([]byte(`{"labels":{"good":"1"}}` + "\n"))
	c.Close()

	waitFor(t, func() bool { return sink.count() == 1 }, "only the valid line ingested")
	if r := s.ReadStats().Rejected; int(r) < len(bad) {
		t.Errorf("rejected = %d, want >= %d", r, len(bad))
	}
}

func TestServer_RateLimitExceeded(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)

	c := dial(t, s)
	// Send more than the per-connection cap in one burst. The window
	// is 1s, so a fast burst stays inside one window.
	total := contract.AnnotationConnRateLimit + 200
	var buf strings.Builder
	for i := 0; i < total; i++ {
		buf.WriteString(`{"labels":{"k":"v"}}` + "\n")
	}
	c.Write([]byte(buf.String()))
	c.Close()

	waitFor(t, func() bool {
		return s.ReadStats().Ingested+s.ReadStats().Rejected >= uint64(total)
	}, "all burst lines processed")
	if got := s.ReadStats().Ingested; got > uint64(contract.AnnotationConnRateLimit) {
		t.Errorf("ingested %d, expected the rate cap %d to hold",
			got, contract.AnnotationConnRateLimit)
	}
	if s.ReadStats().Rejected == 0 {
		t.Error("expected rate-exceeded lines to be rejected")
	}
}

func TestServer_SinkError(t *testing.T) {
	sink := &memSink{fail: true}
	s := startServer(t, sink)

	c := dial(t, s)
	c.Write([]byte(`{"labels":{"k":"v"}}` + "\n"))
	c.Close()
	waitFor(t, func() bool { return s.ReadStats().SinkError >= 1 },
		"sink error counted")
	if s.ReadStats().Ingested != 0 {
		t.Error("a sink-rejected line must not count as ingested")
	}
}

func TestServer_IncarnationResolution(t *testing.T) {
	sink := &memSink{}
	dir := t.TempDir()
	s := &Server{
		socketDir:  dir,
		socketPath: dir + "/" + contract.AnnotationSocketName,
		socketGid:  -1,
		sink:       sink,
		// Deterministic resolver: every PID resolves to start_time 5000.
		resolver: func(pid uint32) annotate.ProcessIncarnation {
			return annotate.ProcessIncarnation{PID: pid, StartTime: 5000}
		},
		conns: make(map[net.Conn]struct{}),
	}
	if err := s.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer s.Close()

	c := dial(t, s)
	// PID only - the server resolves the start time.
	c.Write([]byte(`{"labels":{"k":"v"},"pid":4321}` + "\n"))
	// PID + explicit start_time - the server trusts the pair.
	c.Write([]byte(`{"labels":{"k":"v"},"pid":4321,"start_time":777}` + "\n"))
	c.Close()

	waitFor(t, func() bool { return sink.count() == 2 }, "both ingested")
	rows := sink.snapshot()
	var resolved, explicit bool
	for _, r := range rows {
		if r.Process.PID == 4321 && r.Process.StartTime == 5000 {
			resolved = true
		}
		if r.Process.PID == 4321 && r.Process.StartTime == 777 {
			explicit = true
		}
	}
	if !resolved {
		t.Error("PID-only annotation should get its start_time resolved")
	}
	if !explicit {
		t.Error("PID+start_time annotation should keep the explicit pair")
	}
}

func TestServer_CloseIsIdempotent(t *testing.T) {
	sink := &memSink{}
	s := startServer(t, sink)
	if err := s.Close(); err != nil {
		t.Errorf("first Close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}
