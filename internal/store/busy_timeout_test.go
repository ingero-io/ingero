package store

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/annotate"
)

// TestBusyTimeout_AnnotationUnderWriteContention is the regression test
// for the v0.17.0 SQLITE_BUSY bug: an annotation INSERT contended with
// the event-batch writer and was silently dropped.
//
// Root cause: busy_timeout is a per-connection pragma. The old code set
// it with a single post-open db.Exec, so it landed on exactly one pooled
// connection; every other connection in the *sql.DB pool kept
// busy_timeout=0 and failed instantly with SQLITE_BUSY on any write
// contention. The fix moves busy_timeout into the DSN (_pragma=...) so
// modernc.org/sqlite applies it to every connection it opens.
//
// The competing writer here is a second connection drawn from the
// store's OWN pool (s.db.Conn) - exactly the production scenario, where
// the event-batch / aggregate-flush writer and RecordAnnotation both run
// against s.db. Each round, that connection takes the WAL write lock for
// a bounded interval shorter than busy_timeout, then releases it. With
// the fix, the RecordAnnotation connection waits out that interval and
// succeeds; without it, RecordAnnotation lands on a no-timeout
// connection and returns "database is locked" immediately.
//
// Every RecordAnnotation call must return nil and every row must be
// persisted. Rounds are a synchronized handoff so contention is
// deterministic and the annotation writer is never starved past
// busy_timeout.
//
// Must be run under -race.
func TestBusyTimeout_AnnotationUnderWriteContention(t *testing.T) {
	// A file-backed DB is required: :memory: shared-cache does not
	// exercise the WAL write lock the same way a real file does, and the
	// production failure was against a file DB.
	dbPath := filepath.Join(t.TempDir(), "trace.db")

	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	// Force the pool to keep several connections live so RecordAnnotation
	// is very likely served by a connection other than the one any single
	// post-open pragma Exec would have touched. With the bug, those extra
	// connections have busy_timeout=0.
	const poolSize = 8
	s.db.SetMaxOpenConns(poolSize)

	// Directly assert the fix: every pooled connection must carry
	// busy_timeout from the DSN. Pre-opening them also avoids a lazily
	// dialed connection running its DSN setup mid-round.
	warmConns := make([]interface{ Close() error }, 0, poolSize)
	for i := 0; i < poolSize; i++ {
		c, err := s.db.Conn(context.Background())
		if err != nil {
			t.Fatalf("pre-open pooled connection %d: %v", i, err)
		}
		var bt int
		if err := c.QueryRowContext(context.Background(), "PRAGMA busy_timeout").Scan(&bt); err != nil {
			t.Fatalf("read busy_timeout on connection %d: %v", i, err)
		}
		if bt != busyTimeoutMillis {
			t.Fatalf("pooled connection %d has busy_timeout=%d, want %d (the SQLITE_BUSY bug)",
				i, bt, busyTimeoutMillis)
		}
		warmConns = append(warmConns, c)
	}
	for _, c := range warmConns {
		c.Close() // returns the connection to the pool, still open
	}

	const rounds = 40
	// holdFor is far shorter than busy_timeout (5000ms) so a correctly
	// configured RecordAnnotation connection waits it out and succeeds.
	const holdFor = 150 * time.Millisecond

	// Per round: the holder closes locked once the WAL write lock is
	// held; the annotation writer closes recorded once its INSERT has
	// returned. The holder waits on the previous round's recorded before
	// starting the next round, so the holder cannot race ahead and
	// starve a still-retrying annotation write across many rounds.
	type round struct {
		locked   chan struct{}
		recorded chan struct{}
	}
	rs := make([]round, rounds)
	for i := range rs {
		rs[i] = round{locked: make(chan struct{}), recorded: make(chan struct{})}
	}

	var wg sync.WaitGroup

	// Holder goroutine: a stand-in for the event-batch / aggregate-flush
	// writer. It draws a connection from the store's own pool and, per
	// round, takes the WAL write lock via a write transaction, holds it
	// for holdFor, then commits.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < rounds; i++ {
			// Wait for the previous round's annotation write to finish
			// so the holder never grabs the lock again while an earlier
			// RecordAnnotation is still retrying.
			if i > 0 {
				<-rs[i-1].recorded
			}
			conn, err := s.db.Conn(context.Background())
			if err != nil {
				t.Errorf("round %d holder Conn: %v", i, err)
				close(rs[i].locked)
				continue
			}
			tx, err := conn.BeginTx(context.Background(), nil)
			if err != nil {
				t.Errorf("round %d holder Begin: %v", i, err)
				conn.Close()
				close(rs[i].locked)
				continue
			}
			// A write forces the WAL write lock to be acquired now and
			// held until commit.
			if _, err := tx.ExecContext(context.Background(),
				"INSERT INTO annotations (timestamp, labels) VALUES (?, ?)",
				time.Now().UnixNano(), `{"holder":"1"}`,
			); err != nil {
				t.Errorf("round %d holder INSERT: %v", i, err)
				tx.Rollback()
				conn.Close()
				close(rs[i].locked)
				continue
			}
			close(rs[i].locked)
			time.Sleep(holdFor)
			tx.Commit()
			conn.Close()
		}
	}()

	// Annotation goroutine: the call that used to silently drop. Each
	// round it waits until the holder has the lock, then issues the
	// INSERT directly into the contended state.
	recordErrs := make([]error, rounds)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < rounds; i++ {
			<-rs[i].locked
			a := annotate.Annotation{
				TimestampNs: time.Now().UnixNano(),
				Labels:      map[string]string{"step": "x"},
				Process:     annotate.ProcessIncarnation{PID: uint32(1000 + i)},
			}
			// RecordAnnotation runs while the holder still has the WAL
			// write lock. With the fix, the connection waits out the
			// hold and succeeds; without it, it returns SQLITE_BUSY
			// immediately.
			recordErrs[i] = s.RecordAnnotation(a)
			close(rs[i].recorded)
		}
	}()

	wg.Wait()

	var failed int
	var firstErr error
	for i, err := range recordErrs {
		if err != nil {
			failed++
			t.Logf("round %d failed: %v", i, err)
			if firstErr == nil {
				firstErr = err
			}
		}
	}
	if failed != 0 {
		t.Errorf("RecordAnnotation failed %d/%d times; first error: %v", failed, rounds, firstErr)
		if firstErr != nil && strings.Contains(firstErr.Error(), "locked") {
			t.Errorf("regression: annotation write dropped with SQLITE_BUSY under write contention")
		}
	}

	// Every annotation must actually be on disk.
	got, err := s.QueryAnnotations(AnnotationQuery{Since: time.Hour, Limit: -1})
	if err != nil {
		t.Fatalf("QueryAnnotations: %v", err)
	}
	// Count only the rows the annotation goroutine wrote (label step=x);
	// the holder goroutine also inserted rows.
	var persisted int
	for _, g := range got {
		if g.Labels["step"] == "x" {
			persisted++
		}
	}
	if persisted != rounds {
		t.Errorf("persisted %d annotations, want %d", persisted, rounds)
	}
}

// TestBuildWriteDSN_CarriesBusyTimeout asserts the DSN builders put
// busy_timeout into the connection string for the file, in-memory, and
// read-only cases, so every pooled connection inherits it.
func TestBuildWriteDSN_CarriesBusyTimeout(t *testing.T) {
	cases := []struct {
		name string
		path string
		want []string
	}{
		{"file", "/var/lib/ingero/trace.db", []string{"file:/var/lib/ingero/trace.db", "busy_timeout(5000)"}},
		{"memory", ":memory:", []string{"file::memory:", "cache=shared", "busy_timeout(5000)"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dsn := buildWriteDSN(c.path)
			for _, sub := range c.want {
				if !strings.Contains(dsn, sub) {
					t.Errorf("buildWriteDSN(%q) = %q, missing %q", c.path, dsn, sub)
				}
			}
		})
	}

	ro := buildReadOnlyDSN("/var/lib/ingero/trace.db")
	if !strings.Contains(ro, "busy_timeout(5000)") || !strings.Contains(ro, "_query_only=1") {
		t.Errorf("buildReadOnlyDSN missing busy_timeout or _query_only: %q", ro)
	}
}
