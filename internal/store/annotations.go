package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/ingero-io/ingero/pkg/annotate"
	"github.com/ingero-io/ingero/pkg/events"
)

// annotationsSchema is the table for external annotation rows (agent
// v0.17.0). Annotations are human-meaningful labels injected from
// outside the eBPF event stream (a training step, an epoch, a Ray task
// id). The events table is untouched; annotations are a separate table
// joined at query time.
//
// Process scope is stored as the pid plus the process start_time
// (/proc field 22) the agent resolved at ingest. The query-time join
// does NOT key on start_time directly: events carry no start_time
// column, so both the annotation and the event are mapped to a
// process incarnation interval bounded by process_exec / process_exit
// events, and the join is on that interval. The stored start_time is
// retained as provenance. pid 0 means an unscoped (trace-wide)
// annotation.
//
// labels is a JSON object string. Storing the bag as JSON keeps the
// schema stable as the label vocabulary grows; the validation cap on
// label count and key/value length is enforced at ingest, so the JSON
// blob is bounded.
//
// peer_uid / peer_gid / peer_pid are the SO_PEERCRED provenance of the
// writer captured by the ingest socket. They are 0 for annotations not
// sourced from the socket.
//
// Created in applyAgentSchema so a rolled-over fresh DB carries it.
const annotationsSchema = `
CREATE TABLE IF NOT EXISTS annotations (
	id           INTEGER PRIMARY KEY AUTOINCREMENT,
	timestamp    INTEGER NOT NULL,            -- unix nanos (annotation instant)
	labels       TEXT    NOT NULL,            -- JSON object of key/value strings
	pid          INTEGER NOT NULL DEFAULT 0,  -- process scope (0 = unscoped)
	start_time   INTEGER NOT NULL DEFAULT 0,  -- /proc/<pid>/stat field 22 (incarnation key)
	span_start   INTEGER NOT NULL DEFAULT 0,  -- unix nanos (0 = instant, not a span)
	span_end     INTEGER NOT NULL DEFAULT 0,  -- unix nanos
	peer_uid     INTEGER NOT NULL DEFAULT 0,  -- SO_PEERCRED uid of the writer
	peer_gid     INTEGER NOT NULL DEFAULT 0,  -- SO_PEERCRED gid of the writer
	peer_pid     INTEGER NOT NULL DEFAULT 0   -- SO_PEERCRED pid of the writer
);
CREATE INDEX IF NOT EXISTS idx_annotations_timestamp ON annotations(timestamp);
CREATE INDEX IF NOT EXISTS idx_annotations_pid ON annotations(pid) WHERE pid != 0;
`

// RecordAnnotation persists one annotation row. Used by the ingest
// socket (one call per accepted NDJSON line) and by tests. The labels
// map is serialized to a JSON object; an empty map is rejected by
// annotate.Annotation.Validate before this is reached, but a defensive
// marshal of nil produces "null" which is harmless on read.
//
// rolloverMu is RLock'd so a concurrent rollover file-swap cannot race
// the write against a closed handle, matching the read-path pattern.
func (s *Store) RecordAnnotation(a annotate.Annotation) error {
	labelsJSON, err := json.Marshal(a.Labels)
	if err != nil {
		return fmt.Errorf("marshaling annotation labels: %w", err)
	}

	s.rolloverMu.RLock()
	defer s.rolloverMu.RUnlock()

	_, err = s.db.Exec(`
		INSERT INTO annotations
			(timestamp, labels, pid, start_time, span_start, span_end,
			 peer_uid, peer_gid, peer_pid)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		a.TimestampNs,
		string(labelsJSON),
		a.Process.PID,
		a.Process.StartTime,
		a.SpanStartNs,
		a.SpanEndNs,
		a.Provenance.PeerUID,
		a.Provenance.PeerGID,
		a.Provenance.PeerPID,
	)
	if err != nil {
		return fmt.Errorf("inserting annotation: %w", err)
	}
	return nil
}

// AnnotationQuery filters an annotation read.
type AnnotationQuery struct {
	// Since selects annotations from the last duration. From overrides
	// it when set.
	Since time.Duration
	From  time.Time
	To    time.Time
	// PID, when non-zero, restricts the result to annotations scoped to
	// that PID (and to unscoped annotations, which apply trace-wide).
	PID uint32
	// Limit caps the result. 0 = default 10000, -1 = unlimited.
	Limit int
}

// QueryAnnotations returns annotation rows matching q, newest-first.
// Parameterized SQL only; no string concatenation of caller values.
func (s *Store) QueryAnnotations(q AnnotationQuery) ([]annotate.Annotation, error) {
	query := `SELECT timestamp, labels, pid, start_time, span_start, span_end,
		peer_uid, peer_gid, peer_pid
	FROM annotations
	WHERE 1=1`
	var args []interface{}

	if !q.From.IsZero() {
		query += " AND timestamp >= ?"
		args = append(args, q.From.UnixNano())
	} else if q.Since > 0 {
		query += " AND timestamp >= ?"
		args = append(args, time.Now().Add(-q.Since).UnixNano())
	}
	if !q.To.IsZero() {
		query += " AND timestamp <= ?"
		args = append(args, q.To.UnixNano())
	}
	if q.PID != 0 {
		// Unscoped annotations (pid = 0) apply trace-wide, so they are
		// included alongside the requested PID's own annotations.
		query += " AND (pid = ? OR pid = 0)"
		args = append(args, q.PID)
	}

	query += " ORDER BY timestamp DESC"
	if q.Limit >= 0 {
		limit := q.Limit
		if limit == 0 {
			limit = 10000
		}
		query += " LIMIT ?"
		args = append(args, limit)
	}

	s.rolloverMu.RLock()
	defer s.rolloverMu.RUnlock()

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("querying annotations: %w", err)
	}
	defer rows.Close()

	return scanAnnotations(rows)
}

// scanAnnotations decodes annotation rows from a *sql.Rows whose column
// order matches the SELECT in QueryAnnotations. Shared so the rollover-
// aware join path can reuse it.
func scanAnnotations(rows *sql.Rows) ([]annotate.Annotation, error) {
	var out []annotate.Annotation
	for rows.Next() {
		var (
			ts, spanStart, spanEnd int64
			labelsJSON             string
			pid                    uint32
			startTime              uint64
			peerUID, peerGID       uint32
			peerPID                uint32
		)
		if err := rows.Scan(&ts, &labelsJSON, &pid, &startTime,
			&spanStart, &spanEnd, &peerUID, &peerGID, &peerPID); err != nil {
			return nil, fmt.Errorf("scanning annotation row: %w", err)
		}
		labels := map[string]string{}
		if labelsJSON != "" && labelsJSON != "null" {
			if err := json.Unmarshal([]byte(labelsJSON), &labels); err != nil {
				// A corrupt blob should not abort the whole read; skip
				// the row and keep going.
				continue
			}
		}
		out = append(out, annotate.Annotation{
			TimestampNs: ts,
			Labels:      labels,
			Process:     annotate.ProcessIncarnation{PID: pid, StartTime: startTime},
			SpanStartNs: spanStart,
			SpanEndNs:   spanEnd,
			Provenance: annotate.Provenance{
				PeerUID: peerUID,
				PeerGID: peerGID,
				PeerPID: peerPID,
			},
		})
	}
	return out, rows.Err()
}

// EventWithMeta is the minimal event projection the annotation join
// needs: identity, time, and the op/source tags. The event has no
// process start-time of its own; its incarnation is resolved from
// process_exec / process_exit events at join time (see AnnotateEvents).
// Kept narrow so the join does not depend on the full RichEvent shape.
type EventWithMeta struct {
	TimestampNs int64
	PID         uint32
	Source      uint8
	Op          uint8
}

// incarnationInterval is one [start, exit) lifetime of a PID, derived
// from a process_exec / process_exit event pair. ExitNs is math.MaxInt64
// when no process_exit was observed (the process was still running at
// trace end), so the interval is open-ended.
type incarnationInterval struct {
	PID     uint32
	StartNs int64
	ExitNs  int64
}

// IncarnationIndex maps a timestamp to the PID incarnation that was
// alive at that moment, using process_exec / process_exit events as the
// incarnation boundaries. It is the bridge between an annotation
// (scoped by a /proc start-time key) and an event (which has no
// start-time column): both are mapped to an incarnation interval, and
// the join is on the interval.
type IncarnationIndex struct {
	// byPID holds intervals per PID, kept in exec-time order.
	byPID map[uint32][]incarnationInterval
}

// incarnationAt returns the index of the interval covering tsNs for
// pid, or -1 when no incarnation was alive then.
func (idx *IncarnationIndex) incarnationAt(pid uint32, tsNs int64) int {
	for i, iv := range idx.byPID[pid] {
		if tsNs >= iv.StartNs && tsNs < iv.ExitNs {
			return i
		}
	}
	return -1
}

// buildIncarnationIndex scans the events table for process_exec /
// process_exit events in [from, to] and builds the per-PID incarnation
// intervals. A process_exec opens an interval; the next process_exit
// for the same PID closes it. An interval with no matching exit stays
// open (ExitNs = math.MaxInt64).
//
// rolloverMu is RLock'd by the caller (QueryAnnotatedEvents).
func (s *Store) buildIncarnationIndex(from, to int64) (*IncarnationIndex, error) {
	rows, err := s.db.Query(`
		SELECT timestamp, pid, op
		FROM events
		WHERE source = ? AND op IN (?, ?)
		  AND timestamp >= ? AND timestamp <= ?
		ORDER BY timestamp ASC`,
		uint8(events.SourceHost),
		uint8(events.HostProcessExec),
		uint8(events.HostProcessExit),
		from, to)
	if err != nil {
		return nil, fmt.Errorf("scanning process lifecycle events: %w", err)
	}
	defer rows.Close()

	idx := &IncarnationIndex{byPID: map[uint32][]incarnationInterval{}}
	for rows.Next() {
		var ts int64
		var pid uint32
		var op uint8
		if err := rows.Scan(&ts, &pid, &op); err != nil {
			return nil, fmt.Errorf("scanning lifecycle row: %w", err)
		}
		switch events.HostOp(op) {
		case events.HostProcessExec:
			idx.byPID[pid] = append(idx.byPID[pid], incarnationInterval{
				PID:     pid,
				StartNs: ts,
				ExitNs:  maxInt64,
			})
		case events.HostProcessExit:
			ivs := idx.byPID[pid]
			// Close the most recent still-open interval for this PID.
			for i := len(ivs) - 1; i >= 0; i-- {
				if ivs[i].ExitNs == maxInt64 {
					ivs[i].ExitNs = ts
					break
				}
			}
			idx.byPID[pid] = ivs
		}
	}
	return idx, rows.Err()
}

const maxInt64 = int64(^uint64(0) >> 1)

// EventAnnotations is the per-event annotation result of an
// incarnation-aware join: the merged label set plus the count of
// annotation rows that contributed.
type EventAnnotations struct {
	Labels             map[string]string
	MatchedAnnotations int
}

// AnnotateEvents resolves the annotations that apply to each event in
// evts and returns a parallel slice of EventAnnotations (one per input
// event, same order). It is the query-time join used by `ingero query`
// and `ingero explain`.
//
// The join key is the process incarnation plus the time window, NOT pid
// + time alone:
//
//  1. process_exec / process_exit events in the scan window bound each
//     PID's incarnation intervals.
//  2. Each scoped annotation is mapped to the incarnation interval its
//     timestamp falls in. A reused PID therefore cannot pull a prior
//     incarnation's labels - the annotation and the event must land in
//     the SAME interval.
//  3. An unscoped annotation (pid 0) applies trace-wide; a span
//     annotation matches by its explicit [span_start, span_end].
//
// This runs over one DB file's events + annotations. The rollover-aware
// caller invokes it once per file before merging, so an annotation and
// an event split across a rollover boundary still resolve within their
// own file. Parameterized SQL only on the read paths it calls.
//
// from / to bound the process-lifecycle scan; pass the same range as
// the event query so every relevant exec/exit row is seen.
func (s *Store) AnnotateEvents(evts []EventWithMeta, from, to int64) ([]EventAnnotations, error) {
	out := make([]EventAnnotations, len(evts))
	for i := range out {
		out[i] = EventAnnotations{Labels: map[string]string{}}
	}
	if len(evts) == 0 {
		return out, nil
	}

	anns, err := s.QueryAnnotations(AnnotationQuery{
		From:  time.Unix(0, from),
		To:    time.Unix(0, to),
		Limit: -1,
	})
	if err != nil {
		return nil, err
	}
	if len(anns) == 0 {
		return out, nil
	}

	s.rolloverMu.RLock()
	idx, err := s.buildIncarnationIndex(from, to)
	s.rolloverMu.RUnlock()
	if err != nil {
		return nil, err
	}

	// Resolve every scoped annotation to an incarnation interval index.
	// annInterval[j] is the interval index for anns[j], or -1 when the
	// annotation is unscoped or its incarnation cannot be located.
	annInterval := make([]int, len(anns))
	for j, a := range anns {
		annInterval[j] = -1
		if !a.Process.Scoped() {
			continue
		}
		// An annotation's own instant locates its incarnation. A span
		// annotation uses its span start.
		probe := a.TimestampNs
		if a.IsSpan() {
			probe = a.SpanStartNs
		}
		annInterval[j] = idx.incarnationAt(a.Process.PID, probe)
	}

	for i, e := range evts {
		evInterval := idx.incarnationAt(e.PID, e.TimestampNs)
		// Collect the matching annotations for this event, then merge
		// them OLDEST-first so that on a same-key collision (an
		// advancing step/epoch counter) the most recent annotation
		// at-or-before the event wins. anns is QueryAnnotations'
		// newest-first order, so iterating it in reverse is oldest-first.
		for j := len(anns) - 1; j >= 0; j-- {
			a := anns[j]
			if !annotationAppliesViaIncarnation(a, e, annInterval[j], evInterval) {
				continue
			}
			out[i].MatchedAnnotations++
			for k, v := range a.Labels {
				out[i].Labels[k] = v
			}
		}
	}
	return out, nil
}

// annotationAppliesViaIncarnation is the incarnation-aware match used
// by AnnotateEvents. annIv / evIv are the incarnation interval indices
// resolved for the annotation and the event respectively (-1 = not
// located).
func annotationAppliesViaIncarnation(a annotate.Annotation, e EventWithMeta, annIv, evIv int) bool {
	if a.Process.Scoped() {
		if a.Process.PID != e.PID {
			return false
		}
		// Incarnation guard. The join only matches when both sides
		// resolve to the SAME incarnation interval, OR neither side
		// resolves (a pre-trace process with no observed exec/exit -
		// both are PID-only and conservatively treated as the same
		// run). It must NOT match when exactly one side resolves: a
		// resolved annotation paired with an unresolved event (or vice
		// versa) could be different incarnations of a reused PID, and a
		// bare PID==PID match would cross-attribute.
		switch {
		case annIv >= 0 && evIv >= 0:
			if annIv != evIv {
				return false
			}
		case annIv < 0 && evIv < 0:
			// Both unresolved: conservative same-run fallback, allowed.
		default:
			// Exactly one side resolved: ambiguous, refuse to match.
			return false
		}
	}
	if a.IsSpan() {
		return e.TimestampNs >= a.SpanStartNs && e.TimestampNs <= a.SpanEndNs
	}
	return e.TimestampNs >= a.TimestampNs
}
