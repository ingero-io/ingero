package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/ingero-io/ingero/pkg/annotate"
)

// annotationsSchema is the table for external annotation rows (agent
// v0.17.0). Annotations are human-meaningful labels injected from
// outside the eBPF event stream (a training step, an epoch, a Ray task
// id). The events table is untouched; annotations are a separate table
// joined at query time.
//
// Process scope is stored as the (pid, process_start_time) incarnation
// key, NOT a raw PID, so PID reuse cannot mis-attribute a row. pid 0
// means an unscoped (trace-wide) annotation.
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

// AnnotatedEvent pairs an events.Event with the annotation labels that
// resolve to it. Labels is the merged label set of every annotation
// whose process incarnation matches the event and whose time window
// (instant or span) contains the event timestamp.
//
// MatchedAnnotations is the count of distinct annotation rows that
// contributed, so a caller can tell "one annotation, three labels"
// apart from "three annotations".
type AnnotatedEvent struct {
	Event              EventWithMeta
	Labels             map[string]string
	MatchedAnnotations int
}

// EventWithMeta is the minimal event projection the annotation join
// needs: identity, time, and the process incarnation. Kept narrow so
// the join does not depend on the full RichEvent shape.
type EventWithMeta struct {
	TimestampNs int64
	PID         uint32
	StartTime   uint64
	Source      uint8
	Op          uint8
}

// JoinAnnotations attaches annotation labels to a set of events by
// process incarnation and time window. The join key is the process
// incarnation (pid + start_time) plus the time window, NOT pid + time
// alone, so a reused PID cannot pull in a prior incarnation's labels.
//
// An annotation matches an event when:
//   - the annotation is unscoped (pid 0), OR the annotation's
//     (pid, start_time) equals the event's incarnation; AND
//   - the event timestamp falls in the annotation's window: for an
//     instant annotation the window is a point so the join is on the
//     span enclosing the event for span annotations, and for instant
//     annotations the event is matched when it shares the incarnation
//     and the annotation timestamp is at or before the event (the
//     annotation marks a moment in the run from that point on).
//
// When an annotation has a start_time of 0 (the writer gave a PID but
// the agent could not resolve the incarnation), it matches any event
// with the same PID regardless of start_time - the row is still useful,
// it just cannot benefit from reuse protection.
//
// This is an in-Go join over the supplied events and a one-shot
// annotation read, so a caller doing a rollover-aware merge runs it
// per-file before merging (see the query/explain CLI path).
func JoinAnnotations(evts []EventWithMeta, anns []annotate.Annotation) []AnnotatedEvent {
	out := make([]AnnotatedEvent, len(evts))
	for i, e := range evts {
		ae := AnnotatedEvent{Event: e, Labels: map[string]string{}}
		for _, a := range anns {
			if !annotationMatchesEvent(a, e) {
				continue
			}
			ae.MatchedAnnotations++
			for k, v := range a.Labels {
				ae.Labels[k] = v
			}
		}
		out[i] = ae
	}
	return out
}

// annotationMatchesEvent reports whether annotation a applies to event
// e under the incarnation + time-window rule documented on
// JoinAnnotations.
func annotationMatchesEvent(a annotate.Annotation, e EventWithMeta) bool {
	// Incarnation check.
	if a.Process.Scoped() {
		if a.Process.PID != e.PID {
			return false
		}
		// start_time 0 on the annotation means the incarnation was not
		// resolved; fall back to PID-only matching for that row.
		if a.Process.StartTime != 0 && a.Process.StartTime != e.StartTime {
			return false
		}
	}
	// Time-window check.
	if a.IsSpan() {
		return e.TimestampNs >= a.SpanStartNs && e.TimestampNs <= a.SpanEndNs
	}
	// Instant annotation: it marks a moment in the run. An event at or
	// after the annotation instant carries the label.
	return e.TimestampNs >= a.TimestampNs
}
