// Package annotate defines the external annotation record type and the
// process-incarnation key used to scope an annotation to a process.
//
// An Annotation is a human-meaningful label injected from outside the
// eBPF event stream: a training step, an epoch, a Ray task id, a team
// or cost-center tag. It is deliberately separate from events.Event so
// the eBPF event type stays pure (eBPF-sourced fields only). The agent
// records annotations in its SQLite store; `ingero query` /
// `ingero explain` join them to events by process incarnation and time.
//
// This package is contract-shaped: it holds the in-memory record and
// validation, not the socket or the storage. Those live in
// internal/annotate (ingest) and internal/store (persistence).
package annotate

import (
	"fmt"

	"github.com/ingero-io/ingero/pkg/contract"
)

// ProcessIncarnation identifies one run of a process unambiguously,
// even across PID reuse. A bare PID is not enough: the kernel reuses
// PIDs, so an annotation scoped to PID 1234 could mis-attribute to a
// later, unrelated PID 1234. The pair (PID, StartTime) is unique for
// the life of the system - StartTime is the process start time in
// clock ticks since boot, field 22 of /proc/<pid>/stat, which the
// kernel never reuses for a given PID within a boot.
//
// A zero value (PID == 0) means "no process scope" - the annotation
// applies to the whole trace, not one process.
type ProcessIncarnation struct {
	// PID is the process id (the kernel tgid). 0 means unscoped.
	PID uint32
	// StartTime is /proc/<pid>/stat field 22: process start time in
	// clock ticks after system boot. 0 when the start time could not
	// be resolved (the annotation still carries the PID, but the join
	// degrades to PID + time-window only for that row).
	StartTime uint64
}

// Scoped reports whether the incarnation names a specific process.
func (p ProcessIncarnation) Scoped() bool { return p.PID != 0 }

// String renders the incarnation for logs and debug output.
func (p ProcessIncarnation) String() string {
	if !p.Scoped() {
		return "unscoped"
	}
	return fmt.Sprintf("pid=%d/start=%d", p.PID, p.StartTime)
}

// Provenance records who submitted an annotation. The agent captures
// it from SO_PEERCRED on the ingest socket and stores it on every
// annotation row so a poisoned or surprising row is traceable to the
// writing process even when group access is enabled.
//
// A zero value means the annotation did not arrive over the socket
// (e.g. an in-process test) and has no peer credentials.
type Provenance struct {
	// PeerUID / PeerGID / PeerPID are the SO_PEERCRED values of the
	// connection that submitted the annotation.
	PeerUID uint32
	PeerGID uint32
	PeerPID uint32
}

// Annotation is one external label record. It is separate from
// events.Event by design; see the package doc.
type Annotation struct {
	// TimestampNs is the annotation instant in unix nanoseconds. When a
	// writer omits the timestamp the ingest layer stamps it with the
	// receive time.
	TimestampNs int64

	// Labels is the set of key/value pairs. Non-empty after validation;
	// keys satisfy contract.IsValidAnnotationLabelKey and the map size
	// is at most contract.AnnotationMaxLabelsPerAnnotation.
	Labels map[string]string

	// Process is the optional process incarnation the annotation is
	// scoped to. Unscoped (PID == 0) annotations apply trace-wide.
	Process ProcessIncarnation

	// SpanStartNs / SpanEndNs optionally mark a phase rather than an
	// instant. Both zero means the annotation is an instant at
	// TimestampNs. When set, SpanEndNs >= SpanStartNs holds (enforced
	// by Validate).
	SpanStartNs int64
	SpanEndNs   int64

	// Provenance is the SO_PEERCRED identity of the submitter. Zero for
	// annotations not sourced from the ingest socket.
	Provenance Provenance
}

// IsSpan reports whether the annotation marks a phase (has a span)
// rather than an instant.
func (a Annotation) IsSpan() bool { return a.SpanStartNs != 0 || a.SpanEndNs != 0 }

// Validate enforces the annotation contract limits from pkg/contract.
// It returns a non-nil error describing the first violation. The
// ingest layer calls Validate on every decoded line; a failing line is
// rejected without dropping the listener.
func (a Annotation) Validate() error {
	if len(a.Labels) == 0 {
		return fmt.Errorf("annotation has no labels")
	}
	if len(a.Labels) > contract.AnnotationMaxLabelsPerAnnotation {
		return fmt.Errorf("annotation has %d labels, max %d",
			len(a.Labels), contract.AnnotationMaxLabelsPerAnnotation)
	}
	for k, v := range a.Labels {
		if !contract.IsValidAnnotationLabelKey(k) {
			return fmt.Errorf("invalid label key %q (charset %s, max %d bytes)",
				k, contract.AnnotationLabelKeyCharset, contract.AnnotationMaxLabelKeyLen)
		}
		if len(v) > contract.AnnotationMaxLabelValueLen {
			return fmt.Errorf("label %q value is %d bytes, max %d",
				k, len(v), contract.AnnotationMaxLabelValueLen)
		}
		// Reject control characters in the value. Label values are
		// rendered verbatim into terminal tables by `ingero query` /
		// `ingero explain`; an embedded ESC, CR, or NUL would allow
		// terminal-injection or row-forging in that output.
		for i := 0; i < len(v); i++ {
			if c := v[i]; c < 0x20 || c == 0x7f {
				return fmt.Errorf("label %q value contains a control character (byte 0x%02x)", k, c)
			}
		}
	}
	if a.IsSpan() {
		if a.SpanStartNs == 0 || a.SpanEndNs == 0 {
			return fmt.Errorf("span annotation needs both span_start and span_end")
		}
		if a.SpanEndNs < a.SpanStartNs {
			return fmt.Errorf("span_end (%d) is before span_start (%d)",
				a.SpanEndNs, a.SpanStartNs)
		}
	}
	return nil
}
