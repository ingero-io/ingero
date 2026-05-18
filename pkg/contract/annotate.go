package contract

// External annotation ingest contract (agent v0.17.0).
//
// The annotation feature lets an external workload (a PyTorch Lightning
// callback, a Ray hook, the `ingero annotate` CLI, or any process that
// can write to the agent's annotation UDS) inject human-meaningful
// labels into the agent's trace store. The agent records the label
// rows; `ingero query` / `ingero explain` join them to the eBPF event
// stream by process incarnation and time window.
//
// These constants flow OUTWARD only. The agent owns the protocol; the
// Lightning / Ray integrations and any future Fleet / Echo consumer
// conform to it. The agent imports none of them.
//
// What lives here: the wire protocol framing for the NDJSON ingest
// socket, the validation limits a conforming writer must respect, and
// the well-known label keys the distribution integrations emit. The
// limits are pinned so a writer built against this contract cannot
// drift from what the agent enforces.

// Annotation NDJSON socket protocol.
//
// The agent's annotation ingest socket accepts newline-delimited JSON.
// Each line is one annotation object. The object shape is:
//
//	{
//	  "ts":         <int64 unix nanos, optional; agent stamps now() if absent>,
//	  "labels":     {"<key>": "<value>", ...},
//	  "pid":        <uint32, optional process scope>,
//	  "start_time": <uint64, optional /proc start-time ticks for the pid>,
//	  "span_start": <int64 unix nanos, optional>,
//	  "span_end":   <int64 unix nanos, optional>
//	}
//
// `labels` is required and must be non-empty. `pid` without
// `start_time` is accepted (the agent resolves the incarnation by
// reading /proc at ingest time); `start_time` without `pid` is
// ignored. `span_start`/`span_end` are accepted as a pair; a span with
// end < start is rejected.
const (
	// AnnotationProtocolVersion is the protocol revision. Bumped only
	// on a breaking change to the NDJSON object shape. A writer may
	// send it as the optional "v" field; the agent does not require it
	// and treats an absent or matching value as v1.
	AnnotationProtocolVersion = 1

	// AnnotationSocketName is the fixed filename of the ingest socket
	// inside the agent-owned socket directory. The full path is
	// AnnotationSocketDir + "/" + AnnotationSocketName. The path is not
	// an operator flag so the agent never unlinks an arbitrary path at
	// bind time.
	AnnotationSocketName = "annotate.sock"

	// AnnotationSocketDir is the agent-owned directory that holds the
	// ingest socket when the agent runs with a writable /run (the
	// common privileged-trace case). It lives under /run, not the
	// world-writable /tmp, so a local attacker cannot pre-create the
	// directory and hijack the socket. The agent verifies at bind that
	// the directory is a real directory it owns with no group/other
	// write bits; it refuses to start otherwise.
	//
	// When /run is not writable, the agent falls back to a 0o700
	// directory it creates and owns under the user's home; the path is
	// derived at runtime, not pinned here.
	AnnotationSocketDir = "/run/ingero"
)

// Annotation NDJSON object field names. Pinned so a writer cannot
// drift from the agent's decoder.
const (
	AnnotationFieldTimestamp = "ts"
	AnnotationFieldLabels    = "labels"
	AnnotationFieldPID       = "pid"
	AnnotationFieldStartTime = "start_time"
	AnnotationFieldSpanStart = "span_start"
	AnnotationFieldSpanEnd   = "span_end"
	AnnotationFieldVersion   = "v"
)

// Annotation ingest validation limits. A conforming writer MUST keep
// every annotation within these bounds; the agent rejects any line
// that violates them without dropping the listening socket.
//
// These are pinned in contract_test.go. Loosening a limit is a
// contract change and must update both the constant and the test.
const (
	// AnnotationMaxLabelKeyLen is the maximum byte length of a label
	// key. Keys are short identifiers (step, epoch, task_id); 64 bytes
	// is generous headroom.
	AnnotationMaxLabelKeyLen = 64

	// AnnotationMaxLabelValueLen is the maximum byte length of a label
	// value. Values carry external workload identity (job names, task
	// IDs); 256 bytes covers UUID-shaped and path-shaped identifiers.
	AnnotationMaxLabelValueLen = 256

	// AnnotationMaxLabelsPerAnnotation caps the number of label pairs
	// in a single annotation. Bounds the per-row storage and the
	// JSON-decode work the agent does per ingested line.
	AnnotationMaxLabelsPerAnnotation = 32

	// AnnotationMaxLineBytes is the maximum byte length of a single
	// NDJSON line, measured before JSON decoding. A line longer than
	// this is rejected without an allocation-heavy decode attempt.
	AnnotationMaxLineBytes = 16 * 1024

	// AnnotationConnRateLimit is the maximum number of annotations one
	// connection may submit per AnnotationConnRateWindow. A connection
	// that exceeds it has further lines on that connection rejected
	// until the window rolls; the listener and other connections are
	// unaffected.
	AnnotationConnRateLimit = 1000

	// AnnotationConnRateWindowMs is the rate-limit window in
	// milliseconds for AnnotationConnRateLimit.
	AnnotationConnRateWindowMs = 1000
)

// AnnotationLabelKeyCharset documents the allowed character set for a
// label key: ASCII letters, digits, underscore, dot, and hyphen. A key
// with any other byte is rejected at ingest. Label VALUES are not
// charset-restricted (they carry free-form external identity) but are
// length-capped by AnnotationMaxLabelValueLen.
const AnnotationLabelKeyCharset = "A-Za-z0-9_.-"

// Well-known annotation label keys emitted by the distribution
// integrations. A writer is free to use any key that satisfies the
// charset rule; these are the keys the Lightning / Ray integrations
// standardize on so `ingero query` / `ingero explain` can document a
// stable slicing vocabulary.
const (
	// AnnotationKeyStep is the training step / global iteration index.
	AnnotationKeyStep = "step"
	// AnnotationKeyEpoch is the training epoch index.
	AnnotationKeyEpoch = "epoch"
	// AnnotationKeyTaskID is an external task identifier (a Ray task
	// id, a job-scheduler task id).
	AnnotationKeyTaskID = "task_id"
	// AnnotationKeyPhase is a free-form workload phase marker
	// (e.g. "warmup", "train", "eval", "checkpoint").
	AnnotationKeyPhase = "phase"
	// AnnotationKeyRunID identifies one end-to-end training/serving run.
	AnnotationKeyRunID = "run_id"
)

// IsValidAnnotationLabelKey reports whether key satisfies the
// annotation label-key contract: non-empty, at most
// AnnotationMaxLabelKeyLen bytes, and every byte in
// AnnotationLabelKeyCharset ([A-Za-z0-9_.-]). The agent calls this at
// ingest; a conforming writer can call it to validate before sending.
func IsValidAnnotationLabelKey(key string) bool {
	if key == "" || len(key) > AnnotationMaxLabelKeyLen {
		return false
	}
	for i := 0; i < len(key); i++ {
		c := key[i]
		switch {
		case c >= 'A' && c <= 'Z':
		case c >= 'a' && c <= 'z':
		case c >= '0' && c <= '9':
		case c == '_' || c == '.' || c == '-':
		default:
			return false
		}
	}
	return true
}
