package contract

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestAnnotationConstants_Pinned asserts the annotation socket protocol
// framing and validation limits stay at their published values. A
// writer built against this contract (the Lightning callback, the Ray
// hook) relies on these; drift here silently breaks every external
// integration, so a change must update both the constant and this
// test in the same change.
func TestAnnotationConstants_Pinned(t *testing.T) {
	intPins := map[string]struct{ got, want int }{
		"AnnotationProtocolVersion":        {AnnotationProtocolVersion, 1},
		"AnnotationMaxLabelKeyLen":         {AnnotationMaxLabelKeyLen, 64},
		"AnnotationMaxLabelValueLen":       {AnnotationMaxLabelValueLen, 256},
		"AnnotationMaxLabelsPerAnnotation": {AnnotationMaxLabelsPerAnnotation, 32},
		"AnnotationMaxLineBytes":           {AnnotationMaxLineBytes, 16 * 1024},
		"AnnotationConnRateLimit":          {AnnotationConnRateLimit, 1000},
		"AnnotationConnRateWindowMs":       {AnnotationConnRateWindowMs, 1000},
	}
	for name, p := range intPins {
		if p.got != p.want {
			t.Errorf("%s = %d, pinned %d", name, p.got, p.want)
		}
	}

	strPins := map[string]struct{ got, want string }{
		"AnnotationSocketName":      {AnnotationSocketName, "annotate.sock"},
		"AnnotationSocketDir":       {AnnotationSocketDir, "/run/ingero"},
		"AnnotationLabelKeyCharset": {AnnotationLabelKeyCharset, "A-Za-z0-9_.-"},
		"AnnotationFieldTimestamp":  {AnnotationFieldTimestamp, "ts"},
		"AnnotationFieldLabels":     {AnnotationFieldLabels, "labels"},
		"AnnotationFieldPID":        {AnnotationFieldPID, "pid"},
		"AnnotationFieldStartTime":  {AnnotationFieldStartTime, "start_time"},
		"AnnotationFieldSpanStart":  {AnnotationFieldSpanStart, "span_start"},
		"AnnotationFieldSpanEnd":    {AnnotationFieldSpanEnd, "span_end"},
		"AnnotationFieldVersion":    {AnnotationFieldVersion, "v"},
		"AnnotationKeyStep":         {AnnotationKeyStep, "step"},
		"AnnotationKeyEpoch":        {AnnotationKeyEpoch, "epoch"},
		"AnnotationKeyTaskID":       {AnnotationKeyTaskID, "task_id"},
		"AnnotationKeyPhase":        {AnnotationKeyPhase, "phase"},
		"AnnotationKeyRunID":        {AnnotationKeyRunID, "run_id"},
	}
	for name, p := range strPins {
		if p.got != p.want {
			t.Errorf("%s = %q, pinned %q", name, p.got, p.want)
		}
	}
}

// TestIsValidAnnotationLabelKey covers the charset and length rule for
// label keys.
func TestIsValidAnnotationLabelKey(t *testing.T) {
	valid := []string{"step", "epoch", "task_id", "run.id", "a-b", "A1._-", "x"}
	for _, k := range valid {
		if !IsValidAnnotationLabelKey(k) {
			t.Errorf("expected %q to be a valid label key", k)
		}
	}
	invalid := []string{
		"",          // empty
		"has space", // space
		"slash/key", // slash
		"emoji☃",    // non-ASCII
		"colon:key", // colon
		strings.Repeat("a", AnnotationMaxLabelKeyLen+1), // too long
	}
	for _, k := range invalid {
		if IsValidAnnotationLabelKey(k) {
			t.Errorf("expected %q to be rejected", k)
		}
	}
	// A key exactly at the cap is valid.
	if !IsValidAnnotationLabelKey(strings.Repeat("a", AnnotationMaxLabelKeyLen)) {
		t.Error("key exactly at AnnotationMaxLabelKeyLen should be valid")
	}
}

// TestAnnotationNDJSON_RoundTrip asserts an annotation object using the
// pinned field names round-trips through encoding/json so a writer and
// the agent agree on the wire shape.
func TestAnnotationNDJSON_RoundTrip(t *testing.T) {
	line := `{"ts":1700000000000000000,"labels":{"step":"42"},"pid":1234,"start_time":99887766,"span_start":1700000000000000000,"span_end":1700000001000000000,"v":1}`
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(line), &m); err != nil {
		t.Fatalf("decode annotation NDJSON: %v", err)
	}
	for _, f := range []string{
		AnnotationFieldTimestamp, AnnotationFieldLabels, AnnotationFieldPID,
		AnnotationFieldStartTime, AnnotationFieldSpanStart, AnnotationFieldSpanEnd,
		AnnotationFieldVersion,
	} {
		if _, ok := m[f]; !ok {
			t.Errorf("annotation NDJSON missing pinned field %q", f)
		}
	}
}
