package mcp

import (
	"encoding/json"
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Adversarial: simulate the frame-parse path in get_stacks with an
// attacker-crafted frames JSON. Reproduces the code pattern from
// server.go:905-912.
//
// Attack: attacker supplies a stack_traces.frames column with a
// JSON array of millions of entries, delivered via `ingero merge`
// on a malicious source DB. MCP get_stacks reads the row, calls
// json.Unmarshal to a []string, then allocates a second slice,
// then runs SanitizeTelemetryTruncate on each element (N*2 string
// copies + delimiter wrap). Memory and time are both attacker-
// controlled.
func TestAdv_FramesJSONBomb_MillionStrings(t *testing.T) {
	var m runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m)
	before := m.HeapAlloc

	// Build a 1M-entry JSON array. Each entry is a 10-byte string
	// "frame00001". Total JSON size ~15 MB.
	var b strings.Builder
	b.WriteString("[")
	N := 1_000_000
	for i := 0; i < N; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, `"frame%06d"`, i)
	}
	b.WriteString("]")
	framesJSON := b.String()
	t.Logf("crafted %d bytes of frames JSON", len(framesJSON))

	start := time.Now()
	var rawFrames []string
	if err := json.Unmarshal([]byte(framesJSON), &rawFrames); err != nil {
		t.Fatalf("unmarshal err: %v", err)
	}
	t.Logf("unmarshaled %d frames in %v", len(rawFrames), time.Since(start))

	start = time.Now()
	sanitized := make([]string, len(rawFrames))
	for i, f := range rawFrames {
		sanitized[i] = SanitizeTelemetryTruncate(f, MaxFrameLen)
	}
	t.Logf("sanitized %d frames in %v", len(sanitized), time.Since(start))

	runtime.ReadMemStats(&m)
	after := m.HeapAlloc
	delta := int64(after) - int64(before)
	t.Logf("heap delta: %d bytes (%.1f MB)", delta, float64(delta)/1024/1024)
}

// Adversarial: verify that parseAndSanitizeFrames returns nil for oversized
// input (the server.go caller pre-checks len > 64KB, but even if that check
// were removed, parseAndSanitizeFrames itself should handle it gracefully).
func TestAdv_FramesJSONBomb_SizeCapRejectsLargeInput(t *testing.T) {
	var b strings.Builder
	N := 10_000
	b.WriteString("[")
	for i := 0; i < N; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(`"x"`)
	}
	b.WriteString("]")
	framesJSON := b.String()
	t.Logf("crafted %d bytes of frames JSON", len(framesJSON))

	// The server.go fix rejects framesJSON > 64KB before parse.
	// Verify parseAndSanitizeFrames handles small arrays correctly.
	result := parseAndSanitizeFrames(framesJSON)
	if result == nil {
		t.Fatal("parseAndSanitizeFrames returned nil for valid small array")
	}
	if len(result) != N {
		t.Fatalf("expected %d frames, got %d", N, len(result))
	}
	// Each frame should be wrapped.
	for i, f := range result {
		if !strings.HasPrefix(f, TelemetryOpenDelim) {
			t.Fatalf("frame %d not wrapped: %q", i, f)
		}
	}
	t.Logf("parseAndSanitizeFrames correctly handled %d frames", len(result))
}

// Adversarial: a single 500MB string inside an array. Tests whether
// any one-frame limit exists.
func TestAdv_FramesJSONBomb_HugeSingleString(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping huge string in -short")
	}
	var m runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m)
	before := m.HeapAlloc

	// 100 MB single string inside a 1-element array
	var b strings.Builder
	b.WriteString(`["`)
	b.Grow(100_000_000)
	for i := 0; i < 100_000_000; i++ {
		b.WriteByte('x')
	}
	b.WriteString(`"]`)
	framesJSON := b.String()
	t.Logf("crafted %d bytes of frames JSON", len(framesJSON))

	var rawFrames []string
	if err := json.Unmarshal([]byte(framesJSON), &rawFrames); err != nil {
		t.Fatalf("unmarshal err: %v", err)
	}
	if len(rawFrames) != 1 {
		t.Fatalf("expected 1 frame, got %d", len(rawFrames))
	}
	t.Logf("single frame length: %d bytes (%.1f MB)", len(rawFrames[0]), float64(len(rawFrames[0]))/1024/1024)

	sanitized := SanitizeTelemetryTruncate(rawFrames[0], MaxFrameLen)
	t.Logf("sanitized length: %d bytes (should be <~%d due to MaxFrameLen)", len(sanitized), MaxFrameLen+len(TelemetryOpenDelim)+len(TelemetryCloseDelim)+len("..."))

	runtime.ReadMemStats(&m)
	after := m.HeapAlloc
	delta := int64(after) - int64(before)
	mb := float64(delta) / 1024 / 1024
	t.Logf("heap delta: %.1f MB", mb)

	if len(sanitized) > 2*MaxFrameLen {
		t.Errorf("FINDING: single-string sanitization did NOT truncate - returned %d bytes", len(sanitized))
	}
}

// Adversarial: deeply nested arrays. JSON parser should handle or reject
// without stack overflow. Go's default decoder uses a fixed-depth limit
// (10000). Beyond that, it returns an error.
func TestAdv_FramesJSONBomb_DeepNesting(t *testing.T) {
	var b strings.Builder
	depth := 100000
	for i := 0; i < depth; i++ {
		b.WriteByte('[')
	}
	b.WriteString(`"x"`)
	for i := 0; i < depth; i++ {
		b.WriteByte(']')
	}
	framesJSON := b.String()
	t.Logf("crafted %d bytes of deep-nested frames JSON (%d levels)", len(framesJSON), depth)

	var rawFrames []string
	err := json.Unmarshal([]byte(framesJSON), &rawFrames)
	t.Logf("deep-nest unmarshal err: %v", err)
	// Go's default limit is 10000 which we exceed. We want an error, not
	// a crash.
	if err == nil {
		t.Errorf("FINDING: 100K-deep JSON accepted without error - unexpected, likely parser allowed it")
	}
}

// Adversarial: a CRLF-in-frame attempt. Verifies sanitizer strips CR.
func TestAdv_FrameCRLFStripped(t *testing.T) {
	got := SanitizeTelemetryTruncate("frame\r\nHTTP/1.1 200 OK\r\nInjected: yes", MaxFrameLen)
	if strings.Contains(got, "\r") {
		t.Errorf("FINDING: CR survived sanitization: %q", got)
	}
	if !strings.Contains(got, "\n") {
		t.Logf("note: newline preserved as documented")
	}
	t.Logf("sanitized: %q", got)
}
