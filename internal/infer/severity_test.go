package infer

import (
	"testing"
	"time"
)

func TestParseSeverity(t *testing.T) {
	cases := []struct {
		in   string
		want severityRank
	}{
		{"HIGH", sevHigh},
		{"high", sevHigh},
		{" High ", sevHigh},
		{"MEDIUM", sevMedium},
		{"LOW", sevLow},
		{"", sevNone},
		{"UNKNOWN", sevNone},
		{"hi", sevNone},
	}
	for _, tc := range cases {
		if got := parseSeverity(tc.in); got != tc.want {
			t.Errorf("parseSeverity(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestSeverityGate_NotSetIsNotPaused(t *testing.T) {
	g := newSeverityGate(time.Second)
	if g.IsAtLeast(123, sevHigh, time.Now()) {
		t.Error("unset PID should not be at-or-above HIGH")
	}
}

func TestSeverityGate_HighGatesAtThreshold(t *testing.T) {
	g := newSeverityGate(time.Second)
	now := time.Now()
	g.Set(123, sevHigh, now)
	if !g.IsAtLeast(123, sevHigh, now) {
		t.Error("HIGH should satisfy IsAtLeast(HIGH)")
	}
	if !g.IsAtLeast(123, sevMedium, now) {
		t.Error("HIGH should satisfy IsAtLeast(MEDIUM)")
	}
	if !g.IsAtLeast(123, sevLow, now) {
		t.Error("HIGH should satisfy IsAtLeast(LOW)")
	}
}

func TestSeverityGate_MediumDoesNotMeetHigh(t *testing.T) {
	g := newSeverityGate(time.Second)
	now := time.Now()
	g.Set(456, sevMedium, now)
	if g.IsAtLeast(456, sevHigh, now) {
		t.Error("MEDIUM should not satisfy IsAtLeast(HIGH)")
	}
	if !g.IsAtLeast(456, sevMedium, now) {
		t.Error("MEDIUM should satisfy IsAtLeast(MEDIUM)")
	}
}

func TestSeverityGate_TTLExpiry(t *testing.T) {
	g := newSeverityGate(100 * time.Millisecond)
	t0 := time.Now()
	g.Set(789, sevHigh, t0)
	if !g.IsAtLeast(789, sevHigh, t0.Add(50*time.Millisecond)) {
		t.Error("entry should be live within TTL")
	}
	if g.IsAtLeast(789, sevHigh, t0.Add(200*time.Millisecond)) {
		t.Error("entry should be expired beyond TTL")
	}
	// Expired entries are removed lazily by IsAtLeast.
	if g.Len() != 0 {
		t.Errorf("expired entry should be removed, len=%d", g.Len())
	}
}

func TestSeverityGate_ThresholdNoneDisablesGate(t *testing.T) {
	g := newSeverityGate(time.Second)
	now := time.Now()
	g.Set(111, sevHigh, now)
	if g.IsAtLeast(111, sevNone, now) {
		t.Error("threshold sevNone should never gate")
	}
}

func TestSeverityGate_SetSevNoneClears(t *testing.T) {
	g := newSeverityGate(time.Second)
	now := time.Now()
	g.Set(222, sevHigh, now)
	g.Set(222, sevNone, now)
	if g.IsAtLeast(222, sevHigh, now) {
		t.Error("Set(sevNone) should clear the entry")
	}
}

func TestSeverityGate_PruneExpired(t *testing.T) {
	g := newSeverityGate(100 * time.Millisecond)
	t0 := time.Now()
	g.Set(1, sevHigh, t0)
	g.Set(2, sevMedium, t0)
	g.Set(3, sevLow, t0.Add(50*time.Millisecond))
	g.PruneExpired(t0.Add(150 * time.Millisecond))
	if g.Len() != 1 {
		t.Errorf("after prune len=%d, want 1 (only PID 3 within TTL)", g.Len())
	}
}
