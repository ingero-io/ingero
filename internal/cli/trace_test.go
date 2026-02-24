package cli

import (
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// formatDuration tests
// ---------------------------------------------------------------------------

// TestFormatDuration verifies human-readable duration formatting.
func TestFormatDuration(t *testing.T) {
	tests := []struct {
		name string
		dur  time.Duration
		want string
	}{
		// Zero
		{"zero", 0, "0"},

		// Nanoseconds (< 1µs)
		{"1ns", 1 * time.Nanosecond, "1ns"},
		{"500ns", 500 * time.Nanosecond, "500ns"},
		{"999ns", 999 * time.Nanosecond, "999ns"},

		// Microseconds (< 1ms)
		{"1us", 1 * time.Microsecond, "1.0us"},
		{"1.5us", 1500 * time.Nanosecond, "1.5us"},
		{"9.9us", 9900 * time.Nanosecond, "9.9us"},
		{"10us", 10 * time.Microsecond, "10us"},
		{"456us", 456 * time.Microsecond, "456us"},
		{"999us", 999 * time.Microsecond, "999us"},

		// Milliseconds (< 1s)
		{"1ms", 1 * time.Millisecond, "1.0ms"},
		{"1.5ms", 1500 * time.Microsecond, "1.5ms"},
		{"9.9ms", 9900 * time.Microsecond, "9.9ms"},
		{"10ms", 10 * time.Millisecond, "10ms"},
		{"456ms", 456 * time.Millisecond, "456ms"},

		// Seconds (< 1min)
		{"1s", 1 * time.Second, "1.0s"},
		{"1.5s", 1500 * time.Millisecond, "1.5s"},
		{"9.9s", 9900 * time.Millisecond, "9.9s"},
		{"10s", 10 * time.Second, "10s"},
		{"45s", 45 * time.Second, "45s"},

		// Minutes (>= 1min)
		{"1m0s", 1 * time.Minute, "1m0s"},
		{"1m30s", 90 * time.Second, "1m30s"},
		{"5m0s", 5 * time.Minute, "5m0s"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatDuration(tt.dur)
			if got != tt.want {
				t.Errorf("formatDuration(%v) = %q, want %q", tt.dur, got, tt.want)
			}
		})
	}
}

// TestDebugf verifies the debug helper doesn't panic and respects debugMode.
func TestDebugf(t *testing.T) {
	// debugMode=false: no output, no panic.
	debugMode = false
	debugf("should not appear: %d", 42)

	// debugMode=true: writes to stderr, no panic.
	debugMode = true
	debugf("test message: %s %d", "hello", 42)
	debugMode = false
}
