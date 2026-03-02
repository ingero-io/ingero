package cli

import (
	"testing"
	"time"
)

func TestParseTime(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
		check   func(t *testing.T, got time.Time)
	}{
		{
			name:  "full datetime with seconds",
			input: "2026-03-02 14:30:05",
			check: func(t *testing.T, got time.Time) {
				if got.Hour() != 14 || got.Minute() != 30 || got.Second() != 5 {
					t.Errorf("got %v, want 14:30:05", got)
				}
				if got.Year() != 2026 || got.Month() != 3 || got.Day() != 2 {
					t.Errorf("got %v, want 2026-03-02", got)
				}
			},
		},
		{
			name:  "full datetime without seconds",
			input: "2026-03-02 14:30",
			check: func(t *testing.T, got time.Time) {
				if got.Hour() != 14 || got.Minute() != 30 {
					t.Errorf("got %v, want 14:30", got)
				}
			},
		},
		{
			name:  "ISO format",
			input: "2026-03-02T14:30:05",
			check: func(t *testing.T, got time.Time) {
				if got.Hour() != 14 || got.Minute() != 30 || got.Second() != 5 {
					t.Errorf("got %v, want 14:30:05", got)
				}
			},
		},
		{
			name:  "time only with seconds",
			input: "14:30:05",
			check: func(t *testing.T, got time.Time) {
				now := time.Now()
				if got.Year() != now.Year() || got.Month() != now.Month() || got.Day() != now.Day() {
					t.Errorf("time-only should use today's date, got %v", got)
				}
				if got.Hour() != 14 || got.Minute() != 30 || got.Second() != 5 {
					t.Errorf("got %v, want 14:30:05", got)
				}
			},
		},
		{
			name:  "time only without seconds",
			input: "15:45",
			check: func(t *testing.T, got time.Time) {
				now := time.Now()
				if got.Year() != now.Year() || got.Month() != now.Month() || got.Day() != now.Day() {
					t.Errorf("time-only should use today's date, got %v", got)
				}
				if got.Hour() != 15 || got.Minute() != 45 {
					t.Errorf("got %v, want 15:45", got)
				}
			},
		},
		{
			name:    "invalid format",
			input:   "not-a-time",
			wantErr: true,
		},
		{
			name:    "empty string",
			input:   "",
			wantErr: true,
		},
		{
			name:    "partial date",
			input:   "2026-03",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseTime(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Errorf("parseTime(%q) should return error", tt.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseTime(%q) unexpected error: %v", tt.input, err)
			}
			if tt.check != nil {
				tt.check(t, got)
			}
		})
	}
}

// TestSinglePIDOrZero and TestToUint32Slice are in pidutil_test.go.
