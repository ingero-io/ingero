//go:build linux

package health

import (
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestAdversarial_PersistJSONBombs — adversarial Load inputs.
func TestAdversarial_PersistJSONBombs(t *testing.T) {
	tmp := t.TempDir()
	q := slog.New(slog.NewTextHandler(io.Discard, nil))
	now := time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)

	write := func(name string, data []byte) string {
		path := filepath.Join(tmp, name)
		os.WriteFile(path, data, 0644)
		return path
	}

	deepNest := func(depth int) []byte {
		return []byte(strings.Repeat(`{"a":`, depth) + `1` + strings.Repeat(`}`, depth))
	}
	padded := func(prefix, suffix string, fill int) []byte {
		body := make([]byte, 0, len(prefix)+fill+len(suffix))
		body = append(body, prefix...)
		body = append(body, make([]byte, fill)...)
		for i := len(prefix); i < len(prefix)+fill; i++ {
			body[i] = 'a'
		}
		body = append(body, suffix...)
		return body
	}

	cases := []struct {
		name string
		data []byte
	}{
		{"deep-nesting-1k", deepNest(1000)},
		{"deep-nesting-100k", deepNest(100000)},
		{"huge-float",
			[]byte(`{"schema_version":1,"saved_at":"2026-04-16T12:00:00Z","sample_count":1,"fast_alpha":1e100000,"floor_alpha":0.001,"fast_ema":{},"hard_floor":{}}`)},
		{"unicode-schema",
			[]byte(`{"schema_version":"①","saved_at":"2026-04-16T12:00:00Z"}`)},
		{"just-under-1mb",
			padded(`{"schema_version":1,"saved_at":"2026-04-16T12:00:00Z","sample_count":1,"fast_alpha":0.1,"floor_alpha":0.001,"fast_ema":{"Throughput":0,"Compute":0,"Memory":0,"CPU":0},"hard_floor":{"Throughput":0,"Compute":0,"Memory":0,"CPU":0},"padding":"`, `"}`, 1040000)},
		{"over-1mb-cap",
			padded(`{"schema_version":1,"padding":"`, `"}`, 2000000)},
		{"type-confusion-schema",
			[]byte(`{"schema_version":"1","saved_at":"2026-04-16T12:00:00Z"}`)},
		{"type-confusion-savedat",
			[]byte(`{"schema_version":1,"saved_at":0}`)},
		{"very-long-ema-string",
			[]byte(`{"schema_version":1,"saved_at":"2026-04-16T12:00:00Z","sample_count":1,"fast_alpha":0.1,"floor_alpha":0.001,"fast_ema":{"Throughput":` + strings.Repeat("0", 1000000) + `}}`)},
		{"bom-prefix",
			append([]byte{0xef, 0xbb, 0xbf}, []byte(`{"schema_version":1,"saved_at":"2026-04-16T12:00:00Z"}`)...)},
		{"null-bytes-prefix",
			[]byte("\x00\x00\x00\x00" + `{"schema_version":1}`)},
		{"jsonp-wrapper",
			[]byte(`alert("xss")//{"schema_version":1}`)},
	}

	maxAge := 30 * time.Minute
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := write(tc.name+".json", tc.data)
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("PANIC on %s: %v", tc.name, r)
				}
			}()
			start := time.Now()
			_, status, err := Load(path, maxAge, now, q)
			elapsed := time.Since(start)
			t.Logf("%s: status=%s elapsed=%s err=%v size=%d",
				tc.name, status.String(), elapsed, err, len(tc.data))
			if elapsed > 2*time.Second {
				t.Errorf("SLOW: took %s", elapsed)
			}
		})
	}
}
