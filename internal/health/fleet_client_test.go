package health

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/contract"
)

func basePollerCfg(baseURL string) PollerConfig {
	return PollerConfig{
		BaseURL:   baseURL,
		ClusterID: "test-cluster",
		Interval:  1 * time.Second,
		Timeout:   500 * time.Millisecond,
		Insecure:  true,
	}
}

func TestPollerConfig_Validate(t *testing.T) {
	good := basePollerCfg("fleet:8080")
	tests := []struct {
		name    string
		mutate  func(*PollerConfig)
		wantErr bool
	}{
		{"defaults", func(*PollerConfig) {}, false},
		{"empty_base", func(c *PollerConfig) { c.BaseURL = "" }, true},
		{"empty_cluster", func(c *PollerConfig) { c.ClusterID = "" }, true},
		{"sub_second_interval", func(c *PollerConfig) { c.Interval = 100 * time.Millisecond }, true},
		{"sub_100ms_timeout", func(c *PollerConfig) { c.Timeout = 50 * time.Millisecond }, true},
		{"timeout_exceeds_interval", func(c *PollerConfig) {
			c.Interval = 500 * time.Millisecond
			c.Timeout = 2 * time.Second
		}, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := good
			tc.mutate(&c)
			err := c.Validate()
			if tc.wantErr && err == nil {
				t.Fatal("expected error")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected: %v", err)
			}
		})
	}
}

func TestBuildThresholdURL(t *testing.T) {
	cases := []struct {
		base     string
		cluster  string
		insecure bool
		want     string
	}{
		{"fleet:8080", "prod", true, "http://fleet:8080/api/v1/threshold?cluster_id=prod"},
		{"fleet:8080", "prod", false, "https://fleet:8080/api/v1/threshold?cluster_id=prod"},
		{"https://fleet.example/", "prod", false, "https://fleet.example/api/v1/threshold?cluster_id=prod"},
		{"https://fleet.example/api/v1/threshold", "prod", false, "https://fleet.example/api/v1/threshold?cluster_id=prod"},
		{"[::1]:8080", "prod", true, "http://[::1]:8080/api/v1/threshold?cluster_id=prod"},
	}
	for _, c := range cases {
		got, err := buildThresholdURL(c.base, c.cluster, c.insecure)
		if err != nil {
			t.Errorf("buildThresholdURL(%q) unexpected err: %v", c.base, err)
			continue
		}
		if got != c.want {
			t.Errorf("buildThresholdURL(%q) = %q, want %q", c.base, got, c.want)
		}
	}
}

func TestBuildThresholdURL_Errors(t *testing.T) {
	cases := []string{"", "   ", "ftp://bad"}
	for _, c := range cases {
		if _, err := buildThresholdURL(c, "prod", false); err == nil {
			t.Errorf("buildThresholdURL(%q) expected error", c)
		}
	}
}

// A 200 response with valid JSON flows into the cache.
func TestPollOnce_200OK(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("cluster_id"); got != "test-cluster" {
			t.Errorf("cluster_id = %q", got)
		}
		if got := r.Header.Get("Accept"); got != "application/json" {
			t.Errorf("Accept = %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"threshold": 0.88, "quorum_met": true}`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	p, err := NewPoller(basePollerCfg(srv.URL), cache, discardLogger())
	if err != nil {
		t.Fatalf("NewPoller: %v", err)
	}
	retryAfter, rl := p.PollOnce(context.Background())
	if rl {
		t.Fatal("200 should not set rate-limited flag")
	}
	if retryAfter != 0 {
		t.Fatalf("retryAfter = %v, want 0", retryAfter)
	}
	snap, ok := cache.Get()
	if !ok || snap.Value != 0.88 || !snap.QuorumMet {
		t.Fatalf("cache = %+v, want 0.88/true", snap)
	}
	_, succ, _, _, _, _ := p.Stats()
	if succ != 1 {
		t.Fatalf("successes = %d, want 1", succ)
	}
}

// Fleet cold-start response (threshold=0, quorum_met=false) is stored
// in the cache so Story 3.3's ModeEvaluator can select fleet-cached.
// This bypasses the sanity bounds check that would otherwise reject a
// 0-valued threshold.
func TestPollOnce_ColdStartResponse(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"threshold": 0, "quorum_met": false}`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg(srv.URL), cache, discardLogger())
	_, rl := p.PollOnce(context.Background())
	if rl {
		t.Fatal("200 should not set rate-limited flag")
	}
	snap, ok := cache.Get()
	if !ok {
		t.Fatal("cold-start response should be cached (Set(0, false))")
	}
	if snap.Value != 0 || snap.QuorumMet {
		t.Fatalf("cold-start snapshot = %+v, want {0, false}", snap)
	}
}

// 429 with Retry-After in seconds sets the backoff.
func TestPollOnce_429RetryAfterSeconds(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(http.StatusTooManyRequests)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg(srv.URL), cache, discardLogger())
	retryAfter, rl := p.PollOnce(context.Background())
	if !rl {
		t.Fatal("429 should set rate-limited flag")
	}
	if retryAfter != 7*time.Second {
		t.Fatalf("retryAfter = %v, want 7s", retryAfter)
	}
}

// 429 with Retry-After as HTTP-date.
func TestPollOnce_429RetryAfterHTTPDate(t *testing.T) {
	future := time.Now().Add(5 * time.Second).UTC().Format(http.TimeFormat)
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", future)
		w.WriteHeader(http.StatusTooManyRequests)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg(srv.URL), cache, discardLogger())
	retryAfter, rl := p.PollOnce(context.Background())
	if !rl {
		t.Fatal("429 should set rate-limited flag")
	}
	if retryAfter <= 0 || retryAfter > 6*time.Second {
		t.Fatalf("retryAfter = %v, want ~5s", retryAfter)
	}
}

// 429 without Retry-After falls back to FallbackBackoff (default 2x interval).
func TestPollOnce_429NoRetryAfter_FallbackBackoff(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	cfg := basePollerCfg(srv.URL)
	cfg.FallbackBackoff = 0 // derived from Interval
	p, _ := NewPoller(cfg, cache, discardLogger())
	retryAfter, rl := p.PollOnce(context.Background())
	if !rl {
		t.Fatal("429 should set rate-limited flag")
	}
	if retryAfter != 2*time.Second {
		t.Fatalf("retryAfter = %v, want 2s (interval*2)", retryAfter)
	}
}

// 5xx WITHOUT Retry-After: counts as server error, no rate-limit flag,
// cache untouched. Normal retry cadence resumes.
func TestPollOnce_500NoCacheMutation(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg(srv.URL), cache, discardLogger())
	_, rl := p.PollOnce(context.Background())
	if rl {
		t.Fatal("5xx without Retry-After should not set rate-limited flag")
	}
	if _, ok := cache.Get(); ok {
		t.Fatal("5xx must not mutate cache")
	}
	_, _, _, serverErrs, _, _ := p.Stats()
	if serverErrs != 1 {
		t.Fatalf("serverErrors = %d, want 1", serverErrs)
	}
}

// 503 WITH Retry-After honors the backoff — same treatment as 429.
// RFC 7231 allows Retry-After on 503 for maintenance windows.
func TestPollOnce_503WithRetryAfter(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "12")
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg(srv.URL), cache, discardLogger())
	retryAfter, rl := p.PollOnce(context.Background())
	if !rl {
		t.Fatal("503 with Retry-After should set rate-limited flag")
	}
	if retryAfter != 12*time.Second {
		t.Fatalf("retryAfter = %v, want 12s", retryAfter)
	}
}

// 403 is a client error, not retried.
func TestPollOnce_403ClientError(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg(srv.URL), cache, discardLogger())
	p.PollOnce(context.Background())
	_, _, clientErrs, _, _, _ := p.Stats()
	if clientErrs != 1 {
		t.Fatalf("clientErrors = %d, want 1", clientErrs)
	}
}

// Malformed JSON body is a client-side problem.
func TestPollOnce_MalformedJSON(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{not json`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg(srv.URL), cache, discardLogger())
	p.PollOnce(context.Background())
	if _, ok := cache.Get(); ok {
		t.Fatal("malformed JSON must not mutate cache")
	}
	_, _, clientErrs, _, _, _ := p.Stats()
	if clientErrs != 1 {
		t.Fatalf("clientErrors = %d, want 1", clientErrs)
	}
}

// Oversized response body is rejected by the size guard BEFORE JSON
// decode. Body is deliberately non-JSON so that a future regression of
// the size check would surface as a test failure rather than be masked
// by a JSON-parse failure on the same path.
func TestPollOnce_OversizedBody(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		// 9 KiB of 'x' bytes — not JSON, and well above the 8 KiB cap.
		w.Write([]byte(strings.Repeat("x", 9*1024)))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg(srv.URL), cache, discardLogger())
	p.PollOnce(context.Background())
	if _, ok := cache.Get(); ok {
		t.Fatal("oversized body must not mutate cache")
	}
	_, _, clientErrs, _, _, _ := p.Stats()
	if clientErrs != 1 {
		t.Fatalf("clientErrors = %d, want 1", clientErrs)
	}
}

// Connection error (unreachable endpoint) increments client errors.
func TestPollOnce_Unreachable(t *testing.T) {
	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg("http://127.0.0.1:1/"), cache, discardLogger())
	_, rl := p.PollOnce(context.Background())
	if rl {
		t.Fatal("unreachable should not set rate-limited flag")
	}
	_, _, clientErrs, _, _, _ := p.Stats()
	if clientErrs != 1 {
		t.Fatalf("clientErrors = %d, want 1", clientErrs)
	}
}

// Cancelled ctx returns promptly.
func TestPollOnce_CtxCancelled(t *testing.T) {
	rs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
	}))
	defer rs.Close()
	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg(rs.URL), cache, discardLogger())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	p.PollOnce(ctx)
	// Should have returned; assertion is simply "did not hang." The
	// test's 30s deadline would catch a hang.
}

// Run exits promptly on ctx cancellation. Uses the minimum legal Interval
// (1s) and cancels before the first tick fires, so the test takes
// milliseconds.
func TestRun_CancelsBeforeFirstTick(t *testing.T) {
	cache := NewThresholdCache()
	p, err := NewPoller(basePollerCfg("http://127.0.0.1:1/"), cache, discardLogger())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- p.Run(ctx) }()
	// Cancel well before the 1s first-tick sleep elapses.
	time.Sleep(5 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Fatalf("Run err = %v, want context.Canceled", err)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Run did not return after cancel")
	}
}

// Run skips HTTP polls while PiggybackAvailable() is true and increments
// the suspendedHits counter. Uses a mock clock so we can spin through
// multiple iterations without sleeping.
func TestRun_SuspendsWhenPiggybackAvailable(t *testing.T) {
	var serverCalls int32
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/threshold", func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&serverCalls, 1)
		fmt.Fprint(w, `{"threshold": 0.5, "quorum_met": true}`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cache := NewThresholdCache()
	cache.Set(0.7, true, time.Now())
	// Mark piggyback available so Run should skip the HTTP call.
	h := newHTTPHeaders(map[string]string{
		contract.HeaderThreshold: "0.7",
		contract.HeaderQuorumMet: "true",
	})
	cache.ParseAndSetHTTPHeaders(h, time.Now())
	if !cache.PiggybackAvailable() {
		t.Fatal("setup: piggyback should be available")
	}

	p, _ := NewPoller(basePollerCfg(srv.URL), cache, discardLogger())

	// Mock clock: fire ticks as fast as possible by returning a
	// monotonically-incrementing time so nextFire.Sub(now) goes
	// non-positive immediately.
	var mockNow int64 // ns since epoch
	p.SetClock(func() time.Time {
		// Advance by 10 seconds per call so every scheduled sleep is
		// already in the past.
		atomic.AddInt64(&mockNow, int64(10*time.Second))
		return time.Unix(0, atomic.LoadInt64(&mockNow))
	})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		_ = p.Run(ctx)
		close(done)
	}()

	// Give Run a moment to iterate a few times while piggyback is
	// available; each iteration should be very fast since the mock
	// clock makes nextFire always <= now.
	time.Sleep(20 * time.Millisecond)
	cancel()
	<-done

	if atomic.LoadInt32(&serverCalls) != 0 {
		t.Fatalf("server was called %d times while piggyback was available", serverCalls)
	}
	_, _, _, _, _, suspended := p.Stats()
	if suspended == 0 {
		t.Fatal("suspendedHits counter never incremented")
	}
}

// _ silences unused-import complaints that would trip if the test is
// trimmed during refactoring.
var _ = sync.WaitGroup{}

// ParseRetryAfter is deterministic and handles all standard forms.
func TestParseRetryAfter(t *testing.T) {
	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg("fleet:8080"), cache, discardLogger())
	future := time.Now().Add(10 * time.Second).UTC().Format(http.TimeFormat)
	cases := map[string]bool{ // value -> should be positive
		"":         false, // fallback
		"  ":       false, // fallback
		"0":        true,  // zero seconds = immediate
		"5":        true,
		"abc":      false, // fallback
		"-1":       false, // fallback
		future:     true,
	}
	for in, wantPositive := range cases {
		got := p.parseRetryAfter(in)
		if in == "0" {
			if got != 0 {
				t.Errorf("parseRetryAfter(%q) = %v, want 0", in, got)
			}
			continue
		}
		if wantPositive && got <= 0 {
			t.Errorf("parseRetryAfter(%q) = %v, want positive", in, got)
		}
		if !wantPositive && got != p.cfg.FallbackBackoff {
			t.Errorf("parseRetryAfter(%q) = %v, want FallbackBackoff %v", in, got, p.cfg.FallbackBackoff)
		}
	}
}

// Jitter is within +/-20% of base.
func TestJitterBounds(t *testing.T) {
	cache := NewThresholdCache()
	p, _ := NewPoller(basePollerCfg("fleet:8080"), cache, discardLogger())
	const base = 100 * time.Millisecond
	const n = 1000
	for i := 0; i < n; i++ {
		got := p.jitter(base)
		if got < 80*time.Millisecond || got > 120*time.Millisecond {
			t.Fatalf("jitter out of +/-20%%: got %v for base %v", got, base)
		}
	}
}

// ensure the discard logger helper is used (imported from loop_test.go).
var _ = io.Discard

func init() {
	// Silence any slog.Default() output from tests that pass nil logger.
	slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
}
