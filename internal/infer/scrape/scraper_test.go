package scrape

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/infer/enginedetect"
)

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// targetFromTestServer extracts host/port from an httptest server's
// URL and packages it as a scrape Target.
func targetFromTestServer(t *testing.T, srv *httptest.Server, engine enginedetect.Engine, pid uint32) Target {
	t.Helper()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	host, portStr, _ := net.SplitHostPort(u.Host)
	port, _ := strconv.Atoi(portStr)
	return Target{
		Engine: engine,
		Host:   host,
		Port:   uint16(port),
		Path:   "/metrics",
		PID:    pid,
	}
}

func TestScraper_HappyPath_VLLM(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprint(w, vllmFixture)
	}))
	defer srv.Close()

	var (
		mu       sync.Mutex
		received []ScrapedSample
	)
	sink := func(target Target, samples []ScrapedSample) {
		mu.Lock()
		defer mu.Unlock()
		received = append(received, samples...)
	}

	s := NewScraper(Config{Interval: 50 * time.Millisecond}, sink, quietLogger())
	s.AddTarget(targetFromTestServer(t, srv, enginedetect.VLLM, 1234))

	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = s.Run(ctx) }()
	// Wait long enough for at least 2 ticks (initial + scheduled).
	time.Sleep(150 * time.Millisecond)
	cancel()

	mu.Lock()
	defer mu.Unlock()
	if len(received) == 0 {
		t.Fatal("scraper did not emit any samples")
	}
	// Confirm the OTel canonical names appear and the PID is tagged.
	foundTTFT := false
	for _, s := range received {
		if s.CanonicalName == "gen_ai.client.operation.time_to_first_token" {
			foundTTFT = true
			if s.Labels["ingero.engine.pid"] != "1234" {
				t.Errorf("PID label = %q, want 1234", s.Labels["ingero.engine.pid"])
			}
		}
	}
	if !foundTTFT {
		t.Error("expected gen_ai.client.operation.time_to_first_token in scraped output")
	}
	if got := s.Stats().Scrapes; got < 1 {
		t.Errorf("Stats.Scrapes = %d, want >= 1", got)
	}
}

func TestScraper_EngineDown_Recovers(t *testing.T) {
	// Hit a port that's nothing-listening — connection refused.
	// Scraper should count the error and keep ticking; bringing the
	// engine "up" later should resume successful scrapes.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, vllmFixture)
	}))

	target := targetFromTestServer(t, srv, enginedetect.VLLM, 7777)
	srv.Close() // tear down so first scrape fails

	var sinkCalls atomic_int
	sink := func(target Target, samples []ScrapedSample) {
		sinkCalls.add(1)
	}
	s := NewScraper(Config{Interval: 30 * time.Millisecond, Timeout: 200 * time.Millisecond}, sink, quietLogger())
	s.AddTarget(target)

	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = s.Run(ctx) }()
	time.Sleep(80 * time.Millisecond)
	cancel()

	if got := s.Stats().ScrapeErrors; got == 0 {
		t.Errorf("ScrapeErrors = 0, want > 0 (engine was down)")
	}
	if sinkCalls.get() != 0 {
		t.Errorf("sink called %d times despite engine down", sinkCalls.get())
	}
}

func TestScraper_AddRemoveTarget(t *testing.T) {
	s := NewScraper(Config{}, nil, quietLogger())
	s.AddTarget(Target{PID: 1, Engine: enginedetect.VLLM, Host: "x", Port: 1})
	s.AddTarget(Target{PID: 2, Engine: enginedetect.TGI, Host: "y", Port: 2})
	if len(s.Targets()) != 2 {
		t.Fatalf("Targets() len = %d, want 2", len(s.Targets()))
	}
	s.RemoveTarget(1)
	if len(s.Targets()) != 1 {
		t.Errorf("after remove len = %d, want 1", len(s.Targets()))
	}
	// Re-adding same PID replaces.
	s.AddTarget(Target{PID: 2, Engine: enginedetect.SGLang, Host: "z", Port: 3})
	if got := s.Targets()[0].Engine; got != enginedetect.SGLang {
		t.Errorf("re-add replace failed: got %v, want sglang", got)
	}
}

func TestScraper_NonOK_Status(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	s := NewScraper(Config{Interval: 30 * time.Millisecond}, nil, quietLogger())
	s.AddTarget(targetFromTestServer(t, srv, enginedetect.VLLM, 1))
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = s.Run(ctx) }()
	time.Sleep(80 * time.Millisecond)
	cancel()
	if got := s.Stats().ScrapeErrors; got == 0 {
		t.Errorf("non-2xx response should count as error; got %d", got)
	}
}

func TestScraper_BodySizeCappedAt16MiB(t *testing.T) {
	// Serve a body larger than 16 MiB; scraper should cap read.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		// Write 20 MiB of "vllm:something_we_dont_map 1\n" repeats.
		const line = "vllm:something_we_dont_map 1\n"
		needed := (20 << 20) / len(line)
		w.Write([]byte(strings.Repeat(line, needed)))
	}))
	defer srv.Close()

	s := NewScraper(Config{Interval: 100 * time.Millisecond}, nil, quietLogger())
	s.AddTarget(targetFromTestServer(t, srv, enginedetect.VLLM, 99))
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = s.Run(ctx) }()
	time.Sleep(150 * time.Millisecond)
	cancel()

	if got := s.Stats().BytesRead; got > 17<<20 {
		t.Errorf("BytesRead = %d, want <= 16 MiB cap", got)
	}
}

// atomic_int wraps a uint64 for tests; uses sync.Mutex rather than
// sync/atomic to keep the test imports light.
type atomic_int struct {
	mu sync.Mutex
	v  int
}

func (a *atomic_int) add(n int) {
	a.mu.Lock()
	a.v += n
	a.mu.Unlock()
}
func (a *atomic_int) get() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.v
}
