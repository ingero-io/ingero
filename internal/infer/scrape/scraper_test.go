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

func TestScraper_LookupModelAndEngine(t *testing.T) {
	// Confirms the LookupModel / LookupEngine accessors return the
	// values stamped on AddTarget. Used by the snapshot path to
	// enrich Layer 1 inference metric data points with
	// gen_ai.request.model / gen_ai.system attributes so the
	// Fleet-side cross-pod groupBy can aggregate by served model.
	s := NewScraper(Config{}, nil, quietLogger())
	s.AddTarget(Target{
		PID:    7,
		Engine: enginedetect.VLLM,
		Host:   "127.0.0.1",
		Port:   8000,
		Model:  "meta-llama/Llama-3-7b",
	})
	s.AddTarget(Target{
		PID:    8,
		Engine: enginedetect.TGI,
		Host:   "127.0.0.1",
		Port:   8080,
		Model:  "bigcode/starcoder",
	})

	if got := s.LookupModel(7); got != "meta-llama/Llama-3-7b" {
		t.Errorf("LookupModel(7) = %q, want meta-llama/Llama-3-7b", got)
	}
	if got := s.LookupEngine(7); got != "vllm" {
		t.Errorf("LookupEngine(7) = %q, want vllm", got)
	}
	if got := s.LookupModel(8); got != "bigcode/starcoder" {
		t.Errorf("LookupModel(8) = %q, want bigcode/starcoder", got)
	}
	// Unknown PID returns empty string, not panic.
	if got := s.LookupModel(99); got != "" {
		t.Errorf("LookupModel(unknown) = %q, want empty", got)
	}
	if got := s.LookupEngine(99); got != "" {
		t.Errorf("LookupEngine(unknown) = %q, want empty", got)
	}
	// PID 0 is a sentinel that never matches.
	if got := s.LookupModel(0); got != "" {
		t.Errorf("LookupModel(0) = %q, want empty", got)
	}
}

func TestScraper_LookupModelOnNilScraper(t *testing.T) {
	// The cli/trace.go enrichment path calls these on the scraper
	// pointer, which is nil when --inference-scrape=off. Both
	// accessors must be nil-safe.
	var s *Scraper
	if got := s.LookupModel(1); got != "" {
		t.Errorf("LookupModel on nil = %q, want empty", got)
	}
	if got := s.LookupEngine(1); got != "" {
		t.Errorf("LookupEngine on nil = %q, want empty", got)
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

func TestScraper_Redetect_DiscoverViaPIDLister(t *testing.T) {
	// v0.16.4 #10: the PIDLister returns a freshly-launched engine PID.
	// On the first re-detection tick the scraper should add it as a
	// target. Detector is injected so we don't need a real /proc.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, vllmFixture)
	}))
	defer srv.Close()
	host, portStr, _ := net.SplitHostPort(strings.TrimPrefix(srv.URL, "http://"))
	port, _ := strconv.Atoi(portStr)
	_ = host

	var listed []uint32
	listMu := sync.Mutex{}
	lister := func() []uint32 {
		listMu.Lock()
		defer listMu.Unlock()
		return listed
	}
	detector := func(pid uint32) (enginedetect.Detection, bool) {
		if pid == 4242 {
			return enginedetect.Detection{Engine: enginedetect.VLLM, Port: uint16(port)}, true
		}
		return enginedetect.Detection{}, false
	}

	s := NewScraper(Config{
		Interval:               50 * time.Millisecond,
		RedetectInterval:       30 * time.Millisecond,
		RedetectStableInterval: 30 * time.Millisecond,
		PIDLister:              lister,
		Detector:               detector,
	}, nil, quietLogger())

	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = s.Run(ctx) }()

	// Initially: no PIDs visible. No targets registered.
	time.Sleep(70 * time.Millisecond)
	if got := len(s.Targets()); got != 0 {
		t.Errorf("targets before lister returns anything = %d, want 0", got)
	}

	// "Engine starts up": expose its PID via the lister.
	listMu.Lock()
	listed = []uint32{4242}
	listMu.Unlock()

	// Wait for at least one re-detect tick.
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if len(s.Targets()) == 1 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	cancel()

	if got := len(s.Targets()); got != 1 {
		t.Fatalf("targets after engine startup = %d, want 1", got)
	}
	target := s.Targets()[0]
	if target.PID != 4242 || target.Engine != enginedetect.VLLM {
		t.Errorf("target = %+v, want PID=4242 engine=vllm", target)
	}
	if target.Port != uint16(port) {
		t.Errorf("target.Port = %d, want %d", target.Port, port)
	}
}

func TestScraper_Redetect_RemovesDeadPID(t *testing.T) {
	// v0.16.4 #10: a registered PID that no longer matches a known
	// engine (process died, PID gone) should be removed on the next
	// re-detection tick.
	alive := make(map[uint32]bool)
	aliveMu := sync.Mutex{}
	aliveMu.Lock()
	alive[7] = true
	aliveMu.Unlock()
	detector := func(pid uint32) (enginedetect.Detection, bool) {
		aliveMu.Lock()
		defer aliveMu.Unlock()
		if alive[pid] {
			return enginedetect.Detection{Engine: enginedetect.VLLM, Port: 8000}, true
		}
		return enginedetect.Detection{}, false
	}
	// Install a lister so re-detection runs; both intervals tight
	// so the test doesn't wait the production cadence.
	s := NewScraper(Config{
		Interval:               200 * time.Millisecond,
		RedetectInterval:       30 * time.Millisecond,
		RedetectStableInterval: 30 * time.Millisecond,
		PIDLister:              func() []uint32 { return []uint32{7} },
		Detector:               detector,
	}, nil, quietLogger())
	s.AddTarget(Target{PID: 7, Engine: enginedetect.VLLM, Host: "127.0.0.1", Port: 8000, Path: "/metrics"})

	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = s.Run(ctx) }()
	time.Sleep(80 * time.Millisecond)

	// Engine "dies": detector now returns false for PID 7.
	aliveMu.Lock()
	alive[7] = false
	aliveMu.Unlock()

	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if len(s.Targets()) == 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	cancel()
	if got := len(s.Targets()); got != 0 {
		t.Errorf("dead PID still registered, target count = %d", got)
	}
}

func TestScraper_Redetect_ReplacesOnEngineIdentityChange(t *testing.T) {
	// v0.16.4 #10: PID recycled to a different engine (k8s pod swap
	// reusing the same PID) should be replaced, not duplicated.
	current := enginedetect.Detection{Engine: enginedetect.VLLM, Port: 8000}
	curMu := sync.Mutex{}
	detector := func(pid uint32) (enginedetect.Detection, bool) {
		curMu.Lock()
		defer curMu.Unlock()
		return current, true
	}
	s := NewScraper(Config{
		Interval:               200 * time.Millisecond,
		RedetectInterval:       30 * time.Millisecond,
		RedetectStableInterval: 30 * time.Millisecond,
		PIDLister:              func() []uint32 { return []uint32{9} },
		Detector:               detector,
	}, nil, quietLogger())
	s.AddTarget(Target{PID: 9, Engine: enginedetect.VLLM, Host: "127.0.0.1", Port: 8000, Path: "/metrics"})

	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = s.Run(ctx) }()
	time.Sleep(80 * time.Millisecond)

	// Engine identity changes on PID 9: now it's TGI on a different port.
	curMu.Lock()
	current = enginedetect.Detection{Engine: enginedetect.TGI, Port: 8080}
	curMu.Unlock()

	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		ts := s.Targets()
		if len(ts) == 1 && ts[0].Engine == enginedetect.TGI {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	cancel()
	ts := s.Targets()
	if len(ts) != 1 {
		t.Fatalf("target count = %d, want 1 (replaced not duplicated)", len(ts))
	}
	if ts[0].Engine != enginedetect.TGI || ts[0].Port != 8080 {
		t.Errorf("target after identity change = %+v, want TGI/8080", ts[0])
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
