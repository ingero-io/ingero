package scrape

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ingero-io/ingero/internal/infer/enginedetect"
)

// Default scrape interval and request timeout. Defaults chosen to be
// gentle on the engine — 10s scrape with 2s timeout fires
// well under the engine's typical request rate, and the timeout
// caps any single hung scrape so the agent's wall clock doesn't
// stall.
const (
	defaultInterval = 10 * time.Second
	defaultTimeout  = 2 * time.Second
)

// Target identifies one engine to scrape. The agent constructs a
// Target from a Detection (enginedetect.Detect) plus the listening
// host (typically 127.0.0.1 because /metrics is local to the pod).
type Target struct {
	Engine enginedetect.Engine
	Host   string
	Port   uint16
	Path   string

	// Optional — if set, attached to every emitted ScrapedSample's
	// Labels under the "ingero.engine.pid" key so consumers can
	// correlate scraped metrics back to the eBPF-side workload.
	PID uint32
}

// URL returns the http://host:port/path URL the scraper fetches.
func (t Target) URL() string {
	path := t.Path
	if path == "" {
		path = t.Engine.MetricsPath()
	}
	return fmt.Sprintf("http://%s:%d%s", t.Host, t.Port, path)
}

// Sink is the callback invoked once per successful scrape. The
// caller (cli.traceRunE) owns the OTLP exporter and emits the
// canonical-named samples there.
type Sink func(target Target, samples []ScrapedSample)

// Stats reports cumulative scraper telemetry. Safe to call from any
// goroutine. Read-only snapshot; mutating it is meaningless.
type Stats struct {
	Scrapes      uint64
	ScrapeErrors uint64
	ParseErrors  uint64
	BytesRead    uint64
}

// Scraper periodically fetches engine /metrics endpoints and
// publishes parsed OTel-mapped samples through the configured sink.
// One Scraper per agent process; targets can be added at runtime
// (when new engine PIDs appear) and removed at runtime (when PIDs
// exit).
//
// Concurrency model: a single Run goroutine drives the scrape ticker;
// AddTarget / RemoveTarget take a small mutex; the sink callback
// runs on the scrape goroutine (caller is responsible for not
// blocking it).
type Scraper struct {
	cfg Config
	log *slog.Logger
	cli *http.Client

	mu      sync.Mutex
	targets map[uint32]Target // keyed by PID for fast remove
	sink    Sink

	scrapes      atomic.Uint64
	scrapeErrors atomic.Uint64
	parseErrors  atomic.Uint64
	bytesRead    atomic.Uint64
}

// Config tunes the scraper. Zero-value Interval / Timeout resolve
// to defaults (10s / 2s).
type Config struct {
	Interval time.Duration
	Timeout  time.Duration
}

// NewScraper constructs a stopped Scraper. Caller invokes Run on a
// goroutine to start the periodic loop.
func NewScraper(cfg Config, sink Sink, log *slog.Logger) *Scraper {
	if log == nil {
		log = slog.Default()
	}
	if cfg.Interval <= 0 {
		cfg.Interval = defaultInterval
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultTimeout
	}
	return &Scraper{
		cfg:     cfg,
		log:     log,
		cli:     &http.Client{Timeout: cfg.Timeout},
		targets: make(map[uint32]Target),
		sink:    sink,
	}
}

// AddTarget registers a Target for scraping. Subsequent ticks
// include it. If a target with the same PID already exists it is
// replaced (port-change or engine-restart safe).
func (s *Scraper) AddTarget(t Target) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.targets[t.PID] = t
}

// RemoveTarget drops a PID's target. Subsequent ticks skip it.
// No-op when the PID is not registered.
func (s *Scraper) RemoveTarget(pid uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.targets, pid)
}

// Targets returns a snapshot of the current target set.
func (s *Scraper) Targets() []Target {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Target, 0, len(s.targets))
	for _, t := range s.targets {
		out = append(out, t)
	}
	return out
}

// Run drives the scrape ticker. Blocks until ctx is cancelled.
// Errors during individual scrapes are logged and counted but do
// not propagate — the agent stays up regardless of engine health.
func (s *Scraper) Run(ctx context.Context) error {
	t := time.NewTicker(s.cfg.Interval)
	defer t.Stop()

	// Fire one tick immediately so the first samples land within
	// the first interval rather than after.
	s.tickAll(ctx)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			s.tickAll(ctx)
		}
	}
}

// tickAll runs one scrape pass over every registered target. Each
// target is scraped sequentially (no per-target goroutine) — the
// expected target count is small (one or two engines per pod) and
// the scrape is brief, so concurrency is unnecessary complexity.
func (s *Scraper) tickAll(ctx context.Context) {
	targets := s.Targets()
	for _, t := range targets {
		s.scrape(ctx, t)
	}
}

// scrape fetches one target's /metrics, parses, and pushes through
// the sink. Errors are logged and counted; the sink is not called
// on error.
func (s *Scraper) scrape(ctx context.Context, t Target) {
	url := t.URL()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		s.scrapeErrors.Add(1)
		s.log.Warn("infer scrape: build request failed", "url", url, "err", err.Error())
		return
	}
	req.Header.Set("Accept", "text/plain; version=0.0.4")
	req.Header.Set("User-Agent", "ingero-infer-scrape")
	resp, err := s.cli.Do(req)
	if err != nil {
		s.scrapeErrors.Add(1)
		// Quiet the log on the common case — engine may not be
		// listening yet during pod startup. Use Debug, not Warn,
		// for connection-refused; Warn for malformed responses.
		if errors.Is(err, context.Canceled) {
			return
		}
		s.log.Debug("infer scrape: fetch failed", "url", url, "err", err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		s.scrapeErrors.Add(1)
		s.log.Debug("infer scrape: non-2xx", "url", url, "status", resp.StatusCode)
		return
	}
	// Cap body read at 16 MiB to bound memory if a misbehaving
	// engine emits enormous /metrics output. Prometheus exposition
	// at typical inference scale is < 1 MiB; 16x headroom is plenty.
	const maxBytes = 16 << 20
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes))
	if err != nil {
		s.scrapeErrors.Add(1)
		s.log.Warn("infer scrape: read failed", "url", url, "err", err.Error())
		return
	}
	s.bytesRead.Add(uint64(len(body)))

	parser := parserFor(t.Engine)
	if parser == nil {
		s.scrapeErrors.Add(1)
		return
	}
	samples, err := parser.Parse(body)
	if err != nil {
		s.parseErrors.Add(1)
		s.log.Warn("infer scrape: parse failed", "url", url, "err", err.Error())
		return
	}
	// Tag each sample with the target's PID so consumers can
	// correlate back to the eBPF-side workload key.
	if t.PID != 0 {
		for i := range samples {
			if samples[i].Labels == nil {
				samples[i].Labels = make(map[string]string, 1)
			}
			samples[i].Labels["ingero.engine.pid"] = fmt.Sprintf("%d", t.PID)
		}
	}
	s.scrapes.Add(1)
	if s.sink != nil {
		s.sink(t, samples)
	}
}

// parserFor returns the Parser for an engine, or nil for unknown.
func parserFor(e enginedetect.Engine) Parser {
	switch e {
	case enginedetect.VLLM:
		return VLLMParser{}
	case enginedetect.TGI:
		return TGIParser{}
	case enginedetect.SGLang:
		return SGLangParser{}
	case enginedetect.Triton:
		return TritonParser{}
	}
	return nil
}

// Stats returns a point-in-time snapshot of cumulative counters.
func (s *Scraper) Stats() Stats {
	return Stats{
		Scrapes:      s.scrapes.Load(),
		ScrapeErrors: s.scrapeErrors.Load(),
		ParseErrors:  s.parseErrors.Load(),
		BytesRead:    s.bytesRead.Load(),
	}
}
