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

	// Default fast / slow re-detection cadences. Fast (30s) when no
	// targets are known so the agent picks up a freshly-booted engine
	// quickly; slow (5m) once at least one target is registered, since
	// re-walking /proc every 30s on a busy machine is wasteful when
	// the engine is stable. Adaptive switching between the two happens
	// inside Run.
	defaultRedetectFastInterval = 30 * time.Second
	defaultRedetectSlowInterval = 5 * time.Minute
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

	// Model is the model identifier the engine is serving, parsed
	// from the cmdline at detection time (--model / --model-id).
	// May be empty when the engine doesn't report it on cmdline (the
	// scraper's parsers also extract it from /metrics body labels in
	// some cases). Forwarded into Layer 1 OTLP attributes so the
	// Fleet processor can group cross-pod baselines by model when
	// running peer-relative outlier detection.
	Model string
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

// PIDLister returns the candidate PIDs the re-detection loop should
// inspect on each tick. v0.16.4 #10. Implementations:
//
//   - When the agent is invoked with --pid, the lister returns those
//     PIDs verbatim. Re-detect on a fixed PID set still catches the
//     case where one of those PIDs gets recycled by the kernel and
//     now hosts a different engine (k8s rolling restart, pod swap).
//   - When the agent runs system-wide (--pid empty), the lister walks
//     /proc and returns every PID whose cmdline matches a known
//     engine pattern (enginedetect.ListEnginePIDs). This catches
//     engines that started after the agent.
//
// Nil is allowed and means "do not actively discover new PIDs" — the
// scraper will still re-confirm engine identity on its registered
// targets, just won't add new ones.
type PIDLister func() []uint32

// Detector is the function the scraper calls to classify a PID. The
// production implementation is enginedetect.Detect; tests inject a
// fake to avoid /proc dependence. Nil is allowed and falls back to
// enginedetect.Detect.
type Detector func(pid uint32) (enginedetect.Detection, bool)

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

	// RedetectInterval is the v0.16.4 continuous-re-detection cadence
	// when the scraper has zero registered targets (the "looking for
	// an engine" state). Once at least one target is registered, the
	// loop slows to RedetectStableInterval to avoid burning /proc
	// reads on stable hosts. Zero or negative resolves to
	// defaultRedetectFastInterval (30s). Continuous re-detection is
	// gated on PIDLister != nil so callers (mostly tests) that just
	// want a dumb scraper without lifecycle management opt out by
	// leaving PIDLister nil.
	RedetectInterval time.Duration

	// RedetectStableInterval is the slower cadence used once the
	// scraper has at least one target registered. Zero or negative
	// resolves to defaultRedetectSlowInterval (5m). Test seam.
	RedetectStableInterval time.Duration

	// PIDLister returns the candidate PIDs to (re)inspect each tick.
	// Nil disables continuous re-detection entirely (the scraper
	// reverts to v0.16.2 one-shot behavior — registered targets are
	// trusted and never confirmed). See PIDLister doc for the two
	// production shapes.
	PIDLister PIDLister

	// Detector classifies a PID. Test seam; nil falls back to
	// enginedetect.Detect.
	Detector Detector
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
	if cfg.RedetectInterval <= 0 {
		cfg.RedetectInterval = defaultRedetectFastInterval
	}
	if cfg.RedetectStableInterval <= 0 {
		cfg.RedetectStableInterval = defaultRedetectSlowInterval
	}
	if cfg.Detector == nil {
		cfg.Detector = enginedetect.Detect
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

// LookupModel returns the served model identifier for the given PID,
// or "" if no target is registered for that PID. Used by the agent's
// inference snapshot path to enrich Layer 1 (per-workload) metric
// data points with a gen_ai.request.model attribute so the Fleet
// processor can group cross-pod baselines by model.
func (s *Scraper) LookupModel(pid uint32) string {
	if s == nil || pid == 0 {
		return ""
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.targets[pid]
	if !ok {
		return ""
	}
	return t.Model
}

// LookupEngine returns the served engine identifier (vllm / tgi /
// sglang / triton) for the given PID, or "" when no target is
// registered. Companion to LookupModel for the gen_ai.system
// attribute.
func (s *Scraper) LookupEngine(pid uint32) string {
	if s == nil || pid == 0 {
		return ""
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.targets[pid]
	if !ok {
		return ""
	}
	return string(t.Engine)
}

// Run drives the scrape ticker. Blocks until ctx is cancelled.
// Errors during individual scrapes are logged and counted but do
// not propagate — the agent stays up regardless of engine health.
//
// v0.16.4 #10 adds a second ticker for continuous engine re-detection.
// Cadence is adaptive: defaultRedetectFastInterval (30s, configurable
// via Config.RedetectInterval) when no targets are registered, slowing
// to defaultRedetectSlowInterval (5m, internal constant) once at least
// one target exists. The slow cadence still catches engine restarts
// (PID recycle, k8s rolling pod swap) without wastefully re-walking
// /proc on stable hosts.
func (s *Scraper) Run(ctx context.Context) error {
	scrapeTick := time.NewTicker(s.cfg.Interval)
	defer scrapeTick.Stop()

	// Re-detect feature is opt-in via PIDLister (production callers
	// always set it; tests that want pre-v0.16.4 behavior leave it
	// nil and skip the timer entirely).
	redetectEnabled := s.cfg.PIDLister != nil
	var redetectC <-chan time.Time
	var redetectTimer *time.Timer
	if redetectEnabled {
		redetectTimer = time.NewTimer(s.redetectInterval())
		defer redetectTimer.Stop()
		redetectC = redetectTimer.C

		// Fire one immediate re-detection so a starting agent picks
		// up engines that were already running, without waiting an
		// interval.
		s.tickRedetect()
	}

	// Fire one scrape immediately so the first samples land within
	// the first interval rather than after.
	s.tickAll(ctx)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-scrapeTick.C:
			s.tickAll(ctx)
		case <-redetectC:
			s.tickRedetect()
			redetectTimer.Reset(s.redetectInterval())
		}
	}
}

// redetectInterval picks the next re-detection cadence based on
// whether we currently have any registered targets. The fast/slow
// adaptive logic mirrors the documented v0.16.4 design: fast while
// hunting for an engine, slow once it's found.
func (s *Scraper) redetectInterval() time.Duration {
	s.mu.Lock()
	hasTargets := len(s.targets) > 0
	s.mu.Unlock()
	if hasTargets {
		return s.cfg.RedetectStableInterval
	}
	return s.cfg.RedetectInterval
}

// tickRedetect runs one engine-discovery + identity-confirmation pass.
//
// Phase 1 (confirm): for every registered target, re-Detect on the
// PID. If the cmdline no longer matches a known engine (process died,
// PID recycled to a non-engine), drop the target. If the engine
// identity changed (PID recycled to a *different* engine), replace
// the target with the new Detection so subsequent scrapes hit the
// right /metrics shape.
//
// Phase 2 (discover): if a PIDLister is installed, call it to get
// candidate PIDs, run Detect on each new (not-already-registered) one,
// and register matches. Idempotent on PIDs already registered.
//
// Both phases log a single info line on each registration change so
// operators can see the wire-up history; sustained "no engine" runs
// stay quiet.
func (s *Scraper) tickRedetect() {
	// Snapshot current targets under lock; mutate in place after.
	s.mu.Lock()
	current := make(map[uint32]Target, len(s.targets))
	for pid, t := range s.targets {
		current[pid] = t
	}
	s.mu.Unlock()

	detect := s.cfg.Detector
	if detect == nil {
		detect = enginedetect.Detect
	}

	// Phase 1: confirm each registered target.
	for pid, t := range current {
		det, ok := detect(pid)
		if !ok {
			// PID gone or cmdline no longer matches. Drop.
			s.RemoveTarget(pid)
			s.log.Info("infer scrape: target removed (PID gone or engine ended)",
				"pid", pid, "engine", t.Engine)
			continue
		}
		if det.Engine != t.Engine || det.Port != t.Port {
			// Identity change on the same PID. Replace the target.
			newTarget := Target{
				Engine: det.Engine,
				Host:   t.Host,
				Port:   det.Port,
				Path:   det.Engine.MetricsPath(),
				PID:    pid,
				Model:  det.Model,
			}
			s.AddTarget(newTarget)
			s.log.Info("infer scrape: target replaced (engine identity changed)",
				"pid", pid,
				"old_engine", t.Engine, "old_port", t.Port,
				"new_engine", det.Engine, "new_port", det.Port)
		}
	}

	// Phase 2: discover new PIDs.
	if s.cfg.PIDLister == nil {
		return
	}
	for _, pid := range s.cfg.PIDLister() {
		if pid == 0 {
			continue
		}
		if _, already := current[pid]; already {
			continue
		}
		det, ok := detect(pid)
		if !ok {
			continue
		}
		// Default Host to 127.0.0.1; the agent never knows the engine
		// listens elsewhere unless an operator set --inference-scrape-host
		// on the trace command, in which case the agent constructs the
		// initial Target via AddTarget directly. The scraper itself only
		// sees PIDs from the lister and assumes loopback.
		s.AddTarget(Target{
			Engine: det.Engine,
			Host:   "127.0.0.1",
			Port:   det.Port,
			Path:   det.Engine.MetricsPath(),
			PID:    pid,
			Model:  det.Model,
		})
		s.log.Info("infer scrape: target discovered",
			"pid", pid, "engine", det.Engine, "port", det.Port, "model", det.Model)
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
