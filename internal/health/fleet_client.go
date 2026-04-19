package health

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ingero-io/ingero/pkg/contract"
)

// maxThresholdResponseBody caps the response body read from Fleet's
// threshold endpoint. Fleet sends a tiny JSON object; anything larger is
// suspicious.
const maxThresholdResponseBody = 8 * 1024 // 8 KiB

// PollerConfig configures the GET-endpoint fallback poller.
type PollerConfig struct {
	// BaseURL is the Fleet threshold API root, e.g. "https://fleet:8080".
	// The poller appends contract.ThresholdAPIPath and the cluster_id
	// query parameter.
	BaseURL string
	// ClusterID is sent as the `cluster_id` query parameter.
	ClusterID string
	// Interval is the base polling cadence; actual interval is jittered
	// +/- 20% to de-synchronize across fleet-wide restarts.
	Interval time.Duration
	// Timeout bounds each GET attempt.
	Timeout time.Duration
	// FallbackBackoff is used when the server returns 429 without a
	// Retry-After header. Defaults to Interval * 2.
	FallbackBackoff time.Duration
	// TLSConfig, if non-nil, overrides the poller's HTTP client TLS. Use
	// the same config object as the emitter to reuse mTLS material.
	TLSConfig *tls.Config
	// Insecure selects http:// when BaseURL is a bare host:port.
	Insecure bool
	// Headers are attached to every GET request (e.g., Authorization).
	Headers map[string]string
}

// Validate returns an error on malformed configs.
func (c PollerConfig) Validate() error {
	if strings.TrimSpace(c.BaseURL) == "" {
		return errors.New("poller.base_url is required")
	}
	if strings.TrimSpace(c.ClusterID) == "" {
		return errors.New("poller.cluster_id is required")
	}
	if c.Interval < time.Second {
		return fmt.Errorf("poller.interval must be >= 1s: got %s", c.Interval)
	}
	if c.Timeout < 100*time.Millisecond {
		return fmt.Errorf("poller.timeout must be >= 100ms: got %s", c.Timeout)
	}
	if c.Timeout > c.Interval {
		return fmt.Errorf("poller.timeout (%s) must be <= interval (%s)", c.Timeout, c.Interval)
	}
	return nil
}

// Poller issues periodic GET requests to Fleet's threshold endpoint when
// piggyback headers are unavailable. The poller suspends itself whenever
// the ThresholdCache reports piggyback as available (Fleet is delivering
// the threshold on each push response) and resumes when piggyback
// disappears.
type Poller struct {
	cfg    PollerConfig
	url    string
	cache  *ThresholdCache
	client *http.Client
	log    *slog.Logger
	now    func() time.Time // overridable for tests

	mu  sync.Mutex
	rng *rand.Rand

	requests      atomic.Int64
	successes     atomic.Int64
	clientErrors  atomic.Int64
	serverErrors  atomic.Int64
	rateLimited   atomic.Int64
	suspendedHits atomic.Int64
}

// NewPoller constructs a Poller. Returns an error on invalid config.
// Pass a nil logger to use slog.Default(); cache must not be nil.
func NewPoller(cfg PollerConfig, cache *ThresholdCache, log *slog.Logger) (*Poller, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cache == nil {
		return nil, errors.New("poller: cache must not be nil")
	}
	if log == nil {
		log = slog.Default()
	}

	u, err := buildThresholdURL(cfg.BaseURL, cfg.ClusterID, cfg.Insecure)
	if err != nil {
		return nil, fmt.Errorf("poller.base_url: %w", err)
	}

	// Rotate DNS periodically so polling rebalances across Fleet replicas
	// behind a headless Service. Same rationale as emitter.go.
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.IdleConnTimeout = 30 * time.Second
	tr.MaxIdleConnsPerHost = 1
	client := &http.Client{
		Timeout:   cfg.Timeout,
		Transport: tr,
		// Deny redirects outright — Fleet is a single well-known host;
		// a 3xx to a different origin would (a) leak the mTLS client
		// cert to an unexpected server, (b) potentially apply the
		// client timeout/TLS config to a host that was never pinned.
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	if cfg.TLSConfig != nil {
		trTLS := tr.Clone()
		trTLS.TLSClientConfig = cfg.TLSConfig
		client.Transport = trTLS
	}

	if cfg.FallbackBackoff == 0 {
		cfg.FallbackBackoff = cfg.Interval * 2
	}

	return &Poller{
		cfg:    cfg,
		url:    u,
		cache:  cache,
		client: client,
		log:    log,
		rng:    rand.New(rand.NewSource(time.Now().UnixNano())),
		now:    time.Now,
	}, nil
}

// SetClock overrides `time.Now` for deterministic tests. Must be called
// before Run.
func (p *Poller) SetClock(clock func() time.Time) {
	if clock != nil {
		p.now = clock
	}
}

// Run drives the poller until ctx is cancelled. On each tick: if
// piggyback is currently available, skip the network call; otherwise
// issue one GET request and update the cache.
//
// Cadence model: `nextFire` tracks when the next poll should occur.
// Normally it advances by `jitter(Interval)` after each tick. A
// Retry-After header (on 429 or 5xx) REPLACES that advance with the
// server-provided backoff — not added on top of it.
func (p *Poller) Run(ctx context.Context) error {
	nextFire := p.now().Add(p.jitter(p.cfg.Interval))
	for {
		// Check ctx on every iteration so a zero/negative sleep on a
		// fast-advancing mock clock (tests) does not starve cancellation.
		if err := ctx.Err(); err != nil {
			return err
		}
		sleep := nextFire.Sub(p.now())
		if sleep > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(sleep):
			}
		}

		if p.cache.PiggybackAvailable() {
			p.suspendedHits.Add(1)
			nextFire = p.now().Add(p.jitter(p.cfg.Interval))
			continue
		}

		retryAfter, rateLimited := p.pollOnce(ctx)
		switch {
		case rateLimited && retryAfter > 0:
			// Server asked us to back off — replace the next tick.
			nextFire = p.now().Add(retryAfter)
		default:
			nextFire = p.now().Add(p.jitter(p.cfg.Interval))
		}
	}
}

// PollOnce issues a single GET. Test-only entry point; production callers
// use Run.
func (p *Poller) PollOnce(ctx context.Context) (retryAfter time.Duration, rateLimited bool) {
	return p.pollOnce(ctx)
}

func (p *Poller) pollOnce(ctx context.Context) (time.Duration, bool) {
	p.requests.Add(1)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.url, nil)
	if err != nil {
		p.clientErrors.Add(1)
		p.log.Debug("poller: new request", "err", err.Error())
		return 0, false
	}
	for k, v := range p.cfg.Headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		p.clientErrors.Add(1)
		p.log.Debug("poller: request", "err", err.Error())
		return 0, false
	}
	defer resp.Body.Close()
	body, rerr := io.ReadAll(io.LimitReader(resp.Body, maxThresholdResponseBody+1))
	if rerr != nil {
		p.clientErrors.Add(1)
		p.log.Debug("poller: read body", "err", rerr.Error())
		return 0, false
	}

	switch {
	case resp.StatusCode == http.StatusTooManyRequests:
		p.rateLimited.Add(1)
		return p.parseRetryAfter(resp.Header.Get("Retry-After")), true

	case resp.StatusCode >= 500:
		// 5xx may carry Retry-After per RFC 7231 (e.g. 503 during
		// maintenance). Honor it ONLY when the server explicitly sent
		// one — silent 5xx should fall through to the normal cadence.
		p.serverErrors.Add(1)
		p.log.Debug("poller: server error", "status", resp.StatusCode)
		if raw := strings.TrimSpace(resp.Header.Get("Retry-After")); raw != "" {
			return p.parseRetryAfter(raw), true
		}
		return 0, false

	case resp.StatusCode >= 400 && resp.StatusCode < 500:
		p.clientErrors.Add(1)
		p.log.Debug("poller: client error", "status", resp.StatusCode)
		return 0, false

	case resp.StatusCode < 200 || resp.StatusCode >= 300:
		p.clientErrors.Add(1)
		return 0, false
	}

	if len(body) > maxThresholdResponseBody {
		p.clientErrors.Add(1)
		p.log.Warn("poller: response exceeds max body size", "size", len(body))
		return 0, false
	}

	var parsed struct {
		Threshold float64 `json:"threshold"`
		QuorumMet bool    `json:"quorum_met"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		p.clientErrors.Add(1)
		p.log.Debug("poller: decode", "err", err.Error())
		return 0, false
	}

	now := p.now()

	// Fleet cold-start shape (threshold=0, quorum_met=false) is a valid
	// response meaning "Fleet is reachable but has no threshold yet."
	// It must land in the cache so Story 3.3's ModeEvaluator can select
	// fleet-cached; routing it through parseAndSet would drop it via
	// sanity bounds.
	if parsed.Threshold == 0 && !parsed.QuorumMet {
		p.cache.Set(0, false, now)
		p.successes.Add(1)
		return 0, false
	}

	// Normal case: reuse the ThresholdCache's strict parse path so GET
	// fallback and piggyback headers produce the same downstream
	// behavior. Only count as success when the cache actually absorbed
	// the value — rejected payloads are observability-only.
	tRaw := strconv.FormatFloat(parsed.Threshold, 'f', -1, 64)
	qRaw := strconv.FormatBool(parsed.QuorumMet)
	if p.cache.parseAndSet(tRaw, qRaw, now) {
		p.successes.Add(1)
	}
	return 0, false
}

// parseRetryAfter decodes the two standard forms (RFC 7231):
//   - delay-seconds (integer)
//   - HTTP-date
//
// A malformed or missing value falls back to FallbackBackoff.
func (p *Poller) parseRetryAfter(s string) time.Duration {
	if d, ok := parseRetryAfterHeader(s); ok {
		return d
	}
	return p.cfg.FallbackBackoff
}

// jitter returns base +/- 20%.
func (p *Poller) jitter(base time.Duration) time.Duration {
	p.mu.Lock()
	defer p.mu.Unlock()
	pct := p.rng.Float64()*0.4 - 0.2
	return base + time.Duration(float64(base)*pct)
}

// Stats returns cumulative counters.
func (p *Poller) Stats() (requests, successes, clientErrs, serverErrs, rateLimited, suspended int64) {
	return p.requests.Load(), p.successes.Load(), p.clientErrors.Load(),
		p.serverErrors.Load(), p.rateLimited.Load(), p.suspendedHits.Load()
}

// buildThresholdURL resolves the full GET URL including the cluster_id
// query parameter. Accepts bare host:port or a full URL with optional
// trailing slashes.
func buildThresholdURL(base, clusterID string, insecure bool) (string, error) {
	base = strings.TrimSpace(base)
	if base == "" {
		return "", errors.New("empty base")
	}

	var u *url.URL
	var err error
	if strings.Contains(base, "://") {
		u, err = url.Parse(base)
		if err != nil {
			return "", fmt.Errorf("parse: %w", err)
		}
	} else {
		scheme := "https"
		if insecure {
			scheme = "http"
		}
		u, err = url.Parse(scheme + "://" + base)
		if err != nil {
			return "", fmt.Errorf("parse: %w", err)
		}
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("unsupported scheme %q", u.Scheme)
	}
	if u.Host == "" {
		return "", errors.New("missing host")
	}

	p := strings.TrimRight(u.Path, "/")
	if !strings.HasSuffix(p, contract.ThresholdAPIPath) {
		p += contract.ThresholdAPIPath
	}
	u.Path = p

	q := u.Query()
	q.Set(contract.ParamClusterID, clusterID)
	u.RawQuery = q.Encode()

	return u.String(), nil
}
