package alerter

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

// PagerDutyEventsURL is the v2 Events API endpoint. Hard-coded
// because PD has not changed it in years and a config knob
// invites operator confusion.
const PagerDutyEventsURL = "https://events.pagerduty.com/v2/enqueue"

// PagerDuty delivers events to PagerDuty Events API v2 as alerts.
// Severity defaults to "warning"; operators override per-config.
type PagerDuty struct {
	cfg    *PagerDutyConfig
	client *http.Client
	log    *slog.Logger
	url    string // overridden in tests
}

// TriggerParams is the payload shape an MCP-driven incident creation
// caller supplies. Severity must be one of: info, warning, error, critical.
// DedupKey is optional; when empty, Trigger generates a UUIDv4.
type TriggerParams struct {
	Summary       string
	Severity      string
	Source        string
	DedupKey      string
	CustomDetails map[string]any
}

// NewPagerDuty constructs a PagerDuty backend.
func NewPagerDuty(cfg *PagerDutyConfig, timeout time.Duration, log *slog.Logger) *PagerDuty {
	return &PagerDuty{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
		log:    log,
		url:    PagerDutyEventsURL,
	}
}

func (p *PagerDuty) Name() string { return "pagerduty" }

// Send POSTs a PD Events v2 trigger. The dedup key is the
// straggler event_id when present, falling back to
// node_id+timestamp_ns so PD does not collapse distinct stragglers
// into one incident.
func (p *PagerDuty) Send(ctx context.Context, ev StragglerEvent) error {
	return p.post(ctx, pagerdutyPayload(ev, p.cfg))
}

// Trigger posts a PD Events v2 trigger event with caller-supplied content.
// Returns the resolved DedupKey (the supplied one or a freshly generated UUID).
func (p *PagerDuty) Trigger(ctx context.Context, params TriggerParams) (string, error) {
	if p.cfg == nil || p.cfg.RoutingKey == "" {
		return "", errors.New("pagerduty not configured: set alerter.pagerduty.routing_key")
	}
	if params.Summary == "" {
		return "", errors.New("pagerduty trigger: summary is required")
	}
	switch params.Severity {
	case "info", "warning", "error", "critical":
	default:
		return "", fmt.Errorf("pagerduty trigger: invalid severity %q (want info|warning|error|critical)", params.Severity)
	}

	dedup := params.DedupKey
	if dedup == "" {
		u, err := uuidV4()
		if err != nil {
			return "", fmt.Errorf("pagerduty trigger: generate dedup_key: %w", err)
		}
		dedup = u
	}

	source := params.Source
	if source == "" {
		source = "ingero"
	}

	payload := map[string]any{
		"routing_key":  p.cfg.RoutingKey,
		"event_action": "trigger",
		"dedup_key":    dedup,
		"payload": map[string]any{
			"summary":        params.Summary,
			"severity":       params.Severity,
			"source":         source,
			"component":      "ingero",
			"class":          "ai_investigation",
			"custom_details": params.CustomDetails,
		},
	}
	if err := p.post(ctx, payload); err != nil {
		return "", err
	}
	return dedup, nil
}

// post serializes payload and POSTs it to the PD Events v2 endpoint.
// Errors deliberately omit the routing_key — it is a secret and must
// never appear in logs or wrapped errors.
func (p *PagerDuty) post(ctx context.Context, payload map[string]any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("pagerduty marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("pagerduty new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("pagerduty POST: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		return fmt.Errorf("pagerduty POST: status %d", resp.StatusCode)
	}
	return nil
}

// pagerdutyPayload constructs the Events v2 envelope. Exposed at
// package level so tests can assert structure without HTTP.
func pagerdutyPayload(ev StragglerEvent, cfg *PagerDutyConfig) map[string]any {
	severity := cfg.Severity
	if severity == "" {
		severity = "warning"
	}
	dedup := ev.EventID
	if dedup == "" {
		dedup = fmt.Sprintf("%s/%d", ev.NodeID, ev.TimestampNs)
	}
	summary := fmt.Sprintf("Ingero straggler %s/%s score=%.3f threshold=%.3f",
		ev.ClusterID, ev.NodeID, ev.Score, ev.Threshold)
	if ev.NodeID == "" {
		summary = fmt.Sprintf("Ingero local straggler PID=%d comm=%s", ev.PID, ev.Comm)
	}
	return map[string]any{
		"routing_key":  cfg.RoutingKey,
		"event_action": "trigger",
		"dedup_key":    dedup,
		"payload": map[string]any{
			"summary":   summary,
			"severity":  severity,
			"source":    ev.NodeID,
			"component": "ingero",
			"group":     ev.ClusterID,
			"class":     "gpu_straggler",
			"custom_details": map[string]any{
				"event_id":        ev.EventID,
				"node_id":         ev.NodeID,
				"cluster_id":      ev.ClusterID,
				"score":           ev.Score,
				"threshold":       ev.Threshold,
				"detection_mode":  ev.DetectionMode,
				"dominant_signal": ev.DominantSignal,
				"rank":            ev.Rank,
				"world_size":      ev.WorldSize,
			},
		},
	}
}

// SetPagerDutyURLForTest overrides the destination URL on a PagerDuty
// backend. Exported so tests in sibling packages (e.g. internal/mcp)
// can point a real *PagerDuty at an httptest server without a parallel
// fake.
func SetPagerDutyURLForTest(p *PagerDuty, url string) { p.url = url }

// uuidV4 returns an RFC 4122 v4 UUID. Inlined to avoid a transitive
// dep just for one call site.
func uuidV4() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // RFC 4122 variant
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}
