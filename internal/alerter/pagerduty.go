package alerter

import (
	"bytes"
	"context"
	"encoding/json"
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
	payload := pagerdutyPayload(ev, p.cfg)
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
	// PD returns 202 Accepted on success. 4xx = invalid payload;
	// 5xx = transient; we surface both as errors so the caller can
	// log them.
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
