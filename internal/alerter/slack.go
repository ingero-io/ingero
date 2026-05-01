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

// Slack delivers events to a Slack incoming webhook. Slack's webhook
// returns 200 on success; any non-2xx is surfaced as an error so the
// dispatcher can log it.
type Slack struct {
	cfg    *SlackConfig
	client *http.Client
	log    *slog.Logger
}

// NewSlack constructs a Slack backend. The HTTP client honours the
// shared per-call timeout passed in.
func NewSlack(cfg *SlackConfig, timeout time.Duration, log *slog.Logger) *Slack {
	return &Slack{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
		log:    log,
	}
}

func (s *Slack) Name() string { return "slack" }

// Send POSTs a small Slack message describing the straggler event.
// The payload format is the legacy "text + attachments" shape that
// every incoming-webhook integration accepts; we deliberately do
// not use Block Kit so the message is dumb HTML in workspaces that
// do not render blocks.
func (s *Slack) Send(ctx context.Context, ev StragglerEvent) error {
	payload := slackPayload(ev, s.cfg)
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("slack marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.WebhookURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("slack new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("slack POST: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("slack POST: status %d", resp.StatusCode)
	}
	return nil
}

// slackPayload renders the event body. Exposed at package level so
// tests can assert structure without going through HTTP.
func slackPayload(ev StragglerEvent, cfg *SlackConfig) map[string]any {
	title := fmt.Sprintf("Ingero straggler on %s/%s", ev.ClusterID, ev.NodeID)
	if ev.NodeID == "" {
		title = fmt.Sprintf("Ingero local straggler (PID %d)", ev.PID)
	}
	text := fmt.Sprintf(
		"score=%.3f threshold=%.3f mode=%s dominant=%s rank=%d/%d event_id=%s",
		ev.Score, ev.Threshold, ev.DetectionMode, ev.DominantSignal, ev.Rank, ev.WorldSize, ev.EventID,
	)
	out := map[string]any{
		"text": title + "\n" + text,
	}
	if cfg.Channel != "" {
		out["channel"] = cfg.Channel
	}
	if cfg.Username != "" {
		out["username"] = cfg.Username
	}
	return out
}
