// Package alerter consumes straggler NDJSON events from the agent's
// remediation UDS socket and dispatches them to one or more
// notification backends (Slack incoming webhooks, PagerDuty Events
// API v2). Designed to run as a sidecar next to the ingero agent;
// the agent emits, the alerter routes.
//
// Lifecycle:
//
//   - Run(ctx, cfg) blocks until ctx is cancelled.
//   - It dials the UDS socket, reads NDJSON lines, and for each
//     "type":"straggler_state" or "straggle" event dispatches to
//     every enabled backend in order.
//   - Backends fail independently; one backend's HTTP error does not
//     gate the others. Per-backend retry is the backend's
//     responsibility.
//   - On UDS disconnect, the loop attempts to reconnect with
//     exponential backoff (1s, 2s, 4s, 8s, capped at 30s).
package alerter

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"time"
)

// Config carries the runtime configuration. Loaded from JSON at
// startup; see ingero-alerter --config <path>.
type Config struct {
	UDSPath    string           `json:"uds_path"`
	Slack      *SlackConfig     `json:"slack,omitempty"`
	PagerDuty  *PagerDutyConfig `json:"pagerduty,omitempty"`
	HTTPClient *HTTPTimeoutsCfg `json:"http_client,omitempty"`
}

// SlackConfig is the per-backend config for Slack incoming webhooks.
// WebhookURL is required; Channel/Username are optional overrides.
type SlackConfig struct {
	WebhookURL string `json:"webhook_url"`
	Channel    string `json:"channel,omitempty"`
	Username   string `json:"username,omitempty"`
}

// PagerDutyConfig is the per-backend config for PD Events API v2.
// RoutingKey (also called "integration key") is required.
type PagerDutyConfig struct {
	RoutingKey string `json:"routing_key"`
	Severity   string `json:"severity,omitempty"`
}

// HTTPTimeoutsCfg overrides the default HTTP timeouts used by all
// backends. Tests pin these to short values; production leaves them
// at defaults.
type HTTPTimeoutsCfg struct {
	RequestTimeoutSeconds int `json:"request_timeout_seconds,omitempty"`
}

// Validate returns an error on a configuration that the alerter
// cannot run with. Empty backend list is allowed (no-op alerter
// that just consumes the UDS); missing UDS path is fatal.
func (c *Config) Validate() error {
	if c.UDSPath == "" {
		return errors.New("alerter: uds_path is required")
	}
	if c.Slack != nil && c.Slack.WebhookURL == "" {
		return errors.New("alerter: slack.webhook_url is required when slack is configured")
	}
	if c.PagerDuty != nil && c.PagerDuty.RoutingKey == "" {
		return errors.New("alerter: pagerduty.routing_key is required when pagerduty is configured")
	}
	return nil
}

// Backend is the interface every notification backend implements.
// Send takes a parsed straggler event; the backend serializes it as
// it sees fit and returns a per-call error.
type Backend interface {
	Name() string
	Send(ctx context.Context, ev StragglerEvent) error
}

// StragglerEvent is the parsed UDS message; matches the agent's
// remediate.fleetStragglerStateMessage and remediate.straggleMessage
// shapes. Unused fields are dropped silently to keep the alerter
// forward-compatible with future agent additions.
type StragglerEvent struct {
	Type           string  `json:"type"`
	NodeID         string  `json:"node_id,omitempty"`
	ClusterID      string  `json:"cluster_id,omitempty"`
	PID            uint32  `json:"pid,omitempty"`
	Comm           string  `json:"comm,omitempty"`
	Score          float64 `json:"score,omitempty"`
	Threshold      float64 `json:"threshold,omitempty"`
	DetectionMode  string  `json:"detection_mode,omitempty"`
	DominantSignal string  `json:"dominant_signal,omitempty"`
	EventID        string  `json:"event_id,omitempty"`
	Rank           int     `json:"rank,omitempty"`
	WorldSize      int     `json:"world_size,omitempty"`
	TimestampNs    int64   `json:"timestamp_ns,omitempty"`
}

// Run blocks until ctx is cancelled. Connects to the UDS at
// cfg.UDSPath, reads NDJSON, dispatches to backends. On UDS error
// (no agent / socket vanished), retries with exponential backoff.
func Run(ctx context.Context, cfg *Config, log *slog.Logger) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	if log == nil {
		log = slog.Default()
	}

	backends := buildBackends(cfg, log)
	log.Info("ingero-alerter started", "uds", cfg.UDSPath, "backends", backendNames(backends))

	backoff := time.Second
	const maxBackoff = 30 * time.Second
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		err := connectAndDispatch(ctx, cfg.UDSPath, backends, log)
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil
		}
		log.Warn("UDS disconnected; reconnecting after backoff", "err", err, "backoff", backoff)
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(backoff):
		}
		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

func connectAndDispatch(ctx context.Context, udsPath string, backends []Backend, log *slog.Logger) error {
	conn, err := net.Dial("unix", udsPath)
	if err != nil {
		return fmt.Errorf("dial %s: %w", udsPath, err)
	}
	defer conn.Close()
	log.Info("UDS connected", "path", udsPath)

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return err
		}
		var ev StragglerEvent
		if err := json.Unmarshal(scanner.Bytes(), &ev); err != nil {
			log.Warn("invalid NDJSON line; skipping", "err", err)
			continue
		}
		if ev.Type != "straggler_state" && ev.Type != "straggle" {
			continue
		}
		fanout(ctx, backends, ev, log)
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scanner: %w", err)
	}
	return errors.New("UDS EOF")
}

// fanout dispatches one event to every backend in parallel. We log
// per-backend errors but never abort the loop because of them. A
// flaky Slack integration must not block PD delivery.
func fanout(ctx context.Context, backends []Backend, ev StragglerEvent, log *slog.Logger) {
	for _, b := range backends {
		b := b
		go func() {
			if err := b.Send(ctx, ev); err != nil {
				log.Warn("backend send failed", "backend", b.Name(), "err", err, "event_id", ev.EventID)
			}
		}()
	}
}

// buildBackends constructs the enabled backend list from Config.
// The returned slice is empty when no backend is configured;
// running with an empty backend list is supported (no-op alerter).
func buildBackends(cfg *Config, log *slog.Logger) []Backend {
	timeout := 10 * time.Second
	if cfg.HTTPClient != nil && cfg.HTTPClient.RequestTimeoutSeconds > 0 {
		timeout = time.Duration(cfg.HTTPClient.RequestTimeoutSeconds) * time.Second
	}
	out := make([]Backend, 0, 2)
	if cfg.Slack != nil {
		out = append(out, NewSlack(cfg.Slack, timeout, log))
	}
	if cfg.PagerDuty != nil {
		out = append(out, NewPagerDuty(cfg.PagerDuty, timeout, log))
	}
	return out
}

func backendNames(bs []Backend) []string {
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = b.Name()
	}
	return out
}

// LoadConfigFile parses a YAML file into a Config. Convenience for
// the cmd entry point. Tests can construct Config directly.
func LoadConfigFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	return ParseConfig(data)
}
