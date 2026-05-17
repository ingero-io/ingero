package cli

import (
	"context"
	"log/slog"
	"time"

	"github.com/ingero-io/ingero/internal/infer"
	"github.com/ingero-io/ingero/internal/remediate"
)

// sloWatcherInterval is the cadence at which the watcher drains
// per-workload SLO breaches from the inference engine. Matches the
// 5s default of the other periodic watchers (link_poller,
// zombie_gpu_watcher); the per-tracker sustain count is sized
// against this so 3 sustain ticks = ~15s of sustained breach before
// emission.
const sloWatcherInterval = 5 * time.Second

// startInferenceSloWatcher spawns the goroutine that drains rolling-
// p99 SLO breaches from the inference engine and publishes each one
// over the remediate UDS as an `inference_slo_breach` message.
//
// Returns immediately. The goroutine exits on ctx cancellation. Nil-
// safe on all three inputs: a nil engine, nil remediateSrv, or nil
// log turns the call into a no-op (matches the throttle/link poller
// pattern). The watcher only fires when both --inference and
// --remediate are on (the engine and srv are otherwise nil), so the
// no-op path handles every "feature off" combination cleanly.
//
// Per-emission log line includes the breach ratio so operators can
// correlate against the orchestrator's `inference_slo_breach`
// dispatch log on the EE side.
func startInferenceSloWatcher(
	ctx context.Context,
	engine *infer.Engine,
	srv *remediate.Server,
	nodeID, clusterID string,
	log *slog.Logger,
) {
	if engine == nil || srv == nil {
		return
	}
	if log == nil {
		log = slog.Default()
	}
	go func() {
		t := time.NewTicker(sloWatcherInterval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case now := <-t.C:
				drainSloBreachesOnce(engine, srv, nodeID, clusterID, now, log)
			}
		}
	}()
}

// drainSloBreachesOnce is the testable inner step: one call to the
// engine's DrainSloBreaches, one wire emit per returned breach.
// Caller (the watcher goroutine, or the unit test) supplies `now` so
// the wall-clock dependency stays explicit.
func drainSloBreachesOnce(
	engine *infer.Engine,
	srv *remediate.Server,
	nodeID, clusterID string,
	now time.Time,
	log *slog.Logger,
) {
	for _, b := range engine.DrainSloBreaches(now) {
		emit := remediate.InferenceSloBreach{
			PID:           b.Key.PID,
			P99LatencyNs:  b.Breach.CurrentP99Ns,
			BaselineP99Ns: b.Breach.BaselineP99Ns,
			BreachRatio:   b.Breach.Ratio,
		}
		if err := srv.SendInferenceSloBreach(emit, nodeID, clusterID); err != nil {
			log.Debug("remediate: inference_slo_breach drop",
				"pid", emit.PID, "ratio", emit.BreachRatio, "err", err)
			continue
		}
		log.Info("inference slo watcher: breach emitted",
			"pid", emit.PID,
			"current_p99_ns", emit.P99LatencyNs,
			"baseline_p99_ns", emit.BaselineP99Ns,
			"ratio", emit.BreachRatio,
		)
	}
}
