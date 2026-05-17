package cli

import (
	"context"
	"log/slog"
	"time"

	"github.com/ingero-io/ingero/internal/remediate"
	"github.com/ingero-io/ingero/internal/tcpretransmit"
	"github.com/ingero-io/ingero/pkg/events"
)

// startTcpRetransmitWatcher tees the raw TCP tracer channel through a
// per-PID sustain tracker and publishes one TcpRetransmitStorm wire
// message per detected episode. Returns a derived event channel that
// the caller must use in place of the raw `in` channel; downstream
// consumers (mergeAllEventChannels, the unified event pipeline) read
// from the returned channel exactly as they would from `in`.
//
// The watcher does not alter or drop events on the data path: each
// raw event is observed by the sustain tracker and forwarded to the
// returned channel unchanged. The sustain tracker is sampled on a
// 1s tick; one Sweep returns at most one Storm per PID per episode,
// gated by the tracker's internal suppression window so a flapping
// PID does not flood the orchestrator.
//
// `in` close is mirrored to the returned channel — when the upstream
// tracer drains (ctx cancel + tracer.Close), the watcher closes the
// returned channel after one final Sweep so any tail-edge storm
// emission still lands. The sweep ticker shares ctx.Done() with the
// tee, so cancellation tears down both goroutines.
func startTcpRetransmitWatcher(
	ctx context.Context,
	in <-chan events.Event,
	srv *remediate.Server,
	nodeID, clusterID string,
) <-chan events.Event {
	out := make(chan events.Event, cap(in))
	tracker := tcpretransmit.New()

	// Tee goroutine: observe + forward. Bounded buffer; if the
	// downstream consumer falls behind the select-default drops the
	// event to preserve the data path's latency, matching the
	// tcpTracer's own ring-buffer drop semantics.
	go func() {
		defer close(out)
		for ev := range in {
			tracker.Observe(ev.PID, ev.Timestamp)
			select {
			case out <- ev:
			case <-ctx.Done():
				return
			default:
				// Downstream is full; drop the forward but keep
				// observing so the per-PID rate stays accurate.
			}
		}
	}()

	// Sweep goroutine: 1s tick, emit at most one Storm per PID per
	// episode. Logs at debug on dropped emissions so a misbehaving
	// orchestrator (consumer down) is visible in the agent log
	// without dominating it.
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case t := <-ticker.C:
				for _, storm := range tracker.Sweep(t) {
					if err := srv.SendTcpRetransmitStorm(remediate.TcpRetransmitStorm{
						PID:         storm.PID,
						RatePerSec:  storm.RatePerSec,
						SustainedMs: storm.SustainedMs,
					}, nodeID, clusterID); err != nil {
						slog.Default().Debug("remediate: tcp_retransmit_storm drop",
							"pid", storm.PID, "err", err)
					}
				}
			}
		}
	}()

	return out
}
