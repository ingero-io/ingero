package cli

import (
	"context"
	"log/slog"
	"time"

	"github.com/ingero-io/ingero/internal/cuidle"
	"github.com/ingero-io/ingero/internal/remediate"
	"github.com/ingero-io/ingero/pkg/events"
)

// startCuLaunchIdleWatcher tees the driver tracer's event channel
// through the cuidle sustain tracker. Driver events whose Op is
// cuLaunchKernel update the per-PID last-launch timestamp; a
// background ticker sweeps every second and publishes
// InferenceProcessHang signals over the remediate UDS for PIDs
// whose idle age has crossed the threshold while still alive.
//
// Non-launch driver events (cuMemcpy, cuMemAlloc, etc.) pass through
// untouched. Returns a derived channel that downstream consumers
// (mergeAllEventChannels, the unified event pipeline) read in place
// of the raw driver channel.
//
// The watcher mirrors startTcpRetransmitWatcher's shape:
//   - bounded forward buffer with a select-default drop so a slow
//     downstream consumer cannot block driver-event observation
//   - sweep goroutine on a 1s ticker, both teardown via ctx.Done()
//   - close out when the upstream channel closes
func startCuLaunchIdleWatcher(
	ctx context.Context,
	in <-chan events.Event,
	srv *remediate.Server,
	nodeID, clusterID string,
) <-chan events.Event {
	out := make(chan events.Event, cap(in))
	tracker := cuidle.New()

	go func() {
		defer close(out)
		for ev := range in {
			if ev.Source == events.SourceDriver &&
				events.DriverOp(ev.Op) == events.DriverLaunchKernel {
				tracker.Observe(ev.PID, ev.GPUID, ev.Timestamp)
			}
			select {
			case out <- ev:
			case <-ctx.Done():
				return
			default:
				// Downstream is full; drop the forward but keep
				// observing so the per-PID idle timer stays accurate.
			}
		}
	}()

	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case t := <-ticker.C:
				for _, hang := range tracker.Sweep(t) {
					if err := srv.SendInferenceProcessHang(remediate.InferenceProcessHang{
						PID:    hang.PID,
						GPUID:  hang.GPUID,
						IdleMs: hang.IdleMs,
					}, nodeID, clusterID); err != nil {
						slog.Default().Debug("remediate: inference_process_hang drop",
							"pid", hang.PID, "err", err)
					}
				}
			}
		}
	}()

	return out
}
