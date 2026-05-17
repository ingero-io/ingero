package cli

import (
	"context"
	"log/slog"
	"time"

	"github.com/ingero-io/ingero/internal/ncclhang"
	"github.com/ingero-io/ingero/internal/remediate"
)

// startNcclHangWatcher runs a 1s ticker that drains the ncclhang
// tracker and publishes nccl_hang signals over the remediate UDS.
// The tracker is fed by the existing ncclTracer event loop (one
// Observe per NCCL return event); per-PID inactivity detection
// gates emission on the configured idle threshold + suppression
// window (see ncclhang docs for defaults).
//
// Same shape as the Theme 3 cu_launch_idle_watcher: 1s tick, drop
// emissions logged at debug, ctx cancellation tears down cleanly.
func startNcclHangWatcher(
	ctx context.Context,
	tracker *ncclhang.Tracker,
	srv *remediate.Server,
	nodeID, clusterID string,
) {
	if tracker == nil || srv == nil {
		return
	}
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case t := <-ticker.C:
				for _, hang := range tracker.Sweep(t) {
					if err := srv.SendNcclHang(remediate.NcclHang{
						PID:        hang.PID,
						IdleMs:     hang.IdleMs,
						CommIDHash: hang.CommIDHash,
					}, nodeID, clusterID); err != nil {
						slog.Default().Debug("remediate: nccl_hang drop",
							"pid", hang.PID, "err", err)
					}
				}
			}
		}
	}()
}
