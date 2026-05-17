package cli

import (
	"context"
	"log/slog"
	"time"

	"github.com/ingero-io/ingero/internal/rankdivergence"
	"github.com/ingero-io/ingero/internal/remediate"
)

// startRankDivergenceWatcher runs a 5s ticker that calls the
// rankdivergence tracker's Compute() pass and publishes
// rank_divergence signals over the remediate UDS. The tracker is
// fed by the existing ncclTracer event loop (one Observe per NCCL
// return event with duration > 0); per-comm MAD analysis flags
// outlier ranks that stay divergent across multiple consecutive
// ticks before emission.
//
// The 5s interval is the default Compute window; shorter ticks
// would produce noisier MAD estimates against the same sample
// budget. ctx cancellation tears down cleanly.
func startRankDivergenceWatcher(
	ctx context.Context,
	tracker *rankdivergence.Tracker,
	srv *remediate.Server,
	nodeID, clusterID string,
) {
	if tracker == nil || srv == nil {
		return
	}
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case t := <-ticker.C:
				for _, div := range tracker.Compute(t) {
					if err := srv.SendRankDivergence(remediate.RankDivergence{
						PID:         div.PID,
						Rank:        div.Rank,
						DriftSigma:  div.DriftSigma,
						SustainedMs: div.SustainedMs,
					}, nodeID, clusterID); err != nil {
						slog.Default().Debug("remediate: rank_divergence drop",
							"pid", div.PID, "rank", div.Rank, "err", err)
					}
				}
			}
		}
	}()
}
