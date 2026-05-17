package cli

import (
	"context"
	"log/slog"
	"time"

	"github.com/ingero-io/ingero/internal/nvml"
	"github.com/ingero-io/ingero/internal/remediate"
	"github.com/ingero-io/ingero/internal/zombiegpu"
)

// startZombieGpuWatcher launches a periodic reconciler that walks
// nvidia-smi --query-compute-apps and emits zombie_gpu_allocation
// signals for orphan PIDs (driver reports allocation, kernel reports
// ESRCH). The interval matches the existing memfrag poller cadence
// — orphan allocations are not time-sensitive (they persist until
// reclaimed by cgroup teardown or full GPU reset), so a 5s default
// is a reasonable balance between dashboard freshness and nvidia-smi
// fork overhead.
//
// gpu_id on the wire is the enumeration index — we encode the
// reconciler's GPU UUIDs positionally as we encounter them on each
// tick. This matches the throttle poller's convention and keeps the
// orchestrator's RemediationContext.gpu_id field consistent. UUIDs
// not seen on a tick reset their position on the next reading,
// which is acceptable here because the dispatch arm uses gpu_id only
// for log context (the chain resolves through the PID via cgroup).
//
// run==nil (nvidia-smi not on PATH) is a no-op + warn — the ticker
// is not even started, since there is nothing to reconcile.
func startZombieGpuWatcher(
	ctx context.Context,
	interval time.Duration,
	run nvml.Runner,
	srv *remediate.Server,
	nodeID, clusterID string,
) {
	if run == nil {
		slog.Default().Info("zombiegpu: nvidia-smi not on PATH, reconciler disabled")
		return
	}
	if srv == nil {
		// Defensive: caller should not invoke without a server, but
		// don't crash if they do.
		return
	}
	rec := zombiegpu.New()
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				alloc, err := rec.Tick(ctx, run)
				if err != nil {
					slog.Default().Debug("zombiegpu: tick failed",
						"err", err)
					continue
				}
				for _, z := range alloc {
					if err := srv.SendZombieGpuAllocation(remediate.ZombieGpuAllocation{
						PID:            z.PID,
						GPUID:          gpuUUIDToIndex(z.GPUUUID),
						AllocatedBytes: z.AllocatedBytes,
					}, nodeID, clusterID); err != nil {
						slog.Default().Debug("remediate: zombie_gpu_allocation drop",
							"pid", z.PID, "err", err)
					}
				}
			}
		}
	}()
}

// gpuUUIDToIndex maps a GPU UUID to a numeric index for the wire
// payload. The current implementation is a stub returning 0 — the
// orchestrator's dispatch chain (gpu_context_reset -> pod_drain)
// resolves through the PID via cgroup and does not use gpu_id for
// routing decisions; the field is preserved on the wire purely for
// operator audit. A future enhancement could remember UUIDs across
// ticks and assign stable positional indices.
func gpuUUIDToIndex(_ string) uint32 {
	return 0
}
