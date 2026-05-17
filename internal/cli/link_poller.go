package cli

import (
	"context"
	"log/slog"
	"time"

	"github.com/ingero-io/ingero/internal/nvml"
)

// linkSustainPolls and linkPCIePolls are the per-tracker thresholds
// applied at construction time. The NVLink sustain (2 polls, ~10s at
// the default 5s interval) matches the thermal-throttle floor: long
// enough to drop a single transient retry, short enough to page on
// real degradation. The PCIe sustain (3 polls, ~15s) is one tick
// looser because driver-init and power-state transitions can
// transient-downtrain a lane without operator action being warranted.
const (
	linkNVLinkSustainPolls = 2
	linkPCIeSustainPolls   = 3
)

// startLinkPoller spawns the goroutine that polls NVLink error
// counters and PCIe link state at `interval`. Returns immediately;
// runs until ctx is cancelled. Both runners may be nil; if both are
// nil (no nvidia-smi at all) the poller never starts. If only one is
// non-nil, the corresponding tracker still runs.
//
// `faultSink` is required: a nil sink turns this into a no-op (same
// shape as the throttle poller's --no-remediate mode). Production
// wiring constructs the sink as a closure over
// `remediateSrv.SendHardwareFault(fault, nodeID, clusterID)`.
func startLinkPoller(
	ctx context.Context,
	interval time.Duration,
	nvlinkRun nvml.Runner,
	pcieRun nvml.Runner,
	faultSink HardwareFaultSink,
	log *slog.Logger,
) {
	if faultSink == nil {
		return
	}
	if nvlinkRun == nil && pcieRun == nil {
		if log != nil {
			log.Debug("link poller: nvidia-smi not on PATH, NVLink+PCIe probes disabled")
		}
		return
	}
	if interval <= 0 {
		interval = 5 * time.Second
	}
	if log == nil {
		log = slog.Default()
	}

	nvlinkTracker := nvml.NewNVLinkErrorTracker(linkNVLinkSustainPolls)
	pcieTracker := nvml.NewPCIeDowntrainTracker(linkPCIeSustainPolls)

	go func() {
		// Fire one read immediately so the trackers seed their
		// per-GPU baseline without waiting an interval. Mirrors the
		// throttle poller's first-tick behavior.
		pollLinkOnce(ctx, nvlinkRun, pcieRun, nvlinkTracker, pcieTracker, faultSink, log)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				pollLinkOnce(ctx, nvlinkRun, pcieRun, nvlinkTracker, pcieTracker, faultSink, log)
			}
		}
	}()
}

// pollLinkOnce runs one round of both probes (within the budget the
// runners enforce via their own 2 s timeouts), feeds the results
// into both trackers, and forwards any emissions to faultSink.
//
// A failed nvidia-smi invocation is logged at debug; the trackers
// retain their state and re-arm on the next successful poll. Unlike
// the throttle poller, link state does not feed any OTel gauge here,
// so there is no shared-bitmap staleness concern.
func pollLinkOnce(
	ctx context.Context,
	nvlinkRun nvml.Runner,
	pcieRun nvml.Runner,
	nvlinkTracker *nvml.NVLinkErrorTracker,
	pcieTracker *nvml.PCIeDowntrainTracker,
	faultSink HardwareFaultSink,
	log *slog.Logger,
) {
	readings, err := nvml.GetLinkState(ctx, nvlinkRun, pcieRun)
	if err != nil {
		log.Debug("link poller: query failed", "err", err)
		return
	}
	for _, r := range readings {
		if nvlinkRun != nil {
			if fault := nvlinkTracker.Observe(r.Index, r.NVLinkErrors); fault.Kind != "" {
				log.Info("link poller: nvlink_errors sustained",
					"gpu_id", r.Index, "uuid", r.UUID, "total_errors", r.NVLinkErrors)
				faultSink(fault)
			}
		}
		if pcieRun != nil {
			if fault := pcieTracker.Observe(r); fault.Kind != "" {
				log.Info("link poller: pcie_downtrain sustained",
					"gpu_id", r.Index,
					"uuid", r.UUID,
					"gen_current", r.PCIeGenCurrent,
					"gen_max", r.PCIeGenMax,
					"width_current", r.PCIeWidthCurrent,
					"width_max", r.PCIeWidthMax,
				)
				faultSink(fault)
			}
		}
	}
}
