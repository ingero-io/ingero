package cli

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ingero-io/ingero/internal/nvml"
	"github.com/ingero-io/ingero/internal/stats"
)

// throttleBuf holds the most recent decoded throttle reading per GPU UUID.
// Producer: the poller goroutine in startThrottlePoller. Consumer: the
// snapshot callback (drainThrottleBuf), invoked at every snapshot tick.
//
// Last-value-wins semantics: an OTel gauge is a snapshot of "current state"
// rather than a time series of distinct events, so drains overwrite rather
// than append. This bounds memory at one entry per GPU even if the snapshot
// consumer stalls.
var (
	throttleBufMu sync.Mutex
	throttleBuf   map[string]stats.ThrottleReading
)

// throttleNotSupportedLogged tracks UUIDs we have already logged as
// "[Not Supported]" so the poller does not spam the log every tick on
// consumer-GPU rigs.
var (
	throttleLogMu             sync.Mutex
	throttleNotSupportedLogged map[string]bool
)

// addThrottleReading stores the latest reading for a GPU. Drops on contention?
// No: lock-protected and bounded by GPU count.
func addThrottleReading(r stats.ThrottleReading) {
	throttleBufMu.Lock()
	defer throttleBufMu.Unlock()
	if throttleBuf == nil {
		throttleBuf = map[string]stats.ThrottleReading{}
	}
	throttleBuf[r.UUID] = r
}

// drainThrottleBuf returns the current set of readings (one per GPU) and
// clears the buffer. Called from onSnapshot.
func drainThrottleBuf() []stats.ThrottleReading {
	throttleBufMu.Lock()
	defer throttleBufMu.Unlock()
	if len(throttleBuf) == 0 {
		return nil
	}
	out := make([]stats.ThrottleReading, 0, len(throttleBuf))
	for _, r := range throttleBuf {
		out = append(out, r)
	}
	throttleBuf = nil
	return out
}

// resetThrottleState clears all module-level state. Test-only helper.
func resetThrottleState() {
	throttleBufMu.Lock()
	throttleBuf = nil
	throttleBufMu.Unlock()
	throttleLogMu.Lock()
	throttleNotSupportedLogged = nil
	throttleLogMu.Unlock()
	throttleConsecFailures.Store(0)
	updateCurrentThrottleReasons(0)
	throttleSustainTracker.Reset()
}

// startThrottlePoller spawns the goroutine that polls NVML throttle reasons
// at `interval`. Returns immediately; runs until ctx is cancelled. If the
// runner is nil (no nvidia-smi on PATH), this returns without spawning.
//
// The interval is the bursting floor: a throttle event shorter than the
// poll interval may be missed. This caveat is documented in the CHANGELOG
// so dashboard authors can choose an interval that matches their workload.
//
// `faultSink` is the optional hook that turns sustained thermal throttling
// into a HardwareFault wire emission. Pass nil to disable (tests, agents
// running without `--remediate`); pass a closure over `remediateSrv.SendHardwareFault`
// to enable.
func startThrottlePoller(ctx context.Context, interval time.Duration, run nvml.Runner, log *slog.Logger, faultSink ThermalFaultSink) {
	if run == nil {
		if log != nil {
			log.Debug("throttle poller: nvidia-smi not on PATH, NVML metrics disabled")
		}
		return
	}
	if interval <= 0 {
		interval = 5 * time.Second
	}
	if log == nil {
		log = slog.Default()
	}
	go func() {
		// Fire one read immediately so the first snapshot already has data.
		pollOnce(ctx, run, log, faultSink)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				pollOnce(ctx, run, log, faultSink)
			}
		}
	}()
}

// ThermalFaultSink receives one HardwareFault per emission from the
// sustain-tracker fed inside pollOnce. The agent's main wiring passes
// `remediateSrv.SendHardwareFault(fault, nodeID, clusterID)` as the
// closure here; a nil sink disables emission (tests, no-remediate runs).
type ThermalFaultSink func(fault nvml.HardwareFault)

// throttleSustainTracker turns the per-poll (uuid, buckets, bitmask)
// sequence into a single HardwareFault when thermal throttling stays
// asserted long enough to be operator-action territory. Sibling to
// throttleEdgeDetector: the edge detector counts every rising edge for
// OTel surfacing, the sustain tracker only fires once per sustained
// run. Both share the same poll input; neither's state affects the
// other.
//
// `sustainPolls=2` at the default 5s poll interval = ~10s thermal
// floor. That's intentionally a low floor: a single short spike under
// a synthetic stress test does NOT cross it (consecutive=1), but a
// real sustained thermal slowdown does. Operators who want a softer
// signal can lower the floor; the orchestrator's per-action settling
// time absorbs noise on top.
var throttleSustainTracker = nvml.NewThermalSustainTracker(2)

// throttleConsecFailures counts top-level nvml failures since the last
// successful query. After throttleStaleThreshold consecutive failures the
// poller zeroes the shared bitmap so the inference outlier path cannot
// attach a stale "throttle was active" context to a fresh outlier.
var throttleConsecFailures atomic.Uint32

// throttleStaleThreshold is the consecutive-failure count after which the
// shared bitmap is forced to zero. Three ticks (~15s at the 5s default poll
// interval) covers transient nvidia-smi hiccups (driver restart, GPU reset)
// without flapping the bitmap on every transient.
const throttleStaleThreshold = 3

// pollOnce runs one nvidia-smi query and pushes the decoded readings into
// throttleBuf. Errors are logged at debug; "[Not Supported]" rows are logged
// once per UUID at info to surface the consumer-GPU degraded mode without
// flooding the log.
//
// When `faultSink` is non-nil, each reading also feeds the sustain
// tracker; an emission from the tracker is forwarded immediately so the
// orchestrator sees the hardware_fault wire message within one poll of
// the sustain threshold being crossed.
func pollOnce(ctx context.Context, run nvml.Runner, log *slog.Logger, faultSink ThermalFaultSink) {
	readings, err := nvml.GetCurrentClocksThrottleReasons(ctx, run)
	if err != nil {
		log.Debug("throttle poller: query failed", "err", err)
		if throttleConsecFailures.Add(1) == throttleStaleThreshold {
			updateCurrentThrottleReasons(0)
			log.Warn("throttle poller: clearing stale bitmap after consecutive failures",
				"failures", throttleStaleThreshold, "err", err)
		}
		return
	}
	throttleConsecFailures.Store(0)
	var orFolded uint64
	// gpuIndex tracks the NVML enumeration position across the slice.
	// nvidia-smi reports devices in NVML index order, so the position
	// here is the same `uint32` value the orchestrator uses for gpu_id
	// downstream. Devices that errored (e.g. [Not Supported]) still
	// advance the index so the mapping stays stable across polls; if
	// we skipped them, an [N/A] device on GPU 0 would shift GPU 1 to
	// be reported as index 0.
	var gpuIndex uint32
	for _, r := range readings {
		if r.Err != nil {
			if errors.Is(r.Err, nvml.ErrNotSupported) {
				logThrottleNotSupportedOnce(r.UUID, log)
			} else {
				log.Debug("throttle poller: per-device error", "uuid", r.UUID, "err", r.Err)
			}
			gpuIndex++
			continue
		}
		addThrottleReading(stats.ThrottleReading{
			UUID:          r.UUID,
			Bitmask:       r.Bitmask,
			PowerActive:   r.Buckets.Power,
			ThermalActive: r.Buckets.Thermal,
			SWActive:      r.Buckets.SW,
			HWActive:      r.Buckets.HW,
		})
		// v0.15 item L: feed the edge detector. Each rising edge per
		// (uuid, bucket) is counted once into
		// gpu.throttle.{power,thermal,sw,hw}.event_total. Sub-poll
		// bursts are still missed by design (same floor as the gauge).
		throttleEdgeDetector.Observe(r.UUID, r.Buckets)
		// Theme 1 second half: feed the sustain tracker. Returns the
		// zero HardwareFault below the sustain threshold or while
		// suppressed; only emits the single transition event when
		// sustained throttling becomes operator-action territory.
		if faultSink != nil {
			if fault := throttleSustainTracker.Observe(r.UUID, r.Buckets, r.Bitmask, gpuIndex); fault.Kind != "" {
				log.Info("throttle poller: thermal_throttle sustained",
					"uuid", r.UUID, "gpu_id", gpuIndex, "throttle_reasons", r.Bitmask)
				faultSink(fault)
			}
		}
		orFolded |= r.Bitmask
		gpuIndex++
	}
	// v0.16.3: publish the OR-fold so the inference engine's
	// throttleReader can read the latest bitmap when an outlier
	// fires. Always store, even when zero - that's how we transition
	// the inference outlier context from "throttle was active" to
	// "throttle has cleared" without the engine needing TTL logic.
	updateCurrentThrottleReasons(orFolded)
}

// throttleEdgeDetector is the process-wide edge detector fed from
// pollOnce. v0.15 item L. Snapshot()'d on each Prometheus / OTLP
// push so consumers see cumulative event counters.
var throttleEdgeDetector = nvml.NewThrottleEdgeDetector()

func logThrottleNotSupportedOnce(uuid string, log *slog.Logger) {
	throttleLogMu.Lock()
	defer throttleLogMu.Unlock()
	if throttleNotSupportedLogged == nil {
		throttleNotSupportedLogged = map[string]bool{}
	}
	if throttleNotSupportedLogged[uuid] {
		return
	}
	throttleNotSupportedLogged[uuid] = true
	log.Info("throttle poller: clocks_throttle_reasons not supported on this device, skipping",
		"uuid", uuid)
}
