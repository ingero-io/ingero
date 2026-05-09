package cli

import (
	"context"
	"errors"
	"log/slog"
	"sync"
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
}

// startThrottlePoller spawns the goroutine that polls NVML throttle reasons
// at `interval`. Returns immediately; runs until ctx is cancelled. If the
// runner is nil (no nvidia-smi on PATH), this returns without spawning.
//
// The interval is the bursting floor: a throttle event shorter than the
// poll interval may be missed. This caveat is documented in the CHANGELOG
// so dashboard authors can choose an interval that matches their workload.
func startThrottlePoller(ctx context.Context, interval time.Duration, run nvml.Runner, log *slog.Logger) {
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
		pollOnce(ctx, run, log)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				pollOnce(ctx, run, log)
			}
		}
	}()
}

// pollOnce runs one nvidia-smi query and pushes the decoded readings into
// throttleBuf. Errors are logged at debug; "[Not Supported]" rows are logged
// once per UUID at info to surface the consumer-GPU degraded mode without
// flooding the log.
func pollOnce(ctx context.Context, run nvml.Runner, log *slog.Logger) {
	readings, err := nvml.GetCurrentClocksThrottleReasons(ctx, run)
	if err != nil {
		log.Debug("throttle poller: query failed", "err", err)
		return
	}
	var orFolded uint64
	for _, r := range readings {
		if r.Err != nil {
			if errors.Is(r.Err, nvml.ErrNotSupported) {
				logThrottleNotSupportedOnce(r.UUID, log)
			} else {
				log.Debug("throttle poller: per-device error", "uuid", r.UUID, "err", r.Err)
			}
			continue
		}
		addThrottleReading(stats.ThrottleReading{
			UUID:           r.UUID,
			Bitmask:        r.Bitmask,
			PowerActive:    r.Buckets.Power,
			ThermalActive:  r.Buckets.Thermal,
			SWActive:       r.Buckets.SW,
			HWActive:       r.Buckets.HW,
		})
		// v0.15 item L: feed the edge detector. Each rising edge per
		// (uuid, bucket) is counted once into
		// gpu.throttle.{power,thermal,sw,hw}.event_total. Sub-poll
		// bursts are still missed by design (same floor as the gauge).
		throttleEdgeDetector.Observe(r.UUID, r.Buckets)
		orFolded |= r.Bitmask
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
