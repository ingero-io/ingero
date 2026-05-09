package cli

import (
	"sync"
	"sync/atomic"

	"github.com/ingero-io/ingero/internal/ebpf/memfrag"
)

// v0.16.3 wiring helpers between the closed-driver kprobe / NVML
// poller goroutines and the inference engine.
//
// Two side-channels are needed beyond the existing event hot path:
//
//  1. memfragInferenceHook lets the v0.15 W1 memfrag IOCTL kprobe
//     (which already feeds gpu.memfrag.ioctl_event_total counters)
//     also push into the inference engine's per-PID observable
//     bucket so the phase classifier can fire its decode-pressure
//     rule. The hook is set by configureInferenceEngine when
//     --inference is engaged; recordMemfragEvent calls it (when
//     non-nil) in addition to the existing per-cmd counter.
//
//  2. currentThrottleReasons is the latest aggregated NVML clock
//     throttle bitmap (max-OR across visible GPUs), updated by the
//     throttle poller's pollOnce on each tick. The inference engine
//     reads it via the throttleReader callback installed in
//     configureInferenceEngine; OutlierEvent gets the current
//     bitmap attached when an outlier fires, so a step that
//     coincided with HW_SLOWDOWN is visibly thermal in the UDS
//     envelope and the OTLP attributes.
//
// Both channels degrade silently when --inference is not engaged
// (hook stays nil; atomic stays 0).

var (
	memfragHookMu     sync.RWMutex
	memfragInferenceHook func(memfrag.Event)
)

// setMemfragInferenceHook installs the per-event callback. Called
// once, at engine construction; nil-safe.
func setMemfragInferenceHook(fn func(memfrag.Event)) {
	memfragHookMu.Lock()
	memfragInferenceHook = fn
	memfragHookMu.Unlock()
}

// callMemfragInferenceHook is invoked from recordMemfragEvent in
// memfrag_counters.go. The RLock is taken on the hot path, but
// contention is bounded - the writer is called once at engine setup.
func callMemfragInferenceHook(ev memfrag.Event) {
	memfragHookMu.RLock()
	fn := memfragInferenceHook
	memfragHookMu.RUnlock()
	if fn != nil {
		fn(ev)
	}
}

// currentThrottleReasons is the latest OR-folded NVML throttle
// bitmap across visible GPUs. Atomic so the throttle poller (writer)
// and the inference engine's sync hot path (reader) don't need a
// shared mutex. Zero when the throttle poller is disabled or when
// no GPU is reporting any throttle reason.
var currentThrottleReasons atomic.Uint64

// updateCurrentThrottleReasons is called from pollOnce in
// throttle_poller.go after each NVML query. The bitmap is the OR
// fold across every GPU in this poll cycle, so a multi-GPU host
// surfaces "any GPU is throttled" cleanly without per-GPU plumbing
// into the inference engine.
func updateCurrentThrottleReasons(orFolded uint64) {
	currentThrottleReasons.Store(orFolded)
}

// readCurrentThrottleReasons is the throttleReader callback installed
// on the engine.
func readCurrentThrottleReasons() uint64 {
	return currentThrottleReasons.Load()
}
