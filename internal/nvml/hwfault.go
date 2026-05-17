package nvml

import (
	"sync"
	"time"
)

// HardwareFaultKind classifies the upstream cause of a HardwareFault
// emission. Wire-stable strings; the ingero-ee orchestrator's
// HardwareFault SignalSource dispatch arm (Phase 13) matches on
// these values when routing to node_cordon + pod_drain.
type HardwareFaultKind string

const (
	// FaultKindThermalThrottle is sustained NVML thermal-throttle:
	// the Thermal bucket of DecodeReasons stays true across enough
	// consecutive polls to clear the configured sustain window.
	// Drives catalog rows I12 (inference) and T10 (training).
	FaultKindThermalThrottle HardwareFaultKind = "thermal_throttle"
	// FaultKindXid is an NVML Xid event (graphics engine exception,
	// memory page fault, uncontained ECC, GPU off the bus, etc.).
	// The Xid probe lives in a separate module; this constant is
	// declared here so the wire catalog stays single-sourced.
	FaultKindXid HardwareFaultKind = "xid"
	// FaultKindECC is an uncontained ECC error reported separately
	// from the Xid stream (e.g., from DCGM health monitoring on
	// hosts where the agent has DCGM access).
	FaultKindECC HardwareFaultKind = "ecc"
	// FaultKindNVLink is an NVLink negotiation retry or link-down
	// event. Reserved; emission landing in a future module.
	FaultKindNVLink HardwareFaultKind = "nvlink"
	// FaultKindPCIeDowntrain is a PCIe-link-downtrain event, detected
	// by baseline-drift on cudaMemcpy throughput. Reserved.
	FaultKindPCIeDowntrain HardwareFaultKind = "pcie_downtrain"
)

// HardwareFaultSeverity gates the ingero-ee orchestrator's dispatch.
// `critical` triggers the node_cordon + pod_drain playbook; `warning`
// is observed (log + metric) without an auto-action.
type HardwareFaultSeverity string

const (
	HardwareFaultWarning  HardwareFaultSeverity = "warning"
	HardwareFaultCritical HardwareFaultSeverity = "critical"
)

// HardwareFault is the event the agent emits over the remediate UDS.
// Wire shape is owned by `internal/remediate/server.go::hardwareFaultMessage`;
// this struct is the producer-facing input that callers fill in.
type HardwareFault struct {
	Kind     HardwareFaultKind
	Severity HardwareFaultSeverity
	// GPUID is the NVML device index that produced the fault. For
	// node-wide events (e.g., a host-level NVLink error not tied to a
	// specific device), zero is acceptable; the orchestrator does not
	// gate dispatch on the index for cordon/drain decisions.
	GPUID uint32
	// XidNumber is populated for Kind == FaultKindXid. Common codes:
	// 13 (graphics engine exception), 31 (GPU memory page fault),
	// 43 (reset channel), 48/63 (uncontained ECC), 79 (GPU off the bus).
	// Zero / unset for non-Xid kinds.
	XidNumber uint32
	// PID is optional. NVML faults are device-wide and not always
	// attributable to a single process at detection time; the orchestrator
	// falls back to VramTracker.top_by_utilization() when this is zero.
	PID uint32
	// ThrottleReasons is the OR-folded NVML throttle bitmask at the
	// emission boundary. Optional context the consumer can log; only
	// populated for Kind == FaultKindThermalThrottle.
	ThrottleReasons uint64
	// EventID is a UUIDv4 stamped by the producer when known. Empty is
	// allowed; the remediate Server can fill one in at Send time.
	EventID string
	// Timestamp is the producer-side observation time. Zero means
	// "use Send time"; the remediate Server stamps now() in that case.
	Timestamp time.Time
}

// ThermalSustainTracker turns the per-poll ThrottleBuckets sequence
// into HardwareFault emissions when thermal throttling becomes
// sustained. State machine per GPU UUID: counts consecutive observed
// polls where the Thermal bucket is true; emits a single
// HardwareFault when the count crosses the sustain threshold, and
// then suppresses re-emission until the bucket clears.
//
// Why a separate tracker (not a method on ThrottleEdgeDetector):
// the edge detector counts every rising edge for OTel surfacing.
// Hardware-fault emission must NOT fire on transient single-tick
// throttles (those are normal under power-cap workloads); only
// sustained throttling crosses the operator-action threshold. The
// two surfaces have different semantics, so they live in different
// types.
//
// Caveat: this tracker is the FOSS emitter half. The full Theme 1
// pipeline requires wiring it into `internal/cli/throttle_poller.go`
// so each pollOnce(...) feeds the tracker and forwards any emission
// to `remediate.Server.SendHardwareFault`. The wiring lands in a
// follow-up commit alongside the live integration test.
type ThermalSustainTracker struct {
	mu sync.Mutex
	// sustainPolls is the number of consecutive thermal-true polls
	// required before an emission fires. Two polls at the default
	// 5s interval = 10s sustained thermal, which clears the "single
	// short spike under a synthetic stress test" floor.
	sustainPolls int
	state        map[string]thermalState // keyed by GPU UUID
}

type thermalState struct {
	consecutive int  // current run of thermal-true polls
	emitted     bool // true once we've emitted; gates re-emission
}

// NewThermalSustainTracker returns a tracker that emits after
// sustainPolls consecutive Thermal=true observations on the same
// GPU UUID. Pass 2 for a 10s floor at the default 5s poll interval;
// 1 for "any single Thermal observation emits" (test-only, noisy).
func NewThermalSustainTracker(sustainPolls int) *ThermalSustainTracker {
	if sustainPolls < 1 {
		sustainPolls = 1
	}
	return &ThermalSustainTracker{
		sustainPolls: sustainPolls,
		state:        map[string]thermalState{},
	}
}

// Observe records one poll's (uuid, buckets, bitmask) and returns a
// HardwareFault to emit if this observation crossed the sustain
// threshold. Returns the zero value (Kind == "") when no emission
// should fire — caller checks `fault.Kind != ""`.
//
// State transitions:
//   - Thermal=false -> reset run to 0, clear emitted flag
//   - Thermal=true, consecutive < sustainPolls -> increment, no emit
//   - Thermal=true, consecutive == sustainPolls, not emitted -> EMIT
//   - Thermal=true, already emitted -> no-op (suppressed)
//
// Severity is always Critical for thermal emissions: a sustained
// thermal slowdown on a training or serving GPU is operator-action
// territory by definition. Operators who want a softer signal can
// drop the sustainPolls floor and have the orchestrator's per-action
// settling time absorb the noise.
func (t *ThermalSustainTracker) Observe(
	uuid string,
	buckets ThrottleBuckets,
	bitmask uint64,
	gpuID uint32,
) HardwareFault {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := t.state[uuid]
	if !buckets.Thermal {
		s.consecutive = 0
		s.emitted = false
		t.state[uuid] = s
		return HardwareFault{}
	}
	s.consecutive++
	defer func() { t.state[uuid] = s }()
	if s.emitted || s.consecutive < t.sustainPolls {
		return HardwareFault{}
	}
	s.emitted = true
	return HardwareFault{
		Kind:            FaultKindThermalThrottle,
		Severity:        HardwareFaultCritical,
		GPUID:           gpuID,
		ThrottleReasons: bitmask,
		Timestamp:       time.Now().UTC(),
	}
}

// Reset zeroes all per-UUID state. Test-only.
func (t *ThermalSustainTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.state = map[string]thermalState{}
}
