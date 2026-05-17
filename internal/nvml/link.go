package nvml

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// LinkReading is one sample of NVLink error counters and PCIe link
// state for one GPU. UUID and Index are populated together; the
// per-GPU NVLink and PCIe values are independent (a GPU with no
// NVLink connections reports NVLinkErrors == 0 and the PCIe fields
// are still populated).
type LinkReading struct {
	UUID  string
	Index uint32
	// NVLinkErrors is the sum across all links and all error
	// categories (Replay, Recovery, CRC Flit, CRC Data, ECC, Data
	// Tx/Rx) that `nvidia-smi nvlink -e` reports for this GPU. The
	// sum is monotonic during the agent's lifetime; the sustain
	// tracker keys on the per-poll DELTA, not the absolute value.
	NVLinkErrors uint64
	// PCIeGenCurrent / PCIeGenMax / PCIeWidthCurrent / PCIeWidthMax
	// come from nvidia-smi --query-gpu=pcie.link.gen.current,
	// pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max.
	// Downtrain is detected by GenCurrent < GenMax or
	// WidthCurrent < WidthMax sustained across polls.
	PCIeGenCurrent   uint32
	PCIeGenMax       uint32
	PCIeWidthCurrent uint32
	PCIeWidthMax     uint32
}

// IsPCIeDowntrained returns true when this reading shows the link
// running below its negotiated maximum on either dimension. The
// sustain tracker uses this as the per-poll predicate.
func (r LinkReading) IsPCIeDowntrained() bool {
	if r.PCIeGenMax > 0 && r.PCIeGenCurrent > 0 && r.PCIeGenCurrent < r.PCIeGenMax {
		return true
	}
	if r.PCIeWidthMax > 0 && r.PCIeWidthCurrent > 0 && r.PCIeWidthCurrent < r.PCIeWidthMax {
		return true
	}
	return false
}

// NewNVLinkErrorRunner returns a Runner that invokes
// `nvidia-smi nvlink -e`. Returns nil when nvidia-smi is not on
// PATH. 2 s timeout matches the other nvml runners.
func NewNVLinkErrorRunner() Runner {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil || path == "" {
		return nil
	}
	return func(ctx context.Context) ([]byte, error) {
		ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		return exec.CommandContext(ctx, path, "nvlink", "-e").Output()
	}
}

// NewPCIeLinkRunner returns a Runner that invokes
// `nvidia-smi --query-gpu=index,uuid,pcie.link.gen.current,
// pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max
// --format=csv,noheader,nounits`. nil when nvidia-smi is missing.
func NewPCIeLinkRunner() Runner {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil || path == "" {
		return nil
	}
	return func(ctx context.Context) ([]byte, error) {
		ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		return exec.CommandContext(ctx, path,
			"--query-gpu=index,uuid,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
			"--format=csv,noheader,nounits",
		).Output()
	}
}

// GetLinkState runs both subprocess calls (if available) and folds
// the outputs into one LinkReading per GPU keyed by NVML index.
// Either runner may be nil; the field set populated in the result
// reflects only the runners actually invoked. Returns an empty
// slice and a nil error when both runners are nil (the agent is
// running on a host without nvidia-smi).
func GetLinkState(ctx context.Context, nvlinkRun, pcieRun Runner) ([]LinkReading, error) {
	if nvlinkRun == nil && pcieRun == nil {
		return nil, nil
	}
	readings := map[uint32]*LinkReading{}

	if pcieRun != nil {
		out, err := pcieRun(ctx)
		if err != nil {
			return nil, fmt.Errorf("nvml: pcie query: %w", err)
		}
		pcie, err := parsePCIeCSV(out)
		if err != nil {
			return nil, err
		}
		for _, r := range pcie {
			cp := r
			readings[r.Index] = &cp
		}
	}

	if nvlinkRun != nil {
		out, err := nvlinkRun(ctx)
		if err != nil {
			return nil, fmt.Errorf("nvml: nvlink -e: %w", err)
		}
		errs := parseNVLinkErrors(out)
		for idx, total := range errs {
			if r, ok := readings[idx]; ok {
				r.NVLinkErrors = total
			} else {
				readings[idx] = &LinkReading{Index: idx, NVLinkErrors: total}
			}
		}
	}

	out := make([]LinkReading, 0, len(readings))
	for _, r := range readings {
		out = append(out, *r)
	}
	return out, nil
}

const maxLinkOutput = 64 * 1024

// parsePCIeCSV parses the output of the PCIe query Runner. Each line:
//
//	"<index>, <uuid>, <gen.current>, <gen.max>, <width.current>, <width.max>"
//
// Tolerates "N/A" / "[Not Supported]" tokens (consumer-GPU shows
// these for the gen.max field on some drivers) by zeroing the
// affected field. IsPCIeDowntrained() then ignores the GPU because
// the Max == 0 guard suppresses the comparison.
func parsePCIeCSV(out []byte) ([]LinkReading, error) {
	if len(out) > maxLinkOutput {
		return nil, fmt.Errorf("nvml: pcie output exceeds %d bytes", maxLinkOutput)
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return nil, fmt.Errorf("nvml: pcie query returned empty output")
	}
	var rows []LinkReading
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != 6 {
			return nil, fmt.Errorf("nvml: unexpected pcie line %q", line)
		}
		idx, err := strconv.ParseUint(strings.TrimSpace(parts[0]), 10, 32)
		if err != nil {
			return nil, fmt.Errorf("nvml: parse pcie index %q: %w", parts[0], err)
		}
		row := LinkReading{
			Index:            uint32(idx),
			UUID:             strings.TrimSpace(parts[1]),
			PCIeGenCurrent:   parseLinkUint(parts[2]),
			PCIeGenMax:       parseLinkUint(parts[3]),
			PCIeWidthCurrent: parseLinkUint(parts[4]),
			PCIeWidthMax:     parseLinkUint(parts[5]),
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("nvml: no pcie rows parsed")
	}
	return rows, nil
}

// parseLinkUint returns 0 for any "[Not Supported]" / "N/A" /
// unparseable cell. The sustain tracker treats 0 as "missing" via
// the IsPCIeDowntrained Max guard.
func parseLinkUint(s string) uint32 {
	t := strings.TrimSpace(s)
	v, err := strconv.ParseUint(t, 10, 32)
	if err != nil {
		return 0
	}
	return uint32(v)
}

// parseNVLinkErrors walks `nvidia-smi nvlink -e` output and folds
// all per-link error counters into a per-GPU total keyed by NVML
// index. Real-world output across H100, A100, V100 drivers shows
// nine possible counter names: Replay Errors, Recovery Errors,
// CRC Flit Errors, CRC Data Errors, ECC Errors, Data Tx Errors,
// Data Rx Errors, Tx Throughput, Rx Throughput. The first seven
// are operator-relevant; the throughput counters are dropped so a
// busy GPU does not look like a faulty one.
//
// Lines we accept:
//
//	GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-...)    -> sets current GPU
//	         Link 0: Replay Errors: 42              -> adds 42 to GPU 0
//
// Anything we do not recognise is skipped silently so future driver
// additions do not panic the parser.
func parseNVLinkErrors(out []byte) map[uint32]uint64 {
	totals := map[uint32]uint64{}
	if len(out) > maxLinkOutput {
		return totals
	}
	var cur uint32
	var haveCur bool
	for _, line := range strings.Split(string(out), "\n") {
		trim := strings.TrimSpace(line)
		if trim == "" {
			continue
		}
		if strings.HasPrefix(trim, "GPU ") {
			rest := trim[len("GPU "):]
			colon := strings.Index(rest, ":")
			if colon <= 0 {
				continue
			}
			idx, err := strconv.ParseUint(rest[:colon], 10, 32)
			if err != nil {
				continue
			}
			cur = uint32(idx)
			haveCur = true
			if _, seen := totals[cur]; !seen {
				totals[cur] = 0
			}
			continue
		}
		if !haveCur || !strings.HasPrefix(trim, "Link ") {
			continue
		}
		// "Link 0: Replay Errors: 42"
		colon1 := strings.Index(trim, ":")
		if colon1 < 0 {
			continue
		}
		rest := strings.TrimSpace(trim[colon1+1:])
		colon2 := strings.Index(rest, ":")
		if colon2 < 0 {
			continue
		}
		fieldName := strings.TrimSpace(rest[:colon2])
		fieldVal := strings.TrimSpace(rest[colon2+1:])
		if !isCountedNVLinkField(fieldName) {
			continue
		}
		v, err := strconv.ParseUint(fieldVal, 10, 64)
		if err != nil {
			continue
		}
		totals[cur] += v
	}
	return totals
}

func isCountedNVLinkField(name string) bool {
	switch name {
	case "Replay Errors",
		"Recovery Errors",
		"CRC Flit Errors",
		"CRC Data Errors",
		"ECC Errors",
		"Data Tx Errors",
		"Data Rx Errors":
		return true
	}
	return false
}

// NVLinkErrorTracker turns the per-poll per-GPU error-total sequence
// into a HardwareFault emission when the counter delta stays
// positive across sustainPolls consecutive observations. State
// machine per NVML index, mirroring ThermalSustainTracker:
//
//   - delta == 0 -> reset run, clear emitted flag, store new total
//   - delta > 0, consecutive < sustainPolls -> increment, no emit
//   - delta > 0, consecutive == sustainPolls, not emitted -> EMIT
//   - delta > 0, already emitted -> no-op (suppressed)
//
// Severity is always Critical: a sustained run of NVLink errors is
// operator-action territory by definition (the link will degrade
// before it dies; cordon + drain costs less than waiting for a
// hard fault).
type NVLinkErrorTracker struct {
	mu           sync.Mutex
	sustainPolls int
	state        map[uint32]linkSustainState
}

type linkSustainState struct {
	lastTotal   uint64
	consecutive int
	emitted     bool
	seen        bool
}

// NewNVLinkErrorTracker returns a tracker that emits after
// sustainPolls consecutive polls with a positive error-counter
// delta on the same NVML index. Pass 2 for a 10s floor at the
// default 5s poll interval; 1 for "any positive delta emits"
// (test-only, noisy).
func NewNVLinkErrorTracker(sustainPolls int) *NVLinkErrorTracker {
	if sustainPolls < 1 {
		sustainPolls = 1
	}
	return &NVLinkErrorTracker{
		sustainPolls: sustainPolls,
		state:        map[uint32]linkSustainState{},
	}
}

// Observe records one (index, totalErrors) sample and returns a
// HardwareFault to emit if this observation crossed the sustain
// threshold. Returns the zero value (Kind == "") when no emission
// should fire.
//
// The first observation per index seeds the baseline and never
// emits, regardless of the absolute value (the counter may carry a
// pre-agent error history that the operator has already triaged).
func (t *NVLinkErrorTracker) Observe(index uint32, total uint64) HardwareFault {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := t.state[index]
	defer func() { t.state[index] = s }()
	if !s.seen {
		s.seen = true
		s.lastTotal = total
		return HardwareFault{}
	}
	delta := total - s.lastTotal
	s.lastTotal = total
	if delta == 0 {
		s.consecutive = 0
		s.emitted = false
		return HardwareFault{}
	}
	s.consecutive++
	if s.emitted || s.consecutive < t.sustainPolls {
		return HardwareFault{}
	}
	s.emitted = true
	return HardwareFault{
		Kind:      FaultKindNVLink,
		Severity:  HardwareFaultCritical,
		GPUID:     index,
		Timestamp: time.Now().UTC(),
	}
}

// Reset zeroes per-index state. Test-only.
func (t *NVLinkErrorTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.state = map[uint32]linkSustainState{}
}

// PCIeDowntrainTracker emits a HardwareFault when a GPU's PCIe link
// runs below its negotiated max gen or width across sustainPolls
// consecutive polls. State machine mirrors ThermalSustainTracker
// and NVLinkErrorTracker.
//
// The 3-poll default (~15 s at 5 s interval) covers the transient
// lane-renegotiation window during driver init or power-state
// transitions, which is the false-positive class operators have
// historically called out.
type PCIeDowntrainTracker struct {
	mu           sync.Mutex
	sustainPolls int
	state        map[uint32]pcieState
}

type pcieState struct {
	consecutive int
	emitted     bool
}

// NewPCIeDowntrainTracker returns a tracker that emits after
// sustainPolls consecutive polls of IsPCIeDowntrained == true on
// the same NVML index.
func NewPCIeDowntrainTracker(sustainPolls int) *PCIeDowntrainTracker {
	if sustainPolls < 1 {
		sustainPolls = 1
	}
	return &PCIeDowntrainTracker{
		sustainPolls: sustainPolls,
		state:        map[uint32]pcieState{},
	}
}

// Observe records one LinkReading and returns a HardwareFault to
// emit if downtrain has been sustained long enough. Returns the
// zero value otherwise.
func (t *PCIeDowntrainTracker) Observe(r LinkReading) HardwareFault {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := t.state[r.Index]
	defer func() { t.state[r.Index] = s }()
	if !r.IsPCIeDowntrained() {
		s.consecutive = 0
		s.emitted = false
		return HardwareFault{}
	}
	s.consecutive++
	if s.emitted || s.consecutive < t.sustainPolls {
		return HardwareFault{}
	}
	s.emitted = true
	return HardwareFault{
		Kind:      FaultKindPCIeDowntrain,
		Severity:  HardwareFaultCritical,
		GPUID:     r.Index,
		Timestamp: time.Now().UTC(),
	}
}

// Reset zeroes per-index state. Test-only.
func (t *PCIeDowntrainTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.state = map[uint32]pcieState{}
}
