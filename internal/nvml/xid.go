package nvml

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// XidEvent is one parsed NVRM Xid line from /dev/kmsg or dmesg.
//
// Wire-stable: the upstream nvidia driver has emitted Xid lines in
// the `NVRM: Xid (PCI:<bbbb:bb:dd>): <num>, ...` shape since at
// least driver 340 (2014). Older "NVRM: Xid <num>" forms without the
// PCI prefix were dropped before Maxwell and we do not parse them.
//
// PciBusID is the raw substring captured between the `PCI:` token
// and the closing `)` (e.g. `0000:1a:00`). NVML enumeration index
// resolution is a separate concern; see PciIndexResolver.
type XidEvent struct {
	XidNumber uint32
	PciBusID  string
	// PID is populated when the driver line carries `pid=<n>`. Some
	// Xids attribute to "<unknown>" or omit the field; PID == 0 means
	// "no offender pid in the kmsg line" and the orchestrator falls
	// back to VramTracker the same way thermal_throttle does.
	PID uint32
}

// criticalXids enumerates the Xid codes the operator playbook treats
// as hardware-action territory: cordon the node, drain its pods, page
// SRE. The list is the union of NVIDIA's "GPU Recovery and Sysadmin
// Guide" critical-list and the empirical set our incident reviews
// have flagged. Codes outside this set still emit a hardware_fault
// wire message, but at warning severity (counter-only on the EE side).
//
// Reference codes:
//
//	13 Graphics Engine Exception (kernel-level CUDA fault)
//	31 GPU memory page fault (uncontained MMU fault)
//	43 Reset channel verif error
//	45 Preemptive cleanup -- channel forcibly killed
//	48 Double Bit ECC error (uncontained)
//	56 Display Engine error
//	57 Receiver error
//	58 Bus error
//	62 Internal micro-controller halt
//	63 ECC page retirement recording event
//	64 ECC page retirement failure
//	65 ECC dynamic page blacklisting
//	68 NVDEC error
//	69 Graphics Engine class error
//	73 NVENC error
//	74 NVLINK error
//	79 GPU has fallen off the bus
var criticalXids = map[uint32]bool{
	13: true,
	31: true,
	43: true,
	45: true,
	48: true,
	56: true,
	57: true,
	58: true,
	62: true,
	63: true,
	64: true,
	65: true,
	68: true,
	69: true,
	73: true,
	74: true,
	79: true,
}

// IsCriticalXid returns true when xidNumber is in the operator
// playbook's hardware-action list. Used to derive the wire severity
// (critical vs warning) on the producer side.
func IsCriticalXid(xidNumber uint32) bool {
	return criticalXids[xidNumber]
}

// ParseXidLine recognises an NVRM Xid emission inside a raw line
// from /dev/kmsg or dmesg. Returns (event, true) when the line
// carries an Xid; (zero, false) for any non-Xid line so callers can
// pass an entire kmsg stream without pre-filtering.
//
// Accepts both kmsg-prefixed lines (`4,1234,5678,-;NVRM: Xid ...`)
// and the plain dmesg/console form (`[ 123.456] NVRM: Xid ...`).
// The prefix split is forgiving: we look for `NVRM: Xid` anywhere
// in the line. PID extraction tolerates either the literal
// `pid=<number>` form or `pid=<unknown>` / missing field.
func ParseXidLine(line string) (XidEvent, bool) {
	idx := strings.Index(line, "NVRM: Xid")
	if idx < 0 {
		return XidEvent{}, false
	}
	tail := line[idx+len("NVRM: Xid"):]

	pciStart := strings.Index(tail, "(PCI:")
	if pciStart < 0 {
		return XidEvent{}, false
	}
	pciClose := strings.Index(tail[pciStart:], ")")
	if pciClose < 0 {
		return XidEvent{}, false
	}
	pciBusID := strings.TrimSpace(tail[pciStart+len("(PCI:") : pciStart+pciClose])
	if pciBusID == "" {
		return XidEvent{}, false
	}

	after := tail[pciStart+pciClose+1:]
	colon := strings.Index(after, ":")
	if colon < 0 {
		return XidEvent{}, false
	}
	rest := strings.TrimSpace(after[colon+1:])
	commaIdx := strings.Index(rest, ",")
	var numStr string
	if commaIdx >= 0 {
		numStr = strings.TrimSpace(rest[:commaIdx])
		rest = rest[commaIdx+1:]
	} else {
		numStr = strings.TrimSpace(rest)
		rest = ""
	}
	xid, err := strconv.ParseUint(numStr, 10, 32)
	if err != nil {
		return XidEvent{}, false
	}

	ev := XidEvent{
		XidNumber: uint32(xid),
		PciBusID:  normalizePCIBusID(pciBusID),
	}
	if pid, ok := extractPID(rest); ok {
		ev.PID = pid
	}
	return ev, true
}

// normalizePCIBusID lower-cases and pads the bus_id so it matches
// the form returned by `nvidia-smi --query-gpu=pci.bus_id`
// (8-digit domain), regardless of which form the kmsg line used.
// Inputs we tolerate:
//
//	0000:1a:00       -> 00000000:1a:00.0
//	0000:1A:00.0     -> 00000000:1a:00.0
//	00000000:1A:00.0 -> 00000000:1a:00.0
//	1a:00            -> 00000000:1a:00.0
func normalizePCIBusID(raw string) string {
	s := strings.ToLower(strings.TrimSpace(raw))
	dotIdx := strings.LastIndex(s, ".")
	if dotIdx >= 0 {
		s = s[:dotIdx]
	}
	parts := strings.Split(s, ":")
	var domain, bus, device string
	switch len(parts) {
	case 2:
		domain, bus, device = "0", parts[0], parts[1]
	case 3:
		domain, bus, device = parts[0], parts[1], parts[2]
	default:
		return strings.ToLower(strings.TrimSpace(raw))
	}
	domainNum, err := strconv.ParseUint(domain, 16, 32)
	if err != nil {
		return strings.ToLower(strings.TrimSpace(raw))
	}
	return fmt.Sprintf("%08x:%s:%s.0", domainNum, bus, device)
}

// extractPID pulls a PID out of the trailing portion of an Xid line.
// Returns (0, false) when the line carries `pid=<unknown>`, no
// `pid=` token, or a non-numeric value.
func extractPID(s string) (uint32, bool) {
	pidIdx := strings.Index(s, "pid=")
	if pidIdx < 0 {
		return 0, false
	}
	tail := s[pidIdx+len("pid="):]
	end := len(tail)
	for i, r := range tail {
		if r == ',' || r == ' ' || r == '\'' || r == '"' {
			end = i
			break
		}
	}
	v, err := strconv.ParseUint(strings.TrimSpace(tail[:end]), 10, 32)
	if err != nil {
		return 0, false
	}
	return uint32(v), true
}

// XidToHardwareFault converts a parsed XidEvent into the producer-
// facing HardwareFault. gpuID is the NVML enumeration index resolved
// by PciIndexResolver; pass 0 when resolution fails so the
// orchestrator falls back to VramTracker.top_by_utilization() (same
// fallback the thermal_throttle path uses).
func XidToHardwareFault(ev XidEvent, gpuID uint32) HardwareFault {
	sev := HardwareFaultWarning
	if IsCriticalXid(ev.XidNumber) {
		sev = HardwareFaultCritical
	}
	return HardwareFault{
		Kind:      FaultKindXid,
		Severity:  sev,
		GPUID:     gpuID,
		XidNumber: ev.XidNumber,
		PID:       ev.PID,
		Timestamp: time.Now().UTC(),
	}
}

// NewPciIndexRunner returns a Runner that invokes
// `nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader`.
// Returns nil when nvidia-smi is not on PATH. The 2 s timeout
// matches the other nvidia-smi runners in this package.
func NewPciIndexRunner() Runner {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil || path == "" {
		return nil
	}
	return func(ctx context.Context) ([]byte, error) {
		ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		return exec.CommandContext(ctx, path,
			"--query-gpu=index,pci.bus_id",
			"--format=csv,noheader",
		).Output()
	}
}

// ResolvePciIndex runs the nvidia-smi index,pci.bus_id query once
// and returns a map of normalized pci_bus_id -> NVML enumeration
// index. Callers cache the map for the agent lifetime; PCIe topology
// is fixed at boot on non-hotplug hardware (which covers all NVIDIA
// datacenter GPUs in scope for this agent).
//
// Returns an empty map and a non-nil error when nvidia-smi is
// unavailable or returns garbage; callers degrade to "unknown index"
// (gpu_id == 0 on the wire) without aborting the Xid path.
func ResolvePciIndex(ctx context.Context, run Runner) (map[string]uint32, error) {
	if run == nil {
		return nil, fmt.Errorf("nvml: nvidia-smi not available")
	}
	out, err := run(ctx)
	if err != nil {
		return nil, fmt.Errorf("nvml: nvidia-smi pci.bus_id: %w", err)
	}
	return parsePciIndexCSV(out)
}

const maxPciIndexOutput = 16 * 1024

func parsePciIndexCSV(out []byte) (map[string]uint32, error) {
	if len(out) > maxPciIndexOutput {
		return nil, fmt.Errorf("nvml: pci.bus_id output exceeds %d bytes", maxPciIndexOutput)
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return nil, fmt.Errorf("nvml: pci.bus_id returned empty output")
	}
	m := map[string]uint32{}
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != 2 {
			return nil, fmt.Errorf("nvml: unexpected pci.bus_id line %q", line)
		}
		idxStr := strings.TrimSpace(parts[0])
		busID := normalizePCIBusID(parts[1])
		idx, err := strconv.ParseUint(idxStr, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("nvml: parse index %q: %w", idxStr, err)
		}
		m[busID] = uint32(idx)
	}
	if len(m) == 0 {
		return nil, fmt.Errorf("nvml: no pci.bus_id rows parsed")
	}
	return m, nil
}
