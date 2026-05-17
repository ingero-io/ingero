package cli

import (
	"bufio"
	"context"
	"errors"
	"io"
	"log/slog"
	"os"

	"github.com/ingero-io/ingero/internal/nvml"
)

// HardwareFaultSink is the receiver for any nvml.HardwareFault emission
// outside the thermal_throttle path. Same shape as ThermalFaultSink (kept
// distinct for documentation: thermal lives in the throttle poller; xid
// and link emissions live here). Production wiring constructs both from
// the same SendHardwareFault closure.
type HardwareFaultSink = ThermalFaultSink

// startXidReader spawns the /dev/kmsg follower goroutine. Opens kmsgPath
// (default "/dev/kmsg" when ""), seeks to end so historical noise is
// dropped, and pipes every line through nvml.ParseXidLine. Recognised
// Xid events are converted to HardwareFault via nvml.XidToHardwareFault
// and forwarded to faultSink; gpuID resolution uses pciResolver invoked
// once at startup.
//
// Returns immediately. The reader goroutine exits when ctx is cancelled
// (the watcher goroutine closes the file to unblock the read) or when
// the underlying reader returns a non-recoverable error.
//
// Degrades to a no-op when kmsgPath cannot be opened (typical on
// non-Linux hosts, in dev containers without /dev/kmsg, or when the
// agent lacks read permission). The degrade is logged at debug -- the
// throttle poller has the same "feature absent, agent still runs" shape.
//
// pciResolver may be nil (e.g. nvidia-smi missing). In that case every
// emission carries gpu_id = 0 and the orchestrator falls back to
// VramTracker.top_by_utilization() the same way the thermal path does
// when the PID is unknown.
func startXidReader(
	ctx context.Context,
	kmsgPath string,
	pciResolver nvml.Runner,
	faultSink HardwareFaultSink,
	log *slog.Logger,
) {
	if faultSink == nil {
		return
	}
	if log == nil {
		log = slog.Default()
	}
	if kmsgPath == "" {
		kmsgPath = "/dev/kmsg"
	}

	pciIndex, err := nvml.ResolvePciIndex(ctx, pciResolver)
	if err != nil {
		log.Debug("xid reader: pci.bus_id resolve failed; emissions will carry gpu_id=0",
			"err", err)
		pciIndex = nil
	}

	f, err := os.OpenFile(kmsgPath, os.O_RDONLY, 0)
	if err != nil {
		log.Debug("xid reader: open failed, disabled", "path", kmsgPath, "err", err)
		return
	}
	// Skip the historical kmsg buffer. SEEK_END on /dev/kmsg is
	// supported since kernel 3.5; if a host's kmsg device rejects
	// it we accept replay of the buffer once and the per-PID/Xid
	// dedup the EE side does on `event_id` prevents duplicate
	// action dispatch.
	if _, err := f.Seek(0, io.SeekEnd); err != nil {
		log.Debug("xid reader: seek end not supported, will replay current buffer once",
			"err", err)
	}

	// Close-on-cancel: blocking reads on /dev/kmsg only unblock when
	// the file is closed. The watcher goroutine does the close so
	// ctx cancellation cleanly exits the reader.
	go func() {
		<-ctx.Done()
		_ = f.Close()
	}()

	go runXidReader(ctx, f, pciIndex, faultSink, log)
}

// runXidReader is the testable inner loop. Reads lines from r until
// it sees io.EOF (test path), a closed-file error (production cancel
// path), or ctx.Done. Every Xid line is forwarded to faultSink. Lines
// that do not parse as Xid are ignored, not logged: kmsg is high-
// volume and most lines are unrelated to NVRM.
func runXidReader(
	ctx context.Context,
	r io.Reader,
	pciIndex map[string]uint32,
	faultSink HardwareFaultSink,
	log *slog.Logger,
) {
	br := bufio.NewReader(r)
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		line, err := br.ReadString('\n')
		if len(line) > 0 {
			processXidLine(line, pciIndex, faultSink, log)
		}
		if err != nil {
			// io.EOF is the expected exit for test inputs; closed-file
			// errors are the expected exit for production cancel. Both
			// are logged at debug to surface unexpected disconnects
			// without spamming the log.
			if !errors.Is(err, io.EOF) {
				log.Debug("xid reader: read loop exit", "err", err)
			}
			return
		}
	}
}

// processXidLine isolates the per-line work (parse, resolve, emit) so
// the read loop stays focused on I/O and cancellation. Resolves PCI
// bus_id to NVML index when the map carries an entry; falls back to
// gpu_id=0 otherwise (orchestrator handles that via VramTracker).
func processXidLine(
	line string,
	pciIndex map[string]uint32,
	faultSink HardwareFaultSink,
	log *slog.Logger,
) {
	ev, ok := nvml.ParseXidLine(line)
	if !ok {
		return
	}
	gpuID := uint32(0)
	if idx, found := pciIndex[ev.PciBusID]; found {
		gpuID = idx
	} else if pciIndex != nil {
		log.Debug("xid reader: pci.bus_id not in index, defaulting gpu_id=0",
			"pci_bus_id", ev.PciBusID, "xid", ev.XidNumber)
	}
	fault := nvml.XidToHardwareFault(ev, gpuID)
	log.Info("xid reader: hardware_fault emitted",
		"xid", ev.XidNumber,
		"gpu_id", gpuID,
		"pid", ev.PID,
		"severity", fault.Severity,
		"pci_bus_id", ev.PciBusID,
	)
	faultSink(fault)
}
