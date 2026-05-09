// Package ncclprobe - process discovery surface (item A, v0.14).
//
// Periodically scans /proc for processes that have a libnccl-bearing
// shared object loaded, and surfaces a discoverable METRIC stream so
// operators can see which workloads use NCCL on a given node.
//
// This sits ON TOP of FindLibNCCL (the single-PID probe that
// uprobe attach already uses); the scanner is the multi-PID periodic
// view useful for the Fleet `find_nccl_processes` MCP tool and for
// dashboards that want a "which nodes run NCCL today, version
// distribution" rollup.
//
// Per the v0.14 plan, the scanner emits two distinct metric data
// shapes via NCCLProcess records (drained by the trace command's
// snapshot callback into stats.NCCLProcessReadings):
//
//	gpu.nccl.process_loaded   gauge=1 per discovered PID, labels:
//	                          pid, comm, libnccl_path, libnccl_version
//	gpu.nccl.processes_total  gauge=count per node
//
// Version lookup: parsed from the libnccl SONAME / file basename
// (e.g. libnccl.so.2.21.5 → 2.21.5). When the basename doesn't carry
// a version suffix we walk the ELF dynamic section's DT_SONAME entry,
// which always carries the API SONAME on a vendor-installed libnccl.
// Falls back to "unknown" for non-conforming installs; no dlopen
// fallback is wired since the version is for diagnostics, not for
// gating uprobe attach.

package ncclprobe

import (
	"context"
	"debug/elf"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// NCCLProcess is one discovered process that has a libnccl-bearing
// shared object loaded. The Comm field is the kernel-recorded /proc
// stat comm (16-byte truncated). LibPath is the resolved absolute
// path on disk. LibVersion is the parsed NCCL version (e.g. "2.21.5")
// or the literal "unknown".
type NCCLProcess struct {
	PID        uint32
	Comm       string
	LibPath    string
	LibVersion string
}

// PIDLister returns the set of PIDs the scanner should probe each
// tick. nil/empty default is "every PID under /proc". Tests inject a
// fixed list; production wires either /proc enumeration or the
// agent's existing tracked-PID set.
type PIDLister func() ([]uint32, error)

// Sink consumes one batch per scan tick. Implementations are expected
// to be cheap (map writes); the scanner does not retry on a slow sink
// and does not buffer between ticks.
type Sink func(processes []NCCLProcess)

// Scanner periodically discovers NCCL-loaded processes and forwards
// the result to a Sink. Mirrors the throttle-poller pattern in
// internal/cli/throttle_poller.go (last-value-wins, single goroutine,
// configurable interval).
type Scanner struct {
	lister   PIDLister
	sink     Sink
	interval time.Duration

	// findLibForPID is the per-PID resolver - overridable for tests.
	// In production this is FindLibNCCL.
	findLibForPID func(pid int) string

	// versionFor is the per-path version resolver - overridable for
	// tests. In production this is libNCCLVersion.
	versionFor func(libPath string) string

	mu       sync.Mutex
	last     []NCCLProcess
	lastErr  error
	scanCnt  uint64
	errCount uint64
}

// NewScanner constructs a scanner with production-default helpers.
// Pass interval <= 0 to disable (the caller should not call Run).
func NewScanner(lister PIDLister, sink Sink, interval time.Duration) *Scanner {
	return &Scanner{
		lister:        lister,
		sink:          sink,
		interval:      interval,
		findLibForPID: FindLibNCCL,
		versionFor:    libNCCLVersion,
	}
}

// Run loops at the configured interval until ctx is cancelled or
// interval is non-positive. Fires one scan immediately so the first
// snapshot tick already has data, mirroring throttle-poller.
func (s *Scanner) Run(ctx context.Context) {
	if s.interval <= 0 {
		return
	}
	s.scanOnce(ctx)
	t := time.NewTicker(s.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.scanOnce(ctx)
		}
	}
}

// scanOnce runs one discovery pass and forwards the result to the sink.
func (s *Scanner) scanOnce(ctx context.Context) {
	if err := ctx.Err(); err != nil {
		return
	}
	pids, err := s.lister()
	if err != nil {
		s.mu.Lock()
		s.lastErr = err
		s.errCount++
		s.mu.Unlock()
		return
	}
	out := make([]NCCLProcess, 0, len(pids))
	for _, pid := range pids {
		path := s.findLibForPID(int(pid))
		if path == "" {
			continue
		}
		comm := readProcComm(int(pid))
		out = append(out, NCCLProcess{
			PID:        pid,
			Comm:       comm,
			LibPath:    path,
			LibVersion: s.versionFor(path),
		})
	}
	// Stable order so consumers can compare batches.
	sort.Slice(out, func(i, j int) bool { return out[i].PID < out[j].PID })
	s.mu.Lock()
	s.last = out
	s.lastErr = nil
	s.scanCnt++
	s.mu.Unlock()
	if s.sink != nil {
		s.sink(out)
	}
}

// LastResult returns the most recent batch and the most recent
// PIDLister error (nil on success). Useful for tests and diagnostics.
func (s *Scanner) LastResult() ([]NCCLProcess, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]NCCLProcess(nil), s.last...), s.lastErr
}

// Stats returns scan and error counts since construction.
func (s *Scanner) Stats() (scans, errs uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.scanCnt, s.errCount
}

// ProcPIDLister returns a PIDLister that enumerates every PID under
// /proc. Skips non-numeric entries silently; surfaces the readdir
// error if /proc itself is missing (would be catastrophic anyway).
func ProcPIDLister() PIDLister {
	return func() ([]uint32, error) {
		entries, err := os.ReadDir("/proc")
		if err != nil {
			return nil, fmt.Errorf("/proc: %w", err)
		}
		pids := make([]uint32, 0, len(entries))
		for _, e := range entries {
			n, err := strconv.ParseUint(e.Name(), 10, 32)
			if err != nil {
				continue
			}
			pids = append(pids, uint32(n))
		}
		return pids, nil
	}
}

// readProcComm reads /proc/<pid>/comm, trimming the trailing newline.
// Returns "" on error (process exited mid-scan, /proc not mounted).
func readProcComm(pid int) string {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/comm", pid))
	if err != nil {
		return ""
	}
	return strings.TrimRight(string(data), "\n")
}

// libnameVersionRe matches the trailing M.m.p version on libnccl
// basenames (e.g. libnccl.so.2.21.5 → 2.21.5). NCCL has used the
// 2.X.Y format consistently since 2.0; we accept up to four numeric
// components for forward compatibility.
var libnameVersionRe = regexp.MustCompile(`(?i)libnccl(?:_static)?\.so(?:\.([0-9]+(?:\.[0-9]+){0,3}))?$`)

// libNCCLVersion attempts to resolve the version of the NCCL shared
// object at path. Strategy:
//
//  1. Parse the SONAME from the ELF dynamic section. Vendor-installed
//     libnccl carries DT_SONAME = "libnccl.so.2.MAJOR.MINOR.PATCH" or
//     similar. This is the most authoritative source.
//  2. Fall back to the on-disk basename / readlink target. Distro
//     packages and pip-installed PyTorch both leave a real file at
//     "libnccl.so.2.21.5".
//  3. "unknown" otherwise. Diagnostics-only; not load-bearing on
//     uprobe attach.
//
// Errors during ELF parsing or readlink are non-fatal - the function
// returns "unknown" rather than propagating, since this is metadata
// for a metric label, not a gate on functionality.
func libNCCLVersion(path string) string {
	if v := versionFromSOName(path); v != "" {
		return v
	}
	if v := versionFromBasename(path); v != "" {
		return v
	}
	// readlink may resolve a libnccl.so symlink to libnccl.so.2.21.5.
	target, err := os.Readlink(path)
	if err == nil && target != "" {
		// readlink can return a relative path; resolve relative to
		// the original file's directory before re-parsing.
		if !filepath.IsAbs(target) {
			target = filepath.Join(filepath.Dir(path), target)
		}
		if v := versionFromBasename(target); v != "" {
			return v
		}
	}
	return "unknown"
}

// versionFromSOName parses DT_SONAME out of the ELF dynamic section
// and applies the same M.m.p extractor used for basenames.
func versionFromSOName(path string) string {
	f, err := elf.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	soNames, err := f.DynString(elf.DT_SONAME)
	if err != nil || len(soNames) == 0 {
		return ""
	}
	return versionFromBasename(soNames[0])
}

// versionFromBasename extracts an "M.m.p"-style suffix from a libnccl
// basename. Applied to both DT_SONAME and the on-disk file name.
func versionFromBasename(s string) string {
	base := filepath.Base(s)
	m := libnameVersionRe.FindStringSubmatch(base)
	if m == nil {
		return ""
	}
	return m[1]
}
