package ncclprobe

import (
	"bytes"
	"context"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"golang.org/x/sys/unix"
)

// Event is the parsed userspace shape of struct nccl_event from
// bpf/nccl_trace.bpf.c. Field layout mirrors the C struct exactly.
//
// Op codes match the NCCL_OP_* defines in bpf/common.bpf.h:
//
//	1 = COMM_INIT_RANK   2 = COMM_DESTROY    3 = ALL_REDUCE
//	4 = ALL_GATHER       5 = REDUCE_SCATTER  6 = BCAST
//	7 = SEND             8 = RECV
type Event struct {
	TimestampNs uint64
	PID         uint32
	TID         uint32
	Source      uint8
	Op          uint8
	// _pad / _pad2 are intentional ABI padding to align CgroupID to
	// 8 bytes, mirroring the C struct ingero_event_hdr in
	// bpf/common.bpf.h. binary.Read consumes them; do not remove.
	//lint:ignore U1000 ABI padding for C struct nccl_event hdr
	_pad uint16
	//lint:ignore U1000 ABI padding for C struct nccl_event hdr
	_pad2    uint32
	CgroupID uint64
	Comm         [16]byte
	DurationNs   uint64
	CommIDHash   uint64
	StreamHandle uint64
	CountBytes   uint64
	Rank         uint32
	NRanks       uint32
	Datatype     uint32
	ReduceOp     uint32
	ReturnCode   int32
	PeerRank     uint32 // v0.12.2: nonzero only for ncclSend/Recv
}

// EventSize is the on-the-wire size of one nccl_event record.
// Must match `struct nccl_event` total in bpf/common.bpf.h.
const EventSize = 104

// Tracer attaches uprobes/uretprobes to NCCL collective functions and
// streams parsed Events on its output channel.
type Tracer struct {
	libPath      string // initial path passed to New; "" allowed (lazy-attach via AttachAt)
	objs         ncclTraceObjects
	links        []link.Link
	reader       *ringbuf.Reader
	eventCh      chan Event
	dropped      atomic.Uint64
	parseErr     atomic.Uint64
	closed       atomic.Bool
	attachedSyms int // sum across all AttachAt calls

	// v0.15 F1: support runtime attachment to libnccl paths discovered
	// AFTER startup. Each call to AttachAt adds a set of uprobes
	// against one ELF; mu protects links + attachedPaths + attachedSyms
	// from concurrent calls (the discovery-scanner sink runs in its
	// own goroutine).
	mu            sync.Mutex
	attachedPaths map[string]int // libPath -> sym count attached for that path
	prepareDone   atomic.Bool
}

// New constructs a tracer for the given libnccl-bearing shared object.
// libPath is typically the result of FindLibNCCL but a caller can pass
// any ELF path that exports the NCCL public API.
func New(libPath string) *Tracer {
	return &Tracer{
		libPath: libPath,
		eventCh: make(chan Event, 4096),
	}
}

// Events returns the read-side channel.
func (t *Tracer) Events() <-chan Event { return t.eventCh }

// AttachedProbeCount returns the number of uprobe/uretprobe links the
// tracer has live. Each NCCL symbol that resolves contributes 2 probes
// (entry + return). v0.12.3 (Sys Arch ★1): callers (CLI banner,
// hardware validation) need the truth value rather than a hardcoded
// constant, because older NCCL builds may not export every symbol the
// agent looks for and silently attach fewer probes.
func (t *Tracer) AttachedProbeCount() int { return t.attachedSyms * 2 }

// Dropped returns the count of events dropped due to a full Go-side channel.
func (t *Tracer) Dropped() uint64 { return t.dropped.Load() }

// Attach loads the BPF program and wires uprobe / uretprobe pairs onto
// each NCCL symbol. Pass nil for system-wide tracing or a non-nil
// slice (even empty) to install the per-tenant PID filter; the filter
// goes in BEFORE the first uprobe attaches, eliminating the race where
// probes would briefly trace system-wide. Caller must Close() to detach.
//
// v0.12.2 (Arch audit ★3 attach race): pre-population is the only way
// to guarantee the first NCCL event seen by the kernel respects the
// filter; runtime SetTargetPID still works but cannot retroactively
// drop events fired during the gap between uprobe attach and map write.
//
// v0.12.3 (Sys Arch ★4): explicit slice parameter replaces the variadic
// signature so callers can't accidentally Attach() with no args and get
// system-wide tracing when they meant filtered.
func (t *Tracer) Attach(targetPIDs []uint32) error {
	if err := t.Prepare(targetPIDs); err != nil {
		return err
	}
	if t.libPath == "" {
		// v0.15 F1: zero-eager-libnccl is now allowed; AttachAt will
		// add uprobes once the discovery scanner finds a workload.
		return nil
	}
	return t.AttachAt(t.libPath)
}

// Prepare loads the eBPF spec, installs the PID filter (if any), and
// opens the ringbuf reader. It does NOT attach uprobes; AttachAt does.
// Splitting these out lets one Tracer serve multiple libnccl paths,
// some of which may not exist yet at startup. Idempotent.
//
// v0.15 F1: when no libnccl is found at startup, the agent should still
// stand up the BPF infrastructure so that AttachAt can wire uprobes
// against newly-discovered ELFs at runtime (PyTorch+pip workloads ship
// libnccl in their venv; the discovery scanner finds it after the
// workload boots).
func (t *Tracer) Prepare(targetPIDs []uint32) error {
	if !t.prepareDone.CompareAndSwap(false, true) {
		return nil
	}
	t.attachedPaths = map[string]int{}

	spec, err := loadNcclTrace()
	if err != nil {
		return fmt.Errorf("loading eBPF spec: %w", err)
	}
	if err := spec.LoadAndAssign(&t.objs, nil); err != nil {
		return fmt.Errorf("LoadAndAssign: %w", err)
	}
	if len(targetPIDs) > 0 {
		if err := t.installPIDFilter(targetPIDs); err != nil {
			t.objs.Close()
			return fmt.Errorf("install PID filter: %w", err)
		}
	}
	r, err := ringbuf.NewReader(t.objs.NcclEvents)
	if err != nil {
		t.objs.Close()
		return fmt.Errorf("ringbuf reader: %w", err)
	}
	t.reader = r
	return nil
}

// AttachAt opens the given libnccl-bearing ELF and wires uprobe +
// uretprobe pairs onto each NCCL symbol it exports. Safe to call
// repeatedly with different paths; idempotent per path. Returns nil
// when the path was already attached. Thread-safe (the discovery
// scanner sink runs in its own goroutine).
//
// v0.15 F1: this is the runtime-attach entry point used by the
// libnccl discovery scanner. The same call also covers the eager
// startup path (Tracer.Attach calls this with t.libPath when it is
// non-empty).
func (t *Tracer) AttachAt(libPath string) error {
	if !t.prepareDone.Load() {
		return fmt.Errorf("ncclprobe: AttachAt called before Prepare")
	}
	if libPath == "" {
		return fmt.Errorf("ncclprobe: AttachAt requires a non-empty libPath")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, already := t.attachedPaths[libPath]; already {
		return nil
	}

	exec, err := link.OpenExecutable(libPath)
	if err != nil {
		return fmt.Errorf("OpenExecutable %s: %w", libPath, err)
	}

	type probeDef struct {
		symbol  string
		entry   *ebpf.Program
		retEntr *ebpf.Program
	}
	probes := []probeDef{
		{"ncclCommInitRank", t.objs.UprobeNcclCommInitRank, t.objs.UretprobeNcclCommInitRank},
		{"ncclCommInitAll", t.objs.UprobeNcclCommInitAll, t.objs.UretprobeNcclCommInitAll},
		{"ncclCommDestroy", t.objs.UprobeNcclCommDestroy, t.objs.UretprobeNcclCommDestroy},
		{"ncclAllReduce", t.objs.UprobeNcclAllReduce, t.objs.UretprobeNcclAllReduce},
		{"ncclAllGather", t.objs.UprobeNcclAllGather, t.objs.UretprobeNcclAllGather},
		{"ncclReduceScatter", t.objs.UprobeNcclReduceScatter, t.objs.UretprobeNcclReduceScatter},
		{"ncclBcast", t.objs.UprobeNcclBcast, t.objs.UretprobeNcclBcast},
		{"ncclSend", t.objs.UprobeNcclSend, t.objs.UretprobeNcclSend},
		{"ncclRecv", t.objs.UprobeNcclRecv, t.objs.UretprobeNcclRecv},
	}

	attached := 0
	added := []link.Link{}
	for _, p := range probes {
		sym, err := resolveSymbol(libPath, p.symbol)
		if err != nil {
			log.Printf("ncclprobe: %s not found in %s: %v (skipping; NCCL build may not export this collective)", p.symbol, libPath, err)
			continue
		}
		l, err := exec.Uprobe(sym, p.entry, nil)
		if err != nil {
			log.Printf("ncclprobe: uprobe %s: %v", sym, err)
			continue
		}
		added = append(added, l)
		lr, err := exec.Uretprobe(sym, p.retEntr, nil)
		if err != nil {
			log.Printf("ncclprobe: uretprobe %s: %v", sym, err)
			continue
		}
		added = append(added, lr)
		attached++
	}
	if attached == 0 {
		// Roll back any half-attached links from this call.
		for _, l := range added {
			_ = l.Close()
		}
		return fmt.Errorf("no NCCL symbols resolved in %s", libPath)
	}
	t.links = append(t.links, added...)
	t.attachedSyms += attached
	t.attachedPaths[libPath] = attached
	return nil
}

// AttachedPaths returns a copy of the libnccl paths currently
// attached. Test + observability hook.
func (t *Tracer) AttachedPaths() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]string, 0, len(t.attachedPaths))
	for k := range t.attachedPaths {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Run reads events from the ringbuf until ctx is cancelled or the
// reader returns ringbuf.ErrClosed. Blocks; intended to run in its
// own goroutine.
func (t *Tracer) Run(ctx context.Context) error {
	defer close(t.eventCh)
	for {
		if err := ctx.Err(); err != nil {
			return nil
		}
		rec, err := t.reader.Read()
		if err != nil {
			if errors.Is(err, ringbuf.ErrClosed) {
				return nil
			}
			return fmt.Errorf("ringbuf read: %w", err)
		}
		if len(rec.RawSample) < EventSize {
			t.parseErr.Add(1)
			continue
		}
		var e Event
		if err := binary.Read(bytes.NewReader(rec.RawSample[:EventSize]), binary.LittleEndian, &e); err != nil {
			t.parseErr.Add(1)
			continue
		}
		select {
		case t.eventCh <- e:
		default:
			t.dropped.Add(1)
		}
	}
}

// Close detaches all probes and closes the ringbuf reader.
func (t *Tracer) Close() error {
	if !t.closed.CompareAndSwap(false, true) {
		return nil
	}
	t.detach()
	return nil
}

func (t *Tracer) detach() {
	if t.reader != nil {
		_ = t.reader.Close()
		t.reader = nil
	}
	for _, l := range t.links {
		_ = l.Close()
	}
	t.links = nil
	t.objs.Close()
}

// ----- symbol resolution -----

// resolveSymbol looks up `name` in the dynamic symbol table of `libPath`.
// On miss, scans for versioned variants ("name_v2_X_Y") and returns the
// most-recent versioned match. Implements Workstream A4 per the v0.12.0
// roadmap entry: NCCL 2.x symbol-name version drift.
func resolveSymbol(libPath, name string) (string, error) {
	f, err := elf.Open(libPath)
	if err != nil {
		return "", fmt.Errorf("elf.Open %s: %w", libPath, err)
	}
	defer f.Close()

	syms, err := f.DynamicSymbols()
	if err != nil {
		return "", fmt.Errorf("DynamicSymbols: %w", err)
	}
	// Fast path: exact match.
	for _, s := range syms {
		if s.Name == name {
			return s.Name, nil
		}
	}
	// Versioned fallback: name_v2_X_Y or similar suffix.
	prefix := name + "_v"
	var versioned []string
	for _, s := range syms {
		if strings.HasPrefix(s.Name, prefix) {
			versioned = append(versioned, s.Name)
		}
	}
	if len(versioned) == 0 {
		return "", fmt.Errorf("symbol %s not found", name)
	}
	// NCCL symbol convention: foo_vMAJOR_MINOR_PATCH. Sort by parsed
	// numeric tuple, NOT lexical, so foo_v2_19_3 outranks foo_v2_9_0.
	// Lexical sort would prefer "_v2_9_0" because '1' < '9'.
	sort.Slice(versioned, func(i, j int) bool {
		return compareVersionedSymbol(versioned[i], versioned[j]) < 0
	})
	return versioned[len(versioned)-1], nil
}

// compareVersionedSymbol returns -1/0/+1 like strings.Compare but ranks
// NCCL-style versioned suffixes (foo_vA_B_C) by parsed numeric tuple.
// Symbols that don't parse fall back to lexical compare so we still
// produce a total order.
func compareVersionedSymbol(a, b string) int {
	pa, oka := parseVersionTuple(a)
	pb, okb := parseVersionTuple(b)
	if !oka || !okb {
		switch {
		case a < b:
			return -1
		case a > b:
			return 1
		default:
			return 0
		}
	}
	for i := 0; i < len(pa) && i < len(pb); i++ {
		if pa[i] != pb[i] {
			if pa[i] < pb[i] {
				return -1
			}
			return 1
		}
	}
	switch {
	case len(pa) < len(pb):
		return -1
	case len(pa) > len(pb):
		return 1
	default:
		return 0
	}
}

func parseVersionTuple(sym string) ([]int, bool) {
	idx := strings.LastIndex(sym, "_v")
	if idx < 0 {
		return nil, false
	}
	tail := sym[idx+2:]
	parts := strings.Split(tail, "_")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, false
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, false
	}
	return out, true
}

// FindLibNCCL searches a process's /proc/<pid>/maps for a loaded
// libnccl.so or, failing that, a libtorch_cuda.so / libtorch_global_deps.so
// (PyTorch builds that statically link NCCL into libtorch_cuda).
//
// Returns the absolute path of the first match, or "" if none found.
// Implements the NCCL-static-fallback half of Workstream A3.
//
// Security: the path comes from the target process's /proc/<pid>/maps,
// which a non-root attacker can populate by mmap'ing an arbitrary file
// they wrote in a user-writable directory. We reject any candidate
// outside a small allowlist of canonical library install roots so that
// uprobe attach (which runs as root) only ever opens a vendor-installed
// or pip-installed PyTorch ELF. Drift here would let an attacker steer
// root-side ELF parsing.
func FindLibNCCL(pid int) string {
	maps, err := os.ReadFile(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		return ""
	}
	candidates := []string{
		"libnccl.so",
		"libtorch_cuda.so",
		"libtorch_global_deps.so",
	}
	seen := map[string]bool{}
	for _, line := range strings.Split(string(maps), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := fields[5]
		if seen[path] {
			continue
		}
		seen[path] = true
		base := filepath.Base(path)
		for _, want := range candidates {
			if !strings.HasPrefix(base, want) {
				continue
			}
			if !isSafeLibPath(path) {
				continue
			}
			if _, err := os.Stat(path); err == nil {
				return path
			}
		}
	}
	return ""
}

// isSafeLibPath returns true if path is under a canonical library install
// root and is NOT under a user-writable directory that an unprivileged
// attacker could plant a fake ELF in.
//
// Allowlist roots cover: distro libraries (/usr/lib*, /lib*), local
// installs (/usr/local), NVIDIA GPU stack (/opt/nccl, /opt/cuda,
// /usr/local/cuda*), and Python virtualenvs / system site-packages
// (/opt/conda, /opt/pytorch, /opt/python, /usr/lib/python*). Explicit
// denylist for /tmp, /home, /dev/shm, /var/tmp.
func isSafeLibPath(path string) bool {
	abs, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	abs = filepath.Clean(abs)
	denied := []string{"/tmp/", "/home/", "/dev/shm/", "/var/tmp/", "/run/user/"}
	for _, d := range denied {
		if strings.HasPrefix(abs, d) {
			return false
		}
	}
	allowed := []string{
		"/usr/lib/", "/usr/lib64/", "/lib/", "/lib64/",
		"/usr/local/", "/opt/nccl/", "/opt/cuda/",
		"/opt/conda/", "/opt/pytorch/", "/opt/python",
	}
	for _, a := range allowed {
		if strings.HasPrefix(abs, a) {
			return true
		}
	}
	return false
}

// SetTargetPID adds a PID to the NCCL probe's filter map. v0.12.2
// (LHF #7): when the filter is non-empty, only listed PIDs produce
// NCCL events; multi-tenant deployments call this for each GPU process
// they own to avoid leaking other tenants' NCCL activity into their
// agent's ringbuf.
//
// Sentinel: also inserts an entry at key=0 so the BPF-side
// nccl_pid_map_empty() returns false. Without the sentinel an empty
// map looks identical to "only pid=0", so the gate would be off.
// Mirrors the net-probe pattern at internal/ebpf/net/net.go:117.
//
// Concurrency: BPF map Update is atomic at the kernel level, so
// concurrent SetTargetPID / ClearTargetPIDs calls from multiple
// goroutines won't corrupt the map. Userspace state (the BPF objects
// themselves) isn't mutated here; all serialization happens kernel-side.
//
// v0.12.2 attach-race note: prefer passing PIDs to Attach() rather than
// calling SetTargetPID after Attach returns. The pre-Attach path
// installs the filter before the first uprobe is wired and is the only
// race-free way to seed it; SetTargetPID after Attach lets a small
// window of system-wide events slip through before the filter takes
// effect. v0.12.3 hardware validation (validate-v0.12.sh N4) exercises
// the pre-Attach path with --pid set.
func (t *Tracer) SetTargetPID(pid uint32) error {
	one := uint8(1)
	if err := t.objs.NcclTargetPids.Update(uint32(0), one, ebpf.UpdateAny); err != nil {
		return fmt.Errorf("nccl_target_pids sentinel: %w", err)
	}
	if pid == 0 {
		return nil // sentinel only; no real PID to add
	}
	if err := t.objs.NcclTargetPids.Update(pid, one, ebpf.UpdateAny); err != nil {
		return fmt.Errorf("nccl_target_pids[%d]: %w", pid, err)
	}
	return nil
}

// ClearTargetPIDs empties the filter map, returning the probe to
// system-wide tracing. Useful for tests and for `--nccl` invocations
// without an explicit --pid.
//
// v0.12.2 (Sec audit ★3): propagate Delete errors. A partial clear
// (sentinel deleted, but a real PID stuck) leaves the map in a
// "filter on, allowed-PID set wrong" state; ENOENT-style races are
// rare here because we iterated to collect the keys, but a returned
// error tells the caller the filter state is inconsistent.
func (t *Tracer) ClearTargetPIDs() error {
	// Iterate by lookup-and-delete. The map is small (max 256), so this
	// is a couple-iteration linear pass.
	var key uint32
	var val uint8
	iter := t.objs.NcclTargetPids.Iterate()
	var keys []uint32
	for iter.Next(&key, &val) {
		keys = append(keys, key)
	}
	for _, k := range keys {
		if err := t.objs.NcclTargetPids.Delete(k); err != nil {
			return fmt.Errorf("nccl_target_pids delete[%d]: %w", k, err)
		}
	}
	return nil
}

// installPIDFilter writes the sentinel + every non-zero PID into
// the filter map. Must be called BEFORE Attach wires uprobes; that's
// the only race-free moment to seed the filter. Helper for the Attach
// pre-population path.
func (t *Tracer) installPIDFilter(pids []uint32) error {
	one := uint8(1)
	if err := t.objs.NcclTargetPids.Update(uint32(0), one, ebpf.UpdateAny); err != nil {
		return fmt.Errorf("nccl_target_pids sentinel: %w", err)
	}
	for _, pid := range pids {
		if pid == 0 {
			continue
		}
		if err := t.objs.NcclTargetPids.Update(pid, one, ebpf.UpdateAny); err != nil {
			return fmt.Errorf("nccl_target_pids[%d]: %w", pid, err)
		}
	}
	return nil
}

// HasCapBPF reports whether the current process has CAP_BPF in its
// effective set. v0.12.1 helper for the --nccl friendly-failure check;
// uprobe attach needs CAP_BPF + CAP_PERFMON on Linux >= 5.8.
//
// v0.12.2 (Sec audit ★3 #5): use capget(2) directly via x/sys/unix
// instead of parsing /proc/<pid>/status. Avoids the read+scan TOCTOU
// window between /proc read and a sibling-thread CapBSet drop, and
// uses the v3 capability format (handles >64 caps).
//
// Conservative: returns false on syscall error or pre-5.8 kernels
// (where CAP_BPF didn't exist yet); the caller treats false as "warn
// only, continue and let attach fail with libbpf's real error".
func HasCapBPF() bool {
	const (
		capBPF           = 39 // CAP_BPF since Linux 5.8
		linuxCapVersion3 = 0x20080522
	)
	hdr := unix.CapUserHeader{Version: linuxCapVersion3, Pid: 0} // 0 = self
	var data [2]unix.CapUserData
	if err := unix.Capget(&hdr, &data[0]); err != nil {
		return false
	}
	// CAP_BPF=39 lives in data[1] (capabilities 32..63).
	return data[1].Effective&(1<<(capBPF-32)) != 0
}

// FindLibNCCLSystemwide is the no-target-PID variant: scans common
// install locations on Linux distributions. Used by `ingero check`
// to surface NCCL availability before any GPU process exists.
func FindLibNCCLSystemwide() string {
	roots := []string{
		"/usr/lib/x86_64-linux-gnu",
		"/usr/lib/aarch64-linux-gnu",
		"/usr/lib64",
		"/usr/lib",
		"/usr/local/lib",
		"/opt/nccl/lib",
	}
	patterns := []string{"libnccl.so", "libnccl.so.2", "libnccl.so.2.*"}
	for _, root := range roots {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range entries {
			name := e.Name()
			for _, pat := range patterns {
				if matched, _ := filepath.Match(pat, name); matched {
					return filepath.Join(root, name)
				}
			}
		}
	}
	return ""
}

// ----- helpers used by tests + cmd/ingero -----

// CommString returns the comm field as a UTF-8 string (truncated to first NUL).
func (e Event) CommString() string {
	if i := bytes.IndexByte(e.Comm[:], 0); i >= 0 {
		return string(e.Comm[:i])
	}
	return string(e.Comm[:])
}

// OpName returns a human-readable name for the op code, matching the
// NCCL_OP_* defines in bpf/common.bpf.h.
func (e Event) OpName() string {
	switch e.Op {
	case 1:
		return "ncclCommInitRank"
	case 2:
		return "ncclCommDestroy"
	case 3:
		return "ncclAllReduce"
	case 4:
		return "ncclAllGather"
	case 5:
		return "ncclReduceScatter"
	case 6:
		return "ncclBcast"
	case 7:
		return "ncclSend"
	case 8:
		return "ncclRecv"
	case 9:
		return "ncclCommInitAll"
	default:
		return fmt.Sprintf("nccl_op_%d", e.Op)
	}
}

// Stringer helps with debug logs.
func (e Event) String() string {
	return fmt.Sprintf("nccl op=%s pid=%d tid=%d rank=%d/%d count=%d duration_us=%d comm_id=%016x ret=%d",
		e.OpName(), e.PID, e.TID, e.Rank, e.NRanks, e.CountBytes,
		e.DurationNs/1000, e.CommIDHash, e.ReturnCode)
}

// _ "use the unused vars": keeps the build fail-fast on field renames.
var _ = time.Second
