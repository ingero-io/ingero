// Package events defines shared event types mirroring bpf/common.bpf.h structs.
package events

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"golang.org/x/sys/unix"
)

// CommToString converts a fixed-size [16]int8 byte buffer (as produced by
// bpf2go for `char comm[16]`) into a Go string, trimming at the first NUL.
// Empty input or all-NUL input returns "" — callers must tolerate empty comm
// (BPF helpers can return zero bytes in softirq/edge contexts).
func CommToString(comm [16]int8) string {
	for i, c := range comm {
		if c == 0 {
			if i == 0 {
				return ""
			}
			b := make([]byte, i)
			for j := 0; j < i; j++ {
				b[j] = byte(comm[j])
			}
			return string(b)
		}
	}
	// No NUL found — full 16 bytes are valid (rare; comm is usually NUL-padded).
	b := make([]byte, 16)
	for j := 0; j < 16; j++ {
		b[j] = byte(comm[j])
	}
	return string(b)
}

// ktimeOffset is the difference between wall-clock time and CLOCK_MONOTONIC.
// bpf_ktime_get_ns() uses CLOCK_MONOTONIC, so to convert to wall-clock time
// we add this offset: wall = ktime + ktimeOffset.
var (
	ktimeOnce   sync.Once
	ktimeOffset int64 // nanoseconds: wall_clock_ns - monotonic_ns
)

func initKtimeOffset() {
	var ts unix.Timespec
	// Read CLOCK_MONOTONIC and wall clock as close together as possible.
	_ = unix.ClockGettime(unix.CLOCK_MONOTONIC, &ts)
	monoNs := ts.Nano()
	wallNs := time.Now().UnixNano()
	ktimeOffset = wallNs - monoNs
}

// KtimeToWallClock converts a bpf_ktime_get_ns() value to wall-clock time.
func KtimeToWallClock(ktimeNs uint64) time.Time {
	ktimeOnce.Do(initKtimeOffset)
	return time.Unix(0, int64(ktimeNs)+ktimeOffset)
}

// Source identifies which eBPF program layer produced the event.
// Values MUST match EVENT_SRC_* in bpf/common.bpf.h.
type Source uint8

const (
	SourceCUDA   Source = 1
	SourceNvidia Source = 2
	SourceHost   Source = 3
	SourceDriver Source = 4
	SourceIO     Source = 5
	SourceTCP    Source = 6
	SourceNet       Source = 7
	SourceCUDAGraph Source = 8
)

// String implements fmt.Stringer.
func (s Source) String() string {
	switch s {
	case SourceCUDA:
		return "cuda"
	case SourceNvidia:
		return "nvidia"
	case SourceHost:
		return "host"
	case SourceDriver:
		return "driver"
	case SourceIO:
		return "io"
	case SourceTCP:
		return "tcp"
	case SourceNet:
		return "net"
	case SourceCUDAGraph:
		return "cuda_graph"
	default:
		return fmt.Sprintf("unknown(%d)", s)
	}
}

// CUDAOp identifies the CUDA runtime operation.
//
// These values MUST match the CUDA_OP_* defines in bpf/common.bpf.h.
type CUDAOp uint8

const (
	CUDAMalloc        CUDAOp = 1
	CUDAFree          CUDAOp = 2
	CUDALaunchKernel  CUDAOp = 3
	CUDAMemcpy        CUDAOp = 4
	CUDAStreamSync    CUDAOp = 5
	CUDADeviceSync    CUDAOp = 6
	CUDAMemcpyAsync   CUDAOp = 7
	CUDAMallocManaged CUDAOp = 8
)

// String returns a human-readable name for the CUDA operation.
func (op CUDAOp) String() string {
	switch op {
	case CUDAMalloc:
		return "cudaMalloc"
	case CUDAFree:
		return "cudaFree"
	case CUDALaunchKernel:
		return "cudaLaunchKernel"
	case CUDAMemcpy:
		return "cudaMemcpy"
	case CUDAStreamSync:
		return "cudaStreamSync"
	case CUDADeviceSync:
		return "cudaDeviceSync"
	case CUDAMemcpyAsync:
		return "cudaMemcpyAsync"
	case CUDAMallocManaged:
		return "cudaMallocManaged"
	default:
		return fmt.Sprintf("unknown(%d)", op)
	}
}

// HostOp identifies the host kernel operation.
//
// These values MUST match the HOST_OP_* defines in bpf/common.bpf.h.
type HostOp uint8

const (
	HostSchedSwitch HostOp = 1
	HostSchedWakeup HostOp = 2
	HostPageAlloc   HostOp = 3
	HostOOMKill     HostOp = 4
	HostProcessExec HostOp = 5
	HostProcessExit HostOp = 6
	HostProcessFork HostOp = 7

	// K8s lifecycle events (synthetic, not from eBPF).
	// Op codes 10+ to avoid collisions with eBPF-defined host ops.
	HostPodRestart  HostOp = 10
	HostPodEviction HostOp = 11
	HostPodOOMKill  HostOp = 12

	// Aggregated summary events (synthetic, emitted by the userspace
	// drainAggregationMaps goroutine rather than the BPF ring buffer).
	// Op codes 20+ to leave room for future eBPF-defined host ops.
	//
	// Argument packing conventions:
	//
	//   HostMmPageAllocSummary:
	//     PID      = aggregated non-target PID
	//     TID      = 0
	//     Args[0]  = count of mm_page_alloc events in the drain window
	//     Args[1]  = total bytes allocated across those events
	//     Duration = 0 (unused)
	//
	//   HostSchedSwitchSummary:
	//     PID      = next_pid (incoming task of each aggregated transition)
	//     TID      = prev_pid (outgoing task — packed into TID since the
	//                transition is keyed by (prev_pid << 32) | next_pid)
	//     Args[0]  = count of sched_switch transitions in the drain window
	//     Args[1]  = total off-CPU nanoseconds (may be 0 when the kernel
	//                could not compute a duration for a transition)
	//     Duration = 0 (unused — the aggregate off-cpu total lives in Args[1])
	HostMmPageAllocSummary HostOp = 20
	HostSchedSwitchSummary HostOp = 21
)

// String returns a human-readable name for the host operation.
func (op HostOp) String() string {
	switch op {
	case HostSchedSwitch:
		return "sched_switch"
	case HostSchedWakeup:
		return "sched_wakeup"
	case HostPageAlloc:
		return "mm_page_alloc"
	case HostOOMKill:
		return "oom_kill"
	case HostProcessExec:
		return "process_exec"
	case HostProcessExit:
		return "process_exit"
	case HostProcessFork:
		return "process_fork"
	case HostPodRestart:
		return "pod_restart"
	case HostPodEviction:
		return "pod_eviction"
	case HostPodOOMKill:
		return "pod_oom_kill"
	case HostMmPageAllocSummary:
		return "mm_page_alloc_summary"
	case HostSchedSwitchSummary:
		return "sched_switch_summary"
	default:
		return fmt.Sprintf("host_op(%d)", op)
	}
}

// DriverOp identifies a CUDA driver API operation (libcuda.so).
//
// These values MUST match the DRIVER_OP_* defines in bpf/common.bpf.h.
type DriverOp uint8

const (
	DriverLaunchKernel   DriverOp = 1
	DriverMemcpy         DriverOp = 2
	DriverMemcpyAsync    DriverOp = 3
	DriverCtxSync        DriverOp = 4
	DriverMemAlloc       DriverOp = 5
	DriverMemAllocManaged DriverOp = 6
)

// String returns a human-readable name for the driver operation.
func (op DriverOp) String() string {
	switch op {
	case DriverLaunchKernel:
		return "cuLaunchKernel"
	case DriverMemcpy:
		return "cuMemcpy"
	case DriverMemcpyAsync:
		return "cuMemcpyAsync"
	case DriverCtxSync:
		return "cuCtxSynchronize"
	case DriverMemAlloc:
		return "cuMemAlloc"
	case DriverMemAllocManaged:
		return "cuMemAllocManaged"
	default:
		return fmt.Sprintf("driver_op(%d)", op)
	}
}

// IOOp identifies a block I/O operation.
//
// These values MUST match the IO_OP_* defines in bpf/common.bpf.h.
type IOOp uint8

const (
	IORead    IOOp = 1
	IOWrite   IOOp = 2
	IODiscard IOOp = 3
)

// String returns a human-readable name for the I/O operation.
func (op IOOp) String() string {
	switch op {
	case IORead:
		return "block_read"
	case IOWrite:
		return "block_write"
	case IODiscard:
		return "block_discard"
	default:
		return fmt.Sprintf("io_op(%d)", op)
	}
}

// TCPOp identifies a TCP operation.
//
// These values MUST match the TCP_OP_* defines in bpf/common.bpf.h.
type TCPOp uint8

const (
	TCPRetransmit TCPOp = 1
)

// String returns a human-readable name for the TCP operation.
func (op TCPOp) String() string {
	switch op {
	case TCPRetransmit:
		return "tcp_retransmit"
	default:
		return fmt.Sprintf("tcp_op(%d)", op)
	}
}

// NetOp identifies a network socket operation.
//
// These values MUST match the NET_OP_* defines in bpf/common.bpf.h.
type NetOp uint8

const (
	NetSend NetOp = 1
	NetRecv NetOp = 2
)

// String returns a human-readable name for the network operation.
func (op NetOp) String() string {
	switch op {
	case NetSend:
		return "net_send"
	case NetRecv:
		return "net_recv"
	default:
		return fmt.Sprintf("net_op(%d)", op)
	}
}

// CUDAGraphOp identifies a CUDA Graph lifecycle operation.
//
// These values MUST match the GRAPH_OP_* defines in bpf/common.bpf.h.
type CUDAGraphOp uint8

const (
	GraphBeginCapture CUDAGraphOp = 1
	GraphEndCapture   CUDAGraphOp = 2
	GraphInstantiate  CUDAGraphOp = 3
	GraphLaunch       CUDAGraphOp = 4
)

// String returns a human-readable name for the CUDA Graph operation.
func (op CUDAGraphOp) String() string {
	switch op {
	case GraphBeginCapture:
		return "graphBeginCapture"
	case GraphEndCapture:
		return "graphEndCapture"
	case GraphInstantiate:
		return "graphInstantiate"
	case GraphLaunch:
		return "graphLaunch"
	default:
		return fmt.Sprintf("graph_op(%d)", op)
	}
}

// OpName returns the human-readable name for the Op field of an Event.
// It dispatches based on the Source to pick the right set of op names.
func (e Event) OpName() string {
	switch e.Source {
	case SourceCUDA:
		return CUDAOp(e.Op).String()
	case SourceHost:
		return HostOp(e.Op).String()
	case SourceDriver:
		return DriverOp(e.Op).String()
	case SourceIO:
		return IOOp(e.Op).String()
	case SourceTCP:
		return TCPOp(e.Op).String()
	case SourceNet:
		return NetOp(e.Op).String()
	case SourceCUDAGraph:
		return CUDAGraphOp(e.Op).String()
	default:
		return fmt.Sprintf("op(%d)", e.Op)
	}
}

// ResolveOp maps a human-readable operation name (e.g., "cudaMemcpy",
// "sched_switch", "cuLaunchKernel") to its Source and Op code. The name
// is matched case-insensitively. Returns false if the name is unknown.
func ResolveOp(name string) (Source, uint8, bool) {
	lower := strings.ToLower(name)

	// CUDA Runtime ops.
	cudaOps := map[string]CUDAOp{
		"cudamalloc":        CUDAMalloc,
		"cudafree":          CUDAFree,
		"cudalaunchkernel":  CUDALaunchKernel,
		"cudamemcpy":        CUDAMemcpy,
		"cudastreamsync":    CUDAStreamSync,
		"cudadevicesync":    CUDADeviceSync,
		"cudamemcpyasync":   CUDAMemcpyAsync,
		"cudamallocmanaged": CUDAMallocManaged,
	}
	for k, v := range cudaOps {
		if lower == k {
			return SourceCUDA, uint8(v), true
		}
	}

	// CUDA Driver ops.
	driverOps := map[string]DriverOp{
		"culaunchkernel":      DriverLaunchKernel,
		"cumemcpy":            DriverMemcpy,
		"cumemcpyasync":       DriverMemcpyAsync,
		"cuctxsynchronize":    DriverCtxSync,
		"cumemallocv2":        DriverMemAlloc,
		"cumemalloc":          DriverMemAlloc,
		"cumemallocmanaged":   DriverMemAllocManaged,
		"cumemallocmanagedv2": DriverMemAllocManaged,
	}
	for k, v := range driverOps {
		if lower == k {
			return SourceDriver, uint8(v), true
		}
	}

	// Host ops.
	hostOps := map[string]HostOp{
		"sched_switch":  HostSchedSwitch,
		"sched_wakeup":  HostSchedWakeup,
		"mm_page_alloc": HostPageAlloc,
		"oom_kill":      HostOOMKill,
		"process_exec":  HostProcessExec,
		"process_exit":  HostProcessExit,
		"process_fork":  HostProcessFork,
		"pod_restart":   HostPodRestart,
		"pod_eviction":  HostPodEviction,
		"pod_oom_kill":  HostPodOOMKill,
		"mm_page_alloc_summary": HostMmPageAllocSummary,
		"sched_switch_summary":  HostSchedSwitchSummary,
	}
	for k, v := range hostOps {
		if lower == k {
			return SourceHost, uint8(v), true
		}
	}

	// Block I/O ops.
	ioOps := map[string]IOOp{
		"block_read":    IORead,
		"block_write":   IOWrite,
		"block_discard": IODiscard,
	}
	for k, v := range ioOps {
		if lower == k {
			return SourceIO, uint8(v), true
		}
	}

	// TCP ops.
	tcpOps := map[string]TCPOp{
		"tcp_retransmit": TCPRetransmit,
	}
	for k, v := range tcpOps {
		if lower == k {
			return SourceTCP, uint8(v), true
		}
	}

	// CUDA Graph ops.
	graphOps := map[string]CUDAGraphOp{
		"graphbegincapture": GraphBeginCapture,
		"graphendcapture":   GraphEndCapture,
		"graphinstantiate":  GraphInstantiate,
		"graphlaunch":       GraphLaunch,
	}
	for k, v := range graphOps {
		if lower == k {
			return SourceCUDAGraph, uint8(v), true
		}
	}

	// Network socket ops.
	netOps := map[string]NetOp{
		"net_send": NetSend,
		"net_recv": NetRecv,
	}
	for k, v := range netOps {
		if lower == k {
			return SourceNet, uint8(v), true
		}
	}

	return 0, 0, false
}

// StackFrame represents a single frame in a userspace stack trace.
// Initially populated with just the raw instruction pointer (IP).
// Symbol resolution (Phase B) fills in SymbolName and File/Line.
// CPython frame extraction (Phase C) fills in PyFile/PyFunc/PyLine.
type StackFrame struct {
	IP         uint64 `json:"ip"`               // raw instruction pointer from bpf_get_stack()
	SymbolName string `json:"symbol,omitempty"`  // resolved native symbol (e.g., "cudaMalloc")
	File       string `json:"file,omitempty"`    // shared object or binary path
	Line       int    `json:"line,omitempty"`    // source line (if debug info available)
	PyFile     string `json:"py_file,omitempty"` // Python source file (Phase C)
	PyFunc     string `json:"py_func,omitempty"` // Python function name (Phase C)
	PyLine     int    `json:"py_line,omitempty"` // Python source line (Phase C)
}

// PyFrame represents a single Python source frame in the call stack.
//
// Produced by either the userspace CPython walker (internal/symtab) or
// the in-kernel eBPF walker (bpf/python_walker.bpf.h → parseEvent). The
// events package owns the type so symtab, ebpf, and resolver layers can
// pass frames around without an import cycle.
//
// Line holds PyCodeObject.co_firstlineno — the first line of the function
// definition, not the currently executing line. Precise current-line
// resolution requires decoding co_linetable and is a future enhancement.
type PyFrame struct {
	Filename string `json:"filename,omitempty"` // e.g., "train.py"
	Function string `json:"function,omitempty"` // e.g., "forward"
	Line     int    `json:"line,omitempty"`     // e.g., 47 (co_firstlineno)
}

// String returns a human-readable representation: "train.py:47 in forward()".
func (f PyFrame) String() string {
	if f.Line > 0 {
		return fmt.Sprintf("%s:%d in %s()", f.Filename, f.Line, f.Function)
	}
	return fmt.Sprintf("%s in %s()", f.Filename, f.Function)
}

// Event is the common envelope for all traced events.
// Single struct with Source discriminator — simplifies channel, stats, and storage.
type Event struct {
	Timestamp time.Time     // when the operation completed (kernel monotonic clock)
	PID       uint32        // process ID (tgid in kernel terms)
	TID       uint32        // thread ID (pid in kernel terms — yes, confusing)
	Comm      string        // process name from bpf_get_current_comm() (≤TASK_COMM_LEN=16, may be empty)
	Source    Source        // which eBPF layer: cuda, nvidia, host
	Op        uint8         // operation type (cast to CUDAOp, etc. based on Source)
	Duration  time.Duration // how long the operation took (entry→return)
	GPUID     uint32        // GPU device index (from CUDA)
	// Args contains operation-specific arguments:
	//   cudaMalloc:  Args[0] = allocation size (bytes), Args[1] = devPtr param address
	//   cudaFree:    Args[0] = device pointer being freed, Args[1] = freed size in bytes (0 if unknown)
	//   cudaMemcpy:  Args[0] = byte count, Args[1] = direction (cudaMemcpyKind)
	//   Other ops:   operation-specific
	Args [2]uint64
	RetCode   int32         // CUDA return code (0 = success)
	Stack     []StackFrame  // userspace stack trace (nil when --stack not enabled)
	CGroupID  uint64        // cgroup v2 inode ID (0 or 1 = no meaningful cgroup)

	// PythonFrames is populated by the event parser when the in-kernel
	// Python frame walker (--py-walker=ebpf) captured frames for this
	// event. Nil when the BPF walker is disabled or produced no frames.
	// The resolver uses this slice in preference to invoking the
	// userspace walker.
	PythonFrames []PyFrame `json:"python_frames,omitempty"`

	// CUDA Graph fields (only populated for SourceCUDAGraph events):
	StreamHandle uint64 // stream for BeginCapture/EndCapture/Launch
	GraphHandle  uint64 // graph for EndCapture/Instantiate
	ExecHandle   uint64 // executable for Instantiate/Launch
	CaptureMode  uint32 // for BeginCapture (0=global, 1=thread_local, 2=relaxed)

	// Multi-node identity fields (v0.9). Populated when --node is set.
	Node      string // node identity (hostname or --node flag value)
	Rank      *int   // distributed training rank (nil = not in distributed training)
	LocalRank *int   // local rank within this node (nil = not set)
	WorldSize *int   // total number of ranks (nil = not set)
}
