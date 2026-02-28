// Package events defines shared event types mirroring bpf/common.bpf.h structs.
package events

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"golang.org/x/sys/unix"
)

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
	default:
		return fmt.Sprintf("unknown(%d)", s)
	}
}

// CUDAOp identifies the CUDA runtime operation.
//
// These values MUST match the CUDA_OP_* defines in bpf/common.bpf.h.
type CUDAOp uint8

const (
	CUDAMalloc       CUDAOp = 1
	CUDAFree         CUDAOp = 2
	CUDALaunchKernel CUDAOp = 3
	CUDAMemcpy       CUDAOp = 4
	CUDAStreamSync   CUDAOp = 5
	CUDADeviceSync   CUDAOp = 6
	CUDAMemcpyAsync  CUDAOp = 7
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
	default:
		return fmt.Sprintf("host_op(%d)", op)
	}
}

// DriverOp identifies a CUDA driver API operation (libcuda.so).
//
// These values MUST match the DRIVER_OP_* defines in bpf/common.bpf.h.
type DriverOp uint8

const (
	DriverLaunchKernel DriverOp = 1
	DriverMemcpy       DriverOp = 2
	DriverMemcpyAsync  DriverOp = 3
	DriverCtxSync      DriverOp = 4
	DriverMemAlloc     DriverOp = 5
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
	default:
		return fmt.Sprintf("driver_op(%d)", op)
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
		"cudamalloc":       CUDAMalloc,
		"cudafree":         CUDAFree,
		"cudalaunchkernel": CUDALaunchKernel,
		"cudamemcpy":       CUDAMemcpy,
		"cudastreamsync":   CUDAStreamSync,
		"cudadevicesync":   CUDADeviceSync,
		"cudamemcpyasync":  CUDAMemcpyAsync,
	}
	for k, v := range cudaOps {
		if lower == k {
			return SourceCUDA, uint8(v), true
		}
	}

	// CUDA Driver ops.
	driverOps := map[string]DriverOp{
		"culaunchkernel":   DriverLaunchKernel,
		"cumemcpy":         DriverMemcpy,
		"cumemcpyasync":    DriverMemcpyAsync,
		"cuctxsynchronize": DriverCtxSync,
		"cumemallocv2":     DriverMemAlloc,
		"cumemalloc":       DriverMemAlloc,
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
	}
	for k, v := range hostOps {
		if lower == k {
			return SourceHost, uint8(v), true
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

// Event is the common envelope for all traced events.
// Single struct with Source discriminator — simplifies channel, stats, and storage.
type Event struct {
	Timestamp time.Time     // when the operation completed (kernel monotonic clock)
	PID       uint32        // process ID (tgid in kernel terms)
	TID       uint32        // thread ID (pid in kernel terms — yes, confusing)
	Source    Source        // which eBPF layer: cuda, nvidia, host
	Op        uint8         // operation type (cast to CUDAOp, etc. based on Source)
	Duration  time.Duration // how long the operation took (entry→return)
	GPUID     uint32        // GPU device index (from CUDA)
	Args      [2]uint64     // operation-specific arguments (size, direction, etc.)
	RetCode   int32         // CUDA return code (0 = success)
	Stack     []StackFrame  // userspace stack trace (nil when --stack not enabled)
	CGroupID  uint64        // cgroup v2 inode ID (0 or 1 = no meaningful cgroup)
}
