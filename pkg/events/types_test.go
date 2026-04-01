package events

import (
	"testing"
)

// TestSourceString verifies human-readable source names.
func TestSourceString(t *testing.T) {
	tests := []struct {
		source Source
		want   string
	}{
		{SourceCUDA, "cuda"},
		{SourceNvidia, "nvidia"},
		{SourceHost, "host"},
		{SourceDriver, "driver"},
		{SourceIO, "io"},
		{SourceTCP, "tcp"},
		{SourceNet, "net"},
		{SourceCUDAGraph, "cuda_graph"},
		{Source(0), "unknown(0)"},
		{Source(99), "unknown(99)"},
	}

	for _, tt := range tests {
		got := tt.source.String()
		if got != tt.want {
			t.Errorf("Source(%d).String() = %q, want %q", tt.source, got, tt.want)
		}
	}
}

// TestCUDAOpString verifies human-readable CUDA op names.
func TestCUDAOpString(t *testing.T) {
	tests := []struct {
		op   CUDAOp
		want string
	}{
		{CUDAMalloc, "cudaMalloc"},
		{CUDAFree, "cudaFree"},
		{CUDALaunchKernel, "cudaLaunchKernel"},
		{CUDAMemcpy, "cudaMemcpy"},
		{CUDAStreamSync, "cudaStreamSync"},
		{CUDADeviceSync, "cudaDeviceSync"},
		{CUDAMemcpyAsync, "cudaMemcpyAsync"},
		{CUDAMallocManaged, "cudaMallocManaged"},
		{CUDAOp(0), "unknown(0)"},
		{CUDAOp(99), "unknown(99)"},
	}

	for _, tt := range tests {
		got := tt.op.String()
		if got != tt.want {
			t.Errorf("CUDAOp(%d).String() = %q, want %q", tt.op, got, tt.want)
		}
	}
}

// TestCUDAGraphOpString verifies human-readable CUDA Graph op names.
func TestCUDAGraphOpString(t *testing.T) {
	tests := []struct {
		op   CUDAGraphOp
		want string
	}{
		{GraphBeginCapture, "graphBeginCapture"},
		{GraphEndCapture, "graphEndCapture"},
		{GraphInstantiate, "graphInstantiate"},
		{GraphLaunch, "graphLaunch"},
		{CUDAGraphOp(0), "graph_op(0)"},
		{CUDAGraphOp(99), "graph_op(99)"},
	}

	for _, tt := range tests {
		got := tt.op.String()
		if got != tt.want {
			t.Errorf("CUDAGraphOp(%d).String() = %q, want %q", tt.op, got, tt.want)
		}
	}
}

// TestHostOpString verifies human-readable host op names.
func TestHostOpString(t *testing.T) {
	tests := []struct {
		op   HostOp
		want string
	}{
		{HostSchedSwitch, "sched_switch"},
		{HostSchedWakeup, "sched_wakeup"},
		{HostPageAlloc, "mm_page_alloc"},
		{HostOOMKill, "oom_kill"},
		{HostProcessExec, "process_exec"},
		{HostProcessExit, "process_exit"},
		{HostProcessFork, "process_fork"},
		{HostPodRestart, "pod_restart"},
		{HostPodEviction, "pod_eviction"},
		{HostPodOOMKill, "pod_oom_kill"},
		{HostOp(0), "host_op(0)"},
		{HostOp(99), "host_op(99)"},
	}

	for _, tt := range tests {
		got := tt.op.String()
		if got != tt.want {
			t.Errorf("HostOp(%d).String() = %q, want %q", tt.op, got, tt.want)
		}
	}
}

// TestDriverOpString verifies human-readable driver op names.
func TestDriverOpString(t *testing.T) {
	tests := []struct {
		op   DriverOp
		want string
	}{
		{DriverLaunchKernel, "cuLaunchKernel"},
		{DriverMemcpy, "cuMemcpy"},
		{DriverMemcpyAsync, "cuMemcpyAsync"},
		{DriverCtxSync, "cuCtxSynchronize"},
		{DriverMemAlloc, "cuMemAlloc"},
		{DriverMemAllocManaged, "cuMemAllocManaged"},
		{DriverOp(0), "driver_op(0)"},
		{DriverOp(99), "driver_op(99)"},
	}

	for _, tt := range tests {
		got := tt.op.String()
		if got != tt.want {
			t.Errorf("DriverOp(%d).String() = %q, want %q", tt.op, got, tt.want)
		}
	}
}

// TestEventOpName verifies OpName dispatches correctly by Source.
func TestEventOpName(t *testing.T) {
	tests := []struct {
		name   string
		source Source
		op     uint8
		want   string
	}{
		{"cuda malloc", SourceCUDA, uint8(CUDAMalloc), "cudaMalloc"},
		{"cuda launch", SourceCUDA, uint8(CUDALaunchKernel), "cudaLaunchKernel"},
		{"host sched_switch", SourceHost, uint8(HostSchedSwitch), "sched_switch"},
		{"host sched_wakeup", SourceHost, uint8(HostSchedWakeup), "sched_wakeup"},
		{"host page_alloc", SourceHost, uint8(HostPageAlloc), "mm_page_alloc"},
		{"host oom_kill", SourceHost, uint8(HostOOMKill), "oom_kill"},
		{"host process_exec", SourceHost, uint8(HostProcessExec), "process_exec"},
		{"host process_exit", SourceHost, uint8(HostProcessExit), "process_exit"},
		{"host process_fork", SourceHost, uint8(HostProcessFork), "process_fork"},
		{"host unknown op", SourceHost, 99, "host_op(99)"},
		{"driver launch", SourceDriver, uint8(DriverLaunchKernel), "cuLaunchKernel"},
		{"driver memcpy", SourceDriver, uint8(DriverMemcpy), "cuMemcpy"},
		{"driver ctx sync", SourceDriver, uint8(DriverCtxSync), "cuCtxSynchronize"},
		{"driver memcpy async", SourceDriver, uint8(DriverMemcpyAsync), "cuMemcpyAsync"},
		{"driver mem alloc", SourceDriver, uint8(DriverMemAlloc), "cuMemAlloc"},
		{"cuda malloc managed", SourceCUDA, uint8(CUDAMallocManaged), "cudaMallocManaged"},
		{"driver mem alloc managed", SourceDriver, uint8(DriverMemAllocManaged), "cuMemAllocManaged"},
		{"driver unknown", SourceDriver, 99, "driver_op(99)"},
		{"host pod restart", SourceHost, uint8(HostPodRestart), "pod_restart"},
		{"host pod eviction", SourceHost, uint8(HostPodEviction), "pod_eviction"},
		{"host pod oom kill", SourceHost, uint8(HostPodOOMKill), "pod_oom_kill"},
		{"io read", SourceIO, uint8(IORead), "block_read"},
		{"io write", SourceIO, uint8(IOWrite), "block_write"},
		{"io discard", SourceIO, uint8(IODiscard), "block_discard"},
		{"io unknown", SourceIO, 99, "io_op(99)"},
		{"tcp retransmit", SourceTCP, uint8(TCPRetransmit), "tcp_retransmit"},
		{"tcp unknown", SourceTCP, 99, "tcp_op(99)"},
		{"net send", SourceNet, uint8(NetSend), "net_send"},
		{"net recv", SourceNet, uint8(NetRecv), "net_recv"},
		{"net unknown", SourceNet, 99, "net_op(99)"},
		{"graph begin capture", SourceCUDAGraph, uint8(GraphBeginCapture), "graphBeginCapture"},
		{"graph end capture", SourceCUDAGraph, uint8(GraphEndCapture), "graphEndCapture"},
		{"graph instantiate", SourceCUDAGraph, uint8(GraphInstantiate), "graphInstantiate"},
		{"graph launch", SourceCUDAGraph, uint8(GraphLaunch), "graphLaunch"},
		{"graph unknown", SourceCUDAGraph, 99, "graph_op(99)"},
		{"nvidia unknown", SourceNvidia, 1, "op(1)"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			evt := Event{Source: tt.source, Op: tt.op}
			got := evt.OpName()
			if got != tt.want {
				t.Errorf("Event{Source: %v, Op: %d}.OpName() = %q, want %q",
					tt.source, tt.op, got, tt.want)
			}
		})
	}
}

// TestResolveOp verifies name → (Source, Op) resolution.
func TestResolveOp(t *testing.T) {
	tests := []struct {
		name       string
		wantSource Source
		wantOp     uint8
	}{
		{"cudaMalloc", SourceCUDA, uint8(CUDAMalloc)},
		{"cudaFree", SourceCUDA, uint8(CUDAFree)},
		{"cudaLaunchKernel", SourceCUDA, uint8(CUDALaunchKernel)},
		{"cudaMemcpy", SourceCUDA, uint8(CUDAMemcpy)},
		{"cudaStreamSync", SourceCUDA, uint8(CUDAStreamSync)},
		{"cudaDeviceSync", SourceCUDA, uint8(CUDADeviceSync)},
		{"cudaMemcpyAsync", SourceCUDA, uint8(CUDAMemcpyAsync)},
		{"cuLaunchKernel", SourceDriver, uint8(DriverLaunchKernel)},
		{"cuMemcpy", SourceDriver, uint8(DriverMemcpy)},
		{"cuMemcpyAsync", SourceDriver, uint8(DriverMemcpyAsync)},
		{"cuCtxSynchronize", SourceDriver, uint8(DriverCtxSync)},
		{"cuMemAlloc", SourceDriver, uint8(DriverMemAlloc)},
		{"sched_switch", SourceHost, uint8(HostSchedSwitch)},
		{"sched_wakeup", SourceHost, uint8(HostSchedWakeup)},
		{"mm_page_alloc", SourceHost, uint8(HostPageAlloc)},
		{"oom_kill", SourceHost, uint8(HostOOMKill)},
		{"process_exec", SourceHost, uint8(HostProcessExec)},
		{"process_exit", SourceHost, uint8(HostProcessExit)},
		{"process_fork", SourceHost, uint8(HostProcessFork)},
		{"cudaMallocManaged", SourceCUDA, uint8(CUDAMallocManaged)},
		{"cuMemAllocManaged", SourceDriver, uint8(DriverMemAllocManaged)},
		{"pod_restart", SourceHost, uint8(HostPodRestart)},
		{"pod_eviction", SourceHost, uint8(HostPodEviction)},
		{"pod_oom_kill", SourceHost, uint8(HostPodOOMKill)},
		{"graphBeginCapture", SourceCUDAGraph, uint8(GraphBeginCapture)},
		{"graphEndCapture", SourceCUDAGraph, uint8(GraphEndCapture)},
		{"graphInstantiate", SourceCUDAGraph, uint8(GraphInstantiate)},
		{"graphLaunch", SourceCUDAGraph, uint8(GraphLaunch)},
		{"block_read", SourceIO, uint8(IORead)},
		{"block_write", SourceIO, uint8(IOWrite)},
		{"block_discard", SourceIO, uint8(IODiscard)},
		{"tcp_retransmit", SourceTCP, uint8(TCPRetransmit)},
		{"net_send", SourceNet, uint8(NetSend)},
		{"net_recv", SourceNet, uint8(NetRecv)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src, op, ok := ResolveOp(tt.name)
			if !ok {
				t.Fatalf("ResolveOp(%q) returned not found", tt.name)
			}
			if src != tt.wantSource || op != tt.wantOp {
				t.Errorf("ResolveOp(%q) = (%v, %d), want (%v, %d)",
					tt.name, src, op, tt.wantSource, tt.wantOp)
			}
		})
	}
}

// TestResolveOp_CaseInsensitive verifies case-insensitive lookup.
func TestResolveOp_CaseInsensitive(t *testing.T) {
	cases := []string{"CUDAMALLOC", "CudaMalloc", "cudamalloc", "CULAUNCHKERNEL", "SCHED_SWITCH",
		"CUDAMALLOCMANAGED", "CUMEMALLOCMANAGED", "BLOCK_READ", "TCP_RETRANSMIT", "NET_SEND", "POD_RESTART",
		"GRAPHBEGINCAPTURE", "GRAPHLAUNCH"}
	for _, name := range cases {
		if _, _, ok := ResolveOp(name); !ok {
			t.Errorf("ResolveOp(%q) should resolve (case-insensitive)", name)
		}
	}
}

// TestResolveOp_Aliases verifies alternative names resolve correctly.
func TestResolveOp_Aliases(t *testing.T) {
	// cuMemAllocV2 is an alias for cuMemAlloc (driver symbol versioning).
	src, op, ok := ResolveOp("cuMemAllocV2")
	if !ok {
		t.Fatal("ResolveOp(\"cuMemAllocV2\") returned not found")
	}
	if src != SourceDriver || op != uint8(DriverMemAlloc) {
		t.Errorf("ResolveOp(\"cuMemAllocV2\") = (%v, %d), want (driver, %d)", src, op, DriverMemAlloc)
	}

	// cuMemAllocManagedV2 is an alias for cuMemAllocManaged.
	src2, op2, ok2 := ResolveOp("cuMemAllocManagedV2")
	if !ok2 {
		t.Fatal("ResolveOp(\"cuMemAllocManagedV2\") returned not found")
	}
	if src2 != SourceDriver || op2 != uint8(DriverMemAllocManaged) {
		t.Errorf("ResolveOp(\"cuMemAllocManagedV2\") = (%v, %d), want (driver, %d)", src2, op2, DriverMemAllocManaged)
	}
}

// TestResolveOp_RoundTrip verifies String() → ResolveOp() round-trips for all ops.
func TestResolveOp_RoundTrip(t *testing.T) {
	cudaOps := []CUDAOp{CUDAMalloc, CUDAFree, CUDALaunchKernel, CUDAMemcpy, CUDAStreamSync, CUDADeviceSync, CUDAMemcpyAsync, CUDAMallocManaged}
	for _, op := range cudaOps {
		name := op.String()
		src, resolved, ok := ResolveOp(name)
		if !ok {
			t.Errorf("ResolveOp(%q) failed round-trip", name)
			continue
		}
		if src != SourceCUDA || resolved != uint8(op) {
			t.Errorf("ResolveOp(%q) = (%v, %d), want (cuda, %d)", name, src, resolved, op)
		}
	}

	driverOps := []DriverOp{DriverLaunchKernel, DriverMemcpy, DriverMemcpyAsync, DriverCtxSync, DriverMemAlloc, DriverMemAllocManaged}
	for _, op := range driverOps {
		name := op.String()
		src, resolved, ok := ResolveOp(name)
		if !ok {
			t.Errorf("ResolveOp(%q) failed round-trip", name)
			continue
		}
		if src != SourceDriver || resolved != uint8(op) {
			t.Errorf("ResolveOp(%q) = (%v, %d), want (driver, %d)", name, src, resolved, op)
		}
	}

	hostOps := []HostOp{HostSchedSwitch, HostSchedWakeup, HostPageAlloc, HostOOMKill, HostProcessExec, HostProcessExit, HostProcessFork, HostPodRestart, HostPodEviction, HostPodOOMKill}
	for _, op := range hostOps {
		name := op.String()
		src, resolved, ok := ResolveOp(name)
		if !ok {
			t.Errorf("ResolveOp(%q) failed round-trip", name)
			continue
		}
		if src != SourceHost || resolved != uint8(op) {
			t.Errorf("ResolveOp(%q) = (%v, %d), want (host, %d)", name, src, resolved, op)
		}
	}

	ioOps := []IOOp{IORead, IOWrite, IODiscard}
	for _, op := range ioOps {
		name := op.String()
		src, resolved, ok := ResolveOp(name)
		if !ok {
			t.Errorf("ResolveOp(%q) failed round-trip", name)
			continue
		}
		if src != SourceIO || resolved != uint8(op) {
			t.Errorf("ResolveOp(%q) = (%v, %d), want (io, %d)", name, src, resolved, op)
		}
	}

	tcpOps := []TCPOp{TCPRetransmit}
	for _, op := range tcpOps {
		name := op.String()
		src, resolved, ok := ResolveOp(name)
		if !ok {
			t.Errorf("ResolveOp(%q) failed round-trip", name)
			continue
		}
		if src != SourceTCP || resolved != uint8(op) {
			t.Errorf("ResolveOp(%q) = (%v, %d), want (tcp, %d)", name, src, resolved, op)
		}
	}

	netOps := []NetOp{NetSend, NetRecv}
	for _, op := range netOps {
		name := op.String()
		src, resolved, ok := ResolveOp(name)
		if !ok {
			t.Errorf("ResolveOp(%q) failed round-trip", name)
			continue
		}
		if src != SourceNet || resolved != uint8(op) {
			t.Errorf("ResolveOp(%q) = (%v, %d), want (net, %d)", name, src, resolved, op)
		}
	}

	graphOps := []CUDAGraphOp{GraphBeginCapture, GraphEndCapture, GraphInstantiate, GraphLaunch}
	for _, op := range graphOps {
		name := op.String()
		src, resolved, ok := ResolveOp(name)
		if !ok {
			t.Errorf("ResolveOp(%q) failed round-trip", name)
			continue
		}
		if src != SourceCUDAGraph || resolved != uint8(op) {
			t.Errorf("ResolveOp(%q) = (%v, %d), want (cuda_graph, %d)", name, src, resolved, op)
		}
	}
}

// TestResolveOp_Unknown verifies unknown names return false.
func TestResolveOp_Unknown(t *testing.T) {
	unknowns := []string{"", "nonexistent", "cudaMallocX", "foobar"}
	for _, name := range unknowns {
		if _, _, ok := ResolveOp(name); ok {
			t.Errorf("ResolveOp(%q) should return false", name)
		}
	}
}

