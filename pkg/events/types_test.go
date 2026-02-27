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
		{"driver mem alloc", SourceDriver, uint8(DriverMemAlloc), "cuMemAlloc"},
		{"driver unknown", SourceDriver, 99, "driver_op(99)"},
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
	cases := []string{"CUDAMALLOC", "CudaMalloc", "cudamalloc", "CULAUNCHKERNEL", "SCHED_SWITCH"}
	for _, name := range cases {
		if _, _, ok := ResolveOp(name); !ok {
			t.Errorf("ResolveOp(%q) should resolve (case-insensitive)", name)
		}
	}
}

// TestResolveOp_RoundTrip verifies String() → ResolveOp() round-trips for all ops.
func TestResolveOp_RoundTrip(t *testing.T) {
	cudaOps := []CUDAOp{CUDAMalloc, CUDAFree, CUDALaunchKernel, CUDAMemcpy, CUDAStreamSync, CUDADeviceSync, CUDAMemcpyAsync}
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

	driverOps := []DriverOp{DriverLaunchKernel, DriverMemcpy, DriverMemcpyAsync, DriverCtxSync, DriverMemAlloc}
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

	hostOps := []HostOp{HostSchedSwitch, HostSchedWakeup, HostPageAlloc, HostOOMKill, HostProcessExec, HostProcessExit, HostProcessFork}
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

