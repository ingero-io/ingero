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
