// SPDX-License-Identifier: Apache-2.0
//
// Package pytrace provides Go-side helpers for the in-kernel CPython 3.12
// frame walker (bpf/python_walker.bpf.h).
//
// The walker (and its py_runtime_map) is compiled into another tracer's
// BPF object — currently the cuda tracer, which includes
// python_walker.bpf.h from bpf/cuda_trace.bpf.c. That tracer owns the map;
// this package never loads its own BPF program.
//
// Lifecycle (called from internal/cli):
//  1. Obtain the map from the cuda tracer: m := cudaTracer.PyRuntimeMap()
//  2. On process_exec for a Python 3.12 process:
//     pytrace.SetPyRuntimeState(m, pid, PyRuntimeState{...})
//  3. On process_exit:
//     pytrace.ClearPID(m, pid)
//
// The Go side only marshals struct py_runtime_state (mirroring
// bpf/common.bpf.h) and issues Put/Delete against the caller-provided
// *ebpf.Map.
package pytrace

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/cilium/ebpf"
)

// PyRuntimeState mirrors `struct py_runtime_state` in bpf/common.bpf.h.
// Layout: u64 + 12 * u16 = 32 bytes total. Field order MUST match C.
type PyRuntimeState struct {
	RuntimeAddr                uint64 // address of _PyRuntime in target process
	OffRuntimeInterpretersHead uint16
	OffTstateHead              uint16
	OffTstateNext              uint16
	OffTstateNativeTid         uint16
	OffTstateFrame             uint16
	OffFrameBack               uint16
	OffFrameCode               uint16
	OffCodeFilename            uint16
	OffCodeName                uint16
	OffCodeFirstLineNo         uint16
	OffUnicodeState            uint16
	OffUnicodeData             uint16
}

// pyRuntimeStateSize is the marshaled size of PyRuntimeState in bytes.
// This must match the C struct layout (u64 + 12 * u16 = 32 bytes).
const pyRuntimeStateSize = 8 + 12*2

// MarshalBinary serializes the struct to little-endian bytes for writing
// to the BPF map. The map value is opaque bytes from the kernel's view —
// we control the layout.
func (s *PyRuntimeState) MarshalBinary() ([]byte, error) {
	buf := make([]byte, pyRuntimeStateSize)
	binary.LittleEndian.PutUint64(buf[0:8], s.RuntimeAddr)
	binary.LittleEndian.PutUint16(buf[8:10], s.OffRuntimeInterpretersHead)
	binary.LittleEndian.PutUint16(buf[10:12], s.OffTstateHead)
	binary.LittleEndian.PutUint16(buf[12:14], s.OffTstateNext)
	binary.LittleEndian.PutUint16(buf[14:16], s.OffTstateNativeTid)
	binary.LittleEndian.PutUint16(buf[16:18], s.OffTstateFrame)
	binary.LittleEndian.PutUint16(buf[18:20], s.OffFrameBack)
	binary.LittleEndian.PutUint16(buf[20:22], s.OffFrameCode)
	binary.LittleEndian.PutUint16(buf[22:24], s.OffCodeFilename)
	binary.LittleEndian.PutUint16(buf[24:26], s.OffCodeName)
	binary.LittleEndian.PutUint16(buf[26:28], s.OffCodeFirstLineNo)
	binary.LittleEndian.PutUint16(buf[28:30], s.OffUnicodeState)
	binary.LittleEndian.PutUint16(buf[30:32], s.OffUnicodeData)
	return buf, nil
}

// SetPyRuntimeState writes per-PID CPython runtime state to the supplied
// BPF map. The map must be the py_runtime_map owned by the tracer that
// compiled bpf/python_walker.bpf.h (currently the cuda tracer).
//
// Returns an error if m is nil, if marshaling fails, or if the underlying
// Put syscall fails. Thread-safe: BPF map operations are kernel-atomic.
func SetPyRuntimeState(m *ebpf.Map, pid uint32, state PyRuntimeState) error {
	if m == nil {
		return errors.New("py_runtime_map is nil")
	}
	buf, err := state.MarshalBinary()
	if err != nil {
		return fmt.Errorf("marshaling py runtime state: %w", err)
	}
	if err := m.Put(pid, buf); err != nil {
		return fmt.Errorf("writing py_runtime_map for pid %d: %w", pid, err)
	}
	return nil
}

// ClearPID removes per-PID state from the supplied BPF map. Call on
// process_exit to keep the map bounded as PIDs recycle.
//
// A nil map is a no-op (walker disabled). A missing key is not an error —
// a process may exit before we ever set its state (e.g., Python <3.12,
// short-lived processes, races between exec and the probe firing).
func ClearPID(m *ebpf.Map, pid uint32) error {
	if m == nil {
		return nil
	}
	if err := m.Delete(pid); err != nil && !errors.Is(err, ebpf.ErrKeyNotExist) {
		return fmt.Errorf("deleting py_runtime_map for pid %d: %w", pid, err)
	}
	return nil
}
