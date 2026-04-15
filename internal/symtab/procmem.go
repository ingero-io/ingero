package symtab

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"syscall"

	"golang.org/x/sys/unix"
)

// ProcMem provides read access to a target process's memory.
//
// It has two read paths:
//
//   - /proc/[pid]/mem, opened at construction time. This is the preferred path,
//     but the kernel enforces ptrace permission checks on every read. Under the
//     default Ubuntu 24.04 configuration (kernel.yama.ptrace_scope=1), reads
//     fail with EPERM unless the caller is the ptrace tracer of the target.
//
//   - process_vm_readv(2) — a syscall that copies memory across processes
//     without requiring an actual ptrace attachment. It only requires
//     CAP_SYS_PTRACE on the calling process. This works at ptrace_scope=0
//     and =1, fails at =2 without CAP_SYS_PTRACE, and always fails at =3.
//     Ingero runs as root, which implicitly grants CAP_SYS_PTRACE.
//
// When the /proc/[pid]/mem path returns EPERM/EACCES — either at open time or
// on a read — ProcMem transparently falls back to process_vm_readv and remains
// on that path for all subsequent reads.
type ProcMem struct {
	pid        uint32
	file       *os.File
	useVmReadv bool
}

// OpenProcMem opens /proc/[pid]/mem for reading. If the open fails with
// EPERM (common on ptrace_scope=1 systems), it returns a ProcMem that uses
// process_vm_readv for all reads instead of failing. Other open errors are
// returned as-is so callers can distinguish e.g. ESRCH (process gone).
func OpenProcMem(pid uint32) (*ProcMem, error) {
	path := fmt.Sprintf("/proc/%d/mem", pid)
	f, err := os.Open(path)
	if err != nil {
		if isPermErr(err) {
			// Strict ptrace_scope — can't open /proc/pid/mem, but
			// process_vm_readv may still work with CAP_SYS_PTRACE.
			return &ProcMem{pid: pid, file: nil, useVmReadv: true}, nil
		}
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	return &ProcMem{pid: pid, file: f}, nil
}

// Close releases the file handle if one was opened.
func (m *ProcMem) Close() error {
	if m.file == nil {
		return nil
	}
	return m.file.Close()
}

// isPermErr reports whether err is a permission-style error that warrants
// falling back to process_vm_readv. Unwraps syscall.Errno values wrapped
// inside *os.PathError or fmt.Errorf.
//
// Covers:
//   - EPERM, EACCES: classic "not permitted" (e.g., ptrace_scope=2 without
//     CAP_SYS_PTRACE).
//   - EIO: YAMA returns EIO when /proc/pid/mem reads are blocked because the
//     caller is not an authorized ptrace tracer. At kernel.yama.ptrace_scope=1
//     (Ubuntu default), only a process's direct parent (or an attached tracer)
//     can read its /proc/pid/mem — even as root. Any other reader gets EIO on
//     seek/read, not EPERM. Triggering the process_vm_readv fallback on EIO
//     recovers the common case of tracing a non-child process as root on
//     Ubuntu.
func isPermErr(err error) bool {
	var errno syscall.Errno
	if errors.As(err, &errno) {
		return errno == syscall.EPERM || errno == syscall.EACCES || errno == syscall.EIO
	}
	return false
}

// readVmReadv reads from the target process using process_vm_readv(2).
func (m *ProcMem) readVmReadv(buf []byte, addr uint64) error {
	if len(buf) == 0 {
		return nil
	}
	localIov := []unix.Iovec{
		{Base: &buf[0], Len: uint64(len(buf))},
	}
	remoteIov := []unix.RemoteIovec{
		{Base: uintptr(addr), Len: len(buf)},
	}
	n, err := unix.ProcessVMReadv(int(m.pid), localIov, remoteIov, 0)
	if err != nil {
		return fmt.Errorf("process_vm_readv %d bytes at 0x%x from PID %d: %w", len(buf), addr, m.pid, err)
	}
	if n != len(buf) {
		return fmt.Errorf("process_vm_readv short read: got %d bytes, wanted %d at 0x%x", n, len(buf), addr)
	}
	return nil
}

// ReadAt reads len(buf) bytes from the target process at the given virtual address.
//
// First tries /proc/[pid]/mem (if open). If that returns EPERM/EACCES, it
// "sticks" to process_vm_readv for all subsequent reads on this ProcMem so we
// don't keep paying the failed-syscall cost.
func (m *ProcMem) ReadAt(buf []byte, addr uint64) error {
	if m.useVmReadv || m.file == nil {
		return m.readVmReadv(buf, addr)
	}
	n, err := m.file.ReadAt(buf, int64(addr))
	if err != nil {
		if isPermErr(err) {
			// /proc/pid/mem now denies us (likely ptrace_scope=1). Flip to
			// process_vm_readv for this and all future reads.
			m.useVmReadv = true
			return m.readVmReadv(buf, addr)
		}
		return fmt.Errorf("reading %d bytes at 0x%x from PID %d: %w", len(buf), addr, m.pid, err)
	}
	if n != len(buf) {
		return fmt.Errorf("short read: got %d bytes, wanted %d at 0x%x", n, len(buf), addr)
	}
	return nil
}

// ReadUint64 reads a uint64 from the target process at the given address.
func (m *ProcMem) ReadUint64(addr uint64) (uint64, error) {
	var buf [8]byte
	if err := m.ReadAt(buf[:], addr); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(buf[:]), nil
}

// ReadUint32 reads a uint32 from the target process at the given address.
func (m *ProcMem) ReadUint32(addr uint64) (uint32, error) {
	var buf [4]byte
	if err := m.ReadAt(buf[:], addr); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(buf[:]), nil
}

// ReadInt32 reads an int32 from the target process at the given address.
func (m *ProcMem) ReadInt32(addr uint64) (int32, error) {
	v, err := m.ReadUint32(addr)
	return int32(v), err
}

// ReadPtr reads a pointer (uint64 on x86_64) from the target process.
func (m *ProcMem) ReadPtr(addr uint64) (uint64, error) {
	return m.ReadUint64(addr)
}

// ReadPyUnicodeString reads a Python unicode string object at the given address.
// Returns the string content. Uses the compact ASCII fast path when possible.
//
// CPython string layout (PyASCIIObject for compact ASCII, x86_64):
//
//	offset 0:  ob_refcnt (8 bytes)
//	offset 8:  ob_type (8 bytes)
//	offset 16: length (8 bytes)
//	offset 24: hash (8 bytes)
//	offset 32: state (4 bytes) — contains ascii/compact/kind flags
//	offset 48: data starts (for compact ASCII, after wstr pointer + padding)
//
// For non-ASCII strings, the layout is more complex. We only handle
// compact ASCII — this covers the vast majority of Python filenames
// and function names.
func (m *ProcMem) ReadPyUnicodeString(addr uint64, offsets *PyOffsets, maxLen int) (string, error) {
	if addr == 0 {
		return "", nil
	}

	// Read string length.
	length, err := m.ReadUint64(addr + offsets.UnicodeLength)
	if err != nil {
		return "", err
	}

	// Sanity check: Python filenames/function names are typically short.
	if length == 0 || length > 4096 {
		return "", fmt.Errorf("suspicious string length: %d", length)
	}
	if int(length) > maxLen {
		length = uint64(maxLen)
	}

	// Read the state to determine string kind.
	state, err := m.ReadUint32(addr + offsets.UnicodeState)
	if err != nil {
		return "", err
	}

	// Check if compact ASCII (most common for source filenames).
	// CPython PyASCIIObject.state bitfield layout (3.10-3.12, GCC little-endian):
	//   interned:2  (bits 0-1)
	//   kind:3      (bits 2-4)
	//   compact:1   (bit 5)
	//   ascii:1     (bit 6)
	isCompact := (state>>5)&1 == 1
	isASCII := (state>>6)&1 == 1

	if isASCII && isCompact {
		// Compact ASCII: data is inline at a fixed offset from the object start.
		buf := make([]byte, length)
		if err := m.ReadAt(buf, addr+offsets.UnicodeData); err != nil {
			return "", err
		}
		return string(buf), nil
	}

	// Non-compact: data pointer is stored at the data offset.
	dataPtr, err := m.ReadPtr(addr + offsets.UnicodeData)
	if err != nil {
		return "", err
	}
	if dataPtr == 0 {
		return "", nil
	}

	buf := make([]byte, length)
	if err := m.ReadAt(buf, dataPtr); err != nil {
		return "", err
	}
	return string(buf), nil
}
