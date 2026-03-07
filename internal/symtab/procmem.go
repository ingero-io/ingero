package symtab

import (
	"encoding/binary"
	"fmt"
	"os"
)

// ProcMem provides read access to a target process's memory via /proc/[pid]/mem.
// Requires CAP_SYS_PTRACE or root — same privilege level as eBPF.
type ProcMem struct {
	pid  uint32
	file *os.File
}

// OpenProcMem opens /proc/[pid]/mem for reading.
func OpenProcMem(pid uint32) (*ProcMem, error) {
	path := fmt.Sprintf("/proc/%d/mem", pid)
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	return &ProcMem{pid: pid, file: f}, nil
}

// Close releases the file handle.
func (m *ProcMem) Close() error {
	return m.file.Close()
}

// ReadAt reads len(buf) bytes from the target process at the given virtual address.
func (m *ProcMem) ReadAt(buf []byte, addr uint64) error {
	n, err := m.file.ReadAt(buf, int64(addr))
	if err != nil {
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
