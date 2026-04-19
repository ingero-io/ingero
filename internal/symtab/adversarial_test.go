//go:build linux

package symtab

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
)

// TestAdversarial_MalformedELFs — force-feed malformed inputs to
// ParseELFSymbols and verify no panic, no OOM, no hang.
func TestAdversarial_MalformedELFs(t *testing.T) {
	tmp := t.TempDir()
	write := func(name string, data []byte) string {
		path := tmp + "/" + name
		if err := os.WriteFile(path, data, 0644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
		return path
	}

	type testCase struct {
		name string
		path string
	}

	cases := []testCase{
		{"empty-file", write("empty.elf", []byte{})},
		{"random-1k", func() string {
			b := make([]byte, 1024)
			for i := range b {
				b[i] = byte(i * 31)
			}
			return write("junk.elf", b)
		}()},
		{"truncated-magic", write("trunc.elf", []byte{0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00})},
	}

	// Huge-section-count claim
	{
		type elfHeader struct {
			Ident     [16]byte
			Type      uint16
			Machine   uint16
			Version   uint32
			Entry     uint64
			Phoff     uint64
			Shoff     uint64
			Flags     uint32
			Ehsize    uint16
			Phentsize uint16
			Phnum     uint16
			Shentsize uint16
			Shnum     uint16
			Shstrndx  uint16
		}
		var buf bytes.Buffer
		hdr := elfHeader{
			Type:      2,  // ET_EXEC
			Machine:   62, // EM_X86_64
			Version:   1,
			Ehsize:    64,
			Phentsize: 56,
			Shentsize: 64,
			Shnum:     0xFFFF, // 65535 sections claimed
			Shoff:     64,
		}
		copy(hdr.Ident[:], []byte{0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00})
		binary.Write(&buf, binary.LittleEndian, &hdr)
		cases = append(cases, testCase{"huge-shnum-claim", write("huge-sh.elf", buf.Bytes())})
	}

	// Large sparse-filled file
	{
		big := make([]byte, 5*1024*1024)
		copy(big[:8], []byte{0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00})
		cases = append(cases, testCase{"5mb-header-only", write("big.elf", big)})
	}

	// Special paths
	cases = append(cases,
		testCase{"dev-null", "/dev/null"},
		testCase{"dev-zero-head", "/dev/zero"},
		testCase{"nonexistent", "/nonexistent/binary"},
		testCase{"directory", tmp},
		testCase{"not-elf", "/proc/self/cmdline"},
	)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Run with panic-recovery so one bad input does not abort the suite.
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("PANIC: %v", r)
				}
			}()
			syms, err := ParseELFSymbols(tc.path)
			if err != nil {
				t.Logf("err: %v", err)
				return
			}
			count := 0
			if syms != nil {
				count = len(syms.Symbols)
			}
			t.Logf("parsed OK: %d symbols", count)
		})
	}
}
