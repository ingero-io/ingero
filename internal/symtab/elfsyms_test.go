package symtab

import (
	"os/exec"
	"runtime"
	"testing"
)

func TestParseELFSymbols_SystemLib(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("ELF parsing requires Linux")
	}

	// Test with libc — always available on Linux.
	// Use ldd first: it resolves to the actual ELF binary (e.g., libc.so.6).
	// On modern Ubuntu, /usr/lib/.../libc.so is a GNU linker script (not ELF),
	// so find-based approaches fail with "bad magic number".
	out, err := exec.Command("sh", "-c", "ldd /bin/ls | grep libc | awk '{print $3}'").Output()
	if err != nil || len(out) == 0 {
		t.Skip("could not find libc.so via ldd")
	}

	libcPath := string(out[:len(out)-1]) // trim newline

	elfs, err := ParseELFSymbols(libcPath)
	if err != nil {
		t.Fatalf("ParseELFSymbols(%s): %v", libcPath, err)
	}

	if len(elfs.Symbols) == 0 {
		t.Fatal("no symbols found in libc")
	}

	t.Logf("Parsed %d symbols from %s (PIE=%v, BaseVA=0x%x)", len(elfs.Symbols), libcPath, elfs.PIE, elfs.BaseVA)

	// libc should have well-known symbols.
	found := false
	for _, s := range elfs.Symbols {
		if s.Name == "printf" || s.Name == "write" || s.Name == "malloc" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected to find printf, write, or malloc in libc")
	}
}

func TestELFSymbols_Lookup(t *testing.T) {
	// Create a synthetic symbol table for testing.
	elfs := &ELFSymbols{
		PIE:    false,
		BaseVA: 0,
		Symbols: []ELFSymbol{
			{Name: "funcA", Value: 0x1000, Size: 0x100},
			{Name: "funcB", Value: 0x1200, Size: 0x200},
			{Name: "funcC", Value: 0x2000, Size: 0x50},
		},
	}

	tests := []struct {
		offset     uint64
		wantName   string
		wantOffset uint64
	}{
		{0x1000, "funcA", 0},         // exact start
		{0x1050, "funcA", 0x50},      // middle of funcA
		{0x10FF, "funcA", 0xFF},      // last byte of funcA
		{0x1100, "", 0},              // between funcA and funcB (not in range)
		{0x1200, "funcB", 0},         // start of funcB
		{0x1300, "funcB", 0x100},     // middle of funcB
		{0x2000, "funcC", 0},         // start of funcC
		{0x0FFF, "", 0},              // before first symbol
	}

	for _, tt := range tests {
		name, offset := elfs.Lookup(tt.offset)
		if name != tt.wantName {
			t.Errorf("Lookup(0x%x): name = %q, want %q", tt.offset, name, tt.wantName)
		}
		if name != "" && offset != tt.wantOffset {
			t.Errorf("Lookup(0x%x): offset = 0x%x, want 0x%x", tt.offset, offset, tt.wantOffset)
		}
	}
}

func TestELFSymbols_LookupPIE(t *testing.T) {
	// PIE binary: symbol values are relative to BaseVA.
	elfs := &ELFSymbols{
		PIE:    true,
		BaseVA: 0x1000, // first PT_LOAD starts at vaddr 0x1000
		Symbols: []ELFSymbol{
			{Name: "main", Value: 0x2000, Size: 0x100}, // vaddr 0x2000
		},
	}

	// file_offset for "main" = 0x2000 - 0x1000 = 0x1000
	// When we do Lookup(fileOffset), it adds BaseVA back: 0x1000 + 0x1000 = 0x2000
	name, offset := elfs.Lookup(0x1000)
	if name != "main" {
		t.Errorf("PIE Lookup: name = %q, want %q", name, "main")
	}
	if offset != 0 {
		t.Errorf("PIE Lookup: offset = 0x%x, want 0", offset)
	}

	// Offset within main.
	name, offset = elfs.Lookup(0x1050)
	if name != "main" {
		t.Errorf("PIE Lookup: name = %q, want %q", name, "main")
	}
	if offset != 0x50 {
		t.Errorf("PIE Lookup: offset = 0x%x, want 0x50", offset)
	}
}
