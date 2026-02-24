package symtab

import (
	"debug/elf"
	"fmt"
	"sort"
)

// ELFSymbol represents a function symbol from an ELF binary.
type ELFSymbol struct {
	Name  string
	Value uint64 // virtual address (for non-PIE) or file offset (for PIE)
	Size  uint64
}

// ELFSymbols holds parsed symbols from an ELF file, sorted for binary search.
type ELFSymbols struct {
	Path    string
	Symbols []ELFSymbol
	PIE     bool   // Position Independent Executable (needs base address adjustment)
	BaseVA  uint64 // Base virtual address from first PT_LOAD segment (for PIE adjustment)
}

// ParseELFSymbols reads function symbols from an ELF file's .symtab and .dynsym.
// Returns symbols sorted by address for O(log n) lookup.
func ParseELFSymbols(path string) (*ELFSymbols, error) {
	f, err := elf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening ELF %s: %w", path, err)
	}
	defer f.Close()

	result := &ELFSymbols{Path: path}

	// Determine if PIE (shared library or PIE executable).
	// ET_DYN = shared object (includes PIE executables and .so files).
	result.PIE = f.Type == elf.ET_DYN

	// Find the base virtual address from the first PT_LOAD segment.
	// For PIE/shared libraries, symbols are relative to this base.
	for _, prog := range f.Progs {
		if prog.Type == elf.PT_LOAD && prog.Flags&elf.PF_X != 0 {
			result.BaseVA = prog.Vaddr - prog.Off
			break
		}
	}

	// Collect function symbols from both .symtab and .dynsym.
	var allSyms []ELFSymbol

	// .symtab — full symbol table (may be stripped in release builds).
	if syms, err := f.Symbols(); err == nil {
		for _, s := range syms {
			if elf.ST_TYPE(s.Info) == elf.STT_FUNC && s.Value > 0 {
				allSyms = append(allSyms, ELFSymbol{
					Name:  s.Name,
					Value: s.Value,
					Size:  s.Size,
				})
			}
		}
	}

	// .dynsym — dynamic symbol table (survives stripping).
	if dynsyms, err := f.DynamicSymbols(); err == nil {
		for _, s := range dynsyms {
			if elf.ST_TYPE(s.Info) == elf.STT_FUNC && s.Value > 0 {
				allSyms = append(allSyms, ELFSymbol{
					Name:  s.Name,
					Value: s.Value,
					Size:  s.Size,
				})
			}
		}
	}

	if len(allSyms) == 0 {
		return nil, fmt.Errorf("no function symbols found in %s", path)
	}

	// Deduplicate by address (same symbol may appear in both tables).
	seen := make(map[uint64]bool)
	deduped := allSyms[:0]
	for _, s := range allSyms {
		if !seen[s.Value] {
			seen[s.Value] = true
			deduped = append(deduped, s)
		}
	}

	// Sort by address for binary search.
	sort.Slice(deduped, func(i, j int) bool {
		return deduped[i].Value < deduped[j].Value
	})

	result.Symbols = deduped
	return result, nil
}

// FindSymbolByName looks up a specific symbol by name in an ELF file,
// searching both .symtab and .dynsym. Unlike ParseELFSymbols() which only
// returns STT_FUNC symbols for stack resolution, this searches all symbol
// types (STT_FUNC, STT_OBJECT, STT_NOTYPE).
//
// This exists because _PyRuntime is an STT_OBJECT (data symbol), invisible
// to ParseELFSymbols(). Adding data symbols to the sorted function array
// would pollute the hot-path Lookup() used on every event.
//
// Returns the symbol value (virtual address), whether the binary is PIE,
// the base virtual address, and whether the symbol was found.
func FindSymbolByName(path string, name string) (value uint64, pie bool, baseVA uint64, found bool) {
	f, err := elf.Open(path)
	if err != nil {
		return 0, false, 0, false
	}
	defer f.Close()

	pie = f.Type == elf.ET_DYN

	// Find base VA from first executable PT_LOAD.
	for _, prog := range f.Progs {
		if prog.Type == elf.PT_LOAD && prog.Flags&elf.PF_X != 0 {
			baseVA = prog.Vaddr - prog.Off
			break
		}
	}

	// Search .symtab first (full symbol table, may be stripped).
	if syms, err := f.Symbols(); err == nil {
		for _, s := range syms {
			if s.Name == name && s.Value > 0 {
				return s.Value, pie, baseVA, true
			}
		}
	}

	// Search .dynsym (survives stripping).
	if dynsyms, err := f.DynamicSymbols(); err == nil {
		for _, s := range dynsyms {
			if s.Name == name && s.Value > 0 {
				return s.Value, pie, baseVA, true
			}
		}
	}

	return 0, pie, baseVA, false
}

// Lookup finds the symbol containing the given file offset.
// Uses binary search: O(log n) per lookup.
//
// For PIE binaries: fileOffset = ip - region.Start + region.Offset
// The symbol value is relative to BaseVA, so we compare against:
//
//	fileOffset + BaseVA
//
// For non-PIE: fileOffset is the absolute virtual address.
func (es *ELFSymbols) Lookup(fileOffset uint64) (string, uint64) {
	syms := es.Symbols
	if len(syms) == 0 {
		return "", 0
	}

	// For PIE, adjust the lookup address.
	lookupAddr := fileOffset
	if es.PIE {
		lookupAddr = fileOffset + es.BaseVA
	}

	// Binary search: find the last symbol with Value <= lookupAddr.
	idx := sort.Search(len(syms), func(i int) bool {
		return syms[i].Value > lookupAddr
	})
	idx-- // The symbol just before the first one that's too high.

	if idx < 0 {
		return "", 0
	}

	sym := syms[idx]

	// Verify the address falls within the symbol's range.
	// If Size is known and > 0, check bounds. If Size is 0 (common for .dynsym),
	// accept any address up to the next symbol.
	if sym.Size > 0 && lookupAddr >= sym.Value+sym.Size {
		return "", 0
	}

	offset := lookupAddr - sym.Value
	return sym.Name, offset
}
