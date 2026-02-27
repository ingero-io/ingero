// Package symtab provides userspace symbol resolution for stack traces.
//
// It resolves raw instruction pointers (from bpf_get_stack) into human-readable
// symbol names by parsing /proc/[pid]/maps and ELF symbol tables.
package symtab

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// MapRegion represents a memory-mapped region from /proc/[pid]/maps.
//
// /proc/[pid]/maps format (one line per region):
//
//	7f1234000000-7f1234100000 r-xp 00001000 08:01 12345  /usr/lib/libfoo.so
//	^start        ^end         ^perms ^offset ^dev ^inode ^pathname
//
// For symbol resolution, we need: start, end, offset, and pathname.
// The file offset is crucial for PIE (Position Independent Executables) / ASLR:
//
//	file_offset_of_ip = ip - region.Start + region.Offset
type MapRegion struct {
	Start  uint64 // virtual address start
	End    uint64 // virtual address end (exclusive)
	Offset uint64 // file offset of this mapping
	Perms  string // permissions (r-xp, rw-p, etc.)
	Path   string // file path (empty for anonymous mappings)
}

// IsExecutable returns true if this region has execute permission.
func (r MapRegion) IsExecutable() bool {
	return len(r.Perms) >= 3 && r.Perms[2] == 'x'
}

// Contains returns true if addr falls within this region.
func (r MapRegion) Contains(addr uint64) bool {
	return addr >= r.Start && addr < r.End
}

// ParseProcMaps reads and parses /proc/[pid]/maps.
// Returns only file-backed executable regions (the ones relevant for symbol resolution).
func ParseProcMaps(pid uint32) ([]MapRegion, error) {
	path := fmt.Sprintf("/proc/%d/maps", pid)
	return parseMapsFile(path)
}

// parseMapsFile parses a /proc/pid/maps file (or any file in that format).
func parseMapsFile(path string) ([]MapRegion, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	defer f.Close()

	var regions []MapRegion
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		r, ok := parseMapsLine(line)
		if !ok {
			continue
		}
		// Only keep file-backed executable regions.
		if r.Path != "" && r.IsExecutable() {
			regions = append(regions, r)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}
	return regions, nil
}

// parseMapsLine parses a single line from /proc/pid/maps.
func parseMapsLine(line string) (MapRegion, bool) {
	// Minimum fields: address range, perms, offset, dev, inode
	// Optional: pathname (field 6+)
	fields := strings.Fields(line)
	if len(fields) < 5 {
		return MapRegion{}, false
	}

	// Parse address range: "7f1234000000-7f1234100000"
	addrParts := strings.SplitN(fields[0], "-", 2)
	if len(addrParts) != 2 {
		return MapRegion{}, false
	}
	start, err := strconv.ParseUint(addrParts[0], 16, 64)
	if err != nil {
		return MapRegion{}, false
	}
	end, err := strconv.ParseUint(addrParts[1], 16, 64)
	if err != nil {
		return MapRegion{}, false
	}

	// Permissions: "r-xp"
	perms := fields[1]

	// File offset: "00001000"
	offset, err := strconv.ParseUint(fields[2], 16, 64)
	if err != nil {
		return MapRegion{}, false
	}

	// Pathname (field 6+ joined, may contain spaces).
	var path string
	if len(fields) >= 6 {
		path = strings.Join(fields[5:], " ")
		// Skip pseudo-paths like [stack], [heap], [vdso], etc.
		if strings.HasPrefix(path, "[") {
			path = ""
		}
	}

	return MapRegion{
		Start:  start,
		End:    end,
		Offset: offset,
		Perms:  perms,
		Path:   path,
	}, true
}

// FindRegion returns the MapRegion containing addr, or nil if not found.
// Regions are sorted by Start address (from /proc/maps).
// Uses binary search: O(log n) instead of O(n) per stack frame lookup.
func FindRegion(regions []MapRegion, addr uint64) *MapRegion {
	// Binary search for the last region whose Start <= addr.
	// sort.Search finds the first index where regions[i].Start > addr,
	// so the candidate is at index-1.
	i := sort.Search(len(regions), func(i int) bool {
		return regions[i].Start > addr
	})
	if i == 0 {
		return nil // addr is below all regions
	}
	r := &regions[i-1]
	if r.Contains(addr) {
		return r
	}
	return nil
}
