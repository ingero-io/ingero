// SPDX-License-Identifier: Apache-2.0

package symtab

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ingero-io/ingero/internal/procpath"
)

// harvestScript is a self-contained Python program that discovers CPython
// internal struct field offsets at runtime by correlating values known from
// the public API (os.gettid, sys._getframe, id(), etc.) against raw struct
// memory read via ctypes. No debug symbols, no DWARF, no hardcoded tables —
// works on any CPython 3.10/3.11/3.12 regardless of distro patches.
//
// Output: one KEY=value line per discovered offset (stdout). The Go parser
// only accepts offsets it recognizes; unknown keys are ignored. Fields the
// harvester cannot reliably discover are simply omitted and callers fall
// back to DWARF/hardcoded tables for those.
//
// Runtime budget: << 200ms wall time (dominated by Python startup).
const harvestScript = `
import ctypes, os, sys, threading

def emit(k, v): print(f'{k}={v}')

ctypes.pythonapi.PyThreadState_Get.restype = ctypes.c_void_p
tstate = ctypes.pythonapi.PyThreadState_Get()
tid = threading.get_native_id()
frame = sys._getframe()
frame_addr = id(frame)
code = frame.f_code
code_addr = id(code)

emit('PyMinor', sys.version_info.minor)

# Safe-read helper — dereferencing arbitrary pointers from ctypes can
# segfault (unmapped memory). /proc/self/mem returns EIO cleanly.
_mem_fd = os.open('/proc/self/mem', os.O_RDONLY)
_MAX_VADDR = 0x00007fffffffffff  # canonical x86_64 user VA top

def safe_read(addr, n):
    if addr < 0x1000 or addr > _MAX_VADDR:
        return None
    try:
        b = os.pread(_mem_fd, n, addr)
        return b if len(b) == n else None
    except OSError:
        return None

def u64_at(buf, off):
    return int.from_bytes(buf[off:off+8], 'little')

# --- Find the interpreter frame (_PyInterpreterFrame) pointer and validate it.
# A candidate C from PyFrameObject's bytes is a real interp frame iff reading
# forward from C reveals code_addr (at C + FrameCode). That anchors us to the
# real struct and gives us FrameCode as a byproduct.
interp_frame = 0
frame_code_off = -1
fr_buf = list((ctypes.c_uint64 * 32).from_address(frame_addr))
for fr_val in fr_buf:
    if fr_val < 0x1000 or fr_val > _MAX_VADDR: continue
    data = safe_read(fr_val, 200)
    if not data: continue
    for j in range(25):
        if u64_at(data, j*8) == code_addr:
            interp_frame = fr_val
            frame_code_off = j * 8
            break
    if interp_frame: break

if frame_code_off >= 0: emit('FrameCode', frame_code_off)

# --- TstateFrame + OffCframeCurrentFrame discovery.
#
# CPython 3.12 actually stores the top _PyInterpreterFrame DIRECTLY in
# tstate.current_frame (no cframe indirection). The legacy cframe field
# still exists but is not what callers should walk through. CPython 3.11
# does use _PyCFrame indirection (.cframe -> .current_frame).
#
# Heuristic order:
#   1. Direct: scan tstate for a pointer P such that *P+0 looks like a
#      PyCodeObject (i.e., P is itself an interp_frame whose f_code at
#      offset 0 is the valid code pointer). The harvester process's
#      sys._getframe() stack means the live current frame is the harvester
#      script's own frame; its f_code IS code_addr. So *P+0 == code_addr
#      is the test that picks the DIRECT current_frame field.
#   2. Cframe indirection: P is a cframe, and *P+small_off contains
#      interp_frame (used by 3.11).
tstate_frame_off = -1
cframe_cf_off = -1
ts_wide = (ctypes.c_uint64 * 512).from_address(tstate)
if interp_frame:
    # Strategy 1: P is direct interp_frame (P[0] == code_addr)
    for i in range(512):
        try:
            p = ts_wide[i]
        except IndexError: break
        if p < 0x1000 or p > _MAX_VADDR: continue
        head = safe_read(p, 8)
        if not head: continue
        if u64_at(head, 0) == code_addr:
            tstate_frame_off = i * 8
            cframe_cf_off = 0  # direct, no indirection
            break

if tstate_frame_off < 0 and interp_frame:
    # Strategy 2: cframe indirection (CPython 3.11)
    for i in range(512):
        try:
            p = ts_wide[i]
        except IndexError: break
        if p < 0x1000 or p > _MAX_VADDR: continue
        data = safe_read(p, 64)
        if not data: continue
        # Skip offset 0 — that's the "direct" case already tried.
        for j in range(1, 8):
            if u64_at(data, j*8) == interp_frame:
                tstate_frame_off = i * 8
                cframe_cf_off = j * 8
                break
        if tstate_frame_off >= 0: break

if tstate_frame_off >= 0:
    emit('TstateFrame', tstate_frame_off)
    emit('OffCframeCurrentFrame', cframe_cf_off)

# --- FrameBack via nested frame chain ---
if interp_frame:
    def _inner():
        child = sys._getframe()
        child_addr = id(child); child_code = id(child.f_code)
        cb = list((ctypes.c_uint64 * 32).from_address(child_addr))
        for cv in cb:
            if cv < 0x1000 or cv == interp_frame: continue
            try:
                cib = (ctypes.c_uint64 * 25).from_address(cv)
                # Validate: must have child_code at some offset
                if not any(w == child_code for w in cib): continue
                # Now find parent interp_frame in cib
                for j, w in enumerate(cib):
                    if w == interp_frame:
                        emit('FrameBack', j * 8); return
            except (OSError, ValueError): continue
    _inner()

# --- TstateNativeThreadID: search for kernel TID value ---
ts_buf = (ctypes.c_uint64 * 256).from_address(tstate)
for i in range(256):
    try:
        if ts_buf[i] == tid:
            emit('TstateNativeThreadID', i * 8); break
    except IndexError: break

# --- TstateNext via a peer thread ---
peer = [0]
ev = threading.Event(); done = threading.Event()
def _peer():
    peer[0] = ctypes.pythonapi.PyThreadState_Get()
    ev.set(); done.wait(timeout=5)
threading.Thread(target=_peer, daemon=True).start()
ev.wait(timeout=5)
if peer[0]:
    # next/prev are near the top of tstate struct
    for i in range(8):
        try:
            if ts_buf[i] == peer[0]:
                emit('TstateNext', i * 8); break
        except IndexError: break

# --- InterpTstateHead ---
# Scan 1024 slots (8 KiB) to cover PyInterpreterState in 3.13+ where the
# struct grew past ~7 KB; 3.10–3.12 still find the match in the first 128.
ctypes.pythonapi.PyInterpreterState_Get.restype = ctypes.c_void_p
interp = ctypes.pythonapi.PyInterpreterState_Get()
is_buf = (ctypes.c_uint64 * 1024).from_address(interp)
for needle in (tstate, peer[0]):
    if not needle: continue
    for i in range(2, 1024):
        try:
            if is_buf[i] == needle:
                emit('InterpTstateHead', i * 8); break
        except IndexError: break
    else:
        continue
    break

# --- code object offsets ---
co_buf = (ctypes.c_uint64 * 32).from_address(code_addr)
fn_addr = id(code.co_filename); nm_addr = id(code.co_name)
fn_off = nm_off = -1
for i in range(32):
    try:
        if co_buf[i] == fn_addr and fn_off < 0: fn_off = i * 8
        elif co_buf[i] == nm_addr and nm_off < 0 and i*8 != fn_off: nm_off = i * 8
    except IndexError: break
if fn_off >= 0: emit('CodeFilename', fn_off)
if nm_off >= 0: emit('CodeName', nm_off)

co32 = (ctypes.c_uint32 * 64).from_address(code_addr)
for i in range(64):
    try:
        if co32[i] == code.co_firstlineno:
            emit('CodeFirstLineNo', i * 4); break
    except IndexError: break

# --- unicode ---
ustr = '__main__'; us_addr = id(ustr); slen = len(ustr); first = ord(ustr[0])
us64 = (ctypes.c_uint64 * 8).from_address(us_addr)
for i in range(2, 8):
    try:
        if us64[i] == slen:
            emit('UnicodeLength', i * 8); break
    except IndexError: break

us32 = (ctypes.c_uint32 * 16).from_address(us_addr)
for i in range(16):
    try:
        if i*4 < 16 or i*4 > 40: continue
        v = us32[i]
        kind = (v >> 2) & 0x7; compact = (v >> 5) & 1; ascii_bit = (v >> 6) & 1
        if compact and ascii_bit and kind == 1:
            emit('UnicodeState', i*4); break
    except IndexError: break

us8 = (ctypes.c_ubyte * 64).from_address(us_addr)
for i in range(32, 56):
    try:
        if us8[i] == first and (slen < 2 or us8[i+1] == ord(ustr[1])):
            emit('UnicodeData', i); break
    except IndexError: break

done.set()
`

// HarvestedOffsets holds the subset of PyOffsets fields discovered by the
// runtime harvester. Each pointer is nil when the harvester couldn't
// discover that field; the caller overlays non-nil values onto a fallback
// offset table.
type HarvestedOffsets struct {
	PyMinor                 *int
	TstateNativeThreadID    *uint64
	TstateNext              *uint64
	InterpTstateHead        *uint64
	TstateFrame             *uint64
	OffCframeCurrentFrame   *uint64 // 0 = direct (3.13+); >0 = cframe indirection (3.11/3.12)
	FrameBack               *uint64
	FrameCode               *uint64
	CodeFilename            *uint64
	CodeName                *uint64
	CodeFirstLineNo         *uint64
	UnicodeLength           *uint64
	UnicodeState            *uint64
	UnicodeData             *uint64
}

// HarvestOffsets spawns the given python binary with the embedded harvester
// script, parses the KEY=value output, and returns the discovered offsets.
//
// Timeout is intentionally tight (3s) — the harvester does no real work,
// just Python startup + a dozen ctypes reads. If the subprocess hangs or
// the binary is wrong, we fall back quickly.
//
// Errors are returned for genuine failures (binary missing, subprocess
// non-zero exit with stderr). Empty stdout returns (&HarvestedOffsets{}, nil)
// so the caller can proceed with fallbacks.
//
// pid is the target process whose Python interpreter to impersonate. When
// ingero runs in a container (our /proc/self/ns/mnt differs from the
// target's) and we have root, we chroot the subprocess into /proc/<pid>/root
// so ld-linux finds the target's libpython + stdlib, not ingero's container
// filesystem (which typically lacks Python entirely). Requires euid 0.
// If pid <= 0, or we're not root, or the namespaces match, or the chroot
// setup fails at any step, we fall through to a plain subprocess — the
// same behavior as before this flag existed.
//
// LD_LIBRARY_PATH / LD_PRELOAD / LD_* are stripped from the child env when
// chrooting, since the caller's ld-linux hints reference the caller's
// namespace and won't be valid under the new root.
func HarvestOffsets(pythonBinary string, pid int) (*HarvestedOffsets, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	// If not running inside a chroot, the target-namespace path may not be
	// openable from our own namespace. Try to translate — but only when we
	// *won't* chroot below (in which case the child needs the target-
	// namespace path directly).
	effectiveBinary := pythonBinary
	useChroot := shouldChrootForPID(pid)
	if !useChroot {
		effectiveBinary = procpath.ResolveContainerPath(pid, pythonBinary)
	}

	cmd := exec.CommandContext(ctx, effectiveBinary, "-c", harvestScript)
	if useChroot {
		cmd.SysProcAttr = &syscall.SysProcAttr{
			Chroot: fmt.Sprintf("/proc/%d/root", pid),
		}
		cmd.Dir = "/" // chroot is applied after Dir; cwd must exist post-chroot
		cmd.Env = filterLDEnv(os.Environ())
	}

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("running harvester via %s: %w", effectiveBinary, err)
	}

	return parseHarvesterOutput(out), nil
}

// parseHarvesterOutput parses KEY=value lines emitted by the harvester
// subprocess and returns the discovered offsets. Extracted from
// HarvestOffsets so tests can exercise the parse + sanity-cap logic
// without spawning a Python subprocess.
func parseHarvesterOutput(out []byte) *HarvestedOffsets {
	result := &HarvestedOffsets{}
	scanner := bufio.NewScanner(strings.NewReader(string(out)))
	// Sanity caps on numeric offsets: the harvester's pointer-chasing
	// heuristics can match plausibly-valid but wrong offsets on certain
	// CPython builds (seen on uv-distributed 3.11.15 / 3.12.13 where
	// TstateFrame returned 240 against a true value of 56, and
	// InterpTstateHead returned 1048 against a true value of 16).
	// Values above the caps are dropped so the fallback table survives
	// the overlay. Caps are generous, picked from struct sizes plus
	// headroom; fields within cap are trusted.
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		switch k {
		case "PyMinor":
			if n, err := strconv.Atoi(v); err == nil {
				result.PyMinor = &n
			}
		case "TstateNativeThreadID":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil && n <= 512 {
				result.TstateNativeThreadID = &n
			}
		case "TstateNext":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.TstateNext = &n
			}
		case "InterpTstateHead":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil && n <= 1024 {
				result.InterpTstateHead = &n
			}
		case "TstateFrame":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil && n <= 256 {
				result.TstateFrame = &n
			}
		case "OffCframeCurrentFrame":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.OffCframeCurrentFrame = &n
			}
		case "FrameBack":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.FrameBack = &n
			}
		case "FrameCode":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil && n <= 512 {
				result.FrameCode = &n
			}
		case "CodeFilename":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.CodeFilename = &n
			}
		case "CodeName":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.CodeName = &n
			}
		case "CodeFirstLineNo":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil && n <= 256 {
				result.CodeFirstLineNo = &n
			}
		case "UnicodeLength":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.UnicodeLength = &n
			}
		case "UnicodeState":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.UnicodeState = &n
			}
		case "UnicodeData":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.UnicodeData = &n
			}
		}
	}
	return result
}

// Overlay copies harvested offsets onto an existing PyOffsets table,
// preserving fallback values for fields the harvester didn't discover.
// Safe to call with a nil harvested argument (returns base unchanged).
func (h *HarvestedOffsets) Overlay(base *PyOffsets) *PyOffsets {
	if h == nil || base == nil {
		return base
	}
	out := *base // copy
	if h.TstateNativeThreadID != nil {
		out.TstateNativeThreadID = *h.TstateNativeThreadID
	}
	if h.TstateNext != nil {
		out.TstateNext = *h.TstateNext
	}
	if h.InterpTstateHead != nil {
		out.InterpTstateHead = *h.InterpTstateHead
	}
	if h.TstateFrame != nil {
		out.TstateFrame = *h.TstateFrame
	}
	if h.OffCframeCurrentFrame != nil {
		out.CframeCurrentFrame = *h.OffCframeCurrentFrame
	}
	if h.FrameBack != nil {
		out.FrameBack = *h.FrameBack
	}
	if h.FrameCode != nil {
		out.FrameCode = *h.FrameCode
	}
	if h.CodeFilename != nil {
		out.CodeFilename = *h.CodeFilename
	}
	if h.CodeName != nil {
		out.CodeName = *h.CodeName
	}
	if h.CodeFirstLineNo != nil {
		out.CodeFirstLineNo = *h.CodeFirstLineNo
	}
	if h.UnicodeLength != nil {
		out.UnicodeLength = *h.UnicodeLength
	}
	if h.UnicodeState != nil {
		out.UnicodeState = *h.UnicodeState
	}
	if h.UnicodeData != nil {
		out.UnicodeData = *h.UnicodeData
	}
	out.Version = out.Version + "+harvested"
	return &out
}

// shouldChrootForPID reports whether the harvester subprocess should chroot
// into /proc/<pid>/root to inherit the target's filesystem view. True only
// when pid > 0, euid is 0 (chroot requires it), /proc/<pid>/root exists,
// and the target's mount namespace is distinct from ingero's own (compared
// via /proc/self/ns/mnt vs /proc/<pid>/ns/mnt readlinks). Any step failing
// returns false so the plain-exec fallback takes over.
func shouldChrootForPID(pid int) bool {
	if pid <= 0 || os.Geteuid() != 0 {
		return false
	}
	if _, err := os.Stat(fmt.Sprintf("/proc/%d/root", pid)); err != nil {
		return false
	}
	self, err := os.Readlink("/proc/self/ns/mnt")
	if err != nil {
		return false
	}
	target, err := os.Readlink(fmt.Sprintf("/proc/%d/ns/mnt", pid))
	if err != nil {
		return false
	}
	return self != target
}

// filterLDEnv returns env minus LD_LIBRARY_PATH, LD_PRELOAD, and any other
// LD_* variable — these reference the caller's namespace and will mislead
// the child's ld-linux after a chroot into the target's root.
func filterLDEnv(env []string) []string {
	out := env[:0:0]
	for _, kv := range env {
		if strings.HasPrefix(kv, "LD_") {
			continue
		}
		out = append(out, kv)
	}
	return out
}
