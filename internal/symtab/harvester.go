// SPDX-License-Identifier: Apache-2.0

package symtab

import (
	"bufio"
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
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

# --- Find the interpreter frame (_PyInterpreterFrame) pointer and validate it.
# A candidate C from PyFrameObject's bytes is a real interp frame iff reading
# forward from C reveals code_addr (at C + FrameCode). That anchors us to the
# real struct and gives us FrameCode as a byproduct.
interp_frame = 0
frame_code_off = -1
fr_buf = list((ctypes.c_uint64 * 32).from_address(frame_addr))
for fr_val in fr_buf:
    if fr_val < 0x1000: continue
    try:
        cand = (ctypes.c_uint64 * 25).from_address(fr_val)
        for j, w in enumerate(cand):
            if w == code_addr:
                interp_frame = fr_val
                frame_code_off = j * 8
                break
        if interp_frame: break
    except (OSError, ValueError): continue

if frame_code_off >= 0: emit('FrameCode', frame_code_off)

# --- TstateFrame: search widely in tstate for the validated interp_frame ptr.
# tstate can be large (thousands of bytes); scan aggressively.
if interp_frame:
    ts_wide = (ctypes.c_uint64 * 512).from_address(tstate)
    for i in range(512):
        try:
            if ts_wide[i] == interp_frame:
                emit('TstateFrame', i * 8); break
        except (IndexError): break

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
ctypes.pythonapi.PyInterpreterState_Get.restype = ctypes.c_void_p
interp = ctypes.pythonapi.PyInterpreterState_Get()
is_buf = (ctypes.c_uint64 * 128).from_address(interp)
for needle in (tstate, peer[0]):
    if not needle: continue
    for i in range(2, 128):
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
func HarvestOffsets(pythonBinary string) (*HarvestedOffsets, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, pythonBinary, "-c", harvestScript)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("running harvester via %s: %w", pythonBinary, err)
	}

	result := &HarvestedOffsets{}
	scanner := bufio.NewScanner(strings.NewReader(string(out)))
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
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.TstateNativeThreadID = &n
			}
		case "TstateNext":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.TstateNext = &n
			}
		case "InterpTstateHead":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.InterpTstateHead = &n
			}
		case "TstateFrame":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.TstateFrame = &n
			}
		case "FrameBack":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				result.FrameBack = &n
			}
		case "FrameCode":
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
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
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
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
	return result, nil
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
