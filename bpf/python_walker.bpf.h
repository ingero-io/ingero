// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/*
 * python_walker.bpf.h — inline CPython 3.9..3.14 frame walker for eBPF.
 *
 * Single source of truth for the in-kernel Python frame walker. Included
 * by cuda_trace.bpf.c so the walker runs inline on the same probe that
 * emits the CUDA event — no tail call, no per-program map coordination.
 *
 * Version dispatch: walk_python_frames() reads py_runtime_state.python_minor
 * from the per-PID py_runtime_map and routes to one of three variants:
 *   - 3.9/3.10: PyThreadState.frame -> PyFrameObject*, walk via f_back/f_code.
 *   - 3.11:     PyThreadState.cframe -> _PyCFrame*, then cframe->current_frame
 *               -> _PyInterpreterFrame*, walk via previous/f_code.
 *   - 3.12:     PyThreadState.current_frame -> _PyInterpreterFrame* directly
 *               (or via cframe on distro builds that retain indirection —
 *                harvester-discovered off_cframe_current_frame selects).
 *   - 3.13:     same as 3.12 direct (_PyCFrame dropped upstream).
 *   - 3.14:     same frame layout as 3.13, but f_executable is a _PyStackRef
 *               tagged union — the walker masks low 3 bits off code_ptr.
 * python_minor==0 (legacy 32-byte struct writers) is treated as 3.12 for
 * backward compatibility with pre-v2 clients.
 *
 * NOTE: After editing this file on Linux, run `make generate` so the
 * bpf2go-generated Go bindings pick up any BTF-exported struct changes.
 *
 * Architectural choice (2026-04 refactor):
 *   - The walker was originally authored as its own BPF program
 *     (python_trace.bpf.c) that would be loaded by an internal/ebpf/pytrace
 *     Go package. We collapsed that into a header because the walker has
 *     no probes of its own — it is only called from other probes — and
 *     because BPF maps defined in two separate compiled .o files become
 *     two distinct kernel maps (they are NOT auto-shared across bpf2go
 *     objects). Keeping the map definition in exactly one .c file
 *     (cuda_trace.bpf.c) guarantees a single kernel map and lets the
 *     pytrace Go package access it via the cuda tracer's loaded
 *     objects (t.objs.PyRuntimeMap).
 *
 *   - Consequence: this header MUST be included in exactly one .c file
 *     that will compile into its own .o (the "owner"). Today that owner
 *     is cuda_trace.bpf.c. If additional BPF programs need the walker,
 *     they should either (a) include this header and accept that they
 *     will each get their own private py_runtime_map + py_scratch, or
 *     (b) use BPF map pinning / BPF_F_EXPORTED to share across programs.
 *
 * Generator / coroutine frames: the walker handles these implicitly.
 * _PyInterpreterFrame carries an `owner` byte distinguishing thread-
 * allocated, generator-embedded, PyFrameObject-owned, and C-stack
 * entry frames, but `owner` only describes who allocated the storage,
 * not whether the frame is in the `tstate.current_frame -> previous`
 * chain. A currently-executing generator or coroutine has its embedded
 * frame linked into that chain exactly like a regular frame; only
 * suspended generators are detached, and no cuda can fire from one by
 * definition. So `async def inner(): await cudaop()` and generator
 * `yield from` chains walk correctly without special-case code. See
 * tests/workloads/async_cuda.py for the verification workload.
 *
 * Verifier safety:
 *   - All loops are #pragma unroll with compile-time bounds.
 *   - Every bpf_probe_read_user return value is checked.
 *   - No recursion, no unbounded pointer chases.
 *   - Result buffer (4232 bytes) exceeds the 512-byte BPF stack limit;
 *     callers MUST obtain scratch space via bpf_map_lookup_elem on
 *     py_scratch (per-CPU array) rather than stack-allocating it.
 */

#ifndef __INGERO_PYTHON_WALKER_BPF_H
#define __INGERO_PYTHON_WALKER_BPF_H

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "common.bpf.h"

#define PY_MAX_THREADS 8
#define PY_MAX_INTERPRETERS 4   /* bound the subinterpreter walk for the verifier */
#define PY_STRING_MAX 127  /* leave 1 byte for NUL */

/*
 * py_runtime_map: per-PID Python runtime state. Pushed from userspace
 * when a Python 3.12 process is detected (typically on process_exec).
 *
 * Canonical location: this header. The map is instantiated in whichever
 * .c file includes this header. Today that is cuda_trace.bpf.c only —
 * so `t.objs.PyRuntimeMap` on the CUDA tracer is the single Go-side
 * handle for userspace writers (including internal/ebpf/pytrace).
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 256);
	__type(key, __u32);   /* pid */
	__type(value, struct py_runtime_state);
} py_runtime_map SEC(".maps");

/*
 * py_scratch: per-CPU scratch buffer for py_walk_result. The result
 * struct (~4232 bytes) cannot live on the 512-byte BPF stack, so
 * callers of walk_python_frames() should fetch a scratch buffer via
 *     __u32 zero = 0;
 *     struct py_walk_result *r = bpf_map_lookup_elem(&py_scratch, &zero);
 * and pass it to walk_python_frames(). One entry per CPU keeps accesses
 * race-free since BPF programs run with preemption disabled.
 */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct py_walk_result);
} py_scratch SEC(".maps");

/* Force BTF emission for exported types (used by Go userspace). */
static const struct py_runtime_state *_unused_py_runtime_state __attribute__((unused));
static const struct py_walk_result *_unused_py_walk_result __attribute__((unused));
static const struct py_frame *_unused_py_frame __attribute__((unused));

/*
 * py_debug_stats: per-CPU counters for diagnosing why the walker isn't
 * emitting frames. Slots:
 *   [0] entered_dispatcher          (walker called)
 *   [1] state_lookup_ok             (per-PID state found in py_runtime_map)
 *   [2] entered_312                 (3.12 variant entered)
 *   [3] read_interp_ok              (first read of interpreters_head succeeded)
 *   [4] read_threads_head_ok        (read of threads.head succeeded)
 *   [5] thread_loop_iterations      (sum of iterations across all calls)
 *   [6] thread_match_found          (find_thread_state returned non-zero)
 *   [7] frame_loop_first_iteration  (entered the frame walk loop at least once)
 *   [8] depth_gt_zero               (walker returned with depth > 0)
 *   [9] read_first_native_tid       (read of first tstate's native_thread_id ok)
 *   [29] unicode_non_compact_skipped (read_compact_ascii returned 0 because
 *                                    the target PyUnicodeObject is not a
 *                                    compact-ASCII string; emitted py_frame
 *                                    has an empty filename or funcname)
 */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 32);
	__type(key, __u32);
	__type(value, __u64);
} py_debug_stats SEC(".maps");

static __always_inline void py_debug_inc(__u32 slot) {
	__u64 *c = bpf_map_lookup_elem(&py_debug_stats, &slot);
	if (c) (*c)++;
}

/*
 * read_compact_ascii: reads a CPython compact-ASCII string into dst.
 * Returns 1 on success, 0 on any failure or non-compact string.
 *
 * We only support compact ASCII to keep the BPF program bounded:
 *   - Compact strings have inline data at unicode_addr + off_unicode_data.
 *   - Non-compact strings require a second pointer chase. Full support
 *     means reading PyCompactUnicodeObject.utf8 (or PyUnicodeObject.data
 *     on older layouts), which differs per CPython version and adds
 *     verifier complexity we would rather not carry until there is a
 *     concrete workload that needs it. For now the function increments
 *     py_debug_stats[29] on rejection so operators can distinguish
 *     "silently truncated because non-compact" from "walker bug".
 */
static __always_inline int read_compact_ascii(
    void *dst, __u32 dst_size,
    __u64 unicode_addr,
    const struct py_runtime_state *st)
{
	if (unicode_addr == 0)
		return 0;

	/* Read the state flag word. Compact ASCII has the compact bit and
	 * the ascii bit set in the state bitfield. */
	__u32 state = 0;
	if (bpf_probe_read_user(&state, sizeof(state),
	    (const void *)(unicode_addr + st->off_unicode_state)) != 0)
		return 0;

	/* Bit layout in PyASCIIObject.state (CPython 3.12):
	 *   bits 0-1: interned (2 bits)
	 *   bits 2-4: kind (3 bits)
	 *   bit 5:    compact
	 *   bit 6:    ascii
	 *   bit 7:    ready (removed in 3.12 — always 1)
	 *
	 * We require both compact (1<<5) and ascii (1<<6).
	 */
	__u32 mask = (1U << 5) | (1U << 6);
	if ((state & mask) != mask) {
		((char *)dst)[0] = 0;
		py_debug_inc(29);  /* unicode_non_compact_skipped */
		return 0;
	}

	/* Inline data at unicode_addr + off_unicode_data. Read up to
	 * dst_size bytes via bpf_probe_read_user_str (NUL-terminates and
	 * bounds the read). */
	long n = bpf_probe_read_user_str(
	    dst, dst_size,
	    (const void *)(unicode_addr + st->off_unicode_data));
	if (n <= 0) {
		((char *)dst)[0] = 0;
		return 0;
	}
	return 1;
}

/*
 * find_thread_state: walks the PyInterpreterState chain, and within
 * each interpreter the PyThreadState linked list, to find one matching
 * the given native thread id. Returns the tstate address, or 0.
 *
 * Subinterpreters (PEP 684): the outer loop iterates
 * PyInterpreterState.next up to PY_MAX_INTERPRETERS. Single-interpreter
 * processes see next==NULL on the second iteration and exit early, so
 * there is no behavior change vs the prior single-interp implementation.
 *
 * Bounded by PY_MAX_THREADS per interpreter and PY_MAX_INTERPRETERS
 * across interpreters. Workloads with more Python threads or
 * subinterpreters than those bounds will miss frames for the overflow
 * (acceptable trade-off for verifier safety).
 *
 * Single-thread fallback: on some builds (e.g., Ubuntu 22.04's patched
 * CPython 3.10) PyThreadState.native_thread_id is 0 for the main thread
 * because the interpreter never populates it for the thread that called
 * Py_Initialize(). When the outer walk finishes without finding a
 * native_tid match and exactly one tstate was observed across all
 * interpreters, we fall back to that single tstate. For the common
 * case of single-threaded single-interpreter Python workloads this
 * yields correct frames; multi-threaded workloads with a
 * genuinely-mismatched offset correctly return 0.
 */
static __always_inline __u64 find_thread_state(
    __u32 native_tid, const struct py_runtime_state *st)
{
	/* interpreters_head -> first PyInterpreterState */
	__u64 interp = 0;
	if (bpf_probe_read_user(&interp, sizeof(interp),
	    (const void *)(st->runtime_addr + st->off_runtime_interpreters_head)) != 0)
		return 0;
	py_debug_inc(3);  /* read_interp_ok */
	if (interp == 0)
		return 0;

	__u64 first_tstate = 0;
	int walked = 0;

	#pragma unroll(4)
	for (int ii = 0; ii < PY_MAX_INTERPRETERS; ii++) {
		if (interp == 0)
			break;

		/* threads.head -> first PyThreadState for this interpreter */
		__u64 tstate = 0;
		if (bpf_probe_read_user(&tstate, sizeof(tstate),
		    (const void *)(interp + st->off_tstate_head)) != 0)
			break;
		if (ii == 0)
			py_debug_inc(4);  /* read_threads_head_ok (first interp only) */

		/* Walk this interpreter's thread list (bounded). */
		#pragma unroll(8)
		for (int i = 0; i < PY_MAX_THREADS; i++) {
			if (tstate == 0)
				break;
			if (first_tstate == 0)
				first_tstate = tstate;
			walked++;
			py_debug_inc(5);  /* thread_loop_iterations */

			__u64 cur_tid = 0;
			if (bpf_probe_read_user(&cur_tid, sizeof(cur_tid),
			    (const void *)(tstate + st->off_tstate_native_tid)) != 0)
				break;
			if (ii == 0 && i == 0)
				py_debug_inc(9);  /* read_first_native_tid */

			if ((__u32)cur_tid == native_tid) {
				py_debug_inc(6);  /* thread_match_found */
				return tstate;
			}

			__u64 next = 0;
			if (bpf_probe_read_user(&next, sizeof(next),
			    (const void *)(tstate + st->off_tstate_next)) != 0)
				break;
			tstate = next;
		}

		/* Advance to next PyInterpreterState; NULL exits the outer loop. */
		__u64 next_interp = 0;
		if (bpf_probe_read_user(&next_interp, sizeof(next_interp),
		    (const void *)(interp + st->off_interp_next)) != 0)
			break;
		interp = next_interp;
	}

	if (walked == 1 && first_tstate != 0) {
		py_debug_inc(18);  /* single_thread_fallback */
		return first_tstate;
	}
	return 0;
}

/*
 * walk_python_frames_312: CPython 3.12/3.13/3.14 frame walker.
 *
 * Shared across 3.12, 3.13, and 3.14 because the frame-walk field layout
 * is the same (f_executable at 0, previous at 8; only the surrounding
 * struct sizes and per-version offsets in py_runtime_state differ). In
 * 3.12+ PyThreadState.current_frame is a DIRECT _PyInterpreterFrame*
 * (no cframe indirection — that was 3.11's design). Each frame has:
 *   - .f_code/.f_executable (PyCodeObject*, off_frame_code; 3.14 wraps
 *     this in a _PyStackRef tagged-pointer union — see mask below)
 *   - .previous             (walk pointer, off_frame_back)
 *
 * At uprobe time inside a C extension call (e.g. cudaMalloc), CPython
 * pushes a stack-allocated "entry frame stub" on top. The stub may have
 * uninitialized garbage in the f_code slot (low bits of an unrelated
 * value), so checking `code_ptr != 0` is too weak. We require code_ptr
 * to be a userspace heap pointer (>= 0x100000 and below the canonical
 * x86_64 user VA top) and walk .previous over stubs until we find a
 * real Python frame.
 *
 * _PyStackRef tag bits (3.14): 3.14 changed f_executable from a raw
 * PyObject* to a _PyStackRef union whose low bits carry a tag
 * (Py_TAG_REFCNT=1 in default GIL build, Py_INT_TAG=3 for tagged ints).
 * Masking `code_ptr & ~0x7ULL` strips those tags. This is a safe no-op
 * for 3.10–3.13 because PyCodeObject allocations are always 8-byte
 * aligned, so the low 3 bits are already zero.
 *
 * Preconditions:
 *   - result is non-NULL and already zeroed (depth=0, truncated=0).
 *   - st is the per-PID py_runtime_state already looked up by the
 *     dispatcher; we skip a redundant map lookup here.
 */
static __always_inline int walk_python_frames_312(
    __u32 pid, __u32 native_tid,
    const struct py_runtime_state *st,
    struct py_walk_result *result)
{
	py_debug_inc(2);  /* entered_312 */
	__u64 tstate = find_thread_state(native_tid, st);
	if (tstate == 0)
		return 0;  /* not an error — thread may not exist yet */

	/* Get current frame from matched tstate. */
	__u64 frame = 0;
	if (bpf_probe_read_user(&frame, sizeof(frame),
	    (const void *)(tstate + st->off_tstate_frame)) != 0)
		return 0;

	/* Walk frame chain (bounded). emit-count is bounded by PY_MAX_FRAMES;
	 * traversal-count is bounded separately (we may skip stub frames). */
	int emitted = 0;
	#pragma unroll(16)
	for (int walk_iter = 0; walk_iter < 16; walk_iter++) {
		if (frame == 0 || emitted >= PY_MAX_FRAMES)
			break;
		if (walk_iter == 0)
			py_debug_inc(7);  /* frame_loop_first_iteration */

		/* frame->f_code -> PyCodeObject */
		__u64 code_ptr = 0;
		if (bpf_probe_read_user(&code_ptr, sizeof(code_ptr),
		    (const void *)(frame + st->off_frame_code)) != 0)
			break;

		/* Strip _PyStackRef tag bits. See function doc: safe no-op on
		 * 3.10–3.13, required on 3.14 where f_executable is a tagged
		 * union. Must happen before the range check so a tagged-but-
		 * valid pointer isn't skipped. */
		code_ptr &= ~0x7ULL;

		/* frame->previous (read before deciding so we can skip stubs). */
		__u64 prev = 0;
		if (bpf_probe_read_user(&prev, sizeof(prev),
		    (const void *)(frame + st->off_frame_back)) != 0)
			break;

		/* Heap-range check: stubs (entry frames for C calls) live on the
		 * C stack with garbage in f_code. Real PyCodeObjects are
		 * heap-allocated, well above any static binary mapping. */
		if (code_ptr <= 0x100000 || code_ptr >= 0x800000000000) {
			frame = prev;
			continue;
		}

		struct py_frame *out = &result->frames[emitted];
		out->filename[0] = 0;
		out->funcname[0] = 0;
		out->firstlineno = 0;

		__u64 fn_ptr = 0;
		if (bpf_probe_read_user(&fn_ptr, sizeof(fn_ptr),
		    (const void *)(code_ptr + st->off_code_filename)) == 0)
			read_compact_ascii(out->filename, sizeof(out->filename),
			                   fn_ptr, st);

		__u64 nm_ptr = 0;
		if (bpf_probe_read_user(&nm_ptr, sizeof(nm_ptr),
		    (const void *)(code_ptr + st->off_code_name)) == 0)
			read_compact_ascii(out->funcname, sizeof(out->funcname),
			                   nm_ptr, st);

		bpf_probe_read_user(&out->firstlineno, sizeof(out->firstlineno),
		    (const void *)(code_ptr + st->off_code_firstlineno));

		emitted++;
		result->depth = emitted;
		frame = prev;
	}

	if (frame != 0)
		result->truncated = 1;

	if (result->depth > 0)
		py_debug_inc(8);  /* depth_gt_zero (success) */

	return 0;
}

/*
 * walk_python_frames_311: CPython 3.11 frame walker.
 *
 * 3.11 has one extra pointer chase vs 3.12: PyThreadState.cframe points
 * at a _PyCFrame, whose .current_frame is the top _PyInterpreterFrame.
 * From the interpreter frame the layout matches 3.12 (.previous,
 * .executable), so the rest of the walk is identical.
 *
 * off_tstate_frame -> PyThreadState.cframe offset.
 * off_cframe_current_frame -> _PyCFrame.current_frame offset.
 */
static __always_inline int walk_python_frames_311(
    __u32 pid, __u32 native_tid,
    const struct py_runtime_state *st,
    struct py_walk_result *result)
{
	py_debug_inc(10);  /* entered_311 */
	__u64 tstate = find_thread_state(native_tid, st);
	if (tstate == 0)
		return 0;
	py_debug_inc(11);  /* 311_thread_found */

	/* tstate->cframe -> _PyCFrame* */
	__u64 cframe_ptr = 0;
	if (bpf_probe_read_user(&cframe_ptr, sizeof(cframe_ptr),
	    (const void *)(tstate + st->off_tstate_frame)) != 0)
		return 0;
	py_debug_inc(12);  /* 311_cframe_read_ok */
	if (cframe_ptr == 0)
		return 0;
	py_debug_inc(13);  /* 311_cframe_nonzero */

	/* cframe->current_frame -> _PyInterpreterFrame*
	 *
	 * At uprobe time inside a C extension (e.g. cudaMalloc), CPython has
	 * pushed an "entry frame stub" on top. The stub typically lives at a
	 * STATIC address (in libpython's .data section) and contains garbage
	 * in the f_code slot — not zero, just not a valid heap pointer. So
	 * "f_code != 0" is too weak a check. We require f_code to be a
	 * userspace-heap pointer (>= 0x100000 and <= USERVA_TOP).
	 *
	 * Walk strategy: start at cframe.current_frame and follow the
	 * interp_frame.previous chain (each stub points to the real frame
	 * above it) up to 8 hops, accepting the first frame whose f_code is
	 * heap-range. */
	__u64 cand = 0;
	if (bpf_probe_read_user(&cand, sizeof(cand),
	    (const void *)(cframe_ptr + st->off_cframe_current_frame)) != 0)
		return 0;

	__u64 frame = 0;
	#pragma unroll(8)
	for (int hop = 0; hop < 8; hop++) {
		if (cand == 0) break;
		__u64 first_word = 0;
		if (bpf_probe_read_user(&first_word, sizeof(first_word),
		    (const void *)(cand + st->off_frame_code)) != 0) break;
		/* Heap-range check: stubs (entry frames for C calls) live on the
		 * C stack with garbage in the f_code slot. Real PyCodeObjects are
		 * heap-allocated, well above any static binary mapping. */
		if (first_word > 0x100000 && first_word < 0x800000000000) {
			frame = cand;
			break;
		}
		__u64 prev_frame = 0;
		if (bpf_probe_read_user(&prev_frame, sizeof(prev_frame),
		    (const void *)(cand + st->off_frame_back)) != 0) break;
		cand = prev_frame;
	}
	py_debug_inc(14);  /* 311_interp_frame_read_ok */
	if (frame != 0) py_debug_inc(15);  /* 311_frame_nonzero */

	/* Walk frame chain (bounded). Layout identical to 3.12 from here.
	 *
	 * IMPORTANT: CPython allocates "entry frame" stubs at the top of the
	 * frame stack when transitioning between Python and C extension code
	 * (e.g. inside a C extension called from Python). Entry frames have
	 * f_code == NULL but a valid `previous` pointer linking back to the
	 * nearest real Python frame. At uprobe time for a C function like
	 * cudaMalloc, tstate->cframe->current_frame typically IS one of these
	 * stubs — so on code_ptr == 0 we must FOLLOW the previous chain and
	 * try again rather than breaking out. `i` only increments when we
	 * actually emit a frame (so PY_MAX_FRAMES bounds emitted depth, not
	 * traversal count). We bound traversal separately with a larger cap.
	 */
	int emitted = 0;
	#pragma unroll(16)
	for (int walk_iter = 0; walk_iter < 16; walk_iter++) {
		if (frame == 0 || emitted >= PY_MAX_FRAMES)
			break;
		if (walk_iter == 0) py_debug_inc(7);  /* frame_loop_first_iteration */

		__u64 code_ptr = 0;
		if (bpf_probe_read_user(&code_ptr, sizeof(code_ptr),
		    (const void *)(frame + st->off_frame_code)) != 0)
			break;
		if (walk_iter == 0) py_debug_inc(16);  /* 311_loop_code_read_ok */

		__u64 prev = 0;
		if (bpf_probe_read_user(&prev, sizeof(prev),
		    (const void *)(frame + st->off_frame_back)) != 0)
			break;

		if (code_ptr == 0) {
			/* Entry-frame stub — skip and try parent. */
			frame = prev;
			continue;
		}
		if (walk_iter == 0) py_debug_inc(17);  /* 311_loop_code_nonzero */

		struct py_frame *out = &result->frames[emitted];
		out->filename[0] = 0;
		out->funcname[0] = 0;
		out->firstlineno = 0;

		__u64 fn_ptr = 0;
		if (bpf_probe_read_user(&fn_ptr, sizeof(fn_ptr),
		    (const void *)(code_ptr + st->off_code_filename)) == 0)
			read_compact_ascii(out->filename, sizeof(out->filename),
			                   fn_ptr, st);

		__u64 nm_ptr = 0;
		if (bpf_probe_read_user(&nm_ptr, sizeof(nm_ptr),
		    (const void *)(code_ptr + st->off_code_name)) == 0)
			read_compact_ascii(out->funcname, sizeof(out->funcname),
			                   nm_ptr, st);

		bpf_probe_read_user(&out->firstlineno, sizeof(out->firstlineno),
		    (const void *)(code_ptr + st->off_code_firstlineno));

		emitted++;
		result->depth = emitted;
		frame = prev;
	}

	if (frame != 0)
		result->truncated = 1;
	if (result->depth > 0) py_debug_inc(8);  /* depth_gt_zero (shared with _312) */

	return 0;
}

/*
 * walk_python_frames_310: CPython 3.10 frame walker.
 *
 * 3.10 uses the legacy PyFrameObject (not _PyInterpreterFrame).
 * PyThreadState.frame points directly at the top PyFrameObject. Walk
 * via PyFrameObject.f_back; read PyCodeObject via PyFrameObject.f_code.
 * PyCodeObject layout is unchanged between 3.10 and 3.12, so the string
 * extraction path (read_compact_ascii + off_code_filename/name) is
 * identical — only the frame-walk offsets differ.
 *
 * off_tstate_frame -> PyThreadState.frame offset
 * off_frame_back   -> PyFrameObject.f_back offset
 * off_frame_code   -> PyFrameObject.f_code offset
 */
static __always_inline int walk_python_frames_310(
    __u32 pid, __u32 native_tid,
    const struct py_runtime_state *st,
    struct py_walk_result *result)
{
	__u64 tstate = find_thread_state(native_tid, st);
	if (tstate == 0)
		return 0;

	/* tstate->frame -> PyFrameObject* (direct, no cframe indirection) */
	__u64 frame = 0;
	if (bpf_probe_read_user(&frame, sizeof(frame),
	    (const void *)(tstate + st->off_tstate_frame)) != 0)
		return 0;

	#pragma unroll(16)
	for (int i = 0; i < PY_MAX_FRAMES; i++) {
		if (frame == 0)
			break;

		struct py_frame *out = &result->frames[i];
		out->filename[0] = 0;
		out->funcname[0] = 0;
		out->firstlineno = 0;

		/* frame->f_code -> PyCodeObject* */
		__u64 code_ptr = 0;
		if (bpf_probe_read_user(&code_ptr, sizeof(code_ptr),
		    (const void *)(frame + st->off_frame_code)) != 0)
			break;
		if (code_ptr == 0)
			break;

		__u64 fn_ptr = 0;
		if (bpf_probe_read_user(&fn_ptr, sizeof(fn_ptr),
		    (const void *)(code_ptr + st->off_code_filename)) == 0)
			read_compact_ascii(out->filename, sizeof(out->filename),
			                   fn_ptr, st);

		__u64 nm_ptr = 0;
		if (bpf_probe_read_user(&nm_ptr, sizeof(nm_ptr),
		    (const void *)(code_ptr + st->off_code_name)) == 0)
			read_compact_ascii(out->funcname, sizeof(out->funcname),
			                   nm_ptr, st);

		bpf_probe_read_user(&out->firstlineno, sizeof(out->firstlineno),
		    (const void *)(code_ptr + st->off_code_firstlineno));

		result->depth = i + 1;

		/* Advance via frame->f_back. */
		__u64 prev = 0;
		if (bpf_probe_read_user(&prev, sizeof(prev),
		    (const void *)(frame + st->off_frame_back)) != 0)
			break;
		frame = prev;
	}

	if (frame != 0)
		result->truncated = 1;

	return 0;
}

/*
 * walk_python_frames: dispatcher. Extracts up to PY_MAX_FRAMES frames for
 * the given (pid, native_tid) pair by looking up the per-PID runtime
 * state and dispatching to the version-specific variant.
 *
 * Caller must pass an output buffer (typically obtained from py_scratch
 * — see map comment above).
 *
 * Returns:
 *    0 on success (result->depth indicates frame count; may be 0 if
 *      the thread was not found or has no current frame).
 *   -1 if result is NULL or no per-PID Python state is registered.
 *
 * SAFE: every bpf_probe_read_user is checked. Bounded loops only.
 */
static __always_inline int walk_python_frames(__u32 pid, __u32 native_tid,
                                               struct py_walk_result *result) {
	py_debug_inc(0);  /* entered_dispatcher */
	if (!result)
		return -1;
	result->depth = 0;
	result->truncated = 0;

	struct py_runtime_state *st =
	    bpf_map_lookup_elem(&py_runtime_map, &pid);
	if (!st)
		return -1;
	py_debug_inc(1);  /* state_lookup_ok */

	/* Branch on python_minor. 0 = legacy client (pre-v2 struct), treat as 12
	 * for backward compatibility. */
	__u8 minor = st->python_minor;
	if (minor == 0)
		minor = 12;

	/* CPython layout recap:
	 *   - 3.11 uses cframe indirection (tstate.cframe -> _PyCFrame.current_frame
	 *     -> _PyInterpreterFrame). walker_311 handles the double chase.
	 *   - 3.12 uses direct tstate.current_frame -> _PyInterpreterFrame (the
	 *     cframe field still exists but isn't the walk entry point).
	 *   - 3.13 dropped the _PyCFrame struct entirely; layout matches 3.12.
	 *   - 3.14 same frame field order as 3.13, but f_executable is a tagged
	 *     _PyStackRef union — walker_312 masks the tag bits.
	 *
	 * The `off_cframe_current_frame > 0` escape hatch on case 12 below is a
	 * safety net for distro-patched 3.12 builds that keep the cframe walk
	 * path; normal 3.12 is forced to direct in userspace (trace.go) before
	 * the state is pushed. */
	switch (minor) {
	case 9:
		/* 3.9 uses the same legacy PyFrameObject layout as 3.10. */
		return walk_python_frames_310(pid, native_tid, st, result);
	case 10:
		return walk_python_frames_310(pid, native_tid, st, result);
	case 11:
		return walk_python_frames_311(pid, native_tid, st, result);
	case 12:
		/* Real 3.12 has cframe indirection; 3.13 doesn't. Choose based
		 * on whether the harvester discovered a non-zero cframe offset. */
		if (st->off_cframe_current_frame > 0)
			return walk_python_frames_311(pid, native_tid, st, result);
		return walk_python_frames_312(pid, native_tid, st, result);
	case 13:
	case 14:
		/* 3.13+ dropped cframe; current_frame points directly at
		 * _PyInterpreterFrame. walker_312 handles that layout. */
		return walk_python_frames_312(pid, native_tid, st, result);
	default:
		/* Unsupported minor — return 0 frames; userspace walker fills in. */
		return 0;
	}
}

#endif /* __INGERO_PYTHON_WALKER_BPF_H */
