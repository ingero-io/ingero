// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/*
 * python_walker.bpf.h — inline CPython 3.12 frame walker for eBPF.
 *
 * Single source of truth for the in-kernel Python frame walker. Included
 * by cuda_trace.bpf.c so the walker runs inline on the same probe that
 * emits the CUDA event — no tail call, no per-program map coordination.
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
 * read_compact_ascii: reads a CPython compact-ASCII string into dst.
 * Returns 1 on success, 0 on any failure or non-compact string.
 *
 * We only support compact ASCII to keep the BPF program bounded:
 *   - Compact strings have inline data at unicode_addr + off_unicode_data.
 *   - Non-compact strings require a second pointer chase (skipped — dst
 *     will be an empty NUL string).
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
 * find_thread_state: walks the PyThreadState linked list to find one
 * matching the given native thread id. Returns the tstate address, or 0.
 *
 * Bounded by PY_MAX_THREADS — workloads with more Python threads than
 * that will miss frames for later threads (acceptable trade-off for
 * verifier safety).
 */
static __always_inline __u64 find_thread_state(
    __u32 native_tid, const struct py_runtime_state *st)
{
	/* interpreters_head -> first PyInterpreterState */
	__u64 interp = 0;
	if (bpf_probe_read_user(&interp, sizeof(interp),
	    (const void *)(st->runtime_addr + st->off_runtime_interpreters_head)) != 0)
		return 0;
	if (interp == 0)
		return 0;

	/* threads.head -> first PyThreadState */
	__u64 tstate = 0;
	if (bpf_probe_read_user(&tstate, sizeof(tstate),
	    (const void *)(interp + st->off_tstate_head)) != 0)
		return 0;

	/* Walk thread list (bounded). */
	#pragma unroll(8)
	for (int i = 0; i < PY_MAX_THREADS; i++) {
		if (tstate == 0)
			break;

		__u64 cur_tid = 0;
		if (bpf_probe_read_user(&cur_tid, sizeof(cur_tid),
		    (const void *)(tstate + st->off_tstate_native_tid)) != 0)
			break;

		if ((__u32)cur_tid == native_tid)
			return tstate;

		__u64 next = 0;
		if (bpf_probe_read_user(&next, sizeof(next),
		    (const void *)(tstate + st->off_tstate_next)) != 0)
			break;
		tstate = next;
	}
	return 0;
}

/*
 * walk_python_frames: extracts up to PY_MAX_FRAMES frames for the given
 * (pid, native_tid) pair. Caller must pass an output buffer (typically
 * obtained from py_scratch — see map comment above).
 *
 * Returns:
 *    0 on success (result->depth indicates frame count; may be 0 if
 *      the thread was not found or has no current frame).
 *   -1 if no per-PID Python state is registered in py_runtime_map.
 *
 * SAFE: every bpf_probe_read_user is checked. Bounded loops only.
 */
static __always_inline int walk_python_frames(
    __u32 pid, __u32 native_tid, struct py_walk_result *result)
{
	if (!result)
		return -1;
	result->depth = 0;
	result->truncated = 0;

	struct py_runtime_state *st =
	    bpf_map_lookup_elem(&py_runtime_map, &pid);
	if (!st)
		return -1;

	__u64 tstate = find_thread_state(native_tid, st);
	if (tstate == 0)
		return 0;  /* not an error — thread may not exist yet */

	/* Get current frame from matched tstate. */
	__u64 frame = 0;
	if (bpf_probe_read_user(&frame, sizeof(frame),
	    (const void *)(tstate + st->off_tstate_frame)) != 0)
		return 0;

	/* Walk frame chain (bounded). */
	#pragma unroll(16)
	for (int i = 0; i < PY_MAX_FRAMES; i++) {
		if (frame == 0)
			break;

		struct py_frame *out = &result->frames[i];
		out->filename[0] = 0;
		out->funcname[0] = 0;
		out->firstlineno = 0;

		/* frame->executable -> PyCodeObject */
		__u64 code_ptr = 0;
		if (bpf_probe_read_user(&code_ptr, sizeof(code_ptr),
		    (const void *)(frame + st->off_frame_code)) != 0)
			break;
		if (code_ptr == 0)
			break;

		/* code->co_filename, co_name (PyUnicodeObject pointers) */
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

		/* code->co_firstlineno (int32) */
		bpf_probe_read_user(&out->firstlineno, sizeof(out->firstlineno),
		    (const void *)(code_ptr + st->off_code_firstlineno));

		result->depth = i + 1;

		/* Advance to previous frame. */
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

#endif /* __INGERO_PYTHON_WALKER_BPF_H */
