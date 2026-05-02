// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
//
// nccl_trace.bpf.c - eBPF uprobes for NCCL collective tracing.
//
// Hooks the NCCL public API in libnccl.so (or whatever shared object
// userspace registers). Emits one nccl_event record per collective
// uretprobe with the rank/nranks attached via a comm-handle map
// populated at ncclCommInitRank time.
//
// Per-arch via bpf2go -target amd64,arm64. PT_REGS_PARM* macros expand
// against `struct pt_regs` (x86) or `struct user_pt_regs` (arm64); the
// arm64 declaration comes from common.bpf.h (CO-RE relocated).
//
// Out of scope for v0.12.0 (will land in v0.12.1 or later):
//   - ncclCommInitAll (multi-GPU single-process, requires comm-array iteration)
//   - ncclGroupStart / ncclGroupEnd (correlation only, no data on wire)

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "common.bpf.h"

char LICENSE[] SEC("license") = "Dual BSD/GPL";

/* libbpf 0.5 (Ubuntu 22.04) only ships PT_REGS_PARM1..PARM5. Define
 * arch-aware helpers for PARM6 + PARM7 so we can attach to NCCL
 * collectives that take 6 or 7 args (ncclAllReduce, ncclReduceScatter
 * have 7 incl. cudaStream_t; ncclBcast and ncclAllGather have 6).
 *
 * On amd64 SysV: PARM6=r9, PARM7 lives on the caller's stack at sp+8.
 * On arm64 AAPCS: PARM6=x5, PARM7=x6 (regs[5] / regs[6]).
 */
static __always_inline __u64 ingero_pt_regs_arg6(struct pt_regs *ctx)
{
#if defined(__TARGET_ARCH_x86)
	return BPF_CORE_READ(ctx, r9);
#elif defined(__TARGET_ARCH_arm64)
	/* CO-RE relocate against arm64 user_pt_regs.regs[5]. v0.12.1
	 * upgrade from raw bpf_probe_read_kernel: matches the
	 * BPF_CORE_READ pattern used everywhere else in the codebase
	 * (e.g. cuda_trace.bpf.c) and gets relocation if user_pt_regs
	 * gains fields in a future kernel. */
	return BPF_CORE_READ((struct user_pt_regs *)ctx, regs[5]);
#else
	(void)ctx;
	return 0;
#endif
}

static __always_inline __u64 ingero_pt_regs_arg7(struct pt_regs *ctx)
{
#if defined(__TARGET_ARCH_x86)
	__u64 sp = BPF_CORE_READ(ctx, sp);
	if (!sp)
		return 0;
	__u64 val = 0;
	/* arg7 sits on the caller's stack one slot above the return
	 * address; rsp at uprobe entry points at the return address.
	 */
	bpf_probe_read_user(&val, sizeof(val), (void *)(sp + sizeof(void *)));
	return val;
#elif defined(__TARGET_ARCH_arm64)
	return BPF_CORE_READ((struct user_pt_regs *)ctx, regs[6]);
#else
	(void)ctx;
	return 0;
#endif
}

/* ---------------- Maps ---------------- */

/* events: ringbuf for nccl_event records. 2 MB is plenty: NCCL collectives
 * fire at most a few hundred Hz per rank under typical training loads,
 * 104 bytes per event => ~20k events buffered.
 */
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 2 * 1024 * 1024);
} nccl_events SEC(".maps");

/* config_map: shared with other probes. Same layout. */
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct ingero_config);
} config_map SEC(".maps");

/* nccl_entry_state: per-thread uprobe-time state stash. Keyed by tid
 * (kernel side: kernel TID = userspace gettid()). Holds the timestamp,
 * op code, and the captured args we need at uretprobe time.
 */
struct nccl_entry_state {
	__u64 timestamp_ns;
	__u8  op;
	__u8  _pad[7];
	__u64 comm_ptr;     /* the ncclComm_t opaque ptr (PARM6 for collectives, PARM1 for Init) */
	__u64 comm_out_ptr; /* commInitRank only: where the resulting comm_ptr will be stored */
	__u64 count;        /* element count */
	__u64 stream;       /* CUDA stream */
	__u32 datatype;
	__u32 reduce_op;
	__u32 nranks_arg;   /* commInitRank: nranks PARM2 */
	__u32 rank_arg;     /* commInitRank: rank PARM4 */
	__u32 peer_rank;    /* v0.12.2: ncclSend/Recv: peer PARM4; 0 for collectives */
	__u32 _pad2;        /* keep 8-byte alignment of commid_words below */
	/* v0.12.1 (LHF #15): hash all 128 bytes of ncclUniqueId, not
	 * just the first 8. The first 8 bytes are mostly NCCL magic +
	 * version and collide across distinct comms. We splitmix64-fold
	 * the 16 64-bit chunks at uretprobe time; result is a 64-bit
	 * hash collision-free for typical multi-comm workloads. */
	__u64 commid_words[16];
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8192);
	__type(key, __u32);                    /* tid */
	__type(value, struct nccl_entry_state);
} nccl_entry_map SEC(".maps");

/* nccl_comm_value: per-(pid, comm_ptr) record populated at
 * ncclCommInitRank uretprobe and looked up at every collective.
 */
struct nccl_comm_key {
	__u32 pid;
	__u32 _pad;
	__u64 comm_ptr;
};

struct nccl_comm_value {
	__u64 comm_id_hash;
	__u32 rank;
	__u32 nranks;
	__u64 init_ts;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 4096);             /* 4k unique communicators across all GPU procs */
	__type(key, struct nccl_comm_key);
	__type(value, struct nccl_comm_value);
} nccl_comm_map SEC(".maps");

/* v0.12.2 (LHF #7): per-PID filter for the NCCL probe so multi-tenant
 * workloads don't leak unrelated process events into the agent's
 * ringbuf. Userspace populates this map via Tracer.SetTargetPID; when
 * empty (no entries), the probe traces system-wide. When populated,
 * only PIDs in the map produce events.
 *
 * Convention: the userspace caller inserts a sentinel at key=0 to mean
 * "map is populated" (the kernel can't distinguish empty vs. only-pid-0
 * because pid=0 is the swapper and never traced). nccl_pid_map_empty()
 * checks the sentinel.
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 256);
	__type(key, __u32);
	__type(value, __u8);
} nccl_target_pids SEC(".maps");

static __always_inline bool nccl_is_target(__u32 pid)
{
	return bpf_map_lookup_elem(&nccl_target_pids, &pid) != NULL;
}

static __always_inline bool nccl_pid_map_empty(void)
{
	__u32 zero = 0;
	return bpf_map_lookup_elem(&nccl_target_pids, &zero) == NULL;
}

/* nccl_should_trace returns true when the current PID should produce a
 * NCCL event. Either the filter is empty (system-wide tracing) OR the
 * PID is explicitly in the target set. */
static __always_inline bool nccl_should_trace(__u32 pid)
{
	return nccl_pid_map_empty() || nccl_is_target(pid);
}

/* nccl_should_trace_self is reserved for callers that haven't already
 * computed the pid; the per-uprobe gate inlines the check directly,
 * so this wrapper is currently unused but kept for symmetry with the
 * net probe pattern. Marked __attribute__((unused)) so -Werror is happy. */
static __always_inline __attribute__((unused)) bool nccl_should_trace_self(void)
{
	__u32 pid = bpf_get_current_pid_tgid() >> 32;
	return nccl_should_trace(pid);
}

/* ---------------- Helpers ---------------- */

static __always_inline __u64 splitmix64(__u64 x)
{
	x ^= x >> 30;
	x *= 0xbf58476d1ce4e5b9ULL;
	x ^= x >> 27;
	x *= 0x94d049bb133111ebULL;
	x ^= x >> 31;
	return x;
}

/* fold_commid_128: 64-bit hash of the full 128-byte ncclUniqueId.
 * Verifier-friendly unrolled 16-iteration splitmix64 fold. v0.12.1
 * (LHF #15) replacement for splitmix64(words[0]) which collided
 * across distinct comms sharing the same NCCL magic+version header. */
static __always_inline __u64 fold_commid_128(const __u64 words[16])
{
	__u64 h = 0x9e3779b97f4a7c15ULL; /* arbitrary non-zero seed */
	#pragma unroll
	for (int i = 0; i < 16; i++) {
		h = splitmix64(h ^ words[i]);
	}
	return h;
}

/* save_entry: stash uprobe-time state for the current TID. */
static __always_inline void save_entry(__u32 tid, struct nccl_entry_state *st)
{
	st->timestamp_ns = bpf_ktime_get_ns();
	bpf_map_update_elem(&nccl_entry_map, &tid, st, BPF_ANY);
}

/* lookup_comm: pull rank/nranks/comm_id_hash for a (pid, comm_ptr).
 * Writes zeros to out fields if the lookup misses (process started
 * NCCL before our agent attached, or commInitRank wasn't traced).
 */
static __always_inline void lookup_comm(__u32 pid, __u64 comm_ptr,
                                        __u32 *rank, __u32 *nranks, __u64 *comm_id_hash)
{
	struct nccl_comm_key k = {};
	k.pid = pid;
	k.comm_ptr = comm_ptr;
	struct nccl_comm_value *v = bpf_map_lookup_elem(&nccl_comm_map, &k);
	if (v) {
		*rank = v->rank;
		*nranks = v->nranks;
		*comm_id_hash = v->comm_id_hash;
	} else {
		*rank = 0;
		*nranks = 0;
		*comm_id_hash = 0;
	}
}

/* emit_collective: shared exit path for the *Reduce / *Gather / *Scatter / Bcast
 * uretprobes. count_bytes is 0 here because eBPF can't size ncclDataType_t
 * cheaply; userspace multiplies by datatype_size in pkg/contract.
 */
static __always_inline void emit_collective(struct pt_regs *ctx,
                                            __u32 pid, __u32 tid,
                                            struct nccl_entry_state *entry,
                                            __s32 return_code)
{
	__u64 now = bpf_ktime_get_ns();

	struct nccl_event *evt = bpf_ringbuf_reserve(&nccl_events, sizeof(*evt), 0);
	if (!evt)
		return;

	evt->hdr.timestamp_ns = entry->timestamp_ns;
	evt->hdr.pid = pid;
	evt->hdr.tid = tid;
	evt->hdr.source = EVENT_SRC_NCCL;
	evt->hdr.op = entry->op;
	evt->hdr._pad = 0;
	evt->hdr._pad2 = 0;
	evt->hdr.cgroup_id = bpf_get_current_cgroup_id();
	bpf_get_current_comm(&evt->hdr.comm, sizeof(evt->hdr.comm));

	evt->duration_ns = now - entry->timestamp_ns;
	evt->stream_handle = entry->stream;
	evt->count_bytes = entry->count;   /* element count; userspace multiplies by datatype size */
	evt->datatype = entry->datatype;
	evt->reduce_op = entry->reduce_op;
	evt->return_code = return_code;
	evt->peer_rank = entry->peer_rank; /* v0.12.2: nonzero for ncclSend/Recv only */

	__u32 rank = 0, nranks = 0;
	__u64 comm_id_hash = 0;
	lookup_comm(pid, entry->comm_ptr, &rank, &nranks, &comm_id_hash);
	evt->rank = rank;
	evt->nranks = nranks;
	evt->comm_id_hash = comm_id_hash;

	bpf_ringbuf_submit(evt, 0);
}

/* ---------------- ncclCommInitRank ---------------- *
 * Signature: ncclResult_t ncclCommInitRank(ncclComm_t* comm,
 *                                          int nranks,
 *                                          ncclUniqueId commId,
 *                                          int myrank);
 * Note: ncclUniqueId is a 128-byte struct passed BY VALUE on amd64
 * (split across many arg registers + stack) and BY POINTER on arm64.
 * For v0.12.0 we treat PARM3 as a pointer-or-first-8-bytes-as-value
 * and read the first 8 bytes via bpf_probe_read_user. For amd64 SysV
 * ABI structs >16 bytes are passed by hidden pointer (sret), so PARM3
 * is in fact a pointer to the caller's commId on the stack. For arm64
 * AAPCS structs >16 bytes are passed by reference. So PARM3 is always
 * a pointer in practice on the platforms we target.
 */
SEC("uprobe/ncclCommInitRank")
int BPF_KPROBE(uprobe_nccl_comm_init_rank,
               void *comm_out_ptr, int nranks, void *commId, int rank)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state st = {};
	st.op = NCCL_OP_COMM_INIT_RANK;
	st.comm_out_ptr = (__u64)comm_out_ptr;
	st.nranks_arg = nranks;
	st.rank_arg = rank;

	/* v0.12.1: read all 128 bytes of *commId in one bpf_probe_read_user
	 * call. The first 8 are typically NCCL magic+version (collision
	 * prone in v0.12.0); reading the full 128 bytes covers the unique
	 * portion of the id. The uretprobe folds the 16 64-bit chunks
	 * into a single 64-bit hash via repeated splitmix64.
	 *
	 * On EFAULT (bad pointer): leave the buffer zeroed AND set op=0
	 * so uretprobe skips the nccl_comm_map insert. Prevents a single
	 * attacker-stable bucket all bad-call captures would funnel into.
	 */
	long rc = bpf_probe_read_user(st.commid_words, sizeof(st.commid_words), commId);
	if (rc != 0) {
		__builtin_memset(st.commid_words, 0, sizeof(st.commid_words));
		st.op = 0; /* signal to uretprobe: skip the map update */
	}

	if (!nccl_should_trace((__u32)(bpf_get_current_pid_tgid() >> 32))) return 0;
	save_entry(tid, &st);
	return 0;
}

SEC("uretprobe/ncclCommInitRank")
int BPF_KRETPROBE(uretprobe_nccl_comm_init_rank, int retval)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 pid = pid_tid >> 32;
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state *entry = bpf_map_lookup_elem(&nccl_entry_map, &tid);
	if (!entry)
		return 0;

	/* On success (retval==0) AND the uprobe successfully read commId
	 * (entry->op == NCCL_OP_COMM_INIT_RANK; uprobe sets op=0 on
	 * commId EFAULT), read out the resulting comm pointer and
	 * populate nccl_comm_map. Failure to populate keeps the bucket
	 * unset; subsequent collectives on this comm get rank=0/nranks=0
	 * (lookup miss path).
	 */
	if (retval == 0 && entry->op == NCCL_OP_COMM_INIT_RANK) {
		__u64 comm_ptr = 0;
		long rc2 = bpf_probe_read_user(&comm_ptr, sizeof(comm_ptr), (void *)entry->comm_out_ptr);
		if (rc2 == 0 && comm_ptr != 0) {
			struct nccl_comm_key k = {};
			k.pid = pid;
			k.comm_ptr = comm_ptr;

			struct nccl_comm_value v = {};
			v.comm_id_hash = fold_commid_128(entry->commid_words);
			v.rank = entry->rank_arg;
			v.nranks = entry->nranks_arg;
			v.init_ts = bpf_ktime_get_ns();
			bpf_map_update_elem(&nccl_comm_map, &k, &v, BPF_ANY);
		}
	}

	/* Emit a COMM_INIT_RANK event regardless so userspace can see
	 * communicator lifecycle.
	 */
	__u64 now = bpf_ktime_get_ns();
	struct nccl_event *evt = bpf_ringbuf_reserve(&nccl_events, sizeof(*evt), 0);
	if (evt) {
		evt->hdr.timestamp_ns = entry->timestamp_ns;
		evt->hdr.pid = pid;
		evt->hdr.tid = tid;
		evt->hdr.source = EVENT_SRC_NCCL;
		evt->hdr.op = NCCL_OP_COMM_INIT_RANK;
		evt->hdr._pad = 0;
		evt->hdr._pad2 = 0;
		evt->hdr.cgroup_id = bpf_get_current_cgroup_id();
		bpf_get_current_comm(&evt->hdr.comm, sizeof(evt->hdr.comm));

		evt->duration_ns = now - entry->timestamp_ns;
		evt->comm_id_hash = fold_commid_128(entry->commid_words);
		evt->stream_handle = 0;
		evt->count_bytes = 0;
		evt->rank = entry->rank_arg;
		evt->nranks = entry->nranks_arg;
		evt->datatype = 0;
		evt->reduce_op = 0;
		evt->return_code = retval;
		evt->peer_rank = 0;
		bpf_ringbuf_submit(evt, 0);
	}

	bpf_map_delete_elem(&nccl_entry_map, &tid);
	return 0;
}

/* ---------------- ncclCommDestroy ---------------- *
 * Signature: ncclResult_t ncclCommDestroy(ncclComm_t comm);
 * Removes the (pid, comm_ptr) mapping at uretprobe time.
 */
SEC("uprobe/ncclCommDestroy")
int BPF_KPROBE(uprobe_nccl_comm_destroy, void *comm)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state st = {};
	st.op = NCCL_OP_COMM_DESTROY;
	st.comm_ptr = (__u64)comm;
	if (!nccl_should_trace((__u32)(bpf_get_current_pid_tgid() >> 32))) return 0;
	save_entry(tid, &st);
	return 0;
}

SEC("uretprobe/ncclCommDestroy")
int BPF_KRETPROBE(uretprobe_nccl_comm_destroy, int retval)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 pid = pid_tid >> 32;
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state *entry = bpf_map_lookup_elem(&nccl_entry_map, &tid);
	if (!entry)
		return 0;

	struct nccl_comm_key k = {};
	k.pid = pid;
	k.comm_ptr = entry->comm_ptr;
	bpf_map_delete_elem(&nccl_comm_map, &k);

	__u64 now = bpf_ktime_get_ns();
	struct nccl_event *evt = bpf_ringbuf_reserve(&nccl_events, sizeof(*evt), 0);
	if (evt) {
		evt->hdr.timestamp_ns = entry->timestamp_ns;
		evt->hdr.pid = pid;
		evt->hdr.tid = tid;
		evt->hdr.source = EVENT_SRC_NCCL;
		evt->hdr.op = NCCL_OP_COMM_DESTROY;
		evt->hdr._pad = 0;
		evt->hdr._pad2 = 0;
		evt->hdr.cgroup_id = bpf_get_current_cgroup_id();
		bpf_get_current_comm(&evt->hdr.comm, sizeof(evt->hdr.comm));
		evt->duration_ns = now - entry->timestamp_ns;
		evt->comm_id_hash = 0;
		evt->stream_handle = 0;
		evt->count_bytes = 0;
		evt->rank = 0;
		evt->nranks = 0;
		evt->datatype = 0;
		evt->reduce_op = 0;
		evt->return_code = retval;
		evt->peer_rank = 0;
		bpf_ringbuf_submit(evt, 0);
	}

	bpf_map_delete_elem(&nccl_entry_map, &tid);
	return 0;
}

/* ---------------- ncclAllReduce ---------------- *
 * Signature: ncclResult_t ncclAllReduce(const void* sendbuff,
 *                                       void* recvbuff,
 *                                       size_t count,
 *                                       ncclDataType_t datatype,
 *                                       ncclRedOp_t op,
 *                                       ncclComm_t comm,
 *                                       cudaStream_t stream);
 *
 * 7 args; PARM7 is the cudaStream_t. On amd64 SysV ABI, PARM7 lives on
 * the caller's stack at [rsp + 8] before the call insn. We read it via
 * PT_REGS_SP_CORE on entry. On arm64 AAPCS, PARM7 is in x6 -> PT_REGS_PARM7
 * is the right macro. bpf_tracing.h gives us the right thing per arch.
 */
SEC("uprobe/ncclAllReduce")
int uprobe_nccl_all_reduce(struct pt_regs *ctx)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state st = {};
	st.op = NCCL_OP_ALL_REDUCE;
	st.count = (__u64)PT_REGS_PARM3_CORE(ctx);
	st.datatype = (__u32)PT_REGS_PARM4_CORE(ctx);
	st.reduce_op = (__u32)PT_REGS_PARM5_CORE(ctx);
	st.comm_ptr = ingero_pt_regs_arg6(ctx);
	st.stream = ingero_pt_regs_arg7(ctx);

	if (!nccl_should_trace((__u32)(bpf_get_current_pid_tgid() >> 32))) return 0;
	save_entry(tid, &st);
	return 0;
}

SEC("uretprobe/ncclAllReduce")
int BPF_KRETPROBE(uretprobe_nccl_all_reduce, int retval)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 pid = pid_tid >> 32;
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state *entry = bpf_map_lookup_elem(&nccl_entry_map, &tid);
	if (!entry)
		return 0;
	emit_collective(ctx, pid, tid, entry, retval);
	bpf_map_delete_elem(&nccl_entry_map, &tid);
	return 0;
}

/* ---------------- ncclAllGather ---------------- *
 * Signature: ncclResult_t ncclAllGather(const void* sendbuff,
 *                                       void* recvbuff,
 *                                       size_t sendcount,
 *                                       ncclDataType_t datatype,
 *                                       ncclComm_t comm,
 *                                       cudaStream_t stream);
 * 6 args; stream is PARM6. No reduce_op.
 */
SEC("uprobe/ncclAllGather")
int uprobe_nccl_all_gather(struct pt_regs *ctx)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state st = {};
	st.op = NCCL_OP_ALL_GATHER;
	st.count = (__u64)PT_REGS_PARM3_CORE(ctx);
	st.datatype = (__u32)PT_REGS_PARM4_CORE(ctx);
	st.reduce_op = 0;
	st.comm_ptr = (__u64)PT_REGS_PARM5_CORE(ctx);
	st.stream = ingero_pt_regs_arg6(ctx);
	if (!nccl_should_trace((__u32)(bpf_get_current_pid_tgid() >> 32))) return 0;
	save_entry(tid, &st);
	return 0;
}

SEC("uretprobe/ncclAllGather")
int BPF_KRETPROBE(uretprobe_nccl_all_gather, int retval)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 pid = pid_tid >> 32;
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state *entry = bpf_map_lookup_elem(&nccl_entry_map, &tid);
	if (!entry)
		return 0;
	emit_collective(ctx, pid, tid, entry, retval);
	bpf_map_delete_elem(&nccl_entry_map, &tid);
	return 0;
}

/* ---------------- ncclReduceScatter ---------------- *
 * Same layout as ncclAllReduce (7 args, stream on stack).
 */
SEC("uprobe/ncclReduceScatter")
int uprobe_nccl_reduce_scatter(struct pt_regs *ctx)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state st = {};
	st.op = NCCL_OP_REDUCE_SCATTER;
	st.count = (__u64)PT_REGS_PARM3_CORE(ctx);
	st.datatype = (__u32)PT_REGS_PARM4_CORE(ctx);
	st.reduce_op = (__u32)PT_REGS_PARM5_CORE(ctx);
	st.comm_ptr = ingero_pt_regs_arg6(ctx);
	st.stream = ingero_pt_regs_arg7(ctx);
	if (!nccl_should_trace((__u32)(bpf_get_current_pid_tgid() >> 32))) return 0;
	save_entry(tid, &st);
	return 0;
}

SEC("uretprobe/ncclReduceScatter")
int BPF_KRETPROBE(uretprobe_nccl_reduce_scatter, int retval)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 pid = pid_tid >> 32;
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state *entry = bpf_map_lookup_elem(&nccl_entry_map, &tid);
	if (!entry)
		return 0;
	emit_collective(ctx, pid, tid, entry, retval);
	bpf_map_delete_elem(&nccl_entry_map, &tid);
	return 0;
}

/* ---------------- ncclBcast ---------------- *
 * Signature: ncclResult_t ncclBcast(void* buff, size_t count,
 *                                   ncclDataType_t datatype, int root,
 *                                   ncclComm_t comm, cudaStream_t stream);
 * 6 args; PARM6 is stream.
 */
SEC("uprobe/ncclBcast")
int uprobe_nccl_bcast(struct pt_regs *ctx)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state st = {};
	st.op = NCCL_OP_BCAST;
	st.count = (__u64)PT_REGS_PARM2_CORE(ctx);
	st.datatype = (__u32)PT_REGS_PARM3_CORE(ctx);
	/* PARM4 = root (int): not currently emitted; reserved for future. */
	st.comm_ptr = (__u64)PT_REGS_PARM5_CORE(ctx);
	st.reduce_op = 0;
	st.stream = ingero_pt_regs_arg6(ctx);
	if (!nccl_should_trace((__u32)(bpf_get_current_pid_tgid() >> 32))) return 0;
	save_entry(tid, &st);
	return 0;
}

SEC("uretprobe/ncclBcast")
int BPF_KRETPROBE(uretprobe_nccl_bcast, int retval)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 pid = pid_tid >> 32;
	__u32 tid = (__u32)pid_tid;

	struct nccl_entry_state *entry = bpf_map_lookup_elem(&nccl_entry_map, &tid);
	if (!entry)
		return 0;
	emit_collective(ctx, pid, tid, entry, retval);
	bpf_map_delete_elem(&nccl_entry_map, &tid);
	return 0;
}

/* ---------------- ncclSend / ncclRecv (v0.12.1 LHF #17 long-tail) ---------------- *
 * Both have 6 args (peer is PARM4, comm is PARM5, stream is PARM6).
 * Signature: ncclResult_t ncclSend(const void* sendbuff, size_t count,
 *                                   ncclDataType_t datatype, int peer,
 *                                   ncclComm_t comm, cudaStream_t stream);
 * For pipeline-parallel workloads (DeepSpeed, Megatron) ncclSend/Recv
 * are the dominant traffic, not allreduce. Without these probes the
 * agent shows zero NCCL activity on PP-heavy jobs.
 */
SEC("uprobe/ncclSend")
int uprobe_nccl_send(struct pt_regs *ctx)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 tid = (__u32)pid_tid;
	struct nccl_entry_state st = {};
	st.op = NCCL_OP_SEND;
	st.count = (__u64)PT_REGS_PARM2_CORE(ctx);
	st.datatype = (__u32)PT_REGS_PARM3_CORE(ctx);
	st.peer_rank = (__u32)PT_REGS_PARM4_CORE(ctx); /* v0.12.2: peer rank */
	st.comm_ptr = (__u64)PT_REGS_PARM5_CORE(ctx);
	st.reduce_op = 0;
	st.stream = ingero_pt_regs_arg6(ctx);
	if (!nccl_should_trace((__u32)(bpf_get_current_pid_tgid() >> 32))) return 0;
	save_entry(tid, &st);
	return 0;
}

SEC("uretprobe/ncclSend")
int BPF_KRETPROBE(uretprobe_nccl_send, int retval)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 pid = pid_tid >> 32;
	__u32 tid = (__u32)pid_tid;
	struct nccl_entry_state *entry = bpf_map_lookup_elem(&nccl_entry_map, &tid);
	if (!entry)
		return 0;
	emit_collective(ctx, pid, tid, entry, retval);
	bpf_map_delete_elem(&nccl_entry_map, &tid);
	return 0;
}

SEC("uprobe/ncclRecv")
int uprobe_nccl_recv(struct pt_regs *ctx)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 tid = (__u32)pid_tid;
	struct nccl_entry_state st = {};
	st.op = NCCL_OP_RECV;
	st.count = (__u64)PT_REGS_PARM2_CORE(ctx);
	st.datatype = (__u32)PT_REGS_PARM3_CORE(ctx);
	st.peer_rank = (__u32)PT_REGS_PARM4_CORE(ctx); /* v0.12.2: peer rank */
	st.comm_ptr = (__u64)PT_REGS_PARM5_CORE(ctx);
	st.reduce_op = 0;
	st.stream = ingero_pt_regs_arg6(ctx);
	if (!nccl_should_trace((__u32)(bpf_get_current_pid_tgid() >> 32))) return 0;
	save_entry(tid, &st);
	return 0;
}

SEC("uretprobe/ncclRecv")
int BPF_KRETPROBE(uretprobe_nccl_recv, int retval)
{
	__u64 pid_tid = bpf_get_current_pid_tgid();
	__u32 pid = pid_tid >> 32;
	__u32 tid = (__u32)pid_tid;
	struct nccl_entry_state *entry = bpf_map_lookup_elem(&nccl_entry_map, &tid);
	if (!entry)
		return 0;
	emit_collective(ctx, pid, tid, entry, retval);
	bpf_map_delete_elem(&nccl_entry_map, &tid);
	return 0;
}

/* Force BTF emission for the types bpf2go's `-type` flag wants to read
 * out of the .o. Mirrors the cuda_trace.bpf.c convention. */
const struct nccl_event *_unused_nccl_event_force_btf __attribute__((unused));
const struct nccl_comm_value *_unused_nccl_comm_value_force_btf __attribute__((unused));
const struct nccl_comm_key *_unused_nccl_comm_key_force_btf __attribute__((unused));
const struct nccl_entry_state *_unused_nccl_entry_state_force_btf __attribute__((unused));
const struct ingero_config *_unused_ingero_config_force_btf_nccl __attribute__((unused));
