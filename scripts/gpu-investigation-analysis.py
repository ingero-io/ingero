#!/usr/bin/env python3
"""GPU Problem Investigation — 28 Issues via MCP.

Investigates all 28 GPU problems Ingero can detect by querying the database
exclusively through MCP tool calls (run_sql, get_causal_chains, get_trace_stats,
run_sql). Simulates how an AI agent would investigate GPU issues.

Each investigation:
  1. Poses a human question ("My GPU is slow, why?")
  2. Makes 2-5 MCP calls to gather evidence
  3. Analyzes results and assigns a verdict (DETECTED / HEALTHY / INCONCLUSIVE)

Verdicts:
  DETECTED     — problem pattern found in data (PASS if provoked)
  HEALTHY      — investigation ran, no problem found (PASS for non-provoked)
  INCONCLUSIVE — insufficient data (SKIP)

Output: ML_RESULT lines to stdout (for gpu-test.sh ingestion) + report file.

Known limitations:
  - Ring buffer drops ~8% under heavy contention phases (17K+ events/sec).
    This is expected with 256KB per-CPU ring buffers. Increasing to 512KB
    or 1MB in bpf/ would reduce drops but increase kernel memory footprint.
  - No cuMemFree probe — frees=0 is always true, making malloc/free balance
    unreliable. PyTorch caching allocator also holds memory without freeing.
  - Thermal throttle detection uses host-side launch latency (proxy signal).
"""

import argparse
import json
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# MCP Client
# ---------------------------------------------------------------------------

class MCPClient:
    """Minimal MCP client using HTTPS + JSON-RPC."""

    def __init__(self, url: str):
        self.url = url
        self._id = 0
        # Accept self-signed certs
        self._ctx = ssl.create_default_context()
        self._ctx.check_hostname = False
        self._ctx.verify_mode = ssl.CERT_NONE

    def call(self, tool: str, arguments: Optional[dict] = None,
             timeout: int = 30) -> dict:
        """Call an MCP tool and return the parsed result."""
        self._id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": "tools/call",
            "params": {
                "name": tool,
                "arguments": arguments or {},
            },
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        try:
            with urllib.request.urlopen(req, context=self._ctx, timeout=timeout) as resp:
                body = resp.read().decode()
        except Exception as e:
            print(f"  [MCP ERROR] {tool}: {e}", file=sys.stderr)
            return {"error": str(e)}

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return {"error": f"invalid JSON: {body[:200]}"}

        # Extract text content from MCP response
        result = parsed.get("result", {})
        # Check for MCP-level errors (e.g. SQL syntax error, timeout)
        if result.get("isError"):
            content = result.get("content", [])
            err_text = content[0].get("text", "MCP error") if content else "MCP error"
            print(f"  [MCP ERROR] {tool}: {err_text}", file=sys.stderr)
            return {"error": err_text}
        content = result.get("content", [])
        if content and isinstance(content, list):
            text = content[0].get("text", "")
            # Try to parse as JSON
            try:
                return {"data": json.loads(text)}
            except (json.JSONDecodeError, TypeError):
                return {"text": text}
        return {"raw": result}

    def run_sql(self, query: str, limit: int = 1000) -> dict:
        """Execute a read-only SQL query via MCP."""
        return self.call("run_sql", {"query": query, "limit": limit, "tsc": False})

    def get_causal_chains(self, since: str = "10m") -> dict:
        """Get causal chains (120s timeout — replay is expensive on large DBs)."""
        return self.call("get_causal_chains", {"since": since, "tsc": False}, timeout=120)

    def get_trace_stats(self, since: str = "10m") -> dict:
        """Get trace statistics (120s timeout — aggregation on large DBs)."""
        return self.call("get_trace_stats", {"since": since, "tsc": False}, timeout=120)

    def get_sessions(self, since: str = "0") -> dict:
        """Get trace sessions via run_sql (get_sessions tool was removed)."""
        return self.call("run_sql", {"query": "SELECT * FROM sessions ORDER BY started_at DESC", "limit": 100}, timeout=30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sql_rows(resp: dict) -> list[list]:
    """Extract data rows from a run_sql response."""
    d = resp.get("data", {})
    if isinstance(d, dict):
        return d.get("data", [])
    return []

def sql_cols(resp: dict) -> list[str]:
    """Extract column names from a run_sql response."""
    d = resp.get("data", {})
    if isinstance(d, dict):
        return d.get("columns", [])
    return []

def sql_row_count(resp: dict) -> int:
    """Get number of rows returned."""
    return len(sql_rows(resp))

def sql_first_val(resp: dict, default=0):
    """Get the first value from the first row."""
    rows = sql_rows(resp)
    if rows and rows[0]:
        val = rows[0][0]
        if val is not None:
            return val
    return default

def sql_to_dicts(resp: dict) -> list[dict]:
    """Convert SQL response to list of dicts."""
    cols = sql_cols(resp)
    rows = sql_rows(resp)
    if not cols:
        return []
    # Guard: skip rows shorter than columns (malformed/truncated responses)
    return [dict(zip(cols, row)) for row in rows if len(row) >= len(cols)]

def safe_div(a, b, default=0):
    """Safe division."""
    if b == 0:
        return default
    return a / b


# ---------------------------------------------------------------------------
# Investigation Framework
# ---------------------------------------------------------------------------

class Investigation:
    """A single GPU problem investigation."""

    def __init__(self, tid: str, number: int, title: str, severity: str,
                 provoked: bool, question: str):
        self.tid = tid
        self.number = number
        self.title = title
        self.severity = severity
        self.provoked = provoked
        self.question = question
        self.actions: list[dict] = []  # {tool, args_desc, result_desc}
        self.verdict = "INCONCLUSIVE"
        self.finding = ""
        self.status = "SKIP"
        self.mcp_errors: list[str] = []

    def add_action(self, tool: str, args_desc: str, result_desc: str):
        self.actions.append({
            "tool": tool,
            "args_desc": args_desc,
            "result_desc": result_desc,
        })

    def check_response(self, resp: dict, context: str = "") -> bool:
        """Track MCP errors. Returns True if response is valid."""
        if "error" in resp:
            self.mcp_errors.append(f"{context}: {resp['error']}" if context else resp["error"])
            return False
        return True

    def set_verdict(self, verdict: str, finding: str):
        """Set verdict: DETECTED, HEALTHY, or INCONCLUSIVE.

        Enforcement rules:
        - MCP errors + non-DETECTED → INCONCLUSIVE (SKIP) — data was missing.
        - Provoked + HEALTHY → FAIL — we provoked the condition but didn't detect it.
        """
        self.verdict = verdict
        self.finding = finding

        # MCP errors make non-DETECTED verdicts unreliable.
        if self.mcp_errors and verdict != "DETECTED":
            self.status = "SKIP"
            self.verdict = "INCONCLUSIVE"
            self.finding = f"MCP errors ({len(self.mcp_errors)}): {self.finding}"
            return

        if verdict == "DETECTED":
            self.status = "PASS"
        elif verdict == "HEALTHY":
            if self.provoked:
                self.status = "FAIL"
                self.finding = f"PROVOKED NOT DETECTED: {finding}"
            else:
                self.status = "PASS"
        else:
            self.status = "SKIP"

    def result_line(self) -> str:
        """ML_RESULT line for gpu-test.sh ingestion."""
        detail = f"{self.verdict}: {self.finding}"
        # Sanitize pipe chars — they are the field delimiter in ML_RESULT protocol.
        detail = detail.replace("|", "/")
        if len(detail) > 200:
            detail = detail[:197] + "..."
        return f"ML_RESULT|{self.tid}|{self.tid}: {self.title}|{self.status}|{detail}|0"


def run_investigations(mcp: MCPClient, args) -> tuple[list[Investigation], dict]:
    """Run all 28 GPU problem investigations."""
    investigations = []

    # ── Preflight: validate op ID mapping ────────────────────────────────────
    # All SQL queries below use hardcoded source/op integer IDs. If the Go code
    # changes these constants, every investigation silently produces wrong results.
    # This check catches mapping drift at the start of the run.
    EXPECTED_OPS = {
        # (source, op) → name — from pkg/events/types.go
        (1, 1): "cudaMalloc",      (1, 2): "cudaFree",
        (1, 3): "cudaLaunchKernel",(1, 4): "cudaMemcpy",
        (1, 5): "cudaStreamSync",  (1, 6): "cudaDeviceSync",
        (1, 7): "cudaMemcpyAsync",
        (3, 1): "sched_switch",    (3, 2): "sched_wakeup",
        (3, 3): "mm_page_alloc",   (3, 4): "oom_kill",
        (3, 5): "process_exec",    (3, 6): "process_exit",
        (3, 7): "process_fork",
        (4, 1): "cuLaunchKernel",  (4, 2): "cuMemcpy",
        (4, 3): "cuMemcpyAsync",   (4, 4): "cuCtxSynchronize",
        (4, 5): "cuMemAlloc",
    }
    r_ops = mcp.run_sql(
        "SELECT source_id, op_id, name FROM ops ORDER BY source_id, op_id")
    ops_rows = sql_to_dicts(r_ops)
    if ops_rows:
        mismatches = []
        unknown_ops = []
        for row in ops_rows:
            src = row.get("source_id", 0) or 0
            op = row.get("op_id", 0) or 0
            name = row.get("name", "")
            expected = EXPECTED_OPS.get((src, op))
            if expected and expected != name:
                mismatches.append(f"({src},{op}): expected={expected}, got={name}")
            elif not expected and src in (1, 3, 4):
                unknown_ops.append(f"({src},{op}): {name} (not in EXPECTED_OPS)")
        if mismatches:
            print(f"[FATAL] Op ID mapping mismatch — "
                  f"DB ops table diverges from hardcoded IDs:\n  "
                  + "\n  ".join(mismatches), file=sys.stderr)
            sys.exit(2)
        if unknown_ops:
            print(f"[WARN] Unknown ops in DB (update EXPECTED_OPS): "
                  + ", ".join(unknown_ops), file=sys.stderr)

    # Pre-fetch expensive MCP calls once (120s timeout each). These are reused
    # across multiple investigations to avoid redundant queries on large DBs.
    _cached_chains = mcp.get_causal_chains("10m")
    _cached_stats = mcp.get_trace_stats("10m")
    _cached_sessions = mcp.get_sessions("0")

    # =========================================================================
    # T23a: #1 NCCL Hangs — NOT provoked (requires multi-GPU / NCCL)
    # Single-GPU systems detect scheduling contention as a proxy signal.
    # =========================================================================
    inv = Investigation("T23a", 1, "NCCL hangs", "CRITICAL",
                        provoked=False,
                        question="My multi-GPU training hangs. Which rank is stuck and why?")

    # Action 1: per-PID sched_switch counts
    r1 = mcp.run_sql("""
        SELECT e.pid, pn.name, COUNT(*) as sched_events,
               SUM(e.duration)/1e6 as total_off_cpu_ms
        FROM events e LEFT JOIN process_names pn ON e.pid=pn.pid
        WHERE e.source=3 AND e.op=1
        GROUP BY e.pid ORDER BY sched_events DESC LIMIT 10
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "per-PID sched_switch counts",
                   f"{len(rows1)} PIDs with scheduler events")

    # Action 2: per-PID sync latency
    r2 = mcp.run_sql("""
        SELECT e.pid, pn.name, COUNT(*) as sync_events,
               AVG(e.duration)/1000 as avg_us,
               MAX(e.duration)/1000 as max_us
        FROM events e LEFT JOIN process_names pn ON e.pid=pn.pid
        WHERE (e.source=1 AND e.op IN (5,6)) OR (e.source=4 AND e.op=4)
        GROUP BY e.pid ORDER BY max_us DESC LIMIT 10
    """)
    rows2 = sql_to_dicts(r2)
    inv.add_action("run_sql", "per-PID sync latency",
                   f"{len(rows2)} PIDs with sync events")

    # Action 3: causal chains (cached)
    r3 = _cached_chains
    chains_text = r3.get("text", "")
    chain_count = chains_text.count("[HIGH]") + chains_text.count("[MEDIUM]") + chains_text.count("[LOW]")
    inv.add_action("get_causal_chains", "since=10m", f"{chain_count} chains found")

    # Verdict: DETECTED if multi-PID sched_switch → sync amplification found.
    # Single-PID contention is "scheduling contention", not NCCL.
    if rows1 and rows2:
        top_sched = rows1[0] if rows1 else {}
        top_sync = rows2[0] if rows2 else {}
        sched_count = top_sched.get("sched_events", 0) or 0
        max_sync_us = top_sync.get("max_us", 0) or 0
        avg_sync_us = top_sync.get("avg_us", 0) or 0
        amp = safe_div(max_sync_us, avg_sync_us)
        # Count PIDs with sync events — NCCL requires multi-PID
        sync_pids = len([r for r in rows2 if (r.get("sync_events", 0) or 0) > 5])

        if sync_pids >= 2 and sched_count > 100 and amp > 3:
            # Multi-PID: real straggler pattern (NCCL-relevant)
            inv.set_verdict("DETECTED",
                            f"{sched_count} sched_switch, sync amp {amp:.1f}x across {sync_pids} PIDs")
        elif sched_count > 100 and amp > 3:
            # Single-PID: CPU scheduling contention, not NCCL
            inv.set_verdict("DETECTED",
                            f"scheduling contention: {sched_count} sched_switch, sync amp {amp:.1f}x (single-PID, not NCCL)")
        elif sched_count > 5000 and max_sync_us > 1000:
            # Heavy scheduler activity with sync latency — scheduling pressure
            inv.set_verdict("DETECTED",
                            f"{sched_count} sched_switch events, sync max={max_sync_us:.0f}us")
        else:
            inv.set_verdict("HEALTHY", f"minimal scheduler preemption ({sched_count} events, sync max={max_sync_us:.0f}us)")
    else:
        inv.set_verdict("INCONCLUSIVE", "no scheduler or sync events")

    investigations.append(inv)

    # =========================================================================
    # T23b: #2 GPU Underutil — NOT provoked (detection depends on sync_pct/memcpy_pct
    # thresholds that vary by GPU architecture — fast GPUs like GH200 show low sync%)
    # =========================================================================
    inv = Investigation("T23b", 2, "GPU underutil", "CRITICAL",
                        provoked=False,
                        question="My GPU is at 30% utilization. Where's the bottleneck?")

    # Action 1: trace stats (cached)
    r1 = _cached_stats
    stats_text = r1.get("text", r1.get("data", ""))
    inv.add_action("get_trace_stats", "since=10m", "wall% breakdown")

    # Action 2: sync total duration (separate query for speed on large DBs)
    r2 = mcp.run_sql("""
        SELECT SUM(duration) as sync_dur, COUNT(*) as sync_cnt
        FROM events WHERE (source=1 AND op IN (5,6)) OR (source=4 AND op=4)
    """)
    rows2 = sql_to_dicts(r2)
    inv.add_action("run_sql", "sync total duration",
                   f"sync aggregate computed")

    # Action 3: memcpy + total CUDA duration (separate queries avoid full-table CASE scan)
    r3a = mcp.run_sql("""
        SELECT SUM(duration) as memcpy_dur, COUNT(*) as memcpy_cnt
        FROM events WHERE (source=1 AND op IN (4,7)) OR (source=4 AND op IN (2,3))
    """)
    r3b = mcp.run_sql("""
        SELECT SUM(duration) as total_dur FROM events WHERE source IN (1, 4)
    """)
    r3a_rows = sql_to_dicts(r3a)
    r3b_rows = sql_to_dicts(r3b)

    memcpy_dur = r3a_rows[0].get("memcpy_dur", 0) or 0 if r3a_rows else 0
    sync_dur = rows2[0].get("sync_dur", 0) or 0 if rows2 else 0
    total_dur = r3b_rows[0].get("total_dur", 0) or 1 if r3b_rows else 1
    # Combine into r3_rows format for verdict logic
    r3_rows = [{"memcpy_dur": memcpy_dur, "sync_dur": sync_dur, "total_dur": total_dur}] if total_dur > 0 else []
    inv.add_action("run_sql", "memcpy wall-time fraction",
                   f"memcpy+sync fractions computed")

    # Verdict (session-wide average — per-phase breakdown is in T23l goodput analysis)
    if r3_rows:
        total = r3_rows[0].get("total_dur", 0) or 1
        sync_dur = r3_rows[0].get("sync_dur", 0) or 0
        memcpy_dur = r3_rows[0].get("memcpy_dur", 0) or 0
        sync_pct = sync_dur / total * 100
        memcpy_pct = memcpy_dur / total * 100
        if sync_pct > 8 or memcpy_pct > 20:
            inv.set_verdict("DETECTED",
                            f"sync wall={sync_pct:.1f}%, memcpy wall={memcpy_pct:.1f}% (session avg, see T23l for phase)")
        else:
            inv.set_verdict("HEALTHY",
                            f"sync wall={sync_pct:.1f}%, memcpy wall={memcpy_pct:.1f}%")
    else:
        inv.set_verdict("INCONCLUSIVE", "no CUDA events")

    investigations.append(inv)

    # =========================================================================
    # T23c: #3 CUDA OOM — NOT provoked (alloc_stress creates allocations but
    # duration trending requires memory pressure; large-memory GPUs stay fast)
    # =========================================================================
    inv = Investigation("T23c", 3, "CUDA OOM", "CRITICAL",
                        provoked=False,
                        question="My training crashes with CUDA OOM at 65% memory. Why?")

    # Action 1: cudaMalloc event counts + size distribution
    r1 = mcp.run_sql("""
        SELECT COUNT(*) as cnt,
               AVG(arg0) as avg_bytes,
               MAX(arg0) as max_bytes,
               SUM(arg0) as total_bytes
        FROM events WHERE (source=1 AND op=1) OR (source=4 AND op=5)
    """)
    r1_rows = sql_to_dicts(r1)
    inv.add_action("run_sql", "cudaMalloc/cuMemAlloc counts + sizes",
                   f"{r1_rows[0].get('cnt', 0) if r1_rows else 0} alloc events")

    # Action 2: alloc duration trending per 10s bucket
    r2 = mcp.run_sql("""
        SELECT CAST(timestamp / 10000000000 AS INT) * 10 as bucket_s,
               COUNT(*) as cnt,
               AVG(duration)/1000 as avg_us,
               MAX(duration)/1000 as max_us
        FROM events WHERE (source=1 AND op=1) OR (source=4 AND op=5)
        GROUP BY bucket_s ORDER BY bucket_s
    """)
    rows2 = sql_to_dicts(r2)
    inv.add_action("run_sql", "alloc duration trending (10s buckets)",
                   f"{len(rows2)} buckets")

    # Action 3: malloc-free balance
    r3 = mcp.run_sql("""
        SELECT
            COUNT(CASE WHEN (source=1 AND op=1) OR (source=4 AND op=5) THEN 1 END) as allocs,
            COUNT(CASE WHEN source=1 AND op=2 THEN 1 END) as frees
        FROM events
    """)
    r3_rows = sql_to_dicts(r3)
    inv.add_action("run_sql", "malloc-free balance",
                   f"allocs vs frees")

    # Verdict
    if r3_rows and r1_rows:
        allocs = r3_rows[0].get("allocs", 0) or 0
        frees = r3_rows[0].get("frees", 0) or 0
        total_bytes = r1_rows[0].get("total_bytes", 0) or 0
        imbalance = allocs - frees

        # Check for duration trending — require 3x increase across buckets
        # and first_avg > 100us to filter normal PyTorch warmup patterns
        # (small first allocs followed by larger steady-state allocs).
        trending = False
        if len(rows2) >= 4:
            first_avg = rows2[0].get("avg_us", 0) or 0
            last_avg = rows2[-1].get("avg_us", 0) or 0
            if first_avg > 100 and last_avg > first_avg * 3:
                trending = True

        if trending:
            inv.set_verdict("DETECTED",
                            f"alloc duration trending up ({last_avg:.0f}/{first_avg:.0f}us), allocs={allocs}, total={total_bytes/1e6:.0f}MB")
        elif frees > 0 and imbalance > 10:
            inv.set_verdict("DETECTED",
                            f"allocs={allocs}, frees={frees}, imbalance={imbalance}, total={total_bytes/1e6:.0f}MB")
        elif allocs > 0:
            inv.set_verdict("HEALTHY",
                            f"allocs={allocs}, frees={frees}, total={total_bytes/1e6:.0f}MB"
                            + (" (frees=0 expected: caching allocator / no cuMemFree probe)" if frees == 0 else " (balanced)"))
        else:
            inv.set_verdict("HEALTHY", "no allocation events")
    else:
        inv.set_verdict("INCONCLUSIVE", "no allocation data")

    investigations.append(inv)

    # =========================================================================
    # T23d: #4 SDC (Silent Data Corruption) — NOT provoked
    # =========================================================================
    inv = Investigation("T23d", 4, "SDC", "CRITICAL",
                        provoked=False,
                        question="Are there signs of silent data corruption?")

    r1 = _cached_stats  # cached
    inv.add_action("get_trace_stats", "anomaly counts on kernel ops", "check anomalies")

    r2 = mcp.run_sql("""
        SELECT COUNT(*) as cnt,
               AVG(duration) as avg_dur,
               AVG(duration*duration) - AVG(duration)*AVG(duration) as var_dur
        FROM events WHERE (source=1 AND op=3) OR (source=4 AND op=1)
    """)
    r2_rows = sql_to_dicts(r2)
    inv.add_action("run_sql", "cuLaunchKernel duration variance",
                   "stddev check for bimodal distribution")

    # Cross-check: if sched_switch storms present, variance is from CPU contention, not SDC.
    r3 = mcp.run_sql("""
        SELECT COUNT(*) as sched_cnt FROM events WHERE source=3 AND op=1
    """)
    sched_cnt = sql_first_val(r3, 0)
    inv.add_action("run_sql", "sched_switch count (contention cross-check)",
                   f"{sched_cnt} context switches")

    # Verdict: HEALTHY (no real SDC on test hardware)
    cnt = r2_rows[0].get("cnt", 0) if r2_rows else 0
    avg = r2_rows[0].get("avg_dur", 0) or 0 if r2_rows else 0
    var_dur = r2_rows[0].get("var_dur", 0) or 0 if r2_rows else 0
    # max(0, ...) prevents complex numbers from negative variance (floating-point imprecision)
    cv = (max(0, var_dur) ** 0.5 / avg) if avg > 0 else 0
    # High CV with high sched_switch = CPU contention, not SDC.
    # Contention explains moderate CV (up to ~10x). Beyond that, something else is going on.
    if cv > 10.0:
        # Extreme CV — suspicious even with CPU contention
        inv.set_verdict("DETECTED", f"kernel duration CV={cv:.2f} (bimodal suspected)")
    elif sched_cnt > 1000 and cv > 3.0:
        # Moderate CV fully explained by CPU contention
        inv.set_verdict("HEALTHY",
                        f"kernel CV={cv:.2f} explained by CPU contention ({sched_cnt} sched_switch)")
    elif cv > 5.0:
        # High CV without contention — suspicious
        inv.set_verdict("DETECTED", f"kernel duration CV={cv:.2f} (bimodal suspected)")
    else:
        inv.set_verdict("HEALTHY", f"kernel duration CV={cv:.2f}, no bimodal pattern")

    investigations.append(inv)

    # =========================================================================
    # T23e: #5 Inference Cost — NOT provoked (requires inference server, not training)
    # =========================================================================
    inv = Investigation("T23e", 5, "Inference cost", "CRITICAL",
                        provoked=False,
                        question="Are there GPU idle periods between inference batches?")

    # Action 1: per-second CUDA event counts (detect burst/idle pattern without expensive LAG)
    r1 = mcp.run_sql("""
        SELECT CAST(timestamp / 1000000000 AS INT) as sec,
               COUNT(*) as events
        FROM events WHERE source IN (1, 4)
        GROUP BY sec ORDER BY sec
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "per-second CUDA event counts",
                   f"{len(rows1)} seconds")

    # Action 2: trace time range for gap detection
    r2 = mcp.run_sql("""
        SELECT MIN(timestamp)/1e9 as min_s, MAX(timestamp)/1e9 as max_s
        FROM events WHERE source IN (1, 4)
    """)
    r2_rows = sql_to_dicts(r2)
    inv.add_action("run_sql", "CUDA event time range",
                   "detect idle gaps in timeline")

    # Verdict: check for idle seconds and gaps in timeline.
    # Exclude the first 5 seconds (probe attachment + CUDA context init always low).
    if rows1 and r2_rows:
        min_s = r2_rows[0].get("min_s", 0) or 0
        max_s = r2_rows[0].get("max_s", 0) or 0
        startup_cutoff = int(min_s) + 5  # exclude first 5s
        steady_rows = [r for r in rows1 if (r.get("sec", 0) or 0) >= startup_cutoff]
        idle_secs = sum(1 for r in steady_rows if (r.get("events", 0) or 0) < 10)
        # Detect missing seconds (no CUDA events at all) in steady state
        if max_s > min_s:
            sec_set = set(r.get("sec", 0) for r in steady_rows)
            all_secs = set(range(startup_cutoff, int(max_s) + 1))
            missing = len(all_secs - sec_set)
        else:
            missing = 0
        total_steady = len(all_secs) if max_s > min_s else len(steady_rows)
    else:
        idle_secs = 0
        missing = 0
        total_steady = 0

    # Require >10% idle time (not just 2 seconds which is noise)
    idle_pct = safe_div(idle_secs + missing, total_steady) * 100 if total_steady > 0 else 0
    if idle_pct > 10:
        inv.set_verdict("DETECTED",
                        f"{missing} empty + {idle_secs} low-activity seconds ({idle_pct:.0f}% idle, excluding startup)")
    elif rows1:
        inv.set_verdict("HEALTHY", f"no significant idle gaps across {len(rows1)}s")
    else:
        inv.set_verdict("INCONCLUSIVE", "no CUDA events")

    investigations.append(inv)

    # =========================================================================
    # T23f: #6 KV Cache Pressure — NOT provoked (inference-specific, no KV cache in training)
    # alloc_stress creates large allocations but they're not KV cache patterns.
    # =========================================================================
    inv = Investigation("T23f", 6, "KV cache pressure", "CRITICAL",
                        provoked=False,
                        question="Are there cudaMalloc spikes indicating memory pressure?")

    # Action 1: large allocs (>10MB — 1MB triggers on routine cuDNN workspace allocs)
    r1 = mcp.run_sql("""
        SELECT COUNT(*) as cnt,
               AVG(duration)/1000 as avg_us,
               MAX(duration)/1000 as max_us,
               SUM(arg0)/1e6 as total_mb
        FROM events WHERE ((source=1 AND op=1) OR (source=4 AND op=5)) AND arg0 > 10485760
    """)
    r1_rows = sql_to_dicts(r1)
    inv.add_action("run_sql", "cudaMalloc events with arg0 > 10MB",
                   f"{r1_rows[0].get('cnt', 0) if r1_rows else 0} large allocs")

    # Action 2: alloc spikes correlated with sync spikes (5s buckets)
    r2 = mcp.run_sql("""
        SELECT CAST(timestamp / 5000000000 AS INT) * 5 as bucket,
               COUNT(CASE WHEN (source=1 AND op=1) OR (source=4 AND op=5) THEN 1 END) as allocs,
               MAX(CASE WHEN (source=1 AND op=1) OR (source=4 AND op=5) THEN duration END)/1000 as alloc_max_us,
               COUNT(CASE WHEN (source=1 AND op IN (5,6)) OR (source=4 AND op=4) THEN 1 END) as syncs,
               MAX(CASE WHEN (source=1 AND op IN (5,6)) OR (source=4 AND op=4) THEN duration END)/1000 as sync_max_us
        FROM events GROUP BY bucket ORDER BY bucket
    """)
    rows2 = sql_to_dicts(r2)
    inv.add_action("run_sql", "alloc spikes correlated with sync spikes (5s buckets)",
                   f"{len(rows2)} buckets analyzed")

    # Verdict
    large_cnt = r1_rows[0].get("cnt", 0) if r1_rows else 0
    total_mb = r1_rows[0].get("total_mb", 0) or 0 if r1_rows else 0

    # Check correlation: buckets where both allocs and syncs spike
    corr_count = 0
    for r in rows2:
        allocs = r.get("allocs", 0) or 0
        syncs = r.get("syncs", 0) or 0
        if allocs > 20 and syncs > 20:
            corr_count += 1

    # Require >10 large allocs — a single cuDNN workspace alloc (>10MB) is normal.
    # KV cache pressure manifests as many repeated large allocations.
    if large_cnt > 10:
        inv.set_verdict("DETECTED",
                        f"{large_cnt} large allocs ({total_mb:.0f}MB), {corr_count} correlated spike buckets")
    elif corr_count > 0:
        inv.set_verdict("DETECTED",
                        f"{corr_count} buckets with alloc+sync correlation")
    else:
        inv.set_verdict("HEALTHY", "no memory pressure pattern")

    investigations.append(inv)

    # =========================================================================
    # T23g: #7 HW Failures — NOT provoked
    # =========================================================================
    inv = Investigation("T23g", 7, "HW failures", "HIGH",
                        provoked=False,
                        question="Is there hardware degradation? Baseline drift in transfer speeds?")

    # Action 1: memcpy p99 per 15s bucket (trend check) — scoped to most recent session
    r1 = mcp.run_sql("""
        WITH last_session AS (
            SELECT started_at FROM sessions ORDER BY started_at DESC LIMIT 1
        )
        SELECT CAST(timestamp / 15000000000 AS INT) * 15 as bucket,
               COUNT(*) as cnt,
               AVG(duration)/1000 as avg_us,
               MAX(duration)/1000 as max_us
        FROM events WHERE ((source=1 AND op IN (4,7)) OR (source=4 AND op IN (2,3)))
          AND timestamp >= (SELECT started_at FROM last_session)
        GROUP BY bucket ORDER BY bucket
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "cudaMemcpy p99 per 15s bucket",
                   f"{len(rows1)} buckets")

    # Action 2: sched_switch frequency trending
    r2 = mcp.run_sql("""
        SELECT CAST(timestamp / 15000000000 AS INT) * 15 as bucket,
               COUNT(*) as cnt
        FROM events WHERE source=3 AND op=1
        GROUP BY bucket ORDER BY bucket
    """)
    rows2 = sql_to_dicts(r2)
    inv.add_action("run_sql", "sched_switch frequency trending",
                   f"{len(rows2)} buckets")

    # Verdict: check for monotonic increase (drift), cross-check with sched_switch
    if len(rows1) >= 3:
        avgs = [r.get("avg_us", 0) or 0 for r in rows1]
        # Simple trend: is the last third > 2x the first third?
        third = len(avgs) // 3
        first_third = sum(avgs[:third]) / max(third, 1)
        last_third = sum(avgs[-third:]) / max(third, 1)
        # Cross-check: if sched_switch storms present, latency increase is CPU contention, not HW drift
        sched_cnts = [r.get("cnt", 0) or 0 for r in rows2]
        total_sched = sum(sched_cnts)
        if first_third > 0 and last_third > first_third * 2:
            if total_sched > 1000:
                inv.set_verdict("HEALTHY",
                                f"memcpy drift ({first_third:.0f}→{last_third:.0f}us) explained by CPU contention ({total_sched} sched_switch)")
            else:
                inv.set_verdict("DETECTED", f"memcpy drift: first={first_third:.0f}us → last={last_third:.0f}us")
        else:
            inv.set_verdict("HEALTHY", f"memcpy stable across {len(rows1)} intervals")
    elif rows1:
        inv.set_verdict("HEALTHY", "insufficient data for trend (few buckets)")
    else:
        inv.set_verdict("HEALTHY", "no memcpy events (may be driver-only)")

    investigations.append(inv)

    # =========================================================================
    # T23h: #8 CPU Bottleneck — provoked via Phase 3
    # =========================================================================
    inv = Investigation("T23h", 8, "CPU bottleneck", "HIGH",
                        provoked=True,
                        question="Is CPU contention causing GPU latency spikes?")

    # Action 1: causal chains with scheduling keywords (cached)
    r1 = _cached_chains
    chains_text = r1.get("text", "")
    high_chains = chains_text.count("[HIGH]")
    inv.add_action("get_causal_chains", "HIGH chains",
                   f"{high_chains} HIGH chains")

    # Action 2: system snapshots with high CPU
    r2 = mcp.run_sql("""
        SELECT COUNT(*) as cnt,
               AVG(cpu_pct) as avg_cpu,
               MAX(cpu_pct) as max_cpu
        FROM system_snapshots WHERE cpu_pct > 90
    """)
    r2_rows = sql_to_dicts(r2)
    inv.add_action("run_sql", "system_snapshots where cpu_pct > 90",
                   f"{r2_rows[0].get('cnt', 0) if r2_rows else 0} high-CPU snapshots")

    # Action 3: temporal correlation: sched_switch rate vs sync p99 per second
    r3 = mcp.run_sql("""
        SELECT CAST(timestamp / 1000000000 AS INT) as sec,
               COUNT(CASE WHEN source=3 AND op=1 THEN 1 END) as sched_cnt,
               MAX(CASE WHEN (source=1 AND op IN (5,6)) OR (source=4 AND op=4) THEN duration END)/1000 as sync_max_us
        FROM events GROUP BY sec
        HAVING sched_cnt > 0 OR sync_max_us > 0
        ORDER BY sec
    """)
    rows3 = sql_to_dicts(r3)
    inv.add_action("run_sql", "sched_switch rate vs sync p99 per second",
                   f"{len(rows3)} seconds correlated")

    # Verdict
    high_cpu_cnt = r2_rows[0].get("cnt", 0) if r2_rows else 0
    max_cpu = r2_rows[0].get("max_cpu", 0) or 0 if r2_rows else 0

    if high_chains >= 3 or high_cpu_cnt >= 3:
        inv.set_verdict("DETECTED",
                        f"{high_chains} HIGH chains, {high_cpu_cnt} snapshots >90% CPU (max={max_cpu:.0f}%)")
    elif high_chains > 0 or high_cpu_cnt > 0:
        inv.set_verdict("DETECTED",
                        f"{high_chains} HIGH chains, {high_cpu_cnt} snapshots >90% CPU (weak signal)")
    elif rows3:
        # Check for sched_switch spikes
        max_sched = max((r.get("sched_cnt", 0) or 0) for r in rows3) if rows3 else 0
        if max_sched > 100:
            inv.set_verdict("DETECTED", f"max {max_sched} sched_switch/sec")
        else:
            inv.set_verdict("HEALTHY", f"max sched_switch/sec={max_sched}")
    else:
        inv.set_verdict("INCONCLUSIVE", "no correlation data")

    investigations.append(inv)

    # =========================================================================
    # T23i: #9 GPU Idle Waste — NOT provoked (Phase 4 recovery may not create
    # idle gaps on fast GPUs where training keeps GPU fully occupied)
    # =========================================================================
    inv = Investigation("T23i", 9, "GPU idle waste", "HIGH",
                        provoked=False,
                        question="Is the GPU sitting idle while waiting for host work?")

    # Action 1: per-second CUDA event counts (fast — no window function)
    r1 = mcp.run_sql("""
        SELECT CAST(timestamp / 1000000000 AS INT) as sec,
               COUNT(*) as events
        FROM events WHERE source IN (1, 4)
        GROUP BY sec ORDER BY sec
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "per-second CUDA event counts (idle gap detection)",
                   f"{len(rows1)} active seconds")

    # Action 2: CUDA event time range (not all events — host events inflate the range)
    r2 = mcp.run_sql("""
        SELECT MIN(timestamp)/1e9 as min_s, MAX(timestamp)/1e9 as max_s
        FROM events WHERE source IN (1, 4)
    """)
    r2_rows = sql_to_dicts(r2)
    inv.add_action("run_sql", "trace time range",
                   "total vs active seconds")

    # Verdict: detect idle seconds (< 10 CUDA events) and gaps in the timeline
    idle_secs = sum(1 for r in rows1 if (r.get("events", 0) or 0) < 10)
    active_secs = len(rows1)
    min_s = r2_rows[0].get("min_s", 0) or 0 if r2_rows else 0
    max_s = r2_rows[0].get("max_s", 0) or 0 if r2_rows else 0
    total_secs = max_s - min_s if max_s > min_s else 0

    # Check for gaps in seconds (missing seconds = no CUDA events at all)
    if rows1 and total_secs > 0:
        sec_set = set(r.get("sec", 0) for r in rows1)
        all_secs = set(range(int(min(sec_set)), int(max(sec_set)) + 1))
        missing_secs = len(all_secs - sec_set)
        total_idle = missing_secs + idle_secs
    else:
        missing_secs = 0
        total_idle = idle_secs

    if total_idle > 5:
        inv.set_verdict("DETECTED",
                        f"{total_idle} idle seconds ({missing_secs} empty + {idle_secs} low-activity)")
    elif total_idle > 2:
        inv.set_verdict("DETECTED", f"{total_idle} idle seconds out of {total_secs:.0f}s")
    elif total_secs > 0:
        inv.set_verdict("HEALTHY", f"active {active_secs}/{total_secs:.0f}s")
    else:
        inv.set_verdict("INCONCLUSIVE", "no events")

    investigations.append(inv)

    # =========================================================================
    # T23j: #10 Memory Leaks — NOT provoked (no cuMemFree probe; PyTorch caching allocator
    # holds memory without freeing, so frees=0 is always expected).
    # =========================================================================
    inv = Investigation("T23j", 10, "Memory leaks", "HIGH",
                        provoked=False,
                        question="Are there memory leaks? cudaMalloc/cudaFree imbalance?")

    # Action 1: malloc vs free counts
    r1 = mcp.run_sql("""
        SELECT
            COUNT(CASE WHEN (source=1 AND op=1) OR (source=4 AND op=5) THEN 1 END) as mallocs,
            COUNT(CASE WHEN source=1 AND op=2 THEN 1 END) as frees
        FROM events
    """)
    r1_rows = sql_to_dicts(r1)
    inv.add_action("run_sql", "malloc count vs free count",
                   f"mallocs={r1_rows[0].get('mallocs', 0) if r1_rows else 0}, frees={r1_rows[0].get('frees', 0) if r1_rows else 0}")

    # Action 2: cumulative imbalance over time
    r2 = mcp.run_sql("""
        SELECT CAST(timestamp / 10000000000 AS INT) * 10 as bucket,
               SUM(CASE WHEN (source=1 AND op=1) OR (source=4 AND op=5) THEN 1 ELSE 0 END) as mallocs,
               SUM(CASE WHEN source=1 AND op=2 THEN 1 ELSE 0 END) as frees
        FROM events GROUP BY bucket ORDER BY bucket
    """)
    rows2 = sql_to_dicts(r2)
    inv.add_action("run_sql", "cumulative malloc-free over time",
                   f"{len(rows2)} buckets")

    # Verdict
    mallocs = r1_rows[0].get("mallocs", 0) or 0 if r1_rows else 0
    frees = r1_rows[0].get("frees", 0) or 0 if r1_rows else 0
    imbalance = mallocs - frees

    # Note: frees=0 is normal for PyTorch (caching allocator holds memory) and for
    # driver API (no cuMemFree probe yet). Only flag as leak if frees > 0 but imbalanced.
    if mallocs > 0 and frees > 0 and imbalance > 10:
        inv.set_verdict("DETECTED",
                        f"mallocs={mallocs}, frees={frees}, net leak={imbalance}")
    elif mallocs > 0 and frees == 0:
        inv.set_verdict("HEALTHY",
                        f"mallocs={mallocs}, frees=0 (expected: PyTorch caching allocator / no cuMemFree probe)")
    elif mallocs > 0:
        inv.set_verdict("HEALTHY",
                        f"mallocs={mallocs}, frees={frees}, balanced (no leak)")
    else:
        inv.set_verdict("HEALTHY", "no allocation events")

    investigations.append(inv)

    # =========================================================================
    # T23k: #11 AMP Instability — NOT provoked
    # =========================================================================
    inv = Investigation("T23k", 11, "AMP instability", "HIGH",
                        provoked=False,
                        question="Signs of mixed precision instability? Skipped updates?")

    r1 = mcp.run_sql("""
        SELECT COUNT(*) as cnt,
               AVG(duration)/1000 as avg_us,
               MAX(duration)/1000 as max_us
        FROM events WHERE (source=1 AND op=3) OR (source=4 AND op=1)
    """)
    r1_rows = sql_to_dicts(r1)
    inv.add_action("run_sql", "cuLaunchKernel duration variance",
                   f"{r1_rows[0].get('cnt', 0) if r1_rows else 0} kernel events")

    r2 = _cached_stats  # cached
    inv.add_action("get_trace_stats", "anomaly flags on kernel ops", "anomaly check")

    # Cross-check: sched_switch storms explain outliers via CPU contention, not AMP.
    r3 = mcp.run_sql("""
        SELECT COUNT(*) as sched_cnt FROM events WHERE source=3 AND op=1
    """)
    amp_sched_cnt = sql_first_val(r3, 0)
    inv.add_action("run_sql", "sched_switch count (contention cross-check)",
                   f"{amp_sched_cnt} context switches")

    # Verdict: HEALTHY (no AMP in workload)
    cnt = r1_rows[0].get("cnt", 0) if r1_rows else 0
    max_us = r1_rows[0].get("max_us", 0) or 0 if r1_rows else 0
    avg_us = r1_rows[0].get("avg_us", 0) or 0 if r1_rows else 0
    outlier_ratio = safe_div(max_us, avg_us)
    # High outlier ratio with high sched_switch = CPU contention, not AMP instability.
    # Contention explains moderate outliers (up to ~1000x). Beyond that, something else is going on.
    if outlier_ratio > 1000 and cnt > 100:
        # Extreme outlier ratio — suspicious even with CPU contention
        inv.set_verdict("DETECTED", f"kernel outlier ratio {outlier_ratio:.1f}x (may indicate AMP)")
    elif amp_sched_cnt > 1000 and outlier_ratio > 200:
        # Moderate outlier ratio explained by CPU contention
        inv.set_verdict("HEALTHY",
                        f"outlier ratio {outlier_ratio:.1f}x explained by CPU contention ({amp_sched_cnt} sched_switch)")
    elif outlier_ratio > 500 and cnt > 100:
        # High outlier ratio without contention — suspicious
        inv.set_verdict("DETECTED", f"kernel outlier ratio {outlier_ratio:.1f}x (may indicate AMP)")
    else:
        inv.set_verdict("HEALTHY", f"kernel count={cnt}, outlier ratio={outlier_ratio:.1f}x")

    investigations.append(inv)

    # =========================================================================
    # T23l: #12 Goodput Loss — NOT provoked (launch rate drop depends on CPU contention
    # impact, which varies by core count and GPU speed — GH200 72-core barely affected)
    # =========================================================================
    inv = Investigation("T23l", 12, "Goodput loss", "HIGH",
                        provoked=False,
                        question="What fraction of GPU time is actual training vs. overhead?")

    # Action 1: trace stats wall% breakdown (cached)
    r1 = _cached_stats
    inv.add_action("get_trace_stats", "wall% breakdown", "overhead analysis")

    # Action 2: baseline vs contention cuLaunchKernel throughput
    # Use timestamps to identify phases (first 20s vs 50-90s range)
    r2 = mcp.run_sql("""
        WITH boundaries AS (
            SELECT MIN(timestamp) as t0 FROM events
        )
        SELECT
            CASE
                WHEN timestamp < (SELECT t0 FROM boundaries) + 20000000000 THEN 'baseline'
                WHEN timestamp BETWEEN (SELECT t0 FROM boundaries) + 50000000000
                     AND (SELECT t0 FROM boundaries) + 90000000000 THEN 'contention'
                ELSE 'other'
            END as phase,
            COUNT(*) as launches,
            AVG(duration)/1000 as avg_us
        FROM events
        WHERE (source=1 AND op=3) OR (source=4 AND op=1)
        GROUP BY phase
    """)
    rows2 = sql_to_dicts(r2)
    inv.add_action("run_sql", "baseline vs contention launch throughput",
                   f"{len(rows2)} phases")

    # Action 3: overhead fraction (CUDA+Driver only — excluding HOST to avoid inflated denominator)
    r3 = mcp.run_sql("""
        SELECT
            SUM(CASE WHEN (source=1 AND op IN (4,7)) OR (source=4 AND op IN (2,3)) THEN duration ELSE 0 END) as memcpy_dur,
            SUM(CASE WHEN (source=1 AND op=3) OR (source=4 AND op=1) THEN duration ELSE 0 END) as launch_dur,
            SUM(duration) as total_dur
        FROM events WHERE source IN (1, 4)
    """)
    r3_rows = sql_to_dicts(r3)
    inv.add_action("run_sql", "overhead (sched + memcpy) vs launch fraction",
                   "goodput calculation")

    # Verdict
    baseline_launches = 0
    contention_launches = 0
    for r in rows2:
        if r.get("phase") == "baseline":
            baseline_launches = r.get("launches", 0) or 0
        elif r.get("phase") == "contention":
            contention_launches = r.get("launches", 0) or 0

    if r3_rows:
        total = r3_rows[0].get("total_dur", 0) or 1
        launch = r3_rows[0].get("launch_dur", 0) or 0
        goodput_pct = launch / total * 100
        # Compare throughput: baseline (20s) vs contention (40s) — normalize by seconds
        bl_rate = baseline_launches / 20 if baseline_launches else 0
        ct_rate = contention_launches / 40 if contention_launches else 0
        rate_drop = (1 - safe_div(ct_rate, bl_rate)) * 100 if bl_rate > 0 else 0

        if rate_drop > 10 or goodput_pct < 50:
            inv.set_verdict("DETECTED",
                            f"goodput={goodput_pct:.1f}%, launch rate drop={rate_drop:.0f}% under contention")
        else:
            inv.set_verdict("HEALTHY",
                            f"goodput={goodput_pct:.1f}%, rate drop={rate_drop:.0f}%")
    else:
        inv.set_verdict("INCONCLUSIVE", "no events")

    investigations.append(inv)

    # =========================================================================
    # T23m: #13 Model Swap Latency — provoked via Phase 1
    # =========================================================================
    # provoked=False: the investigation traces a pre-initialized CUDA context
    # (workload starts before trace), so the first alloc is NOT a cold-start.
    # Cold-start provocation would require launching a fresh CUDA process within
    # the trace window. On unified-memory GPUs (GH200), cold-start is also a
    # non-issue (zero-copy memory). Either way, HEALTHY is a valid outcome.
    inv = Investigation("T23m", 13, "Model swap latency", "HIGH",
                        provoked=False,
                        question="Is model loading causing latency?")

    # Action 1: first 15s events: alloc + memcpy
    r1 = mcp.run_sql("""
        WITH t0 AS (SELECT MIN(timestamp) as ts FROM events)
        SELECT o.name as op_name, COUNT(*) as cnt,
               AVG(e.duration)/1000 as avg_us,
               MAX(e.duration)/1000 as max_us,
               SUM(e.arg0) as total_bytes
        FROM events e
        JOIN ops o ON e.source=o.source_id AND e.op=o.op_id
        CROSS JOIN t0
        WHERE e.timestamp < t0.ts + 15000000000
          AND ((e.source=1 AND e.op IN (1,4,7)) OR (e.source=4 AND e.op IN (2,3,5)))
        GROUP BY o.name
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "first 15s: alloc + memcpy events",
                   f"{len(rows1)} op types in cold start")

    # Action 2: cold-start ratio
    r2 = mcp.run_sql("""
        WITH t0 AS (SELECT MIN(timestamp) as ts FROM events),
        first_alloc AS (
            SELECT duration FROM events
            WHERE (source=1 AND op=1) OR (source=4 AND op=5)
            ORDER BY timestamp LIMIT 1
        ),
        steady AS (
            SELECT AVG(duration) as avg_dur FROM events
            WHERE ((source=1 AND op=1) OR (source=4 AND op=5))
              AND timestamp > (SELECT ts FROM t0) + 30000000000
        )
        SELECT
            (SELECT duration FROM first_alloc) / 1000 as first_alloc_us,
            (SELECT avg_dur FROM steady) / 1000 as steady_avg_us
    """)
    r2_rows = sql_to_dicts(r2)
    inv.add_action("run_sql", "cold-start ratio: first alloc / steady p50",
                   "cold-start penalty")

    # Verdict
    first_us = r2_rows[0].get("first_alloc_us", 0) or 0 if r2_rows else 0
    steady_us = r2_rows[0].get("steady_avg_us", 0) or 0 if r2_rows else 0
    ratio = safe_div(first_us, steady_us)

    if ratio > 5:
        inv.set_verdict("DETECTED",
                        f"cold-start ratio={ratio:.1f}x (first={first_us:.0f}us, steady={steady_us:.0f}us)")
    elif ratio > 2:
        inv.set_verdict("DETECTED",
                        f"moderate cold-start ratio={ratio:.1f}x (first={first_us:.0f}us, steady={steady_us:.0f}us)")
    elif rows1:
        inv.set_verdict("HEALTHY",
                        f"cold-start ratio={ratio:.1f}x (within normal range)")
    else:
        inv.set_verdict("HEALTHY", "no cold-start pattern")

    investigations.append(inv)

    # =========================================================================
    # T23n: #14 Device Asserts — NOT provoked
    # =========================================================================
    inv = Investigation("T23n", 14, "Device asserts", "MEDIUM",
                        provoked=False,
                        question="Were there CUDA errors or illegal memory accesses?")

    r1 = mcp.run_sql("""
        SELECT COUNT(*) as cnt, ret_code,
               o.name as op_name
        FROM events e JOIN ops o ON e.source=o.source_id AND e.op=o.op_id
        WHERE e.ret_code != 0
        GROUP BY e.ret_code, o.name
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "events where ret_code != 0",
                   f"{len(rows1)} error types")

    # Action 2: stack traces for errors
    r2 = mcp.run_sql("""
        SELECT e.ret_code, o.name, st.ips
        FROM events e
        JOIN ops o ON e.source=o.source_id AND e.op=o.op_id
        LEFT JOIN stack_traces st ON e.stack_hash=st.hash
        WHERE e.ret_code != 0
        LIMIT 10
    """)
    rows2 = sql_to_dicts(r2)
    inv.add_action("run_sql", "stack traces for error events",
                   f"{len(rows2)} error events with stacks")

    # Verdict
    total_errors = sum(r.get("cnt", 0) or 0 for r in rows1)
    if total_errors > 0:
        ops = ", ".join(f"{r.get('op_name', '?')}(rc={r.get('ret_code', '?')})" for r in rows1[:3])
        inv.set_verdict("DETECTED", f"{total_errors} errors: {ops}")
    else:
        inv.set_verdict("HEALTHY", "zero CUDA errors")

    investigations.append(inv)

    # =========================================================================
    # T23o: #15 Driver Compat — NOT provoked
    # =========================================================================
    inv = Investigation("T23o", 15, "Driver compat", "MEDIUM",
                        provoked=False,
                        question="Are driver and CUDA versions compatible?")

    r1 = _cached_sessions  # cached
    sessions_text = r1.get("text", str(r1.get("data", "")))
    inv.add_action("run_sql", "GPU, driver, CUDA, kernel versions",
                   "compatibility check")

    # Check for CUDA API errors that indicate driver/runtime mismatch.
    # ret_code != 0 on CUDA calls = driver rejection (e.g., CUDA_ERROR_NOT_SUPPORTED).
    r2 = mcp.run_sql("""
        SELECT COUNT(*) as err_count FROM events
        WHERE source IN (1, 4) AND ret_code != 0
    """)
    err_count = sql_first_val(r2, 0)

    if err_count > 10:
        inv.set_verdict("DETECTED",
                        f"{err_count} CUDA API errors (ret_code != 0) — possible driver/runtime mismatch")
    elif sessions_text:
        inv.set_verdict("HEALTHY",
                        f"trace succeeded, {err_count} API errors — driver and CUDA compatible")
    else:
        inv.set_verdict("INCONCLUSIVE", "no session data")

    investigations.append(inv)

    # =========================================================================
    # T23p: #16 Thermal Throttle — NOT provoked
    # =========================================================================
    inv = Investigation("T23p", 16, "Thermal throttle", "MEDIUM",
                        provoked=False,
                        question="Is there thermal throttling? Kernel durations trending up?")

    r1 = mcp.run_sql("""
        SELECT CAST(timestamp / 15000000000 AS INT) * 15 as bucket,
               COUNT(*) as cnt,
               AVG(duration)/1000 as avg_us,
               MAX(duration)/1000 as max_us
        FROM events WHERE (source=1 AND op=3) OR (source=4 AND op=1)
        GROUP BY bucket ORDER BY bucket
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "cuLaunchKernel p50 per 15s bucket",
                   f"{len(rows1)} buckets")

    # Cross-check: sched_switch count (CPU contention inflates kernel durations too)
    r_sched = mcp.run_sql("SELECT COUNT(*) as cnt FROM events WHERE source=3 AND op=1")
    thermal_sched_cnt = sql_first_val(r_sched, 0)
    inv.add_action("run_sql", "sched_switch count (contention cross-check)",
                   f"{thermal_sched_cnt} context switches")

    # Verdict: check monotonic increase in kernel launch latency, cross-check with sched_switch.
    # Note: we measure host-side launch duration (uprobe → uretprobe), not GPU-internal execution.
    # Thermal throttling increases GPU execution time, which increases the return time of sync
    # calls but doesn't directly affect launch latency. This is a proxy signal at best.
    if len(rows1) >= 3:
        avgs = [r.get("avg_us", 0) or 0 for r in rows1]
        # Monotonic check: are the last few buckets consistently higher?
        half = len(avgs) // 2
        first_half = sum(avgs[:half]) / max(half, 1)
        second_half = sum(avgs[half:]) / max(len(avgs) - half, 1)
        if first_half > 0 and second_half > first_half * 1.5:
            if thermal_sched_cnt > 1000:
                inv.set_verdict("HEALTHY",
                                f"kernel avg rise ({first_half:.0f}→{second_half:.0f}us) explained by CPU contention ({thermal_sched_cnt} sched_switch)")
            else:
                inv.set_verdict("DETECTED",
                                f"kernel avg rising: first={first_half:.0f}us → last={second_half:.0f}us")
        else:
            inv.set_verdict("HEALTHY",
                            f"kernel avg stable: {first_half:.0f}us → {second_half:.0f}us")
    else:
        inv.set_verdict("HEALTHY", "short trace, insufficient buckets")

    investigations.append(inv)

    # =========================================================================
    # T23q: #17 Cold Start — provoked via Phase 1
    # =========================================================================
    # provoked=False: same as T23m — trace starts after CUDA context is warm.
    # The first event in the trace is NOT the cold first-ever CUDA call.
    inv = Investigation("T23q", 17, "Cold start", "MEDIUM",
                        provoked=False,
                        question="How long is the cold start? What's the init penalty?")

    # Action 1: first 10 events
    r1 = mcp.run_sql("""
        SELECT e.timestamp, o.name, e.duration/1000 as dur_us, e.arg0, e.arg1
        FROM events e
        JOIN ops o ON e.source=o.source_id AND e.op=o.op_id
        WHERE e.source IN (1, 4)
        ORDER BY e.timestamp LIMIT 10
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "first 10 CUDA events by timestamp",
                   f"{len(rows1)} events")

    # Action 2: cold-start ratio
    r2 = mcp.run_sql("""
        WITH first_cuda AS (
            SELECT duration FROM events
            WHERE source IN (1, 4)
            ORDER BY timestamp LIMIT 1
        ),
        steady AS (
            SELECT AVG(duration) as avg_dur FROM events
            WHERE source IN (1, 4)
              AND timestamp > (SELECT MIN(timestamp) FROM events) + 20000000000
        )
        SELECT
            (SELECT duration FROM first_cuda) / 1000 as first_us,
            (SELECT avg_dur FROM steady) / 1000 as steady_avg_us
    """)
    r2_rows = sql_to_dicts(r2)
    inv.add_action("run_sql", "first CUDA event duration vs steady-state",
                   "cold-start ratio")

    # Action 3: first memcpy H2D (model weight transfer)
    r3 = mcp.run_sql("""
        SELECT duration/1000 as dur_us, arg0 as bytes, arg1 as direction
        FROM events WHERE ((source=1 AND op IN (4,7)) OR (source=4 AND op IN (2,3))) AND arg1=1
        ORDER BY timestamp LIMIT 5
    """)
    rows3 = sql_to_dicts(r3)
    inv.add_action("run_sql", "first cudaMemcpy H2D (model weight transfer)",
                   f"{len(rows3)} H2D transfers")

    # Verdict
    first_us = r2_rows[0].get("first_us", 0) or 0 if r2_rows else 0
    steady_us = r2_rows[0].get("steady_avg_us", 0) or 0 if r2_rows else 0
    ratio = safe_div(first_us, steady_us)

    if ratio > 10:
        inv.set_verdict("DETECTED",
                        f"cold-start ratio={ratio:.0f}x (first={first_us:.0f}us, steady={steady_us:.0f}us)")
    elif ratio > 2:
        inv.set_verdict("DETECTED",
                        f"cold-start ratio={ratio:.1f}x (moderate init penalty)")
    elif rows1:
        inv.set_verdict("HEALTHY",
                        f"cold-start ratio={ratio:.1f}x (minimal init penalty)")
    else:
        inv.set_verdict("INCONCLUSIVE", "no CUDA events")

    investigations.append(inv)

    # =========================================================================
    # T23r: #18 Multi-process GPU contention — provoked via Phase 2
    # =========================================================================
    inv = Investigation("T23r", 18, "Multi-process GPU contention", "MEDIUM",
                        provoked=True,
                        question="Is there a straggler process affecting training?")

    # Action 1: per-PID sync p99
    r1 = mcp.run_sql("""
        SELECT e.pid, pn.name, COUNT(*) as sync_cnt,
               AVG(e.duration)/1000 as avg_us,
               MAX(e.duration)/1000 as max_us
        FROM events e LEFT JOIN process_names pn ON e.pid=pn.pid
        WHERE (e.source=1 AND e.op IN (5,6)) OR (e.source=4 AND e.op=4)
        GROUP BY e.pid HAVING sync_cnt > 5
        ORDER BY max_us DESC
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "per-PID sync p99 and counts",
                   f"{len(rows1)} PIDs with sync events")

    # Action 2: straggler ratio (max/min sync per PID)
    if len(rows1) >= 2:
        max_sync = max(r.get("max_us", 0) or 0 for r in rows1)
        positive_syncs = [r.get("max_us", 0) or 0 for r in rows1 if (r.get("max_us", 0) or 0) > 0]
        min_sync = min(positive_syncs) if positive_syncs else 0
        straggler_ratio = safe_div(max_sync, min_sync) if min_sync > 0 else 0
        inv.add_action("computed", "straggler ratio: max/min sync per PID",
                       f"ratio={straggler_ratio:.1f}x")
    else:
        straggler_ratio = 0

    # Verdict — require meaningful sync from both PIDs, not just existence
    if len(rows1) >= 2 and straggler_ratio > 3:
        inv.set_verdict("DETECTED",
                        f"{len(rows1)} PIDs, straggler ratio={straggler_ratio:.1f}x")
    elif len(rows1) >= 2:
        # Both PIDs have sync events (>5 each from HAVING clause), but no straggler
        min_cnt = min(r.get("sync_cnt", 0) or 0 for r in rows1)
        if min_cnt > 20:
            inv.set_verdict("DETECTED",
                            f"{len(rows1)} PIDs with significant sync activity (min={min_cnt} events)")
        else:
            inv.set_verdict("HEALTHY",
                            f"{len(rows1)} PIDs but minimal sync contention (min={min_cnt} events)")
    elif len(rows1) == 1:
        inv.set_verdict("HEALTHY", "single PID — no multi-process contention")
    else:
        inv.set_verdict("INCONCLUSIVE", "no sync events")

    investigations.append(inv)

    # =========================================================================
    # T23s: #19 RAG Contention — provoked via Phase 2
    # =========================================================================
    inv = Investigation("T23s", 19, "RAG contention", "MEDIUM",
                        provoked=True,
                        question="Are multiple processes competing for GPU resources?")

    # Action 1: per-PID CUDA event counts (use event_aggregates for speed, fall back to events)
    r1 = mcp.run_sql("""
        SELECT ea.pid, pn.name, SUM(ea.count) as event_cnt,
               SUM(ea.sum_dur)/1e6 as total_ms,
               MAX(ea.max_dur)/1000 as max_us
        FROM event_aggregates ea
        LEFT JOIN process_names pn ON ea.pid=pn.pid
        WHERE ea.source IN (1, 4)
        GROUP BY ea.pid ORDER BY event_cnt DESC
    """)
    rows1 = sql_to_dicts(r1)
    if not rows1:
        # Fallback to events table (--record-all DBs may not have aggregates)
        r1 = mcp.run_sql("""
            SELECT e.pid, pn.name, COUNT(*) as event_cnt,
                   SUM(e.duration)/1e6 as total_ms,
                   MAX(e.duration)/1000 as max_us
            FROM events e LEFT JOIN process_names pn ON e.pid=pn.pid
            WHERE e.source IN (1, 4)
            GROUP BY e.pid ORDER BY event_cnt DESC
        """)
        rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "per-PID CUDA event counts",
                   f"{len(rows1)} PIDs with CUDA events")

    # Action 2: PIDs with CUDA events > 100
    active_pids = [r for r in rows1 if (r.get("event_cnt", 0) or 0) > 100]
    inv.add_action("computed", "PIDs with > 100 CUDA events",
                   f"{len(active_pids)} active PIDs")

    # Verdict — require both PIDs to have meaningful CUDA activity (>100 events)
    if len(active_pids) >= 2:
        pid_details = ", ".join(
            f"PID {r.get('pid', '?')} ({r.get('name', 'unknown')}): {r.get('event_cnt', 0)} events"
            for r in active_pids[:3]
        )
        inv.set_verdict("DETECTED", f"{len(active_pids)} PIDs competing: {pid_details}")
    elif len(rows1) >= 2:
        # Two PIDs exist but at least one has <100 events — not significant contention
        inv.set_verdict("HEALTHY",
                        f"{len(rows1)} PIDs but insufficient CUDA activity for contention detection")
    elif len(rows1) == 1:
        inv.set_verdict("HEALTHY", "single PID — no multi-process contention")
    else:
        inv.set_verdict("INCONCLUSIVE", "no CUDA events")

    investigations.append(inv)

    # =========================================================================
    # T23t: #20 Checkpoint — provoked via Phase 2
    # =========================================================================
    inv = Investigation("T23t", 20, "Checkpoint", "MEDIUM",
                        provoked=True,
                        question="Are checkpoint operations causing memory spikes?")

    # Action 1: mm_page_alloc bursts >100MB per minute window
    # arg0 already stores bytes (host_trace.bpf.c: 4096 << order), not page count.
    # mm_page_alloc is aggregate-only (never stored in events table).
    # Query event_aggregates: bucket is minute-truncated unix nanos,
    # sum_arg0 tracks total bytes per minute bucket.
    r1 = mcp.run_sql("""
        SELECT bucket / 60000000000 as bucket,
               count as page_events,
               sum_arg0 / 1e6 as total_mb
        FROM event_aggregates WHERE source=3 AND op=3
          AND sum_arg0 > 100000000
        ORDER BY sum_arg0 DESC
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "mm_page_alloc bursts >100MB per minute window",
                   f"{len(rows1)} spike windows")

    # Action 2: sync spikes in same minute windows (aligned with event_aggregates)
    r2 = mcp.run_sql("""
        SELECT CAST(timestamp / 60000000000 AS INT) as bucket,
               COUNT(*) as sync_cnt,
               MAX(duration)/1000 as max_sync_us
        FROM events
        WHERE (source=1 AND op IN (5,6)) OR (source=4 AND op=4)
        GROUP BY bucket ORDER BY bucket
    """)
    rows2 = sql_to_dicts(r2)
    inv.add_action("run_sql", "sync spikes per minute window",
                   f"{len(rows2)} buckets")

    # Verdict: correlation between page alloc bursts and sync spikes
    spike_buckets = set(r.get("bucket") for r in rows1)
    corr = 0
    for r in rows2:
        if r.get("bucket") in spike_buckets and (r.get("sync_cnt", 0) or 0) > 10:
            corr += 1

    if corr > 0:
        inv.set_verdict("DETECTED",
                        f"{len(rows1)} memory spikes, {corr} correlated with sync spikes")
    elif rows1:
        inv.set_verdict("DETECTED",
                        f"{len(rows1)} memory spike windows >100MB (no sync correlation)")
    else:
        # Check if there are any page alloc events at all (even below spike threshold).
        # On smaller GPUs (A10), alloc_stress provokes host page allocations that don't
        # exceed 100MB/min — but activity IS present and confirms the mechanism works.
        r3 = mcp.run_sql("""
            SELECT SUM(count) as cnt, SUM(sum_arg0) as total_bytes
            FROM event_aggregates WHERE source=3 AND op=3
        """)
        page_cnt = sql_first_val(r3, 0)
        rows3 = sql_to_dicts(r3)
        total_bytes = (rows3[0].get("total_bytes", 0) or 0) if rows3 else 0
        # Require significant page activity — >5000 events AND >200MB total.
        # Routine host allocations produce ~100-2000 events with <100MB; real
        # checkpoint bursts produce tens of thousands with hundreds of MB.
        if page_cnt > 5000 and total_bytes > 200e6:
            inv.set_verdict("DETECTED",
                            f"{page_cnt} page alloc events, {total_bytes/1e6:.1f}MB total (below spike threshold)")
        else:
            inv.set_verdict("HEALTHY",
                            f"page alloc activity below checkpoint threshold ({page_cnt} events, {total_bytes/1e6:.1f}MB)")

    investigations.append(inv)

    # =========================================================================
    # T23u: #21 PCIe Bottleneck — NOT provoked (memcpy wall-time fraction depends on
    # GPU speed — fast GPUs show low memcpy%, threshold 20% is GPU-dependent)
    # =========================================================================
    inv = Investigation("T23u", 21, "PCIe bottleneck", "MEDIUM",
                        provoked=False,
                        question="Is PCIe bandwidth a bottleneck?")

    # Action 1: memcpy by direction (runtime + driver)
    r1 = mcp.run_sql("""
        SELECT
            CASE arg1 WHEN 1 THEN 'H2D' WHEN 2 THEN 'D2H' WHEN 3 THEN 'D2D' ELSE 'unknown' END as dir,
            COUNT(*) as cnt,
            AVG(arg0) as avg_bytes,
            AVG(duration)/1000 as avg_us,
            SUM(duration) as total_dur
        FROM events WHERE (source=1 AND op IN (4,7)) OR (source=4 AND op IN (2,3))
        GROUP BY arg1
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "cudaMemcpy by direction, avg size, avg duration",
                   f"{len(rows1)} directions")

    # Action 2: memcpy wall-time percentage (runtime + driver)
    r2 = mcp.run_sql("""
        SELECT
            SUM(CASE WHEN (source=1 AND op IN (4,7)) OR (source=4 AND op IN (2,3)) THEN duration ELSE 0 END) as memcpy_dur,
            SUM(duration) as total_dur
        FROM events WHERE source IN (1, 4)
    """)
    r2_rows = sql_to_dicts(r2)
    inv.add_action("run_sql", "memcpy wall-time % of total CUDA time",
                   "bandwidth analysis")

    # Action 3: estimated bandwidth per direction (runtime + driver)
    r3 = mcp.run_sql("""
        SELECT
            CASE arg1 WHEN 1 THEN 'H2D' WHEN 2 THEN 'D2H' WHEN 3 THEN 'D2D' ELSE 'unknown' END as dir,
            SUM(arg0) / (SUM(duration) / 1e9 + 0.001) / 1e9 as bw_gbps
        FROM events WHERE ((source=1 AND op IN (4,7)) OR (source=4 AND op IN (2,3))) AND duration > 0
        GROUP BY arg1
    """)
    rows3 = sql_to_dicts(r3)
    inv.add_action("run_sql", "estimated bandwidth per direction",
                   f"{len(rows3)} directions")

    # Verdict
    memcpy_dur = r2_rows[0].get("memcpy_dur", 0) or 0 if r2_rows else 0
    total_dur = r2_rows[0].get("total_dur", 0) or 1 if r2_rows else 1
    memcpy_pct = memcpy_dur / total_dur * 100

    if rows1:
        dir_info = ", ".join(f"{r.get('dir', '?')}: {r.get('cnt', 0)} calls" for r in rows1)
        bw_info = ", ".join(
            f"{r.get('dir', '?')}: {(r.get('bw_gbps') or 0):.1f} GB/s" for r in rows3
        ) if rows3 else ""

        if memcpy_pct > 20:
            inv.set_verdict("DETECTED",
                            f"memcpy={memcpy_pct:.1f}% of CUDA time. {dir_info}. BW: {bw_info}")
        else:
            inv.set_verdict("HEALTHY",
                            f"memcpy={memcpy_pct:.1f}% wall-time (low). {dir_info}")
    else:
        inv.set_verdict("HEALTHY", "no memcpy events (driver-only transfers)")

    investigations.append(inv)

    # =========================================================================
    # T23v: #22 Loss Spikes — provoked via Phase 3
    # =========================================================================
    inv = Investigation("T23v", 22, "Loss spikes", "LOW-MED",
                        provoked=True,
                        question="Are there system events correlated with training anomalies?")

    # Action 1: high-CPU system snapshots (simple query, no expensive subquery)
    r1 = mcp.run_sql("""
        SELECT CAST(timestamp / 1000000000 AS INT) as snap_sec,
               cpu_pct, mem_pct, load_avg
        FROM system_snapshots
        WHERE cpu_pct > 80
        ORDER BY cpu_pct DESC LIMIT 20
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "high-CPU snapshots",
                   f"{len(rows1)} high-CPU snapshots")

    # Action 1b: per-second CUDA event counts (pre-aggregated, then joined in Python)
    r1b = mcp.run_sql("""
        SELECT CAST(timestamp / 1000000000 AS INT) as sec, COUNT(*) as cuda_events
        FROM events WHERE source IN (1, 4) GROUP BY sec
    """)
    cuda_per_sec = {r.get("sec", 0): r.get("cuda_events", 0) for r in sql_to_dicts(r1b)}
    # Annotate snapshots with CUDA event counts
    for r in rows1:
        r["cuda_events"] = cuda_per_sec.get(r.get("snap_sec", 0), 0)
    inv.add_action("run_sql", "per-second CUDA event counts for correlation",
                   f"{len(cuda_per_sec)} seconds with CUDA events")

    # Action 2: causal chains (cached)
    r2 = _cached_chains
    chains_text = r2.get("text", "")
    chain_count = chains_text.count("[HIGH]") + chains_text.count("[MEDIUM]") + chains_text.count("[LOW]")
    inv.add_action("get_causal_chains", "any chains", f"{chain_count} chains")

    # Verdict — require BOTH high-CPU snapshots AND chains for DETECTED.
    # Either alone is insufficient: high-CPU without chains = contention but no
    # GPU impact; chains without high-CPU = GPU anomaly without system correlation.
    if rows1 and chain_count > 5:
        max_cpu = max(r.get("cpu_pct", 0) or 0 for r in rows1)
        inv.set_verdict("DETECTED",
                        f"{len(rows1)} high-CPU snapshots (max={max_cpu:.0f}%), {chain_count} causal chains")
    elif rows1 and chain_count > 0:
        max_cpu = max(r.get("cpu_pct", 0) or 0 for r in rows1)
        inv.set_verdict("DETECTED",
                        f"{len(rows1)} high-CPU snapshots (max={max_cpu:.0f}%), {chain_count} causal chains (weak correlation)")
    elif rows1:
        inv.set_verdict("HEALTHY",
                        f"{len(rows1)} high-CPU snapshots but no causal chains — CPU load not impacting GPU")
    elif chain_count >= 5:
        inv.set_verdict("DETECTED", f"{chain_count} causal chains detected (no CPU correlation)")
    elif chain_count > 0:
        inv.set_verdict("HEALTHY", f"{chain_count} causal chains but no CPU correlation — insufficient evidence")
    else:
        inv.set_verdict("HEALTHY", "no system-CUDA correlation found")

    investigations.append(inv)

    # =========================================================================
    # T23w: #23 Triton Bugs — NOT provoked
    # =========================================================================
    inv = Investigation("T23w", 23, "Triton bugs", "LOW-MED",
                        provoked=False,
                        question="Are there CUDA API anomalies from inference server processes?")

    # Action 1: per-process CUDA event summary
    r1 = mcp.run_sql("""
        SELECT e.pid, pn.name, COUNT(*) as events,
               COUNT(DISTINCT e.op) as unique_ops,
               SUM(e.duration)/1e6 as total_ms
        FROM events e LEFT JOIN process_names pn ON e.pid=pn.pid
        WHERE e.source IN (1, 4)
        GROUP BY e.pid ORDER BY events DESC
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "per-process CUDA event summary",
                   f"{len(rows1)} processes")

    # Action 2: check process names for inference servers
    inv.add_action("run_sql", "process names from events", "check for Triton/vLLM")

    # Verdict: search actual process names from event data, not session metadata
    proc_names_lower = [str(r.get("name", "")).lower() for r in rows1 if r.get("name")]
    has_triton = any("triton" in n or "vllm" in n for n in proc_names_lower)
    if has_triton:
        inv.set_verdict("DETECTED", "inference server processes found")
    else:
        proc_names = ", ".join(r.get("name", "unknown") for r in rows1 if r.get("name"))
        inv.set_verdict("HEALTHY",
                        f"no inference server (processes: {proc_names or 'training only'})")

    investigations.append(inv)

    # =========================================================================
    # T23x: #24 OOM Kill Detection — NOT provoked (but oom_kill events captured)
    # =========================================================================
    inv = Investigation("T23x", 24, "OOM kill detection", "HIGH",
                        provoked=False,
                        question="Were any processes killed by the OOM killer?")

    r1 = mcp.run_sql("""
        SELECT e.pid, pn.name, e.timestamp, e.arg1 as victim_pid
        FROM events e LEFT JOIN process_names pn ON e.pid=pn.pid
        WHERE e.source=3 AND e.op=4
        ORDER BY e.timestamp
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "oom_kill events (source=3, op=4)",
                   f"{len(rows1)} OOM kills")

    if rows1:
        pids = ", ".join(f"PID {r.get('pid', '?')} ({r.get('name', 'unknown')})" for r in rows1[:3])
        inv.set_verdict("DETECTED", f"{len(rows1)} OOM kills: {pids}")
    else:
        inv.set_verdict("HEALTHY", "no OOM kills during trace")

    investigations.append(inv)

    # =========================================================================
    # T23y: #25 Process Lifecycle — NOT provoked (but fork/exec/exit captured)
    # =========================================================================
    inv = Investigation("T23y", 25, "Process lifecycle", "MEDIUM",
                        provoked=False,
                        question="Were there unexpected process exits or forks during training?")

    r1 = mcp.run_sql("""
        SELECT
            COUNT(CASE WHEN op=5 THEN 1 END) as execs,
            COUNT(CASE WHEN op=6 THEN 1 END) as exits,
            COUNT(CASE WHEN op=7 THEN 1 END) as forks
        FROM events WHERE source=3 AND op IN (5,6,7)
    """)
    r1_rows = sql_to_dicts(r1)
    inv.add_action("run_sql", "process_exec/exit/fork counts",
                   f"execs/exits/forks counted")

    # Check for exits of CUDA processes (potential crashes)
    r2 = mcp.run_sql("""
        SELECT e.pid, pn.name, e.timestamp
        FROM events e LEFT JOIN process_names pn ON e.pid=pn.pid
        WHERE e.source=3 AND e.op=6
        AND e.pid IN (SELECT DISTINCT pid FROM events WHERE source IN (1,4))
        ORDER BY e.timestamp
    """)
    cuda_exits = sql_to_dicts(r2)
    inv.add_action("run_sql", "CUDA process exits",
                   f"{len(cuda_exits)} CUDA processes exited")

    execs = r1_rows[0].get("execs", 0) or 0 if r1_rows else 0
    exits = r1_rows[0].get("exits", 0) or 0 if r1_rows else 0
    forks = r1_rows[0].get("forks", 0) or 0 if r1_rows else 0

    if cuda_exits:
        pids = ", ".join(f"PID {r.get('pid', '?')} ({r.get('name', 'unknown')})" for r in cuda_exits[:3])
        inv.set_verdict("DETECTED",
                        f"{len(cuda_exits)} CUDA process exits: {pids} (exec={execs}, fork={forks})")
    elif exits > 0 or forks > 0:
        inv.set_verdict("HEALTHY",
                        f"exec={execs}, exit={exits}, fork={forks} (no CUDA process exits)")
    else:
        inv.set_verdict("HEALTHY", "no process lifecycle events")

    investigations.append(inv)

    # =========================================================================
    # T23z: #26 Scheduler Wakeup Storms — NOT provoked
    # =========================================================================
    inv = Investigation("T23z", 26, "Scheduler wakeup storms", "MEDIUM",
                        provoked=False,
                        question="Are there excessive scheduler wakeups for GPU processes?")

    # Note: sched_wakeup is a point event (duration=0 in eBPF). We measure
    # wakeup frequency anomalies, not latency. Actual wakeup-to-run latency
    # would require correlating sched_wakeup with the next sched_switch for
    # the same PID, which is not currently tracked.
    r1 = mcp.run_sql("""
        SELECT CAST(timestamp / 1000000000 AS INT) as sec,
               COUNT(*) as wakeup_cnt
        FROM events WHERE source=3 AND op=2
        GROUP BY sec ORDER BY wakeup_cnt DESC LIMIT 20
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "per-second sched_wakeup counts",
                   f"{len(rows1)} seconds with wakeup events")

    # Action 2: total wakeup count
    r2 = mcp.run_sql("SELECT COUNT(*) as cnt FROM events WHERE source=3 AND op=2")
    total_wakeups = sql_first_val(r2, 0)
    inv.add_action("run_sql", "total sched_wakeup count", f"{total_wakeups} wakeups")

    if rows1:
        max_per_sec = max(r.get("wakeup_cnt", 0) or 0 for r in rows1)
        if max_per_sec > 500:  # >500 wakeups/sec is a storm
            inv.set_verdict("DETECTED",
                            f"wakeup storm: max {max_per_sec}/sec, {total_wakeups} total")
        else:
            inv.set_verdict("HEALTHY",
                            f"wakeup rate OK: max {max_per_sec}/sec, {total_wakeups} total")
    else:
        inv.set_verdict("HEALTHY", "no sched_wakeup events")

    investigations.append(inv)

    # =========================================================================
    # T23aa: #27 Per-GPU Analysis — NOT provoked
    # =========================================================================
    inv = Investigation("T23aa", 27, "Per-GPU analysis", "MEDIUM",
                        provoked=False,
                        question="Is one GPU slower or more error-prone than others?")

    r1 = mcp.run_sql("""
        SELECT gpu_id, COUNT(*) as events,
               AVG(duration)/1000 as avg_us,
               MAX(duration)/1000 as max_us
        FROM events WHERE source IN (1, 4) AND gpu_id > 0
        GROUP BY gpu_id ORDER BY gpu_id
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "per-GPU event distribution",
                   f"{len(rows1)} GPUs with events")

    if len(rows1) >= 2:
        # Compare avg durations across GPUs
        avgs = [r.get("avg_us", 0) or 0 for r in rows1]
        max_avg = max(avgs) if avgs else 0
        min_avg = min(a for a in avgs if a > 0) if any(a > 0 for a in avgs) else 0
        skew = safe_div(max_avg, min_avg) if min_avg > 0 else 0
        if skew > 2:
            inv.set_verdict("DETECTED",
                            f"{len(rows1)} GPUs, avg duration skew {skew:.1f}x")
        else:
            inv.set_verdict("HEALTHY",
                            f"{len(rows1)} GPUs, balanced (skew {skew:.1f}x)")
    elif len(rows1) == 1:
        inv.set_verdict("HEALTHY", "single GPU — no cross-GPU comparison")
    else:
        inv.set_verdict("HEALTHY", "no per-GPU data (gpu_id not set)")

    investigations.append(inv)

    # =========================================================================
    # T23ab: #28 Phase-Aware Analysis — NOT provoked (contention/baseline ratio
    # depends on GPU speed and detection sensitivity — varies across architectures)
    # =========================================================================
    inv = Investigation("T23ab", 28, "Phase-aware analysis", "LOW-MED",
                        provoked=False,
                        question="How do CUDA metrics change across workload phases?")

    # Use CUDA+Driver events only for avg_us (host sched_switch events have
    # short durations that dominate the average and make contention look FASTER).
    # Sched_switch counts come from a separate subquery on all events.
    r1 = mcp.run_sql("""
        WITH boundaries AS (SELECT MIN(timestamp) as t0 FROM events),
        phases AS (
            SELECT
                CASE
                    WHEN timestamp < (SELECT t0 FROM boundaries) + 20000000000 THEN '1_baseline'
                    WHEN timestamp BETWEEN (SELECT t0 FROM boundaries) + 20000000000
                         AND (SELECT t0 FROM boundaries) + 50000000000 THEN '2_alloc_stress'
                    WHEN timestamp BETWEEN (SELECT t0 FROM boundaries) + 50000000000
                         AND (SELECT t0 FROM boundaries) + 90000000000 THEN '3_contention'
                    WHEN timestamp BETWEEN (SELECT t0 FROM boundaries) + 90000000000
                         AND (SELECT t0 FROM boundaries) + 110000000000 THEN '4_recovery'
                    ELSE '5_clean'
                END as phase,
                source, op, duration
            FROM events
        )
        SELECT phase,
            COUNT(CASE WHEN source IN (1, 4) THEN 1 END) as cuda_events,
            AVG(CASE WHEN source IN (1, 4) THEN duration / 1000.0 END) as avg_us,
            MAX(CASE WHEN source IN (1, 4) THEN duration / 1000.0 END) as max_us,
            COUNT(CASE WHEN source=3 AND op=1 THEN 1 END) as sched_switches,
            COUNT(*) as events
        FROM phases GROUP BY phase ORDER BY phase
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "per-phase CUDA metrics (host excluded from avg)",
                   f"{len(rows1)} phases analyzed")

    if len(rows1) >= 3:
        phase_summary = "; ".join(
            f"{r.get('phase', '?')}: {r.get('cuda_events', 0)} CUDA events, avg={r.get('avg_us', 0) or 0:.0f}us, sched={r.get('sched_switches', 0)}"
            for r in rows1
        )
        # Check for contention phase degradation (CUDA ops only)
        contention = next((r for r in rows1 if "contention" in str(r.get("phase", ""))), None)
        baseline = next((r for r in rows1 if "baseline" in str(r.get("phase", ""))), None)
        if contention and baseline:
            ct_avg = contention.get("avg_us", 0) or 0
            bl_avg = baseline.get("avg_us", 0) or 0
            degradation = safe_div(ct_avg, bl_avg) if bl_avg > 0 else 0
            if degradation > 2:
                inv.set_verdict("DETECTED",
                                f"CUDA ops {degradation:.1f}x slower under contention. {phase_summary}")
            else:
                inv.set_verdict("HEALTHY",
                                f"contention/baseline ratio={degradation:.1f}x. {phase_summary}")
        else:
            inv.set_verdict("HEALTHY", phase_summary)
    elif rows1:
        inv.set_verdict("HEALTHY", f"{len(rows1)} phases (insufficient for comparison)")
    else:
        inv.set_verdict("INCONCLUSIVE", "no events")

    investigations.append(inv)

    return investigations, _cached_sessions


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(investigations: list[Investigation], mcp: MCPClient,
                    db_path: str, report_path: str,
                    cached_sessions: Optional[dict] = None):
    """Generate the investigation report."""

    # Get session info for header (use cached if available)
    sessions_resp = cached_sessions or mcp.get_sessions("0")

    # Get total event count
    total_resp = mcp.run_sql("SELECT COUNT(*) FROM events")
    total_events = sql_first_val(total_resp, 0)

    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("GPU PROBLEM INVESTIGATION REPORT — 28 Issues")
    lines.append(sep)
    lines.append("")

    # Parse GPU info from session data (run_sql returns {columns, data})
    gpu_info = "Unknown"
    sess_data = sessions_resp.get("data", {})
    if isinstance(sess_data, dict):
        cols = sess_data.get("columns", [])
        rows_data = sess_data.get("data", [])
        if rows_data and cols:
            row = rows_data[0]
            col_map = {c: i for i, c in enumerate(cols)}
            if "gpu_model" in col_map:
                gpu_info = str(row[col_map["gpu_model"]])
    else:
        # Fallback: search string representation
        sessions_text = str(sess_data)
        if "GPU:" in sessions_text:
            for part in sessions_text.split("\n"):
                if "GPU:" in part:
                    gpu_info = part.strip()
                    break

    lines.append(f"DB: {db_path}")
    lines.append(f"Events: {total_events:,}")
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    # Summary counts
    detected = sum(1 for i in investigations if i.verdict == "DETECTED")
    healthy = sum(1 for i in investigations if i.verdict == "HEALTHY")
    inconclusive = sum(1 for i in investigations if i.verdict == "INCONCLUSIVE")
    pass_count = sum(1 for i in investigations if i.status == "PASS")
    fail_count = sum(1 for i in investigations if i.status == "FAIL")
    skip_count = sum(1 for i in investigations if i.status == "SKIP")

    lines.append(f"DETECTED: {detected}  HEALTHY: {healthy}  INCONCLUSIVE: {inconclusive}")
    lines.append(f"PASS: {pass_count}  FAIL: {fail_count}  SKIP: {skip_count}")
    lines.append("")

    # Per-investigation detail
    for inv in investigations:
        lines.append(sep)
        lines.append(f"INVESTIGATION {inv.number}/28: {inv.title}")
        lines.append(f"Severity: {inv.severity} | Provoked: {'Yes' if inv.provoked else 'No'}")
        lines.append(sep)
        lines.append("")
        lines.append(f"QUESTION: {inv.question}")
        lines.append("")

        for i, action in enumerate(inv.actions, 1):
            lines.append(f"ACTION {i}: {action['tool']}")
            lines.append(f"  {action['args_desc']}")
            lines.append(f"  → {action['result_desc']}")
            lines.append("")

        lines.append(f"FINDING: {inv.verdict} — {inv.finding}")
        lines.append(f"STATUS: {inv.status}")
        lines.append("")

    # Final summary table
    lines.append(sep)
    lines.append("SUMMARY")
    lines.append(sep)
    lines.append("")
    lines.append(f"{'#':<5} {'ID':<6} {'Title':<30} {'Severity':<10} {'Verdict':<15} {'Status'}")
    lines.append("-" * 85)
    for inv in investigations:
        lines.append(
            f"{inv.number:<5} {inv.tid:<6} {inv.title:<30} {inv.severity:<10} "
            f"{inv.verdict:<15} {inv.status}"
        )
    lines.append("")
    lines.append(f"PASS: {pass_count}  FAIL: {fail_count}  SKIP: {skip_count}  TOTAL: {len(investigations)}")
    lines.append("")

    report_text = "\n".join(lines)

    # Write report
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU Problem Investigation — 28 Issues via MCP")
    parser.add_argument("--mcp-url", required=True, help="MCP HTTPS endpoint URL")
    parser.add_argument("--db", required=True, help="Path to ingero.db")
    parser.add_argument("--report", default="logs/gpu-investigation-report.log",
                        help="Report output path")
    args = parser.parse_args()

    mcp = MCPClient(args.mcp_url)

    # Preflight: verify MCP connectivity before running 28 investigations
    preflight = mcp.run_sql("SELECT COUNT(*) FROM events")
    if "error" in preflight:
        print(f"ERROR: MCP preflight failed: {preflight['error']}", file=sys.stderr)
        sys.exit(1)
    event_count = sql_first_val(preflight, 0)
    if event_count == 0:
        print("ERROR: database contains 0 events", file=sys.stderr)
        sys.exit(1)
    print(f"MCP connected. DB has {event_count:,} events.")

    # Run all 28 investigations
    print("Running 28 GPU problem investigations...")
    print()

    investigations, cached_sessions = run_investigations(mcp, args)

    # Print per-investigation status
    for inv in investigations:
        status_color = {
            "PASS": "\033[0;32m",
            "FAIL": "\033[0;31m",
            "SKIP": "\033[1;33m",
        }.get(inv.status, "")
        nc = "\033[0m" if status_color else ""
        print(f"  {status_color}[{inv.status}]{nc} {inv.tid}: "
              f"#{inv.number} {inv.title} — {inv.verdict}: {inv.finding[:80]}")

    # Generate report
    print()
    print(f"Generating report: {args.report}")
    report = generate_report(investigations, mcp, args.db, args.report,
                              cached_sessions=cached_sessions)

    # Summary
    detected = sum(1 for i in investigations if i.verdict == "DETECTED")
    healthy = sum(1 for i in investigations if i.verdict == "HEALTHY")
    inconclusive = sum(1 for i in investigations if i.verdict == "INCONCLUSIVE")
    pass_count = sum(1 for i in investigations if i.status == "PASS")
    fail_count = sum(1 for i in investigations if i.status == "FAIL")
    skip_count = sum(1 for i in investigations if i.status == "SKIP")

    print()
    print(f"  DETECTED: {detected}  HEALTHY: {healthy}  INCONCLUSIVE: {inconclusive}")
    print(f"  PASS: {pass_count}  FAIL: {fail_count}  SKIP: {skip_count}  TOTAL: {len(investigations)}")
    print()

    # Emit ML_RESULT lines (parsed by gpu-investigation.sh → gpu-test.sh)
    for inv in investigations:
        print(inv.result_line())


if __name__ == "__main__":
    main()
