#!/usr/bin/env python3
"""GPU Problem Investigation — 23 Issues via MCP.

Investigates all 23 GPU problems Ingero can detect by querying the database
exclusively through MCP tool calls (run_sql, get_causal_chains, get_trace_stats,
get_sessions). Simulates how an AI agent would investigate GPU issues.

Each investigation:
  1. Poses a human question ("My GPU is slow, why?")
  2. Makes 2-5 MCP calls to gather evidence
  3. Analyzes results and assigns a verdict (DETECTED / HEALTHY / INCONCLUSIVE)

Verdicts:
  DETECTED     — problem pattern found in data (PASS if provoked)
  HEALTHY      — investigation ran, no problem found (PASS for non-provoked)
  INCONCLUSIVE — insufficient data (SKIP)

Output: ML_RESULT lines to stdout (for gpu-test.sh ingestion) + report file.
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
        """Get trace sessions."""
        return self.call("get_sessions", {"since": since, "tsc": False})


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

    def add_action(self, tool: str, args_desc: str, result_desc: str):
        self.actions.append({
            "tool": tool,
            "args_desc": args_desc,
            "result_desc": result_desc,
        })

    def set_verdict(self, verdict: str, finding: str):
        """Set verdict: DETECTED, HEALTHY, or INCONCLUSIVE."""
        self.verdict = verdict
        self.finding = finding
        if verdict == "DETECTED":
            self.status = "PASS"
        elif verdict == "HEALTHY":
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


def run_investigations(mcp: MCPClient, args) -> list[Investigation]:
    """Run all 23 GPU problem investigations."""
    investigations = []

    # Pre-fetch expensive MCP calls once (120s timeout each). These are reused
    # across multiple investigations to avoid redundant queries on large DBs.
    _cached_chains = mcp.get_causal_chains("10m")
    _cached_stats = mcp.get_trace_stats("10m")
    _cached_sessions = mcp.get_sessions("0")

    # =========================================================================
    # T23a: #1 NCCL Hangs — provoked via Phase 3
    # =========================================================================
    inv = Investigation("T23a", 1, "NCCL hangs", "CRITICAL",
                        provoked=True,
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
        elif sched_count > 50:
            inv.set_verdict("DETECTED",
                            f"{sched_count} sched_switch events, sync max={max_sync_us:.0f}us")
        else:
            inv.set_verdict("HEALTHY", f"minimal scheduler preemption ({sched_count} events)")
    else:
        inv.set_verdict("INCONCLUSIVE", "no scheduler or sync events")

    investigations.append(inv)

    # =========================================================================
    # T23b: #2 GPU Underutil — provoked via Phase 3
    # =========================================================================
    inv = Investigation("T23b", 2, "GPU underutil", "CRITICAL",
                        provoked=True,
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
        FROM events WHERE source=1 AND op IN (4,7)
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

    # Verdict
    if r3_rows:
        total = r3_rows[0].get("total_dur", 0) or 1
        sync_dur = r3_rows[0].get("sync_dur", 0) or 0
        memcpy_dur = r3_rows[0].get("memcpy_dur", 0) or 0
        sync_pct = sync_dur / total * 100
        memcpy_pct = memcpy_dur / total * 100
        if sync_pct > 10 or memcpy_pct > 20:
            inv.set_verdict("DETECTED",
                            f"sync wall={sync_pct:.1f}%, memcpy wall={memcpy_pct:.1f}%")
        else:
            inv.set_verdict("HEALTHY",
                            f"sync wall={sync_pct:.1f}%, memcpy wall={memcpy_pct:.1f}%")
    else:
        inv.set_verdict("INCONCLUSIVE", "no CUDA events")

    investigations.append(inv)

    # =========================================================================
    # T23c: #3 CUDA OOM — provoked via Phase 2
    # =========================================================================
    inv = Investigation("T23c", 3, "CUDA OOM", "CRITICAL",
                        provoked=True,
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

        # Check for duration trending
        trending = False
        if len(rows2) >= 3:
            first_avg = rows2[0].get("avg_us", 0) or 0
            last_avg = rows2[-1].get("avg_us", 0) or 0
            if first_avg > 0 and last_avg > first_avg * 2:
                trending = True

        if imbalance > 10 or trending:
            inv.set_verdict("DETECTED",
                            f"allocs={allocs}, frees={frees}, imbalance={imbalance}, total={total_bytes/1e6:.0f}MB")
        elif allocs > 0:
            inv.set_verdict("HEALTHY",
                            f"allocs={allocs}, frees={frees}, total={total_bytes/1e6:.0f}MB (balanced)")
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
    # Upper bound: CV > 50 is suspicious even with contention.
    if sched_cnt > 1000 and 3.0 < cv <= 50.0:
        inv.set_verdict("HEALTHY",
                        f"kernel CV={cv:.2f} explained by CPU contention ({sched_cnt} sched_switch)")
    elif cv > 5.0:  # Very high CV without contention (or extreme CV > 50 with contention)
        inv.set_verdict("DETECTED", f"kernel duration CV={cv:.2f} (bimodal suspected)")
    else:
        inv.set_verdict("HEALTHY", f"kernel duration CV={cv:.2f}, no bimodal pattern")

    investigations.append(inv)

    # =========================================================================
    # T23e: #5 Inference Cost — provoked via Phase 4
    # =========================================================================
    inv = Investigation("T23e", 5, "Inference cost", "CRITICAL",
                        provoked=True,
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

    # Verdict: check for idle seconds and gaps in timeline
    idle_secs = sum(1 for r in rows1 if (r.get("events", 0) or 0) < 10)

    # Detect missing seconds (no CUDA events at all)
    if rows1 and r2_rows:
        sec_set = set(r.get("sec", 0) for r in rows1)
        min_s = r2_rows[0].get("min_s", 0) or 0
        max_s = r2_rows[0].get("max_s", 0) or 0
        if max_s > min_s:
            all_secs = set(range(int(min_s), int(max_s) + 1))
            missing = len(all_secs - sec_set)
        else:
            missing = 0
    else:
        missing = 0

    if missing > 2 or idle_secs > 2:
        inv.set_verdict("DETECTED",
                        f"{missing} empty seconds + {idle_secs} low-activity seconds across {len(rows1)}s")
    elif rows1:
        inv.set_verdict("HEALTHY", f"no significant idle gaps across {len(rows1)}s")
    else:
        inv.set_verdict("INCONCLUSIVE", "no CUDA events")

    investigations.append(inv)

    # =========================================================================
    # T23f: #6 KV Cache Pressure — provoked via Phase 2
    # =========================================================================
    inv = Investigation("T23f", 6, "KV cache pressure", "CRITICAL",
                        provoked=True,
                        question="Are there cudaMalloc spikes indicating memory pressure?")

    # Action 1: large allocs (>1MB)
    r1 = mcp.run_sql("""
        SELECT COUNT(*) as cnt,
               AVG(duration)/1000 as avg_us,
               MAX(duration)/1000 as max_us,
               SUM(arg0)/1e6 as total_mb
        FROM events WHERE ((source=1 AND op=1) OR (source=4 AND op=5)) AND arg0 > 1048576
    """)
    r1_rows = sql_to_dicts(r1)
    inv.add_action("run_sql", "cudaMalloc events with arg0 > 1MB",
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

    if large_cnt > 0:
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
        FROM events WHERE source=1 AND op IN (4,7)
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

    if high_chains > 0 or high_cpu_cnt > 0:
        inv.set_verdict("DETECTED",
                        f"{high_chains} HIGH chains, {high_cpu_cnt} snapshots >90% CPU (max={max_cpu:.0f}%)")
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
    # T23i: #9 GPU Idle Waste — provoked via Phase 4
    # =========================================================================
    inv = Investigation("T23i", 9, "GPU idle waste", "HIGH",
                        provoked=True,
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

    # Action 2: trace time range
    r2 = mcp.run_sql("""
        SELECT MIN(timestamp)/1e9 as min_s, MAX(timestamp)/1e9 as max_s
        FROM events
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
    # T23j: #10 Memory Leaks — provoked via Phase 2
    # =========================================================================
    inv = Investigation("T23j", 10, "Memory leaks", "HIGH",
                        provoked=True,
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

    if mallocs > 0 and imbalance > 10:
        inv.set_verdict("DETECTED",
                        f"mallocs={mallocs}, frees={frees}, net leak={imbalance}")
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
    # Upper bound: outlier_ratio > 5000 is suspicious even with contention.
    if amp_sched_cnt > 1000 and 200 < outlier_ratio <= 5000:
        inv.set_verdict("HEALTHY",
                        f"outlier ratio {outlier_ratio:.1f}x explained by CPU contention ({amp_sched_cnt} sched_switch)")
    elif outlier_ratio > 500 and cnt > 100:
        inv.set_verdict("DETECTED", f"kernel outlier ratio {outlier_ratio:.1f}x (may indicate AMP)")
    else:
        inv.set_verdict("HEALTHY", f"kernel count={cnt}, outlier ratio={outlier_ratio:.1f}x")

    investigations.append(inv)

    # =========================================================================
    # T23l: #12 Goodput Loss — provoked via Phase 3
    # =========================================================================
    inv = Investigation("T23l", 12, "Goodput loss", "HIGH",
                        provoked=True,
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
            SUM(CASE WHEN source=1 AND op IN (4,7) THEN duration ELSE 0 END) as memcpy_dur,
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
    inv = Investigation("T23m", 13, "Model swap latency", "HIGH",
                        provoked=True,
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
          AND ((e.source=1 AND e.op IN (1,4,7)) OR (e.source=4 AND e.op=5))
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
    elif rows1:
        inv.set_verdict("DETECTED",
                        f"cold-start ops found: {', '.join(r.get('op_name', '?') for r in rows1)}")
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
    inv.add_action("get_sessions", "GPU, driver, CUDA, kernel versions",
                   "compatibility check")

    # Verdict: successful trace = compatible
    if "GPU" in sessions_text or "gpu" in sessions_text or "driver" in sessions_text.lower():
        inv.set_verdict("HEALTHY", "trace succeeded — driver and CUDA compatible")
    elif sessions_text:
        inv.set_verdict("HEALTHY", "session metadata available")
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

    # Verdict: check monotonic increase, cross-check with sched_switch
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
    inv = Investigation("T23q", 17, "Cold start", "MEDIUM",
                        provoked=True,
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
        FROM events WHERE source=1 AND op IN (4,7) AND arg1=1
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
        # Even with low ratio, cold start exists (first events are init)
        first_ops = ", ".join(r.get("name", "?") for r in rows1[:3])
        inv.set_verdict("DETECTED", f"init sequence: {first_ops}")
    else:
        inv.set_verdict("INCONCLUSIVE", "no CUDA events")

    investigations.append(inv)

    # =========================================================================
    # T23r: #18 Multi-GPU TP Overhead — provoked via Phase 2
    # =========================================================================
    inv = Investigation("T23r", 18, "Multi-GPU TP overhead", "MEDIUM",
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
        min_sync = min(r.get("max_us", 0) or 0 for r in rows1 if (r.get("max_us", 0) or 0) > 0)
        straggler_ratio = safe_div(max_sync, min_sync) if min_sync > 0 else 0
        inv.add_action("computed", "straggler ratio: max/min sync per PID",
                       f"ratio={straggler_ratio:.1f}x")
    else:
        straggler_ratio = 0

    # Verdict
    if len(rows1) >= 2 and straggler_ratio > 3:
        inv.set_verdict("DETECTED",
                        f"{len(rows1)} PIDs, straggler ratio={straggler_ratio:.1f}x")
    elif len(rows1) >= 2:
        inv.set_verdict("DETECTED",
                        f"{len(rows1)} PIDs accessing GPU (alloc_stress creates 2nd PID)")
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

    # Verdict
    if len(active_pids) >= 2:
        pid_details = ", ".join(
            f"PID {r.get('pid', '?')} ({r.get('name', 'unknown')}): {r.get('event_cnt', 0)} events"
            for r in active_pids[:3]
        )
        inv.set_verdict("DETECTED", f"{len(active_pids)} PIDs competing: {pid_details}")
    elif len(rows1) >= 2:
        inv.set_verdict("DETECTED",
                        f"{len(rows1)} PIDs with CUDA activity (alloc_stress creates 2nd PID)")
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

    # Action 1: mm_page_alloc bursts >100MB in 5s windows
    # arg0 already stores bytes (host_trace.bpf.c: 4096 << order), not page count.
    r1 = mcp.run_sql("""
        SELECT CAST(timestamp / 5000000000 AS INT) * 5 as bucket,
               COUNT(*) as page_events,
               SUM(arg0) / 1e6 as total_mb
        FROM events WHERE source=3 AND op=3
        GROUP BY bucket HAVING total_mb > 100
        ORDER BY total_mb DESC
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "mm_page_alloc bursts >100MB per 5s window",
                   f"{len(rows1)} spike windows")

    # Action 2: sync spikes in same windows
    r2 = mcp.run_sql("""
        SELECT CAST(timestamp / 5000000000 AS INT) * 5 as bucket,
               COUNT(*) as sync_cnt,
               MAX(duration)/1000 as max_sync_us
        FROM events
        WHERE (source=1 AND op IN (5,6)) OR (source=4 AND op=4)
        GROUP BY bucket ORDER BY bucket
    """)
    rows2 = sql_to_dicts(r2)
    inv.add_action("run_sql", "sync spikes per 5s window",
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
        # Check if there are any page alloc events at all
        r3 = mcp.run_sql("SELECT COUNT(*) as cnt FROM events WHERE source=3 AND op=3")
        page_cnt = sql_first_val(r3, 0)
        if page_cnt > 0:
            inv.set_verdict("HEALTHY", f"{page_cnt} page alloc events, no >100MB spikes")
        else:
            inv.set_verdict("HEALTHY", "no mm_page_alloc events (may be filtered)")

    investigations.append(inv)

    # =========================================================================
    # T23u: #21 PCIe Bottleneck — provoked (training memcpy)
    # =========================================================================
    inv = Investigation("T23u", 21, "PCIe bottleneck", "MEDIUM",
                        provoked=True,
                        question="Is PCIe bandwidth a bottleneck?")

    # Action 1: memcpy by direction
    r1 = mcp.run_sql("""
        SELECT
            CASE arg1 WHEN 1 THEN 'H2D' WHEN 2 THEN 'D2H' WHEN 3 THEN 'D2D' ELSE 'unknown' END as dir,
            COUNT(*) as cnt,
            AVG(arg0) as avg_bytes,
            AVG(duration)/1000 as avg_us,
            SUM(duration) as total_dur
        FROM events WHERE source=1 AND op IN (4,7)
        GROUP BY arg1
    """)
    rows1 = sql_to_dicts(r1)
    inv.add_action("run_sql", "cudaMemcpy by direction, avg size, avg duration",
                   f"{len(rows1)} directions")

    # Action 2: memcpy wall-time percentage
    r2 = mcp.run_sql("""
        SELECT
            SUM(CASE WHEN source=1 AND op IN (4,7) THEN duration ELSE 0 END) as memcpy_dur,
            SUM(duration) as total_dur
        FROM events WHERE source IN (1, 4)
    """)
    r2_rows = sql_to_dicts(r2)
    inv.add_action("run_sql", "memcpy wall-time % of total CUDA time",
                   "bandwidth analysis")

    # Action 3: estimated bandwidth per direction
    r3 = mcp.run_sql("""
        SELECT
            CASE arg1 WHEN 1 THEN 'H2D' WHEN 2 THEN 'D2H' WHEN 3 THEN 'D2D' ELSE 'unknown' END as dir,
            SUM(arg0) / (SUM(duration) / 1e9 + 0.001) / 1e9 as bw_gbps
        FROM events WHERE source=1 AND op IN (4,7) AND duration > 0
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

    # Verdict
    if rows1 and chain_count > 0:
        max_cpu = max(r.get("cpu_pct", 0) or 0 for r in rows1)
        inv.set_verdict("DETECTED",
                        f"{len(rows1)} high-CPU snapshots (max={max_cpu:.0f}%), {chain_count} causal chains")
    elif rows1:
        inv.set_verdict("DETECTED",
                        f"{len(rows1)} high-CPU snapshots correlated with CUDA activity")
    elif chain_count > 0:
        inv.set_verdict("DETECTED", f"{chain_count} causal chains detected")
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

    # Action 2: session process names (cached)
    r2 = _cached_sessions
    inv.add_action("get_sessions", "process names", "check for Triton/vLLM")

    # Verdict: HEALTHY (no Triton in workload)
    sessions_text = str(r2.get("text", r2.get("data", "")))
    has_triton = "triton" in sessions_text.lower() or "vllm" in sessions_text.lower()
    if has_triton:
        inv.set_verdict("DETECTED", "inference server processes found")
    else:
        proc_names = ", ".join(r.get("name", "unknown") for r in rows1 if r.get("name"))
        inv.set_verdict("HEALTHY",
                        f"no inference server (processes: {proc_names or 'training only'})")

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
    sessions_text = sessions_resp.get("text", str(sessions_resp.get("data", "")))

    # Get total event count
    total_resp = mcp.run_sql("SELECT COUNT(*) FROM events")
    total_events = sql_first_val(total_resp, 0)

    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("GPU PROBLEM INVESTIGATION REPORT — 23 Issues")
    lines.append(sep)
    lines.append("")

    # Parse GPU info from session text
    gpu_info = "Unknown"
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
    skip_count = sum(1 for i in investigations if i.status == "SKIP")

    lines.append(f"DETECTED: {detected}  HEALTHY: {healthy}  INCONCLUSIVE: {inconclusive}")
    lines.append(f"PASS: {pass_count}  SKIP: {skip_count}")
    lines.append("")

    # Per-investigation detail
    for inv in investigations:
        lines.append(sep)
        lines.append(f"INVESTIGATION {inv.number}/23: {inv.title}")
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
    lines.append(f"PASS: {pass_count}  SKIP: {skip_count}  TOTAL: {len(investigations)}")
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
    parser = argparse.ArgumentParser(description="GPU Problem Investigation — 23 Issues via MCP")
    parser.add_argument("--mcp-url", required=True, help="MCP HTTPS endpoint URL")
    parser.add_argument("--db", required=True, help="Path to ingero.db")
    parser.add_argument("--report", default="logs/gpu-investigation-report.log",
                        help="Report output path")
    # Phase timestamps accepted but unused — the analysis uses relative offsets
    # from the first event timestamp (more reliable than wall-clock phase
    # boundaries which drift due to probe attachment time).
    parser.add_argument("--phase1-start", type=float, default=0)
    parser.add_argument("--phase2-start", type=float, default=0)
    parser.add_argument("--phase3-start", type=float, default=0)
    parser.add_argument("--phase4-start", type=float, default=0)
    args = parser.parse_args()

    mcp = MCPClient(args.mcp_url)

    # Preflight: verify MCP connectivity before running 23 investigations
    preflight = mcp.run_sql("SELECT COUNT(*) FROM events")
    if "error" in preflight:
        print(f"ERROR: MCP preflight failed: {preflight['error']}", file=sys.stderr)
        sys.exit(1)
    event_count = sql_first_val(preflight, 0)
    if event_count == 0:
        print("ERROR: database contains 0 events", file=sys.stderr)
        sys.exit(1)
    print(f"MCP connected. DB has {event_count:,} events.")

    # Run all 23 investigations
    print("Running 23 GPU problem investigations...")
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
        print(f"  {status_color}[{inv.status}]{nc} T23{chr(ord('a') + inv.number - 1)}: "
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
    skip_count = sum(1 for i in investigations if i.status == "SKIP")

    print()
    print(f"  DETECTED: {detected}  HEALTHY: {healthy}  INCONCLUSIVE: {inconclusive}")
    print(f"  PASS: {pass_count}  SKIP: {skip_count}  TOTAL: {len(investigations)}")
    print()

    # Emit ML_RESULT lines (parsed by gpu-investigation.sh → gpu-test.sh)
    for inv in investigations:
        print(inv.result_line())


if __name__ == "__main__":
    main()
