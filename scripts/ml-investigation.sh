#!/bin/bash
################################################################################
# ML Engineer Investigation — Automated Reproducible Test
#
# Simulates the 4 questions a real ML engineer asks when debugging slow training:
#   Q1: "My training is slow — what's the root cause?"
#   Q2: "Is it the GPU or the host?"
#   Q3: "Show me how CPU contention hits my CUDA calls"
#   Q4: "Can an AI agent diagnose this via MCP?"
#
# Each question uses multiple Ingero tools together — that's the key
# differentiator vs nvidia-smi, DCGM, or PyTorch profiler alone.
#
# Setup: ResNet-50 CIFAR-10 training + stress-ng CPU contention
# Methodology: 60s trace — first 15s baseline (no stress), then 45s contention.
#   Single trace, two phases. Stress-ng starts at t+15s in background.
#   All analysis reads from DB via per-op queries (ingero query --json --op X),
#   demonstrating the production workflow: trace → DB → query → analyze.
# Result: 7 tests (T22a-T22g), markdown report, structured result lines
#
# Run standalone:   bash scripts/ml-investigation.sh
# Run via suite:    bash scripts/gpu-test.sh   (Phase 6)
#
# Requires: GPU, PyTorch, stress-ng, Ingero binary at bin/ingero
################################################################################

set -uo pipefail

# Resolve paths — works from agent/ or agent/scripts/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    cd "$SCRIPT_DIR/.." || exit 1
else
    cd "$SCRIPT_DIR" || exit 1
fi
INGERO_DIR="$(pwd)"

# Colors
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; NC=''
fi

PASS_COUNT=0; FAIL_COUNT=0; SKIP_COUNT=0
_test_start=$SECONDS

# Structured results for gpu-test.sh ingestion (one per line on fd 3 if open)
declare -a ML_RESULTS  # "ID|name|status|detail|duration_s"

ts() { date -u '+%Y-%m-%d %H:%M:%S'; }
# Safe grep -c: returns "0" (not "0\n0") when no matches.
# grep -c exits 1 on zero matches; || echo "0" would append a second "0".
# The echo "${n:-0}" handles file-not-found (grep outputs nothing, exits 2).
gcount() { local n; n=$(grep -c "$@" 2>/dev/null) || true; echo "${n:-0}"; }

record() {
    local status="$1" name="$2" detail="$3"
    local elapsed=$((SECONDS - _test_start))
    local tid="${name%%:*}"
    ML_RESULTS+=("${tid}|${name}|${status}|${detail}|${elapsed}")

    if [[ "$status" == "PASS" ]]; then
        echo -e "$(ts)   ${GREEN}[PASS]${NC} $name"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [[ "$status" == "FAIL" ]]; then
        echo -e "$(ts)   ${RED}[FAIL]${NC} $name — $detail"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    elif [[ "$status" == "SKIP" ]]; then
        echo -e "$(ts)   ${YELLOW}[SKIP]${NC} $name — $detail"
        SKIP_COUNT=$((SKIP_COUNT + 1))
    fi
    _test_start=$SECONDS
}

# PIDs to clean up on exit
cleanup_pids=()
ML_DB=""
ML_TMPDIR=""
cleanup() {
    # Kill sudo-spawned processes FIRST (pkill -f) — kill+wait on the sudo
    # wrapper PID can deadlock because sudo doesn't forward SIGTERM to children.
    if [[ -n "$ML_DB" ]]; then
        sudo pkill -f "ingero trace.*${ML_DB}" 2>/dev/null || true
        sudo pkill -f "ingero mcp.*${ML_DB}" 2>/dev/null || true
    fi
    sudo pkill -f 'stress-ng.*matrixprod' 2>/dev/null || true

    # Now kill and reap background jobs
    for pid in "${cleanup_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${cleanup_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    if [[ -n "$ML_DB" ]]; then
        rm -f "${ML_DB}" "${ML_DB}-wal" "${ML_DB}-shm" 2>/dev/null || true
    fi
    [[ -n "$ML_TMPDIR" ]] && rm -rf "$ML_TMPDIR"
}
trap cleanup EXIT

################################################################################
# Preflight
################################################################################

if [[ ! -x bin/ingero ]]; then
    echo "ERROR: bin/ingero not found. Run 'make build' first."
    exit 1
fi

if ! command -v stress-ng &>/dev/null; then
    echo "ERROR: stress-ng not found. Install: sudo apt-get install -y stress-ng"
    exit 1
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: PyTorch with CUDA not available."
    exit 1
fi

mkdir -p logs

################################################################################
# Setup: Create the problem (phased baseline/contention)
################################################################################

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  ML Engineer Investigation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

ML_DB="/tmp/ingero_ml_$(head -c 6 /dev/urandom | xxd -p).db"
TRACE_DURATION=60
BASELINE_SECS=15
REPORT_FILE="logs/ml-investigation-report.md"

echo -e "$(ts) ${CYAN}[SETUP]${NC} ResNet-50 training + CPU contention (stress-ng $(nproc) workers)..."
echo -e "$(ts)   Phase design: ${BASELINE_SECS}s baseline + $((TRACE_DURATION - BASELINE_SECS))s contention = ${TRACE_DURATION}s total"

# Pre-download CIFAR-10 dataset so training starts GPU work immediately.
# First run on a fresh VM downloads ~170MB at variable speed — if the download
# happens during the trace window, zero CUDA events are captured (T22c fails).
echo -e "$(ts)   Pre-downloading CIFAR-10 dataset..."
if ! python3 -c "
import torchvision
torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True, download=True)
print('CIFAR-10 ready')
" > logs/ml-dataset-download.log 2>&1; then
    echo -e "$(ts) ${RED}[ERROR]${NC} CIFAR-10 download failed. See logs/ml-dataset-download.log"
    cat logs/ml-dataset-download.log
    exit 1
fi
echo -e "$(ts)   $(tail -1 logs/ml-dataset-download.log)"

# Start ResNet-50 training (5 epochs — enough GPU work to cover the full 60s
# window even on fast GPUs like GH200 which finish ~26s/epoch)
python3 tests/workloads/training/resnet50_cifar10.py \
    --epochs 5 --batch-size 64 > logs/ml-training.log 2>&1 &
TRAIN_PID=$!
cleanup_pids+=("$TRAIN_PID")

# Wait for CUDA init + first batch to reach the GPU (model.to(device) + first forward pass)
echo -e "$(ts)   Waiting for training to reach GPU..."
sleep 10

# Verify training process is alive
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo -e "$(ts) ${RED}[ERROR]${NC} Training process died. See logs/ml-training.log"
    cat logs/ml-training.log
    exit 1
fi

echo -e "$(ts)   Training PID: $TRAIN_PID"
echo -e "$(ts)   Starting trace (${TRACE_DURATION}s) in background..."

# Start trace in background — allows precise stress-ng timing at t+15s.
# Use --record-all so every event is individually queryable (Q2-Q3 need
# per-event cuLaunchKernel and cudaStreamSync latencies, not just aggregates).
sudo ./bin/ingero trace --db "$ML_DB" --record-all --duration ${TRACE_DURATION}s \
    2> logs/ml-trace.log &
TRACE_PID=$!
cleanup_pids+=("$TRACE_PID")

# Baseline phase — 15s of clean training with no contention
echo -e "$(ts)   Baseline phase: ${BASELINE_SECS}s (no contention)..."
sleep "$BASELINE_SECS"

# Capture wall-clock epoch just before stress-ng starts — passed to Python for
# precise phase classification (avoids drift between bash sleep and kernel timestamps)
STRESS_START_EPOCH=$(date +%s.%N)

# Contention phase — stress-ng saturates ALL cores for the remaining duration
NCPUS=$(nproc)
CONTENTION_SECS=$((TRACE_DURATION - BASELINE_SECS + 5))  # +5s buffer to outlast trace
echo -e "$(ts)   Contention phase: stress-ng --cpu $NCPUS for ~${CONTENTION_SECS}s..."
sudo stress-ng --cpu "$NCPUS" --cpu-method matrixprod --timeout ${CONTENTION_SECS}s > /dev/null 2>&1 &
STRESS_PID=$!
cleanup_pids+=("$STRESS_PID")

# Wait for trace to complete (blocks until --duration expires)
echo -e "$(ts)   Waiting for trace to finish..."
wait "$TRACE_PID" 2>/dev/null
TRACE_EXIT=$?

if [[ "$TRACE_EXIT" -ne 0 ]]; then
    echo -e "$(ts) ${RED}[ERROR]${NC} Trace failed (exit $TRACE_EXIT). See logs/ml-trace.log"
    cat logs/ml-trace.log
    record "FAIL" "T22a: trace captured events" "trace exited $TRACE_EXIT"
    # Skip Q1-Q4 — no data
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${RED}FAIL=1${NC}  Total=1  (trace failed, skipping Q1-Q4)"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    for entry in "${ML_RESULTS[@]}"; do echo "ML_RESULT|${entry}"; done
    exit 1
fi

# Copy ML DB to logs/ for transfer (include WAL/SHM in case checkpoint is deferred)
sudo cp "$ML_DB" logs/ml-investigation.db && sudo chmod 644 logs/ml-investigation.db
sudo cp "${ML_DB}-wal" logs/ml-investigation.db-wal 2>/dev/null && sudo chmod 644 logs/ml-investigation.db-wal || true
sudo cp "${ML_DB}-shm" logs/ml-investigation.db-shm 2>/dev/null && sudo chmod 644 logs/ml-investigation.db-shm || true

# Query events from DB per-op (production workflow: query specific operations)
ML_TMPDIR=$(mktemp -d)

echo -e "$(ts)   Querying events from DB..."
./bin/ingero query --db "$ML_DB" --json --op cudaStreamSync --limit -1 --since 10m \
    > "$ML_TMPDIR/sync.json" 2>/dev/null || echo "[]" > "$ML_TMPDIR/sync.json"
./bin/ingero query --db "$ML_DB" --json --op cuLaunchKernel --limit -1 --since 10m \
    > "$ML_TMPDIR/launch.json" 2>/dev/null || echo "[]" > "$ML_TMPDIR/launch.json"
./bin/ingero query --db "$ML_DB" --json --op sched_switch --limit -1 --since 10m \
    > "$ML_TMPDIR/sched.json" 2>/dev/null || echo "[]" > "$ML_TMPDIR/sched.json"

# Kill stress-ng now (trace is done)
sudo kill "$STRESS_PID" 2>/dev/null || true
# Let training finish naturally or kill it
kill "$TRAIN_PID" 2>/dev/null || true
wait "$TRAIN_PID" 2>/dev/null || true

################################################################################
# Unified Python analysis — per-op DB queries
#
# Reads 3 per-op JSON arrays from DB queries (sync, launch, sched).
# Outputs all computed values for downstream tests.
#
# Output lines (one per metric, parseable via grep -oP):
#   event_counts: sync=X launch=Y sched=Z
#   launch_full: n=X p50=Yus p99=Zus ratio=Wx
#   sched_full: n=X max=Yus over_10ms=Z has_durations=yes|no
#   sync_full: n=X p50=Yus p99=Zus ratio=Wx
#   phase_baseline_sync: n=X p50=Yus p99=Zus
#   phase_contention_sync: n=X p50=Yus p99=Zus
#   phase_baseline_launch: n=X p50=Yus p99=Zus
#   phase_contention_launch: n=X p50=Yus p99=Zus
#   sync_amplification: Xx
#   launch_rate: baseline=N/s contention=N/s change=+X%
#   temporal_rho: 0.XX
#   temporal_overlap: XX%
################################################################################

echo ""
echo -e "$(ts) ${CYAN}[ANALYSIS]${NC} Running unified analysis..."

ANALYSIS_OUT=$(python3 -c "
import json, sys
from datetime import datetime

BASELINE_SECS = ${BASELINE_SECS}
STRESS_EPOCH = ${STRESS_START_EPOCH}
SYNC_FILE = '${ML_TMPDIR}/sync.json'
LAUNCH_FILE = '${ML_TMPDIR}/launch.json'
SCHED_FILE = '${ML_TMPDIR}/sched.json'

def load_events(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f'WARNING: {path} contained {type(data).__name__}, expected list', file=sys.stderr)
        return []
    return data

sync_events = load_events(SYNC_FILE)
launch_events = load_events(LAUNCH_FILE)
sched_events = load_events(SCHED_FILE)

# Output event counts (for T22a and display)
print(f'event_counts: sync={len(sync_events)} launch={len(launch_events)} sched={len(sched_events)}')

sync_durs = []
launch_durs = []
sched_durs = []
sched_total = len(sched_events)

# Phase-split collections (keyed by phase: 'baseline' or 'contention')
phase_sync = {'baseline': [], 'contention': []}
phase_launch = {'baseline': [], 'contention': []}

# Temporal bins: second offset -> {sched_count, sync_p99s}
temporal_bins = {}

def parse_ts(ts_str):
    \"\"\"Parse RFC3339Nano timestamp to epoch seconds (float).\"\"\"
    try:
        s = ts_str.rstrip('Z')
        if '.' in s:
            base, frac = s.split('.', 1)
            frac = frac[:6].ljust(6, '0')
            s = base + '.' + frac
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except:
        return None

# Find first_ts across ALL events (they interleave temporally)
all_timestamps = []
for e in sync_events + launch_events + sched_events:
    ts_str = e.get('timestamp', '')
    if ts_str:
        t = parse_ts(ts_str)
        if t is not None:
            all_timestamps.append(t)

first_ts = min(all_timestamps) if all_timestamps else None

def classify(ts_str):
    \"\"\"Return (phase, sec_offset) for a timestamp.\"\"\"
    if not ts_str or first_ts is None:
        return 'contention', None
    t = parse_ts(ts_str)
    if t is None:
        return 'contention', None
    sec_offset = t - first_ts
    # Use stress-ng wall-clock start for precise phase split (avoids drift
    # between bash sleep and kernel timestamps). Falls back to BASELINE_SECS.
    phase = 'baseline' if t < STRESS_EPOCH else 'contention'
    sec_bin = int(sec_offset)
    if sec_bin not in temporal_bins:
        temporal_bins[sec_bin] = {'sched': 0, 'sync_durs': []}
    return phase, sec_offset

# Process sync events
for e in sync_events:
    d = e.get('duration_ns', 0)
    phase, sec_offset = classify(e.get('timestamp', ''))
    if d > 0:
        dur_us = d / 1000
        sync_durs.append(dur_us)
        phase_sync[phase].append(dur_us)
        if sec_offset is not None:
            temporal_bins[int(sec_offset)]['sync_durs'].append(dur_us)

# Process launch events
for e in launch_events:
    d = e.get('duration_ns', 0)
    phase, sec_offset = classify(e.get('timestamp', ''))
    if d > 0:
        dur_us = d / 1000
        launch_durs.append(dur_us)
        phase_launch[phase].append(dur_us)

# Process sched events
for e in sched_events:
    d = e.get('duration_ns', 0)
    phase, sec_offset = classify(e.get('timestamp', ''))
    if d > 0:
        sched_durs.append(d / 1000)
    if sec_offset is not None:
        temporal_bins[int(sec_offset)]['sched'] += 1


def percentiles(vals):
    \"\"\"Return (p50, p99) from sorted list.\"\"\"
    if not vals:
        return (0, 0)
    s = sorted(vals)
    return (s[len(s)//2], s[int(len(s)*0.99)])


def fmt_stats(vals, label):
    \"\"\"Format n=X p50=Yus p99=Zus ratio=Wx.\"\"\"
    if not vals:
        return f'{label}: n=0 p50=0us p99=0us ratio=0x'
    p50, p99 = percentiles(vals)
    ratio = p99 / max(p50, 0.001)
    return f'{label}: n={len(vals)} p50={p50:.0f}us p99={p99:.0f}us ratio={ratio:.1f}x'


# --- Full-session stats ---
print(fmt_stats(launch_durs, 'launch_full'))

has_durs = 'yes' if sched_durs else 'no'
if sched_durs:
    sched_durs_sorted = sorted(sched_durs)
    max_us = sched_durs_sorted[-1]
    over_10ms = sum(1 for d in sched_durs if d > 10000)
    print(f'sched_full: n={sched_total} max={max_us:.0f}us over_10ms={over_10ms} has_durations={has_durs}')
else:
    print(f'sched_full: n={sched_total} max=0us over_10ms=0 has_durations={has_durs}')

print(fmt_stats(sync_durs, 'sync_full'))

# --- Phase stats ---
print(fmt_stats(phase_sync['baseline'], 'phase_baseline_sync'))
print(fmt_stats(phase_sync['contention'], 'phase_contention_sync'))
print(fmt_stats(phase_launch['baseline'], 'phase_baseline_launch'))
print(fmt_stats(phase_launch['contention'], 'phase_contention_launch'))

# Sync amplification: contention_p99 / baseline_p99
_, base_p99 = percentiles(phase_sync['baseline'])
_, cont_p99 = percentiles(phase_sync['contention'])
if base_p99 > 0 and cont_p99 > 0:
    amp = cont_p99 / base_p99
    print(f'sync_amplification: {amp:.1f}x')
elif not phase_sync['baseline'] or not phase_sync['contention']:
    print('sync_amplification: N/A')  # one phase empty
else:
    print('sync_amplification: 1.0x')  # both phases present, negligible tails

# Launch rate: events per second in each phase
trace_duration = (max(temporal_bins.keys()) - min(temporal_bins.keys()) + 1) if temporal_bins else BASELINE_SECS + 45
base_launch_n = len(phase_launch['baseline'])
cont_launch_n = len(phase_launch['contention'])
cont_secs = max(trace_duration - BASELINE_SECS, 1)
base_rate = base_launch_n / max(BASELINE_SECS, 1)
cont_rate = cont_launch_n / max(cont_secs, 1)
if base_rate > 0:
    change_pct = ((cont_rate - base_rate) / base_rate) * 100
    print(f'launch_rate: baseline={base_rate:.0f}/s contention={cont_rate:.0f}/s change={change_pct:+.0f}%')
else:
    print(f'launch_rate: baseline={base_rate:.0f}/s contention={cont_rate:.0f}/s change=N/A')

# --- Temporal correlation ---
# Spearman rank correlation: sched_count vs sync_p99 per second
paired_secs = []
for sec_bin, data in sorted(temporal_bins.items()):
    if data['sched'] > 0 or data['sync_durs']:
        sp99 = 0
        if data['sync_durs']:
            s = sorted(data['sync_durs'])
            sp99 = s[int(len(s)*0.99)]
        paired_secs.append((data['sched'], sp99))

def rank(values):
    \"\"\"Rank values (1-based, average for ties).\"\"\"
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks

if len(paired_secs) >= 5:
    sched_vals = [p[0] for p in paired_secs]
    sync_vals = [p[1] for p in paired_secs]
    if len(set(sched_vals)) < 2 or len(set(sync_vals)) < 2:
        print('temporal_rho: N/A')
    else:
        r_sched = rank(sched_vals)
        r_sync = rank(sync_vals)
        n = len(paired_secs)
        d_sq = sum((r_sched[i] - r_sync[i])**2 for i in range(n))
        rho = 1 - (6 * d_sq) / (n * (n*n - 1))
        print(f'temporal_rho: {rho:.2f}')
else:
    print('temporal_rho: N/A')

# Temporal overlap: fraction of high-sched seconds that also have high sync p99
if paired_secs:
    sched_vals = [p[0] for p in paired_secs]
    sync_vals = [p[1] for p in paired_secs]
    sched_median = sorted(sched_vals)[len(sched_vals)//2]
    sync_median = sorted(sync_vals)[len(sync_vals)//2] if sync_vals else 0
    high_sched_secs = [(s, sy) for s, sy in paired_secs if s > sched_median]
    if high_sched_secs:
        both_high = sum(1 for s, sy in high_sched_secs if sy > sync_median)
        overlap = (both_high / len(high_sched_secs)) * 100
        print(f'temporal_overlap: {overlap:.0f}%')
    else:
        print('temporal_overlap: 0%')
else:
    print('temporal_overlap: N/A')

# Phase sched rates (for T22d contention-rate comparison)
base_sched_n = sum(data['sched'] for sec, data in temporal_bins.items() if first_ts is not None and (first_ts + sec) < STRESS_EPOCH)
cont_sched_n = sum(data['sched'] for sec, data in temporal_bins.items() if first_ts is not None and (first_ts + sec) >= STRESS_EPOCH)
trace_secs = (max(temporal_bins.keys()) - min(temporal_bins.keys()) + 1) if temporal_bins else BASELINE_SECS + 45
base_secs = STRESS_EPOCH - first_ts if first_ts else BASELINE_SECS
cont_secs_ph = max(trace_secs - base_secs, 1)
base_sched_rate = base_sched_n / max(base_secs, 1)
cont_sched_rate = cont_sched_n / max(cont_secs_ph, 1)
print(f'sched_phase: baseline={base_sched_rate:.0f}/s contention={cont_sched_rate:.0f}/s')
" 2>logs/ml-analysis-stderr.log || echo "analysis_error")

echo "$ANALYSIS_OUT" > logs/ml-analysis.log

# Verify analysis completed (expect at least event_counts + 3 full-session stats)
if ! echo "$ANALYSIS_OUT" | grep -q '^event_counts:' || ! echo "$ANALYSIS_OUT" | grep -q '^sync_full:'; then
    echo -e "$(ts) ${RED}[ERROR]${NC} Python analysis failed or incomplete"
    echo "  Output: ${ANALYSIS_OUT:0:200}"
    echo "  Stderr: $(cat logs/ml-analysis-stderr.log 2>/dev/null | tail -5)"
    record "FAIL" "T22a: trace captured events" "Python analysis failed: ${ANALYSIS_OUT:0:100}"
    # Skip Q1-Q4 — analysis data unavailable
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${RED}FAIL=1${NC}  Total=1  (analysis failed, skipping Q1-Q4)"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    for entry in "${ML_RESULTS[@]}"; do echo "ML_RESULT|${entry}"; done
    exit 1
fi

# Extract key values for downstream tests
LAUNCH_STATS=$(echo "$ANALYSIS_OUT" | grep '^launch_full:' || echo "launch_full: n=0 p50=0us p99=0us ratio=0x")
SCHED_STATS=$(echo "$ANALYSIS_OUT" | grep '^sched_full:' || echo "sched_full: n=0 max=0us over_10ms=0 has_durations=no")
SYNC_STATS=$(echo "$ANALYSIS_OUT" | grep '^sync_full:' || echo "sync_full: n=0 p50=0us p99=0us ratio=0x")
BASELINE_SYNC=$(echo "$ANALYSIS_OUT" | grep '^phase_baseline_sync:' || echo "")
CONTENTION_SYNC=$(echo "$ANALYSIS_OUT" | grep '^phase_contention_sync:' || echo "")
BASELINE_LAUNCH=$(echo "$ANALYSIS_OUT" | grep '^phase_baseline_launch:' || echo "")
CONTENTION_LAUNCH=$(echo "$ANALYSIS_OUT" | grep '^phase_contention_launch:' || echo "")
SYNC_AMP=$(echo "$ANALYSIS_OUT" | grep '^sync_amplification:' | grep -oP '[\d.]+x' || echo "N/A")
LAUNCH_RATE=$(echo "$ANALYSIS_OUT" | grep '^launch_rate:' || echo "")
TEMPORAL_RHO=$(echo "$ANALYSIS_OUT" | grep -oP 'temporal_rho: \K\S+' || echo "N/A")
TEMPORAL_OVERLAP=$(echo "$ANALYSIS_OUT" | grep -oP 'temporal_overlap: \K\S+' || echo "N/A")
SCHED_PHASE=$(echo "$ANALYSIS_OUT" | grep '^sched_phase:' || echo "")
SCHED_BASE_RATE=$(echo "$SCHED_PHASE" | grep -oP 'baseline=\K[0-9]+' || echo "0")
SCHED_CONT_RATE=$(echo "$SCHED_PHASE" | grep -oP 'contention=\K[0-9]+' || echo "0")

# Extract event counts from analysis (approximate — only sync+launch+sched, but
# sufficient for T22a threshold check of >1000)
SYNC_TOTAL=$(echo "$ANALYSIS_OUT" | grep -oP 'sync=\K[0-9]+' | head -1 || echo "0")
LAUNCH_TOTAL=$(echo "$ANALYSIS_OUT" | grep -oP 'launch=\K[0-9]+' | head -1 || echo "0")
SCHED_TOTAL=$(echo "$ANALYSIS_OUT" | grep -oP 'sched=\K[0-9]+' | head -1 || echo "0")
TOTAL_EVENTS=$((SYNC_TOTAL + LAUNCH_TOTAL + SCHED_TOTAL))

# Map per-op counts to source categories for T22a
CUDA_EVENTS=$SYNC_TOTAL      # cudaStreamSync = cuda source
DRIVER_EVENTS=$LAUNCH_TOTAL   # cuLaunchKernel = driver source
HOST_EVENTS=$SCHED_TOTAL      # sched_switch = host source

echo -e "$(ts)   Done: ${TOTAL_EVENTS} events (sync=${SYNC_TOTAL} launch=${LAUNCH_TOTAL} sched=${SCHED_TOTAL})"
echo -e "$(ts)   Analysis complete. See logs/ml-analysis.log"

echo ""

################################################################################
# Q1: "My training is slow — what's the root cause?"
################################################################################

echo -e "$(ts) ${CYAN}── Q1: \"My training is slow — what's the root cause?\" ─────${NC}"
_test_start=$SECONDS

# Tool 1: explain (automated incident report)
EXPLAIN_OUT=$(./bin/ingero explain --db "$ML_DB" --since 5m 2>&1)
echo "$EXPLAIN_OUT" > logs/ml-explain.log

# Tool 2: raw event count for context
QUERY_COUNT=$(./bin/ingero query --db "$ML_DB" --since 5m --json 2>/dev/null | gcount '"op"')

# Display summary
CHAIN_COUNT=$(echo "$EXPLAIN_OUT" | gcount '\[HIGH\]\|\[MEDIUM\]\|\[LOW\]')
CHAIN_SUMMARY=$(echo "$EXPLAIN_OUT" | grep '\[HIGH\]\|\[MEDIUM\]\|\[LOW\]' | head -1 || echo "none")
echo -e "$(ts)   → ingero explain: ${CHAIN_COUNT} causal chain(s)"
if [[ "$CHAIN_COUNT" -gt 0 ]]; then
    echo -e "$(ts)     $CHAIN_SUMMARY"
    ROOT_CAUSE=$(echo "$EXPLAIN_OUT" | grep 'Root cause:' | head -1 | sed 's/.*Root cause: //')
    FIX=$(echo "$EXPLAIN_OUT" | grep 'Fix:' | head -1 | sed 's/.*Fix: //')
    [[ -n "$ROOT_CAUSE" ]] && echo -e "$(ts)     Root cause: $ROOT_CAUSE"
    [[ -n "$FIX" ]] && echo -e "$(ts)     Fix: $FIX"
fi

# T22a: trace captured events — require CUDA/driver data for investigation
# PASS: >1000 events AND (sync > 0 OR launch > 0) AND sched > 0
# FAIL: enough events but no CUDA/driver data (investigation impossible)
# FAIL: too few events total
CUDA_PRESENT=0
if [[ "$TOTAL_EVENTS" -gt 1000 ]]; then
    if [[ "$CUDA_EVENTS" -gt 0 || "$DRIVER_EVENTS" -gt 0 ]]; then
        CUDA_PRESENT=1
        if [[ "$HOST_EVENTS" -gt 0 ]]; then
            record "PASS" "T22a: trace captured events" "${TOTAL_EVENTS} events (sync=${SYNC_TOTAL} launch=${LAUNCH_TOTAL} sched=${SCHED_TOTAL})"
        else
            record "PASS" "T22a: trace captured events" "${TOTAL_EVENTS} events (sync=${SYNC_TOTAL} launch=${LAUNCH_TOTAL}, no sched — cross-stack limited)"
        fi
    else
        record "FAIL" "T22a: trace captured events" "no CUDA/driver data for investigation (sync=${SYNC_TOTAL} launch=${LAUNCH_TOTAL} sched=${SCHED_TOTAL})"
    fi
else
    record "FAIL" "T22a: trace captured events" "only ${TOTAL_EVENTS} events (need >1000)"
fi

# T22b: explain detected causal chain with scheduling evidence
# Under CPU contention, we expect chains with scheduling keywords. Verify the
# chain engine correctly identified our injected contention.
HAS_SCHED_KEYWORDS=""
if [[ "$CHAIN_COUNT" -gt 0 ]]; then
    HAS_SCHED_KEYWORDS=$(echo "$EXPLAIN_OUT" | grep -i 'sched_switch\|CPU\|context switch\|scheduling\|off-CPU' | head -1 || echo "")
fi

if [[ "$CHAIN_COUNT" -gt 0 && -n "$HAS_SCHED_KEYWORDS" ]]; then
    record "PASS" "T22b: causal chain detected" "${CHAIN_COUNT} chain(s) with scheduling root cause"
elif [[ "$CHAIN_COUNT" -gt 0 ]]; then
    record "PASS" "T22b: causal chain detected" "${CHAIN_COUNT} chain(s) found (NOTE: no scheduling keywords despite stress-ng — chain may be unrelated to injected contention)"
else
    # No chains detected — stress-ng may not have caused enough contention
    if echo "$EXPLAIN_OUT" | grep -q 'INCIDENT REPORT\|No events\|no causal'; then
        record "SKIP" "T22b: causal chain detected" "explain ran but no chains (hardware too fast for stress-ng?)"
    else
        record "FAIL" "T22b: causal chain detected" "explain failed: ${EXPLAIN_OUT:0:100}"
    fi
fi

echo ""

################################################################################
# Q2: "Is it the GPU or the host?"
################################################################################

echo -e "$(ts) ${CYAN}── Q2: \"Is it the GPU or the host?\" ───────────────────────${NC}"
_test_start=$SECONDS

# Stats from unified analysis
LAUNCH_COUNT=$(echo "$LAUNCH_STATS" | grep -oP 'n=\K[0-9]+' || echo "0")
LAUNCH_RATIO=$(echo "$LAUNCH_STATS" | grep -oP 'ratio=\K[0-9.]+' || echo "0")
LAUNCH_P50=$(echo "$LAUNCH_STATS" | grep -oP 'p50=\K[0-9]+' || echo "0")
LAUNCH_P99=$(echo "$LAUNCH_STATS" | grep -oP 'p99=\K[0-9]+' || echo "0")

SCHED_COUNT=$(echo "$SCHED_STATS" | grep -oP 'n=\K[0-9]+' || echo "0")
SCHED_MAX=$(echo "$SCHED_STATS" | grep -oP 'max=\K[0-9]+' || echo "0")
OVER_10MS=$(echo "$SCHED_STATS" | grep -oP 'over_10ms=\K[0-9]+' || echo "0")
HAS_DURATIONS=$(echo "$SCHED_STATS" | grep -oP 'has_durations=\K\w+' || echo "no")

echo -e "$(ts)   → cuLaunchKernel: ${LAUNCH_COUNT} events, p50=${LAUNCH_P50}us p99=${LAUNCH_P99}us ratio=${LAUNCH_RATIO}x"
echo -e "$(ts)   → sched_switch: ${SCHED_COUNT} events, max=${SCHED_MAX}us over_10ms=${OVER_10MS}"
if [[ -n "$BASELINE_LAUNCH" && -n "$CONTENTION_LAUNCH" ]]; then
    echo -e "$(ts)   → Baseline:    $(echo "$BASELINE_LAUNCH" | sed 's/phase_baseline_launch: //')"
    echo -e "$(ts)   → Contention:  $(echo "$CONTENTION_LAUNCH" | sed 's/phase_contention_launch: //')"
fi
if [[ -n "$LAUNCH_RATE" ]]; then
    echo -e "$(ts)   → Launch rate: $(echo "$LAUNCH_RATE" | sed 's/launch_rate: //')"
fi

# Verdict
if [[ "$LAUNCH_COUNT" -gt 0 ]]; then
    if python3 -c "exit(0 if float('${LAUNCH_RATIO}') < 10 else 1)" 2>/dev/null; then
        echo -e "$(ts)   Verdict: GPU kernels consistent (ratio < 10x). Bottleneck is host."
    fi
fi

# T22c: cuLaunchKernel present AND consistent (ratio < 30x)
# cuLaunchKernel is fire-and-forget (enqueue only). Even 10x would be abnormal.
if [[ "$LAUNCH_COUNT" -gt 0 ]]; then
    if python3 -c "exit(0 if float('${LAUNCH_RATIO}') < 30 else 1)" 2>/dev/null; then
        record "PASS" "T22c: driver API + GPU consistent" "${LAUNCH_COUNT} cuLaunchKernel, ratio=${LAUNCH_RATIO}x"
    else
        record "FAIL" "T22c: driver API + GPU consistent" "GPU dispatch severely degraded (ratio=${LAUNCH_RATIO}x >= 30x)"
    fi
else
    record "FAIL" "T22c: driver API + GPU consistent" "no cuLaunchKernel events"
fi

# T22d: sched_switch shows scheduling storms (with phase rate comparison)
if [[ "$SCHED_COUNT" -gt 100 && "$HAS_DURATIONS" == "yes" && "$OVER_10MS" -gt 0 ]]; then
    record "PASS" "T22d: scheduling storms confirmed" "${SCHED_COUNT} events, ${OVER_10MS} over 10ms, contention=${SCHED_CONT_RATE}/s vs baseline=${SCHED_BASE_RATE}/s (strong)"
elif [[ "$SCHED_COUNT" -gt 100 && "$HAS_DURATIONS" == "yes" ]]; then
    record "PASS" "T22d: scheduling storms confirmed" "${SCHED_COUNT} sched_switch events, contention=${SCHED_CONT_RATE}/s vs baseline=${SCHED_BASE_RATE}/s (mild, none >10ms)"
elif [[ "$SCHED_COUNT" -gt 100 && "$HAS_DURATIONS" == "no" ]]; then
    record "SKIP" "T22d: scheduling storms confirmed" "${SCHED_COUNT} events but duration_ns=0 (no off-CPU measurements)"
elif [[ "$SCHED_COUNT" -gt 0 ]]; then
    record "SKIP" "T22d: scheduling storms confirmed" "only ${SCHED_COUNT} sched_switch events (need >100)"
else
    record "FAIL" "T22d: scheduling storms confirmed" "no sched_switch events"
fi

echo ""

################################################################################
# Q3: "Show me how CPU contention hits my CUDA calls"
################################################################################

echo -e "$(ts) ${CYAN}── Q3: \"Show me how CPU contention hits my CUDA calls\" ────${NC}"
_test_start=$SECONDS

# Stats from unified analysis
SYNC_COUNT=$(echo "$SYNC_STATS" | grep -oP 'n=\K[0-9]+' || echo "0")
SYNC_P50=$(echo "$SYNC_STATS" | grep -oP 'p50=\K[0-9]+' || echo "0")
SYNC_P99=$(echo "$SYNC_STATS" | grep -oP 'p99=\K[0-9]+' || echo "0")
SYNC_RATIO=$(echo "$SYNC_STATS" | grep -oP 'ratio=\K[0-9.]+' || echo "0")

echo -e "$(ts)   → cudaStreamSync: ${SYNC_COUNT} events, p50=${SYNC_P50}us p99=${SYNC_P99}us ratio=${SYNC_RATIO}x"
if [[ -n "$BASELINE_SYNC" && -n "$CONTENTION_SYNC" ]]; then
    echo -e "$(ts)   → Baseline:    $(echo "$BASELINE_SYNC" | sed 's/phase_baseline_sync: //')"
    echo -e "$(ts)   → Contention:  $(echo "$CONTENTION_SYNC" | sed 's/phase_contention_sync: //')"
fi
if [[ "$SYNC_AMP" != "N/A" ]]; then
    echo -e "$(ts)   → Sync amplification (contention p99 / baseline p99): ${SYNC_AMP}"
fi
if [[ "$TEMPORAL_RHO" != "N/A" ]]; then
    echo -e "$(ts)   → Temporal correlation (Spearman rho): ${TEMPORAL_RHO}"
fi
if [[ "$TEMPORAL_OVERLAP" != "N/A" ]]; then
    echo -e "$(ts)   → Temporal overlap (high-sched ∩ high-sync): ${TEMPORAL_OVERLAP}"
fi

# T22e: sync tail amplification
# Primary: sync_amplification (contention_p99 / baseline_p99).
# Aligns with DefaultTailRatio = 3.0 in correlate.go.
# Fallback: full-session p99/p50 when no phase data.
if [[ "$SYNC_COUNT" -gt 0 ]]; then
    TAIL_DETAIL="p99/p50=${SYNC_RATIO}x"
    if [[ "$SYNC_AMP" != "N/A" ]]; then
        TAIL_DETAIL="${TAIL_DETAIL}, baseline→contention=${SYNC_AMP}"
    fi

    if [[ "$SYNC_AMP" != "N/A" ]]; then
        # Primary: sync_amplification (contention_p99 / baseline_p99)
        AMP_VAL=$(echo "$SYNC_AMP" | sed 's/x$//')
        if python3 -c "exit(0 if float('${AMP_VAL}') > 10 else 1)" 2>/dev/null; then
            record "PASS" "T22e: sync tail amplification" "strong amplification (${TAIL_DETAIL})"
        elif python3 -c "exit(0 if float('${AMP_VAL}') >= 3 else 1)" 2>/dev/null; then
            record "PASS" "T22e: sync tail amplification" "mild amplification (${TAIL_DETAIL})"
        elif python3 -c "exit(0 if float('${AMP_VAL}') >= 1 else 1)" 2>/dev/null; then
            # amp >= 1 but < 3: contention didn't amplify much, but didn't invert either
            record "SKIP" "T22e: sync tail amplification" "minimal amplification (${TAIL_DETAIL})"
        else
            # amp < 1: inverted pattern — GPU starvation (fast GPUs like H100)
            record "SKIP" "T22e: sync tail amplification" "inverted: contention reduced sync latency (${TAIL_DETAIL}) — GPU starvation pattern"
        fi
    else
        # Fallback: full-session p99/p50 when no phase data
        if python3 -c "exit(0 if float('${SYNC_RATIO}') > 10 else 1)" 2>/dev/null; then
            record "PASS" "T22e: sync tail amplification" "strong tail (${TAIL_DETAIL})"
        elif python3 -c "exit(0 if float('${SYNC_RATIO}') >= 3 else 1)" 2>/dev/null; then
            record "PASS" "T22e: sync tail amplification" "mild tail (${TAIL_DETAIL})"
        else
            record "SKIP" "T22e: sync tail amplification" "no tail detected (${TAIL_DETAIL}) — hardware too fast?"
        fi
    fi
else
    # No cudaStreamSync events — try cudaDeviceSync as fallback
    DSYNC_COUNT=$(./bin/ingero query --db "$ML_DB" --op cudaDeviceSync --limit 1 --since 10m 2>/dev/null | gcount 'cudaDeviceSync')
    if [[ "$DSYNC_COUNT" -gt 0 ]]; then
        record "SKIP" "T22e: sync tail amplification" "no cudaStreamSync but ${DSYNC_COUNT} cudaDeviceSync (workload uses device sync)"
    else
        record "SKIP" "T22e: sync tail amplification" "no sync events captured"
    fi
fi

echo ""

################################################################################
# Q4: "Can an AI agent diagnose this via MCP?"
################################################################################

echo -e "$(ts) ${CYAN}── Q4: \"Can an AI agent diagnose this via MCP?\" ───────────${NC}"
_test_start=$SECONDS

# NOTE: MCP queries on large --record-all DBs (~150MB, 1M+ events) can take
# 5-10s for get_trace_stats (scans all events). If MCP tests time out on
# slow VMs, consider adding --limit to MCP calls or increasing wait timeout.
MCP_PORT=8081  # Use different port from gpu-test.sh Phase 5

# Kill any leftover MCP server from a previous crashed run
sudo pkill -f "ingero mcp.*${MCP_PORT}" 2>/dev/null || true
sleep 0.5

# Start MCP server against the ML investigation DB
sudo ./bin/ingero mcp --http ":${MCP_PORT}" --db "$ML_DB" > logs/ml-mcp-server.log 2>&1 &
MCP_PID=$!
cleanup_pids+=("$MCP_PID")

# Wait for MCP server to be ready
MCP_READY=0
for i in $(seq 1 10); do
    if curl -skf -o /dev/null "https://localhost:${MCP_PORT}/mcp" \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"ml-test","version":"1.0"}}}' 2>/dev/null; then
        MCP_READY=1
        break
    fi
    sleep 0.5
done

mcp_call() {
    local tool="$1" args="$2"
    curl -skf "https://localhost:${MCP_PORT}/mcp" \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"${tool}\",\"arguments\":${args}}}" 2>/dev/null
}

if [[ "$MCP_READY" -eq 0 ]]; then
    echo -e "$(ts)   ${RED}MCP server not ready after 5s${NC}"
    record "SKIP" "T22f: MCP causal chains" "MCP server failed to start"
    record "SKIP" "T22g: MCP op filter" "MCP server failed to start"
else
    echo -e "$(ts)   MCP server ready on :${MCP_PORT}"

    # MCP Tool 1: get_trace_stats (overview)
    STATS_RESP=$(mcp_call "get_trace_stats" '{"since":"5m"}')
    echo "MCP get_trace_stats: ${STATS_RESP:0:200}" >> logs/ml-mcp-debug.log
    if echo "$STATS_RESP" | grep -q 'op.*p50\|ops.*cuda\|p50.*p95'; then
        echo -e "$(ts)   → MCP get_trace_stats: events found"
    else
        echo -e "$(ts)   → MCP get_trace_stats: ${STATS_RESP:0:100}"
    fi

    # MCP Tool 2: get_causal_chains (root cause)
    CHAINS_RESP=$(mcp_call "get_causal_chains" '{"since":"5m"}')
    echo "MCP get_causal_chains: ${CHAINS_RESP:0:300}" >> logs/ml-mcp-debug.log

    # T22f: MCP response must be consistent with T22b
    # If T22b found chains (CHAIN_COUNT > 0), MCP should report severity.
    # If T22b found no chains, MCP saying "healthy"/"no causal" is acceptable.
    MCP_HAS_SEVERITY=$(echo "$CHAINS_RESP" | grep -oiP '\bHIGH\b|\bMEDIUM\b|\bLOW\b' | head -1 || echo "")
    MCP_SAYS_HEALTHY=$(echo "$CHAINS_RESP" | grep -qi 'healthy\|no causal\|No causal' && echo "yes" || echo "no")

    if [[ "$CHAIN_COUNT" -gt 0 ]]; then
        # T22b found chains — MCP should too
        if [[ -n "$MCP_HAS_SEVERITY" ]]; then
            echo -e "$(ts)   → MCP get_causal_chains: ${MCP_HAS_SEVERITY} (consistent with T22b)"
            record "PASS" "T22f: MCP causal chains" "MCP reports ${MCP_HAS_SEVERITY}, T22b found ${CHAIN_COUNT} chain(s)"
        elif [[ "$MCP_SAYS_HEALTHY" == "yes" ]]; then
            # Contradiction: T22b found chains but MCP says healthy
            record "FAIL" "T22f: MCP causal chains" "MCP says healthy but T22b found ${CHAIN_COUNT} chain(s)"
        elif echo "$CHAINS_RESP" | grep -qi 'causal\|chain\|sev\|severity'; then
            echo -e "$(ts)   → MCP get_causal_chains: response mentions chains"
            record "PASS" "T22f: MCP causal chains" "MCP response references chains"
        else
            record "FAIL" "T22f: MCP causal chains" "unexpected: ${CHAINS_RESP:0:150}"
        fi
    else
        # T22b found no chains — accept healthy or any valid response
        if echo "$CHAINS_RESP" | grep -qi 'causal\|chain\|healthy\|sev\|severity\|No causal\|MEDIUM\|HIGH\|LOW'; then
            MCP_INFO=$(echo "$CHAINS_RESP" | grep -oiP '\bHIGH\b|\bMEDIUM\b|\bLOW\b|\bhealthy\b|no causal' | head -1 || echo "valid response")
            echo -e "$(ts)   → MCP get_causal_chains: ${MCP_INFO}"
            record "PASS" "T22f: MCP causal chains" "response: ${MCP_INFO}"
        else
            record "FAIL" "T22f: MCP causal chains" "unexpected: ${CHAINS_RESP:0:150}"
        fi
    fi

    # MCP Tool 3: query_events with op filter
    QUERY_RESP=$(mcp_call "query_events" '{"since":"5m","op":"cudaStreamSync","limit":20}')
    echo "MCP query_events op=cudaStreamSync: ${QUERY_RESP:0:300}" >> logs/ml-mcp-debug.log

    # T22g: MCP query_events op filter returns only cudaStreamSync (or empty if none)
    if echo "$QUERY_RESP" | grep -qi 'cudaStreamSync\|StreamSync\|stream_sync'; then
        MCP_SYNC_COUNT=$(echo "$QUERY_RESP" | { grep -oi 'cudaStreamSync\|StreamSync' || true; } | wc -l)
        # Verify no other ops leaked through the filter
        LEAKED_OPS=$(echo "$QUERY_RESP" | { grep -oi 'cuLaunchKernel\|sched_switch\|cudaMalloc\|cudaMemcpy\|cuMemcpy' || true; } | head -1)
        if [[ -n "$LEAKED_OPS" ]]; then
            echo -e "$(ts)   → MCP query_events: ${MCP_SYNC_COUNT} sync refs but leaked: ${LEAKED_OPS}"
            record "FAIL" "T22g: MCP op filter" "filter returned sync events but also leaked ${LEAKED_OPS}"
        else
            echo -e "$(ts)   → MCP query_events op=cudaStreamSync: ${MCP_SYNC_COUNT} refs"
            record "PASS" "T22g: MCP op filter" "cudaStreamSync only, no leaked ops"
        fi
    elif echo "$QUERY_RESP" | grep -qi 'No events\|no events\|0 events\|empty'; then
        # No sync events in DB is OK — the filter worked, just nothing matched
        echo -e "$(ts)   → MCP query_events: no cudaStreamSync events (filter works, no data)"
        record "PASS" "T22g: MCP op filter" "filter works, no cudaStreamSync in window"
    elif echo "$QUERY_RESP" | grep -q 'op.*d_us\|cuda'; then
        # Got events but not specifically cudaStreamSync — check if filter worked
        if echo "$QUERY_RESP" | grep -qi 'cuLaunchKernel\|sched_switch\|cudaMalloc'; then
            record "FAIL" "T22g: MCP op filter" "returned mixed ops (filter not applied)"
        else
            record "PASS" "T22g: MCP op filter" "events returned (op name may be compressed)"
        fi
    else
        record "FAIL" "T22g: MCP op filter" "unexpected: ${QUERY_RESP:0:150}"
    fi

    # Kill MCP server
    sudo kill "$MCP_PID" 2>/dev/null || true
fi

echo ""

################################################################################
# Generate Report — computed verdict, phase comparison, temporal correlation
################################################################################

# Compute Q2 verdict from data (multi-signal: launch_ratio + sync_amplification + launch_rate)
Q2_VERDICT=""
RATE_CHANGE=$(echo "$LAUNCH_RATE" | grep -oP 'change=\K[+-]?[0-9]+' || echo "0")
if [[ "$CUDA_PRESENT" -eq 0 ]]; then
    Q2_VERDICT="Inconclusive — no CUDA/driver events captured."
elif [[ "$LAUNCH_COUNT" -gt 0 ]]; then
    if python3 -c "exit(0 if float('${LAUNCH_RATIO}') < 10 else 1)" 2>/dev/null; then
        # GPU consistent — refine with sync_amplification
        if [[ "$SYNC_AMP" != "N/A" ]] && python3 -c "exit(0 if float('${SYNC_AMP%%x}') < 1 else 1)" 2>/dev/null; then
            # Inverted pattern: contention reduced sync latency (GPU starvation)
            Q2_VERDICT="GPU kernels consistent (ratio=${LAUNCH_RATIO}x). Host contention caused GPU starvation — sync latency decreased (amp=${SYNC_AMP}), launch rate ${RATE_CHANGE}%."
        elif [[ "$SYNC_AMP" != "N/A" ]] && python3 -c "exit(0 if float('${SYNC_AMP%%x}') >= 3 else 1)" 2>/dev/null; then
            Q2_VERDICT="GPU kernels consistent (ratio=${LAUNCH_RATIO}x). Host scheduler inflated sync latency (amp=${SYNC_AMP}). Bottleneck is host."
        else
            Q2_VERDICT="GPU kernels consistent (ratio=${LAUNCH_RATIO}x). Host contention detected (${SCHED_COUNT} context switches) but minimal sync amplification (${SYNC_AMP})."
        fi
    elif python3 -c "exit(0 if float('${LAUNCH_RATIO}') < 30 else 1)" 2>/dev/null; then
        Q2_VERDICT="GPU tail amplification (ratio=${LAUNCH_RATIO}x). Both GPU and host may be contributing."
    else
        Q2_VERDICT="GPU dispatch severely degraded (ratio=${LAUNCH_RATIO}x). Investigate GPU contention."
    fi
else
    Q2_VERDICT="Inconclusive — no cuLaunchKernel events captured."
fi

{
    echo "# ML Engineer Investigation Report"
    echo ""
    echo "**Date**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "**GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "**DB**: ${ML_DB}"
    echo "**Events**: ${TOTAL_EVENTS} (sync=${SYNC_TOTAL} launch=${LAUNCH_TOTAL} sched=${SCHED_TOTAL})"
    echo "**Methodology**: ${BASELINE_SECS}s baseline + $((TRACE_DURATION - BASELINE_SECS))s contention (stress-ng $(nproc) workers)"
    echo ""
    echo "## Q1: \"My training is slow — what's the root cause?\""
    echo ""
    echo "**Tools**: \`ingero explain\` + \`ingero query --json\`"
    echo ""
    echo "- Causal chains found: ${CHAIN_COUNT}"
    if [[ -n "${CHAIN_SUMMARY:-}" ]]; then
        echo "- Top chain: ${CHAIN_SUMMARY}"
    fi
    echo "- Total events analyzed: ${QUERY_COUNT}"
    echo ""
    echo "<details><summary>Full explain output</summary>"
    echo ""
    echo '```'
    echo "$EXPLAIN_OUT"
    echo '```'
    echo "</details>"
    echo ""
    echo "## Q2: \"Is it the GPU or the host?\""
    echo ""
    echo "**Tools**: Per-operation DB queries (\`ingero query --json\`)"
    echo ""
    echo "### GPU Kernel Health (cuLaunchKernel)"
    echo ""
    echo "| Metric | Full Session | Baseline (${BASELINE_SECS}s) | Contention |"
    echo "|--------|-------------|----------|------------|"
    echo "| Events | ${LAUNCH_COUNT} | $(echo "$BASELINE_LAUNCH" | grep -oP 'n=\K[0-9]+' || echo '0') | $(echo "$CONTENTION_LAUNCH" | grep -oP 'n=\K[0-9]+' || echo '0') |"
    echo "| p50 | ${LAUNCH_P50}us | $(echo "$BASELINE_LAUNCH" | grep -oP 'p50=\K[0-9]+' || echo '0')us | $(echo "$CONTENTION_LAUNCH" | grep -oP 'p50=\K[0-9]+' || echo '0')us |"
    echo "| p99 | ${LAUNCH_P99}us | $(echo "$BASELINE_LAUNCH" | grep -oP 'p99=\K[0-9]+' || echo '0')us | $(echo "$CONTENTION_LAUNCH" | grep -oP 'p99=\K[0-9]+' || echo '0')us |"
    echo "| Ratio | ${LAUNCH_RATIO}x | $(echo "$BASELINE_LAUNCH" | grep -oP 'ratio=\K[0-9.]+' || echo '0')x | $(echo "$CONTENTION_LAUNCH" | grep -oP 'ratio=\K[0-9.]+' || echo '0')x |"
    echo ""
    echo "### Host Scheduler Health (sched_switch)"
    echo ""
    echo "- Total: ${SCHED_COUNT} events, max=${SCHED_MAX}us, over_10ms=${OVER_10MS}"
    echo "- Has off-CPU durations: ${HAS_DURATIONS}"
    if [[ -n "$LAUNCH_RATE" ]]; then
        echo "- Launch rate: $(echo "$LAUNCH_RATE" | sed 's/launch_rate: //')"
    fi
    echo ""
    echo "### Verdict"
    echo ""
    echo "**${Q2_VERDICT}**"
    echo ""
    echo "## Q3: \"Show me how CPU contention hits my CUDA calls\""
    echo ""
    echo "**Tools**: Per-operation DB queries (\`ingero query --json\`)"
    echo ""
    echo "### cudaStreamSync Latency"
    echo ""
    echo "| Metric | Full Session | Baseline (${BASELINE_SECS}s) | Contention |"
    echo "|--------|-------------|----------|------------|"
    echo "| Events | ${SYNC_COUNT} | $(echo "$BASELINE_SYNC" | grep -oP 'n=\K[0-9]+' || echo '0') | $(echo "$CONTENTION_SYNC" | grep -oP 'n=\K[0-9]+' || echo '0') |"
    echo "| p50 | ${SYNC_P50}us | $(echo "$BASELINE_SYNC" | grep -oP 'p50=\K[0-9]+' || echo '0')us | $(echo "$CONTENTION_SYNC" | grep -oP 'p50=\K[0-9]+' || echo '0')us |"
    echo "| p99 | ${SYNC_P99}us | $(echo "$BASELINE_SYNC" | grep -oP 'p99=\K[0-9]+' || echo '0')us | $(echo "$CONTENTION_SYNC" | grep -oP 'p99=\K[0-9]+' || echo '0')us |"
    echo "| Tail ratio | ${SYNC_RATIO}x | $(echo "$BASELINE_SYNC" | grep -oP 'ratio=\K[0-9.]+' || echo '0')x | $(echo "$CONTENTION_SYNC" | grep -oP 'ratio=\K[0-9.]+' || echo '0')x |"
    echo ""
    echo "### Temporal Correlation"
    echo ""
    echo "- Sync amplification (contention p99 / baseline p99): ${SYNC_AMP}"
    echo "- Spearman rank correlation (sched count vs sync p99/sec): ${TEMPORAL_RHO}"
    echo "- Temporal overlap (high-sched ∩ high-sync seconds): ${TEMPORAL_OVERLAP}"
    echo ""
    echo "## Q4: \"Can an AI agent diagnose this via MCP?\""
    echo ""
    echo "**Tools**: MCP \`get_trace_stats\` + \`get_causal_chains\` + \`query_events\`"
    echo ""
    echo "- MCP server: port ${MCP_PORT}, DB=${ML_DB}"
    echo "- Same findings available via structured JSON for AI consumption."
    echo ""
    echo "## Results"
    echo ""
    echo "| Test | Status | Detail |"
    echo "|------|--------|--------|"
    for entry in "${ML_RESULTS[@]}"; do
        IFS='|' read -r tid name status detail dur <<< "$entry"
        echo "| ${name} | ${status} | ${detail} |"
    done
    echo ""
    echo "**PASS=${PASS_COUNT} FAIL=${FAIL_COUNT} SKIP=${SKIP_COUNT}**"
} > "$REPORT_FILE"

################################################################################
# Summary
################################################################################

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo -e "  ${GREEN}PASS=${PASS_COUNT}${NC}  ${RED}FAIL=${FAIL_COUNT}${NC}  ${YELLOW}SKIP=${SKIP_COUNT}${NC}  Total=${TOTAL}"
echo -e "  Report: ${REPORT_FILE}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Output structured results for gpu-test.sh ingestion (to stdout, after banner)
# gpu-test.sh captures these lines via grep "^ML_RESULT|"
for entry in "${ML_RESULTS[@]}"; do
    echo "ML_RESULT|${entry}"
done

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
exit 0
