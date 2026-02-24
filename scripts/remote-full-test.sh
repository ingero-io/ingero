#!/bin/bash
# Full integration test + comprehensive log capture on GPU VM
# Run as: bash /tmp/remote-full-test.sh
set -euo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:$HOME/.local/bin:$PATH
cd ~/workspace/ingero

LOGDIR=~/workspace/ingero/logs
mkdir -p "$LOGDIR"

# ── Complete timeline ────────────────────────────────────────────────────────
# Single chronological log connecting every action and its output files.
TIMELINE="$LOGDIR/session-timeline.log"
: > "$TIMELINE"  # truncate

ts() { date -u '+%Y-%m-%d %H:%M:%S'; }

tl() {
    # Append a line to the timeline (and echo to console).
    local msg="$(ts)  $*"
    echo "$msg" | tee -a "$TIMELINE"
}

tl "=== Ingero GPU Test Session Started ==="
tl "Host: $(hostname), Kernel: $(uname -r)"

# ── Phase 1: System Info ─────────────────────────────────────────────────────
tl ""
tl "── Phase 1: System Info Capture ──"

tl "START  nvidia-smi → nvidia-smi.log"
nvidia-smi > "$LOGDIR/nvidia-smi.log" 2>&1
tl "DONE   nvidia-smi.log ($(wc -l < "$LOGDIR/nvidia-smi.log") lines)"

tl "START  uname + os-release → uname.log"
uname -a > "$LOGDIR/uname.log"
cat /etc/os-release >> "$LOGDIR/uname.log"
tl "DONE   uname.log"

tl "START  ingero check --debug → check-debug.log"
sudo ./bin/ingero check --debug > "$LOGDIR/check-debug.log" 2>&1
tl "DONE   check-debug.log ($(wc -l < "$LOGDIR/check-debug.log") lines)"

tl "START  ingero version → version.log"
./bin/ingero version > "$LOGDIR/version.log" 2>&1
tl "DONE   version.log: $(cat "$LOGDIR/version.log" | head -1)"

# ── Phase 2: Integration Tests ───────────────────────────────────────────────
tl ""
tl "── Phase 2: Integration Tests ──"

tl "START  pip install test workload requirements"
pip3 install --quiet -r tests/workloads/requirements.txt 2>&1 | tail -3
tl "DONE   pip install"

tl "START  gpu-integration-test.sh → integration-full.log"
bash scripts/gpu-integration-test.sh 2>&1 | tee "$LOGDIR/integration-full.log"
cp integration-test-report.log "$LOGDIR/" 2>/dev/null || true
tl "DONE   integration-full.log ($(wc -l < "$LOGDIR/integration-full.log") lines)"

# ── Phase 3: Debug Log Captures ──────────────────────────────────────────────
tl ""
tl "── Phase 3: Debug Log Captures ──"

# 3a: Trace ResNet50
tl "START  3a: Launch ResNet50 workload (1 epoch)"
python3 tests/workloads/training/resnet50_cifar10.py --epochs 1 &>/dev/null &
WORKLOAD_PID=$!
sleep 3
CUDA_PID=$(pgrep -f "resnet50_cifar10" | head -1) || CUDA_PID=$WORKLOAD_PID
tl "       Workload PID=$WORKLOAD_PID, CUDA_PID=$CUDA_PID"
tl "START  3a: ingero trace --debug --json --pid $CUDA_PID --duration 45s → watch-resnet50.json + .log"
sudo ./bin/ingero trace --debug --json --pid "$CUDA_PID" --duration 45s \
  > "$LOGDIR/watch-resnet50.json" 2> "$LOGDIR/watch-resnet50.log" || true
wait $WORKLOAD_PID 2>/dev/null || true
tl "DONE   3a: watch-resnet50.json ($(wc -l < "$LOGDIR/watch-resnet50.json") lines), watch-resnet50.log ($(wc -l < "$LOGDIR/watch-resnet50.log") lines)"

sleep 2

# 3b: Trace launch_storm
tl "START  3b: Launch launch_storm workload"
python3 tests/workloads/synthetic/launch_storm.py &>/dev/null &
WORKLOAD_PID=$!
sleep 3
CUDA_PID=$(pgrep -f "launch_storm" | head -1) || CUDA_PID=$WORKLOAD_PID
tl "       Workload PID=$WORKLOAD_PID, CUDA_PID=$CUDA_PID"
tl "START  3b: ingero trace --debug --json --pid $CUDA_PID --duration 30s → watch-launch-storm.json + .log"
sudo ./bin/ingero trace --debug --json --pid "$CUDA_PID" --duration 30s \
  > "$LOGDIR/watch-launch-storm.json" 2> "$LOGDIR/watch-launch-storm.log" || true
wait $WORKLOAD_PID 2>/dev/null || true
tl "DONE   3b: watch-launch-storm.json ($(wc -l < "$LOGDIR/watch-launch-storm.json") lines), watch-launch-storm.log ($(wc -l < "$LOGDIR/watch-launch-storm.log") lines)"

sleep 2

# 3c: Trace CPU contention
tl "START  3c: Launch stress-ng (4 CPU workers, 30s) + matmul workload (28s)"
stress-ng --cpu 4 --timeout 30s &>/dev/null &
STRESS_PID=$!
python3 -c "
import torch
import time
a = torch.randn(2048, 2048, device='cuda')
b = torch.randn(2048, 2048, device='cuda')
start = time.time()
while time.time() - start < 28:
    c = torch.mm(a, b)
    torch.cuda.synchronize()
" &>/dev/null &
WORKLOAD_PID=$!
sleep 2
CUDA_PID=$(pgrep -f "python3 -c" | head -1) || CUDA_PID=$WORKLOAD_PID
tl "       Stress PID=$STRESS_PID, Workload PID=$WORKLOAD_PID, CUDA_PID=$CUDA_PID"
tl "START  3c: ingero trace --debug --json --pid $CUDA_PID --duration 25s → watch-cpu-contention.json + .log"
sudo ./bin/ingero trace --debug --json --pid "$CUDA_PID" --duration 25s \
  > "$LOGDIR/watch-cpu-contention.json" 2> "$LOGDIR/watch-cpu-contention.log" || true
wait $WORKLOAD_PID 2>/dev/null || true
kill $STRESS_PID 2>/dev/null || true
wait $STRESS_PID 2>/dev/null || true
tl "DONE   3c: watch-cpu-contention.json ($(wc -l < "$LOGDIR/watch-cpu-contention.json") lines), watch-cpu-contention.log ($(wc -l < "$LOGDIR/watch-cpu-contention.log") lines)"

sleep 2

# 3d: Trace with record + query round-trip
tl "START  3d: Launch matmul workload (18s)"
python3 -c "
import torch, time
a = torch.randn(1024, 1024, device='cuda')
start = time.time()
while time.time() - start < 18:
    b = torch.mm(a, a)
    torch.cuda.synchronize()
" &>/dev/null &
WORKLOAD_PID=$!
sleep 2
CUDA_PID=$(pgrep -f "python3 -c" | head -1) || CUDA_PID=$WORKLOAD_PID
tl "START  3d: ingero trace --debug --json --pid $CUDA_PID --duration 15s → watch-record.log"
sudo ./bin/ingero trace --debug --json --pid "$CUDA_PID" --duration 15s \
  > /dev/null 2> "$LOGDIR/watch-record.log" || true
wait $WORKLOAD_PID 2>/dev/null || true
tl "START  3d: ingero query --since 5m --json → query-all.json"
sudo ./bin/ingero query --since 5m --json > "$LOGDIR/query-all.json" 2>/dev/null || true
tl "DONE   3d: watch-record.log ($(wc -l < "$LOGDIR/watch-record.log") lines), query-all.json ($(wc -l < "$LOGDIR/query-all.json") lines)"

sleep 2

# 3e: Explain on stored data (trace during contention was captured in 3c)
tl "START  3e: ingero explain --debug --since 5m → explain-debug.log"
./bin/ingero explain --debug --since 5m \
  > "$LOGDIR/explain-debug.log" 2>&1 || true
tl "DONE   3e: explain-debug.log ($(wc -l < "$LOGDIR/explain-debug.log") lines)"

sleep 2

# 3f: Demo synthetic
tl "START  3f: ingero demo --no-gpu → demo-synthetic.log"
sudo timeout 60 ./bin/ingero demo --no-gpu > "$LOGDIR/demo-synthetic.log" 2>&1 || true
tl "DONE   3f: demo-synthetic.log ($(wc -l < "$LOGDIR/demo-synthetic.log") lines)"

# 3g: Demo GPU incident
tl "START  3g: ingero demo --gpu incident --debug → demo-gpu-incident.log"
sudo timeout 60 ./bin/ingero demo --gpu incident --debug \
  > "$LOGDIR/demo-gpu-incident.log" 2>&1 || true
tl "DONE   3g: demo-gpu-incident.log ($(wc -l < "$LOGDIR/demo-gpu-incident.log") lines)"

sleep 2

# 3h: Trace without --pid (dynamic PID tracking)
tl "START  3h: Launch matmul workload (22s), no --pid"
python3 -c "
import torch, time
a = torch.randn(1024, 1024, device='cuda')
start = time.time()
while time.time() - start < 22:
    b = torch.mm(a, a)
    torch.cuda.synchronize()
" &>/dev/null &
WORKLOAD_PID=$!
sleep 2
tl "START  3h: ingero trace --debug --json --duration 20s → watch-nopid.json + .log"
sudo ./bin/ingero trace --debug --json --duration 20s \
  > "$LOGDIR/watch-nopid.json" 2> "$LOGDIR/watch-nopid.log" || true
wait $WORKLOAD_PID 2>/dev/null || true
tl "DONE   3h: watch-nopid.json ($(wc -l < "$LOGDIR/watch-nopid.json") lines), watch-nopid.log ($(wc -l < "$LOGDIR/watch-nopid.log") lines)"

# ── Phase 4: Summary ─────────────────────────────────────────────────────────
tl ""
tl "── Phase 4: Log Summary ──"
tl "Files in $LOGDIR:"
ls -lh "$LOGDIR/" | while IFS= read -r line; do tl "  $line"; done
tl ""
tl "=== Ingero GPU Test Session Complete ==="
