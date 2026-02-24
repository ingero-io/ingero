#!/bin/bash
# Lambda Labs GPU Integration Test Script
# Run on remote: tests ingero with and without --debug, records findings.
# Works on any GPU type (H100, A100, A10, L40, L4, etc.)
export PATH=/usr/local/go/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/go/bin:$HOME/.local/bin:$PATH
cd ~/workspace/ingero
mkdir -p logs
exec > >(tee logs/integration-test-report.log) 2>&1

GPU_NAME=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
echo "================================================================"
echo "  Ingero GPU Integration Test — $(date)"
echo "  $GPU_NAME"
echo "================================================================"

# --- Test 1: ingero check ---
echo ""
echo "=== TEST 1: ingero check ==="
sudo ./bin/ingero check 2>&1 | tee logs/check.log
echo "TEST 1: DONE"

# --- Test 2: ingero demo --no-gpu (all scenarios) ---
echo ""
echo "=== TEST 2: ingero demo --no-gpu (all 6 scenarios) ==="
for scenario in incident cold-start memcpy-bottleneck periodic-spike cpu-contention gpu-steal; do
    echo "--- Scenario: $scenario ---"
    ./bin/ingero demo --no-gpu "$scenario" --json 2>logs/demo-${scenario}-stderr.log | head -5
    echo "  exit=$?"
done
echo "TEST 2: DONE"

# --- Test 3: ingero trace (real GPU workload, no --debug) ---
echo ""
echo "=== TEST 3: ingero trace --duration 15s (clean, no --debug) ==="
python3 -c "
import torch
device = torch.device('cuda')
for i in range(2000):
    t = torch.randn(512, 512, device=device)
    r = torch.mm(t, t)
    torch.cuda.synchronize()
" &
PY_PID=$!
sleep 1
sudo ./bin/ingero trace --json --duration 15s > logs/watch-clean.json 2> logs/watch-clean.log
wait $PY_PID 2>/dev/null || true
echo "Events captured (clean):"
python3 -c "
import json, sys
events = []
for line in open('logs/watch-clean.json'):
    try: events.append(json.loads(line))
    except: pass
from collections import Counter
ops = Counter(e.get('op','?') for e in events if 'op' in e)
print(f'  Total events: {len(events)}')
for op, count in ops.most_common():
    print(f'  {op}: {count}')
" 2>/dev/null || echo "  (could not parse JSON)"
echo "TEST 3: DONE"

# --- Test 4: ingero trace --debug (real GPU workload) ---
echo ""
echo "=== TEST 4: ingero trace --debug --duration 15s ==="
python3 -c "
import torch
device = torch.device('cuda')
for i in range(2000):
    t = torch.randn(512, 512, device=device)
    r = torch.mm(t, t)
    torch.cuda.synchronize()
" &
PY_PID=$!
sleep 1
sudo ./bin/ingero trace --debug --json --duration 15s > logs/watch-debug.json 2> logs/watch-debug.log
wait $PY_PID 2>/dev/null || true
echo "Events captured (debug):"
python3 -c "
import json, sys
events = []
for line in open('logs/watch-debug.json'):
    try: events.append(json.loads(line))
    except: pass
from collections import Counter
ops = Counter(e.get('op','?') for e in events if 'op' in e)
print(f'  Total events: {len(events)}')
for op, count in ops.most_common():
    print(f'  {op}: {count}')
" 2>/dev/null || echo "  (could not parse JSON)"
echo "Debug output lines: $(wc -l < logs/watch-debug.log)"
echo "TEST 4: DONE"

# --- Test 5: ingero trace (records to SQLite by default) ---
echo ""
echo "=== TEST 5: ingero trace --duration 10s ==="
python3 -c "
import torch
device = torch.device('cuda')
for i in range(1000):
    t = torch.randn(256, 256, device=device)
    r = torch.mm(t, t)
    torch.cuda.synchronize()
" &
PY_PID=$!
sleep 1
sudo ./bin/ingero trace --json --duration 10s > logs/watch-record.json 2> logs/watch-record.log
wait $PY_PID 2>/dev/null || true
echo "TEST 5: DONE"

# --- Test 6: ingero query (read back from SQLite) ---
echo ""
echo "=== TEST 6: ingero query --since 60s --json ==="
sudo ./bin/ingero query --since 60s --json --limit 20 > logs/query.json 2> logs/query.log
echo "Queried events:"
python3 -c "
import json
events = []
for line in open('logs/query.json'):
    try: events.append(json.loads(line))
    except: pass
print(f'  Total returned: {len(events)}')
" 2>/dev/null || echo "  (could not parse)"
echo "TEST 6: DONE"

# --- Test 7: ingero explain (DB-only, no sudo needed) ---
echo ""
echo "=== TEST 7: ingero explain --debug --since 180s ==="
./bin/ingero explain --debug --since 180s > logs/explain-debug.log 2>&1
tail -20 logs/explain-debug.log
echo "TEST 7: DONE"

# --- Test 8: ingero demo --gpu (real GPU probes) ---
echo ""
echo "=== TEST 8: ingero demo incident (GPU mode) ==="
sudo ./bin/ingero demo incident --json > logs/demo-gpu-incident.json 2> logs/demo-gpu-incident.log
echo "GPU demo events:"
python3 -c "
import json
events = []
for line in open('logs/demo-gpu-incident.json'):
    try: events.append(json.loads(line))
    except: pass
from collections import Counter
ops = Counter(e.get('op','?') for e in events if 'op' in e)
print(f'  Total events: {len(events)}')
for op, count in ops.most_common():
    print(f'  {op}: {count}')
" 2>/dev/null || echo "  (could not parse)"
echo "TEST 8: DONE"

# --- Test 9: ingero trace (stacks on by default) ---
echo ""
echo "=== TEST 9: ingero trace --json --duration 10s (stacks on by default) ==="
python3 -c "
import torch
device = torch.device('cuda')
for i in range(1000):
    t = torch.randn(256, 256, device=device)
    r = torch.mm(t, t)
    torch.cuda.synchronize()
" &
PY_PID=$!
sleep 1
sudo ./bin/ingero trace --json --duration 10s > logs/watch-stack.json 2> logs/watch-stack.log
wait $PY_PID 2>/dev/null || true
echo "Stack trace events:"
python3 -c "
import json
events = []
for line in open('logs/watch-stack.json'):
    try: events.append(json.loads(line))
    except: pass
stack_events = [e for e in events if e.get('stack')]
print(f'  Total events: {len(events)}')
print(f'  Events with stack: {len(stack_events)}')
if stack_events:
    e = stack_events[0]
    print(f'  Sample stack ({e.get(\"op\",\"?\")}, depth={len(e[\"stack\"])}):')
    for f in e['stack'][:5]:
        sym = f.get('symbol','')
        ip = f.get('ip','')
        fi = f.get('file','')
        pf = f.get('py_file','')
        if pf:
            print(f'    [Python] {pf}:{f.get(\"py_line\",\"?\")} in {f.get(\"py_func\",\"?\")}()')
        elif sym:
            print(f'    [Native] {sym} ({fi})')
        else:
            print(f'    {ip}')
" 2>/dev/null || echo "  (could not parse JSON)"
echo "TEST 9: DONE"

# --- Test 10: Tier 1 synthetic workloads ---
echo ""
echo "=== TEST 10: Tier 1 Synthetic Workloads ==="
cd ~/workspace/ingero/tests/workloads
pip3 install --quiet torch torchvision numpy 2>/dev/null || true
for script in synthetic/alloc_stress.py synthetic/memcpy_stress.py synthetic/launch_storm.py synthetic/sync_stall.py; do
    echo "--- $script ---"
    timeout 60 python3 "$script" 2>&1 | tail -5
    echo "  exit=$?"
done
echo "TEST 10: DONE"

# --- Summary ---
echo ""
echo "================================================================"
echo "  Integration Test Complete — $(date)"
echo "================================================================"
echo ""
echo "Log files in ~/workspace/ingero/logs/:"
ls -la ~/workspace/ingero/logs/
