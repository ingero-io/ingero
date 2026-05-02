#!/bin/bash
################################################################################
# v0.12 + v0.11 + v0.10.1 hardware validation, runs on a Lambda VM after
# `make lambda-sync` and `make build` have populated bin/.
#
# This script extends validate-v0.11.sh (queued in /tmp/lambda-poll/ during
# v0.11.0) with:
#   - NCCL probe attach assertion (libnccl.so or libtorch_cuda.so)
#   - per-arch BPF object presence under internal/ebpf/ncclprobe
#   - --fleet-push-interval bumped to 6s so the emitter's "timeout (5s)
#     <= push_interval" guard passes (was 1s in validate-v0.11.sh,
#     causing WARN-only A1/B1 captures)
#
# Exit codes:
#   0  all assertions passed (or warnings only)
#   1  hard failure
#   2  test environment broken (e.g. missing nvidia-smi); not a regression
################################################################################

set -uo pipefail
cd "$HOME/workspace/ingero"

# Lambda Cloud images install Go under /usr/local/go/bin but interactive
# shells don't always source it. Make sure tests that shell out to `go`
# (Workstream C N2 below) can find the toolchain.
export PATH="/usr/local/go/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

VAL_DIR=logs/v0.12-validation
mkdir -p "$VAL_DIR"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS+1)); }
bad()  { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL+1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; WARN=$((WARN+1)); }
PASS=0; FAIL=0; WARN=0

ARCH=$(uname -m)
KERNEL=$(uname -r)
echo "================================================================"
echo "  v0.12 + v0.11 + v0.10.1 hardware validation"
echo "  arch=$ARCH  kernel=$KERNEL"
echo "  date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

############### v0.10.1 sanity ################################################

echo
echo "--- v0.10.1 T1: ingero check ---"
sudo ./bin/ingero check 2>&1 | tee "$VAL_DIR/01-check.txt"
RC=${PIPESTATUS[0]}
[ "$RC" -eq 0 ] && ok "v0.10.1 T1: check rc=0" || bad "v0.10.1 T1: check rc=$RC"

# Pick a python with torch for the workload.
PYTHON=python3
if ! python3 -c "import torch" 2>/dev/null; then
    if [ -x /opt/pytorch/bin/python3 ]; then
        PYTHON=/opt/pytorch/bin/python3
    fi
fi

cat > /tmp/long_workload.py <<'PYEOF'
import torch, time
dev = torch.device('cuda:0')
end = time.time() + 45
A = torch.randn(2048, 2048, device=dev)
B = torch.randn(2048, 2048, device=dev)
n = 0
while time.time() < end:
    C = A @ B
    torch.cuda.synchronize()
    n += 1
print(f'finished {n} matmuls')
PYEOF

echo
echo "--- v0.10.1 T2-T3: trace --debug --duration 30s ---"
"$PYTHON" /tmp/long_workload.py > /tmp/wl.out 2>&1 &
WL=$!
sleep 3
sudo ./bin/ingero trace --debug --pid "$WL" --duration 30s --json \
    > "$VAL_DIR/03-trace.json" 2> "$VAL_DIR/03-trace.log"
TRACE_RC=$?
wait "$WL" 2>/dev/null

if grep -qE "bad CO-RE relocation|invalid func unknown" "$VAL_DIR/03-trace.log"; then
    bad "v0.10.1 T3: CO-RE relocation error in trace stderr (saiyam #35 regression)"
else
    ok "v0.10.1 T3: no CO-RE relocation errors"
fi

PROBE_LINES=$(grep -cE "[0-9]+ probes? attached|[0-9]+ tracepoints? attached" "$VAL_DIR/03-trace.log" || echo 0)
PROBE_LINES=$(echo "$PROBE_LINES" | head -1 | tr -d '[:space:]')
if [ "$PROBE_LINES" -ge 5 ]; then
    ok "v0.10.1 T3: >=5 probe-group attach lines ($PROBE_LINES seen)"
else
    bad "v0.10.1 T3: only $PROBE_LINES probe-group attach lines"
fi

EV=$(grep -c '^\s*{' "$VAL_DIR/03-trace.json" 2>/dev/null || echo 0)
EV=$(echo "$EV" | head -1 | tr -d '[:space:]')
if [ "$EV" -ge 100 ]; then
    ok "v0.10.1 T3: $EV events recorded"
else
    bad "v0.10.1 T3: only $EV events"
fi

############### v0.11 A1+A2: rank/world_size on OTLP resource + event_id #####

echo
echo "--- v0.11 A1+A2: OTLP push shape (uses fleet-push --stub) ---"
PORT=$(comm -23 <(seq 18000 18999 | sort) <(ss -tan 2>/dev/null | awk '{print $4}' | grep -oE ':[0-9]+$' | sort -u) | shuf -n 1)
PORT=${PORT:-18888}

cat > /tmp/otlp_capture.py <<PYEOF
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get('Content-Length','0'))
        b = self.rfile.read(n)
        sys.stdout.buffer.write(b); sys.stdout.write('\n'); sys.stdout.flush()
        self.send_response(200); self.end_headers()
    def log_message(self, *a, **k): pass
http.server.HTTPServer(('127.0.0.1', $PORT), H).serve_forever()
PYEOF

"$PYTHON" /tmp/otlp_capture.py > "$VAL_DIR/04-otlp-bodies.ndjson" 2>&1 &
CAP_PID=$!
sleep 1

# v0.12 fix: --fleet-push-interval bumped from 1s (validate-v0.11.sh) to 6s
# so the emitter's "timeout (5s) <= push_interval" guard passes and the
# A1/B1 captures actually assert instead of warning.
sudo ./bin/ingero fleet-push \
    --fleet-endpoint "http://127.0.0.1:$PORT" --fleet-insecure \
    --fleet-cluster-id v0.12-validation \
    --fleet-node-id "$(hostname)" \
    --fleet-world-size 4 --fleet-node-rank 1 \
    --stub --warmup-samples 0 \
    --fleet-push-interval 6s \
    > "$VAL_DIR/04-fleet-push.log" 2>&1 &
FP=$!
sleep 9
sudo kill "$FP" 2>/dev/null
kill "$CAP_PID" 2>/dev/null

BODY=$(head -1 "$VAL_DIR/04-otlp-bodies.ndjson" 2>/dev/null)
if [ -n "$BODY" ]; then
    if echo "$BODY" | python3 -c "
import json,sys
d=json.loads(sys.stdin.read())
rm=d['resourceMetrics'][0]
res_attrs={a['key']:a['value'] for a in rm['resource']['attributes']}
have_world = 'ingero.world_size' in res_attrs
have_rank  = 'ingero.node.rank' in res_attrs
print('A1_RES_OK' if (have_world and have_rank) else 'A1_RES_MISS', res_attrs.keys())
" | grep -q "A1_RES_OK"; then
        ok "v0.11 A1: ingero.world_size + ingero.node.rank on resource block"
    else
        bad "v0.11 A1: rank/world_size NOT on resource block"
    fi
else
    warn "v0.11 A1: no OTLP body captured (port-bind or fleet-push stub issue)"
fi

############### v0.11 B1: cost gauges ##########################################

echo
echo "--- v0.11 B1: ingero.node.info + ingero.node.world_size gauges ---"
if [ -n "$BODY" ]; then
    if echo "$BODY" | python3 -c "
import json,sys
d=json.loads(sys.stdin.read())
metrics={m['name']:m for m in d['resourceMetrics'][0]['scopeMetrics'][0]['metrics']}
ok_info = 'ingero.node.info' in metrics
ok_ws   = 'ingero.node.world_size' in metrics
print('B1_OK' if (ok_info and ok_ws) else 'B1_MISS', list(metrics.keys()))
" | grep -q "B1_OK"; then
        ok "v0.11 B1: ingero.node.info + ingero.node.world_size both present"
    else
        bad "v0.11 B1: cost gauges missing from OTLP push"
    fi
else
    warn "v0.11 B1: no OTLP body to inspect"
fi

############### v0.11 C1: support-bundle ######################################

echo
echo "--- v0.11 C1: ingero check --support-bundle ---"
# v0.12.1: capture stderr to a file so a transient write failure is
# visible in the artifacts; retry the tar listing twice with a small
# sleep so a slow filesystem flush (observed once on aarch64 in the
# v0.12.1 cycle) doesn't trip a false negative.
SB_LOG="$VAL_DIR/05-support-bundle-cmd.log"
sudo rm -f "$VAL_DIR/05-support-bundle.tgz"
sudo ./bin/ingero check --support-bundle "$VAL_DIR/05-support-bundle.tgz" \
    > "$SB_LOG" 2>&1
SB_RC=$?
sync
C1_OK=0
for attempt in 1 2 3; do
    if [ -s "$VAL_DIR/05-support-bundle.tgz" ] && \
       tar tzf "$VAL_DIR/05-support-bundle.tgz" 2>&1 | grep -q "ingero-support/metadata.txt"; then
        C1_OK=1
        break
    fi
    sleep 1
done
if [ "$C1_OK" -eq 1 ]; then
    ok "v0.11 C1: support-bundle tarball valid + has metadata.txt"
elif [ ! -s "$VAL_DIR/05-support-bundle.tgz" ]; then
    bad "v0.11 C1: tarball not produced (cmd rc=$SB_RC, see $SB_LOG)"
else
    bad "v0.11 C1: tarball missing metadata.txt (cmd rc=$SB_RC, see $SB_LOG)"
fi

############### v0.11 C6: migrate ##############################################

echo
echo "--- v0.11 C6: ingero migrate --dry-run on fresh DB ---"
sudo rm -f /tmp/migrate-test.db
if sudo ./bin/ingero migrate --dry-run --db-path /tmp/migrate-test.db > "$VAL_DIR/06-migrate.txt" 2>&1; then
    if grep -q "no migrations pending" "$VAL_DIR/06-migrate.txt"; then
        ok "v0.11 C6: migrate --dry-run reports no migrations pending"
    else
        warn "v0.11 C6: migrate --dry-run output unexpected: $(cat $VAL_DIR/06-migrate.txt | head -3)"
    fi
else
    bad "v0.11 C6: migrate --dry-run exited non-zero"
fi

############### v0.12 N1: per-arch NCCL .o present ############################

echo
echo "--- v0.12 N1: per-arch NCCL .bpf.o objects shipped ---"
NCCL_X86_O=internal/ebpf/ncclprobe/nccltrace_x86_bpfel.o
NCCL_ARM64_O=internal/ebpf/ncclprobe/nccltrace_arm64_bpfel.o
if [ -s "$NCCL_X86_O" ] && [ -s "$NCCL_ARM64_O" ]; then
    ok "v0.12 N1: both nccltrace_*_bpfel.o files present"
else
    bad "v0.12 N1: missing one or both per-arch NCCL .o (saw x86=$([ -s $NCCL_X86_O ] && echo y || echo n) arm64=$([ -s $NCCL_ARM64_O ] && echo y || echo n))"
fi

############### v0.12 N2: NCCL parity test ####################################

echo
echo "--- v0.12 N2: cross-arch CO-RE parity covers ncclprobe ---"
if go test -count=1 -run TestPerArchCORERelocationsMatchArch ./internal/ebpf/parity/ \
    > "$VAL_DIR/07-nccl-parity.log" 2>&1; then
    if grep -q "ncclprobe" "$VAL_DIR/07-nccl-parity.log" 2>/dev/null; then
        ok "v0.12 N2: parity test ran and exercised ncclprobe"
    else
        # Test passed but didn't print ncclprobe; check verbose mode
        go test -v -count=1 -run TestPerArchCORERelocationsMatchArch ./internal/ebpf/parity/ \
            > "$VAL_DIR/07-nccl-parity.log" 2>&1
        if grep -q "ncclprobe" "$VAL_DIR/07-nccl-parity.log"; then
            ok "v0.12 N2: parity test exercised ncclprobe (verbose log)"
        else
            warn "v0.12 N2: parity test passed but ncclprobe coverage not visible"
        fi
    fi
else
    bad "v0.12 N2: parity test failed (saiyam #35 / NCCL CO-RE regression?)"
fi

############### v0.12 N3: libnccl discovery ###################################

echo
echo "--- v0.12 N3: libnccl.so or libtorch_cuda.so discoverable ---"
LIBNCCL_HINT=$(go run ./cmd/ingero check --json 2>/dev/null | grep -oE '"libnccl_path"\s*:\s*"[^"]*"' | head -1 || echo "")
if [ -n "$LIBNCCL_HINT" ] && [ "$LIBNCCL_HINT" != '"libnccl_path":""' ]; then
    ok "v0.12 N3: libnccl discoverable: $LIBNCCL_HINT"
else
    # check fallback: just stat known paths for amd64 PyTorch / Lambda
    for cand in /usr/lib/x86_64-linux-gnu/libnccl.so.2 \
                /usr/local/lib/libnccl.so.2 \
                /opt/pytorch/lib/python*/site-packages/torch/lib/libtorch_cuda.so; do
        if compgen -G "$cand" > /dev/null 2>&1; then
            ok "v0.12 N3: libnccl-bearing object on disk: $cand"
            FOUND_NCCL=1
            break
        fi
    done
    if [ -z "${FOUND_NCCL:-}" ]; then
        warn "v0.12 N3: no libnccl.so or libtorch_cuda.so on disk (PyTorch not installed in this image?)"
    fi
fi

############### v0.12 N4: NCCL probe attach on real kernel ####################

echo
echo "--- v0.12 N4: ingero trace --nccl probe attach ---"
"$PYTHON" /tmp/long_workload.py > /tmp/wl-nccl.out 2>&1 &
WL2=$!
sleep 3
sudo ./bin/ingero trace --nccl --debug --pid "$WL2" --duration 5s --json \
    > "$VAL_DIR/08-nccl-trace.json" 2> "$VAL_DIR/08-nccl-trace.log"
N4_TRACE_RC=$?
wait "$WL2" 2>/dev/null

if grep -q "NCCL tracing: attached" "$VAL_DIR/08-nccl-trace.log"; then
    PROBES=$(grep -oE "attached [0-9]+ probes" "$VAL_DIR/08-nccl-trace.log" | head -1 | grep -oE "[0-9]+")
    # v0.12.1 (QA #2): lock probe count to the documented invariant.
    # 8 collectives x (uprobe + uretprobe) = 16. A future regression
    # to 12 (drop of Send/Recv) would otherwise pass silently.
    if [ "$PROBES" = "16" ]; then
        ok "v0.12 N4: NCCL probes attached on real kernel (16 probes)"
    else
        bad "v0.12 N4: probe count = $PROBES, want 16 (Send/Recv probe drop?)"
    fi
elif grep -q "no libnccl.so / libtorch_cuda.so found" "$VAL_DIR/08-nccl-trace.log"; then
    warn "v0.12 N4: NCCL flag honored but no libnccl.so on disk (test environment)"
elif grep -qE "bad CO-RE relocation|invalid func unknown" "$VAL_DIR/08-nccl-trace.log"; then
    bad "v0.12 N4: CO-RE relocation error attaching NCCL probes (saiyam #35-style regression?)"
elif grep -q "NCCL tracing unavailable" "$VAL_DIR/08-nccl-trace.log"; then
    bad "v0.12 N4: NCCL probe attach failed (verifier reject?)"
else
    warn "v0.12 N4: NCCL trace ran but no clear attach signal (rc=$N4_TRACE_RC)"
fi

############### v0.12.2 N5: 2-rank torchrun NCCL emission shape ################
# Validates the v0.12.0 contract that real NCCL collectives produce events
# with nranks == world_size and a non-zero comm_id_hash. Skips cleanly when
# the box has fewer than 2 GPUs or torchrun is unavailable.

echo
echo "--- v0.12.2 N5: 2-rank torchrun all-reduce emits nranks=2 + comm_id_hash ---"
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    warn "v0.12.2 N5: only $GPU_COUNT GPU(s); single-GPU box, skipping multi-rank assertion"
elif ! "$PYTHON" -c "import torch.distributed" 2>/dev/null; then
    warn "v0.12.2 N5: torch.distributed not importable; skipping"
elif ! command -v torchrun >/dev/null 2>&1 && ! "$PYTHON" -m torch.distributed.run --help >/dev/null 2>&1; then
    warn "v0.12.2 N5: torchrun unavailable; skipping"
else
    cat > /tmp/nccl_2rank.py <<'PYEOF'
import os, time, torch, torch.distributed as dist
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
ws = dist.get_world_size()
torch.cuda.set_device(rank)
end = time.time() + 30
n = 0
while time.time() < end:
    t = torch.ones(1024, 1024, device=f'cuda:{rank}') * (rank + 1)
    dist.all_reduce(t)
    torch.cuda.synchronize()
    n += 1
print(f'rank {rank}/{ws} did {n} all-reduces', flush=True)
dist.destroy_process_group()
PYEOF
    # Pick a free TCP port for c10d. Same trick as A1+A2 above.
    C10D_PORT=$(comm -23 <(seq 29400 29999 | sort) <(ss -tan 2>/dev/null | awk '{print $4}' | grep -oE ':[0-9]+$' | sort -u) | shuf -n 1)
    C10D_PORT=${C10D_PORT:-29500}
    if command -v torchrun >/dev/null 2>&1; then
        TORCHRUN="torchrun"
    else
        TORCHRUN="$PYTHON -m torch.distributed.run"
    fi
    $TORCHRUN --nproc_per_node=2 --master_port="$C10D_PORT" /tmp/nccl_2rank.py \
        > /tmp/wl-2rank.out 2>&1 &
    WL5=$!
    sleep 4
    # Trace either child rank — both share the same comm; either's NCCL events
    # carry the same comm_id_hash and nranks=2.
    RANK_PID=$(pgrep -P "$WL5" | head -1)
    if [ -z "$RANK_PID" ]; then
        warn "v0.12.2 N5: torchrun started but no child rank PID found"
    else
        sudo ./bin/ingero trace --nccl --debug --pid "$RANK_PID" --duration 10s --json \
            > "$VAL_DIR/09-nccl-2rank.json" 2> "$VAL_DIR/09-nccl-2rank.log"
        wait "$WL5" 2>/dev/null

        # Pull NCCL events out of the JSON stream (one event per line).
        # An "all_reduce" event's nranks comes from PARM5 of ncclAllReduce
        # = the comm's nranks; comm_id_hash is the 16-hex-char sha of the
        # commId blob.
        NRANKS_2=$(grep -oE '"nranks":\s*2' "$VAL_DIR/09-nccl-2rank.json" 2>/dev/null | head -1)
        NONZERO_HASH=$(grep -oE '"comm_id_hash":\s*"[0-9a-f]{16}"' "$VAL_DIR/09-nccl-2rank.json" 2>/dev/null \
            | grep -v '"comm_id_hash":\s*"0000000000000000"' | head -1)
        if [ -n "$NRANKS_2" ] && [ -n "$NONZERO_HASH" ]; then
            ok "v0.12.2 N5: 2-rank NCCL events carry nranks=2 + non-zero comm_id_hash"
        elif [ -n "$NRANKS_2" ]; then
            bad "v0.12.2 N5: nranks=2 seen but comm_id_hash is zero/missing"
        elif [ -n "$NONZERO_HASH" ]; then
            bad "v0.12.2 N5: non-zero comm_id_hash seen but nranks != 2"
        else
            bad "v0.12.2 N5: no NCCL event with nranks=2 + non-zero comm_id_hash captured"
        fi
    fi
fi

############### v0.11 D4: gpu-test.sh per-feature assertion lines #############

echo
echo "--- v0.11 D4: presence of T07e/f/g/h labels in gpu-test.sh ---"
if grep -qE "T07e: host kernel|T07f: block I/O|T07g: net syscalls|T07h: tcp retransmit" scripts/gpu-test.sh; then
    ok "v0.11 D4: per-feature integration assertion labels present in gpu-test.sh"
else
    bad "v0.11 D4: per-feature labels missing from gpu-test.sh"
fi

############### Summary #######################################################

echo
echo "================================================================"
echo "  Summary: PASS=$PASS  WARN=$WARN  FAIL=$FAIL"
echo "  Artifacts: $VAL_DIR/"
ls -la "$VAL_DIR/" | tail -10
echo "================================================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
