#!/usr/bin/env bash
# install-python-versions.sh — provision a range of CPython versions for the
# per-version walker-matrix test harness.
#
# What this does:
#   1. Install `uv` if missing (idempotent — skips if already on PATH).
#   2. Install CPython 3.9 through 3.14 via `uv python install`.
#   3. Create a venv per version at /tmp/venv-<version>.
#   4. Install torch into each venv where cu124 wheels exist (3.9..3.13);
#      3.14 has no torch wheels as of 2026-04, so that venv gets no torch
#      and the workload is expected to drive cuda via ctypes instead.
#   5. Emit a manifest at /tmp/walker-matrix-pythons.txt — one line per
#      version: `<minor>:<python-bin-path>:<has_torch>`. The walker-matrix
#      harness reads this to decide which venvs to run and how.
#
# Idempotent: re-running is cheap; already-installed pythons and existing
# venvs are reused. Torch install is the slow step; we skip it if import
# torch already works in the venv.

set -euo pipefail

TARGET_VERSIONS=(3.9 3.10 3.11 3.12 3.13 3.14)
MANIFEST=/tmp/walker-matrix-pythons.txt
VENV_PREFIX=/tmp/venv
# cu124 torch wheels cover 3.9..3.13. 3.14 has no torch wheels upstream
# at the time this script was written; the matrix workload handles that
# by falling back to ctypes-only cuda calls.
TORCH_VERSIONS=(3.9 3.10 3.11 3.12 3.13)

log() { echo "[install-pythons] $*"; }

# 1. Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1090,SC1091
    . "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { log "uv install failed"; exit 1; }

# 2. Install each target python. `uv python install` is idempotent.
log "installing python versions: ${TARGET_VERSIONS[*]}"
uv python install "${TARGET_VERSIONS[@]}"

# Helper: does this version appear in the torch-supported list?
has_torch_support() {
    local v=$1
    for tv in "${TORCH_VERSIONS[@]}"; do
        [[ "$tv" == "$v" ]] && return 0
    done
    return 1
}

# 3 + 4. Per-version venv + torch.
: > "$MANIFEST"
for v in "${TARGET_VERSIONS[@]}"; do
    venv="${VENV_PREFIX}-${v}"
    py="${venv}/bin/python"
    if [[ ! -x "$py" ]]; then
        log "creating venv ${venv}"
        uv venv --python "$v" "$venv"
    fi

    has_torch="no"
    if has_torch_support "$v"; then
        if "$py" -c "import torch" 2>/dev/null; then
            has_torch="yes"
        else
            log "installing torch (cu124) into ${venv}"
            # torch 2.5.1+cu124 is the last release with cp39 wheels.
            # Newer torch works for 3.10..3.13 but pinning keeps the
            # matrix reproducible.
            if uv pip install --python "$py" --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 >/dev/null 2>&1; then
                has_torch="yes"
            else
                log "  torch install failed for ${v}, workload will use ctypes fallback"
            fi
        fi
    fi

    # uv resolves the venv's python to a symlink; record the resolved path
    # so the harness attaches uprobes to the right inode.
    resolved=$(readlink -f "$py")
    echo "${v}:${resolved}:${has_torch}" >> "$MANIFEST"
    log "  ${v} -> ${resolved} (torch=${has_torch})"
done

log "manifest written to ${MANIFEST}"
cat "$MANIFEST"
