#!/usr/bin/env bash
# Test 1: Install + check from a fresh release tarball.
#
# Asserts:
#   - The v0.14.0 release tarball downloads + extracts cleanly.
#   - `ingero check` exits 0 against the freshly-extracted binary.
#   - "All checks passed" is present in stdout.
#   - No WARN or FAIL line in the check output.
#
# Hardware: any Linux amd64 host with a GPU (A10 minimum). NVIDIA driver
# present.
#
# Invoke:
#   bash tests/e2e/install-from-release.sh
#
# Optional env:
#   INGERO_VERSION   release tag to test (default v0.14.0)
#   INGERO_ARCH      amd64 | arm64 (default = uname -m mapped)
#
# Expected runtime: ~30s (mostly download + check).
set -euo pipefail

VERSION="${INGERO_VERSION:-v0.14.0}"
case "$(uname -m)" in
  x86_64) ARCH_DEFAULT="amd64" ;;
  aarch64|arm64) ARCH_DEFAULT="arm64" ;;
  *) ARCH_DEFAULT="$(uname -m)" ;;
esac
ARCH="${INGERO_ARCH:-$ARCH_DEFAULT}"

WORK=$(mktemp -d)
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "=== Test 1: install-from-release ($VERSION linux $ARCH) ==="
echo "==> [1/3] Download release tarball"
TARBALL="ingero_${VERSION#v}_linux_${ARCH}.tar.gz"
URL="https://github.com/ingero-io/ingero/releases/download/${VERSION}/${TARBALL}"
if ! curl -fsSL -o "$WORK/$TARBALL" "$URL"; then
  echo "FAIL: could not download $URL"
  exit 1
fi
echo "OK: tarball downloaded ($(stat -c%s "$WORK/$TARBALL") bytes)"

echo "==> [2/3] Extract"
tar -xzf "$WORK/$TARBALL" -C "$WORK"
BIN="$WORK/ingero"
if [[ ! -x "$BIN" ]]; then
  # Some release shapes nest the binary in a versioned subdir.
  BIN=$(find "$WORK" -type f -name ingero -perm -u+x | head -1 || true)
fi
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "FAIL: ingero binary not found after extract"
  ls -la "$WORK"
  exit 1
fi
echo "OK: binary extracted at $BIN"

echo "==> [3/3] Run ingero check"
LOG="$WORK/check.log"
if ! sudo "$BIN" check >"$LOG" 2>&1; then
  echo "FAIL: ingero check exited non-zero"
  cat "$LOG"
  exit 1
fi

if ! grep -q "All checks passed" "$LOG"; then
  echo "FAIL: 'All checks passed' line missing"
  cat "$LOG"
  exit 1
fi

if grep -E '\b(WARN|FAIL)\b' "$LOG" | grep -vi "0 WARN\|0 FAIL" >/dev/null; then
  echo "FAIL: WARN or FAIL line present in check output"
  grep -E '\b(WARN|FAIL)\b' "$LOG"
  exit 1
fi

echo "OK: all checks passed cleanly"
echo "PASS: install-from-release"
