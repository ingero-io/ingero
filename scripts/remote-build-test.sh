#!/bin/bash
# Run on GPU VM: generate eBPF, build, test
set -e
export PATH=/usr/local/go/bin:$HOME/go/bin:$HOME/.local/bin:$PATH
cd ~/workspace/ingero

echo "=== Generate eBPF bindings ==="
make generate 2>&1

echo ""
echo "=== Build ==="
make build 2>&1

echo ""
echo "=== Test ==="
make test 2>&1

echo ""
echo "=== Binary version ==="
./bin/ingero version 2>/dev/null || echo "(version requires root for some checks)"

echo ""
echo "=== Check ==="
sudo ./bin/ingero check --debug 2>&1

echo ""
echo "=== BUILD COMPLETE ==="
