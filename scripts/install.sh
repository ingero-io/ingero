#!/bin/bash
# Ingero install script — downloads latest release binary.
# Usage: curl -fsSL https://get.ingero.io | sh
set -euo pipefail

REPO="ingero-io/ingero"
INSTALL_DIR="/usr/local/bin"

# Detect architecture.
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
    *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
if [ "$OS" != "linux" ]; then
    echo "Ingero only supports Linux (got: $OS)"
    exit 1
fi

echo "Ingero installer"
echo "  OS: $OS"
echo "  Arch: $ARCH"
echo ""

# Get latest release tag.
LATEST=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/' || echo "")
if [ -z "$LATEST" ]; then
    echo "Could not determine latest release. Downloading from main..."
    LATEST="latest"
fi
echo "  Version: $LATEST"

# Download binary.
URL="https://github.com/$REPO/releases/download/$LATEST/ingero-${OS}-${ARCH}"
echo "  Downloading: $URL"
TMPFILE=$(mktemp)
if ! curl -fsSL -o "$TMPFILE" "$URL" 2>/dev/null; then
    echo "Download failed. Check https://github.com/$REPO/releases"
    rm -f "$TMPFILE"
    exit 1
fi

chmod +x "$TMPFILE"

# Install.
if [ -w "$INSTALL_DIR" ]; then
    mv "$TMPFILE" "$INSTALL_DIR/ingero"
else
    echo "  Installing to $INSTALL_DIR (requires sudo)..."
    sudo mv "$TMPFILE" "$INSTALL_DIR/ingero"
fi

echo ""
echo "Ingero installed successfully!"
echo "  Run: sudo ingero check"
echo "  Demo: ingero demo --no-gpu"
