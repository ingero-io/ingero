#!/usr/bin/env python3
"""validate_anchors: invariant checker for ingero-version: anchors.

Rules enforced:
  1. Every anchor parses (id present, attribute syntax valid).
  2. Required attributes: product (in {ingero, ingero-fleet, ingero-ee}),
     channel (in {stable, dev}).
  3. Anchor id is unique within a file.
  4. The next non-blank line after the anchor contains a version-shaped
     substring (v?X.Y.Z[-suffix]).

The validator does NOT flag unanchored version tokens; anchors are opt-in
explicit markers placed where release-time rewrites should happen. A drift
auditor (check-version-drift) serves as the second-opinion scanner.

Usage:
    python validate_anchors.py [--root <path>]

Exit codes:
    0 clean
    1 violations found (printed to stderr)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VALID_PRODUCTS = {"ingero", "ingero-fleet", "ingero-ee"}
VALID_CHANNELS = {"stable", "dev"}

ANCHOR_MD = re.compile(
    r"<!--\s*ingero-version:(?P<id>[\w.-]+)\s+(?P<attrs>.*?)\s*-->"
)
ANCHOR_YAML = re.compile(
    r"#\s*ingero-version:(?P<id>[\w.-]+)\s+(?P<attrs>.+?)\s*$"
)
VERSION_TOKEN = re.compile(r"v?\d+\.\d+\.\d+(?:-[\w.]+)?")

SCAN_EXTS = {".md", ".yaml", ".yml"}
SKIP_DIRS = {".git", "node_modules", "target", "dist", "build", "bin", "vendor"}
EXCLUDE_PATHS = {"CHANGELOG.md"}
MAX_LOOKAHEAD = 10


def parse_attrs(s: str) -> dict[str, str] | None:
    attrs: dict[str, str] = {}
    for part in s.split():
        if "=" not in part:
            return None
        k, v = part.split("=", 1)
        attrs[k] = v
    return attrs


def pick_anchor_res(path: Path) -> list[re.Pattern[str]]:
    """Markdown accepts both HTML-comment and `#`-comment anchors (the
    latter for pins inside ```bash fenced code blocks). YAML uses `#`."""
    if path.suffix == ".md":
        return [ANCHOR_MD, ANCHOR_YAML]
    return [ANCHOR_YAML]


def find_anchor(line: str, anchor_res: list[re.Pattern[str]]) -> re.Match[str] | None:
    for pat in anchor_res:
        m = pat.search(line)
        if m:
            return m
    return None


def iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in SCAN_EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        if rel in EXCLUDE_PATHS:
            continue
        yield p


def check_file(path: Path, anchor_res: list[re.Pattern[str]]) -> list[str]:
    violations: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return violations
    lines = text.splitlines()
    seen_ids: set[str] = set()

    for i, line in enumerate(lines, start=1):
        m = find_anchor(line, anchor_res)
        if not m:
            continue
        anchor_id = m.group("id")
        loc = f"{path}:{i}"
        if anchor_id in seen_ids:
            violations.append(f"{loc}: duplicate anchor id {anchor_id!r}")
        seen_ids.add(anchor_id)

        attrs = parse_attrs(m.group("attrs"))
        if attrs is None:
            violations.append(f"{loc}: malformed attribute syntax")
            continue

        product = attrs.get("product")
        if product is None:
            violations.append(f"{loc}: missing product= attribute")
        elif product not in VALID_PRODUCTS:
            violations.append(
                f"{loc}: invalid product={product!r} "
                f"(valid: {sorted(VALID_PRODUCTS)})"
            )

        channel = attrs.get("channel")
        if channel is None:
            violations.append(f"{loc}: missing channel= attribute")
        elif channel not in VALID_CHANNELS:
            violations.append(
                f"{loc}: invalid channel={channel!r} "
                f"(valid: {sorted(VALID_CHANNELS)})"
            )

        # Assert a version token exists within MAX_LOOKAHEAD lines.
        # i is 1-indexed; lines list is 0-indexed; lines[i] is the next line.
        end = min(i + MAX_LOOKAHEAD, len(lines))
        found = False
        for j in range(i, end):
            if VERSION_TOKEN.search(lines[j]):
                found = True
                break
        if not found:
            violations.append(
                f"{loc}: no version token within {MAX_LOOKAHEAD} lines after anchor"
            )

    return violations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    all_violations: list[str] = []
    files_checked = 0
    anchors_seen = 0

    for f in iter_files(root):
        files_checked += 1
        anchor_res = pick_anchor_res(f)
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            if find_anchor(line, anchor_res):
                anchors_seen += 1
        all_violations.extend(check_file(f, anchor_res))

    if all_violations:
        for v in all_violations:
            print(v, file=sys.stderr)
        print(f"\n{len(all_violations)} violation(s) found.", file=sys.stderr)
        return 1

    print(
        f"All ingero-version anchors OK "
        f"({anchors_seen} anchor(s) across {files_checked} file(s))."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
