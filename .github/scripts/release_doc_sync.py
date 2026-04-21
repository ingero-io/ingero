#!/usr/bin/env python3
"""release_doc_sync: rewrite version pins marked with ingero-version: anchors.

Given a target tag, product, and channel, walks the repo and rewrites the
version token on the line following each matching anchor.

Anchor format:
    Markdown HTML (outside code blocks, truly invisible):
        <!-- ingero-version:<id> product=<name> channel=<stable|dev> -->
    Markdown shell-comment (inside ```bash code blocks, valid bash comment):
        # ingero-version:<id> product=<name> channel=<stable|dev>
    YAML (same syntax as shell-comment):
        # ingero-version:<id> product=<name> channel=<stable|dev>

The first line within MAX_LOOKAHEAD lines after the anchor that contains
a version-shaped substring (v?X.Y.Z or v?X.Y.Z-suffix) is the target.
The first such substring on that line is replaced with the target version.
The leading 'v' prefix is preserved from the existing pin.

For channel=dev, target gets -dev appended (e.g. 0.10.0 -> 0.10.0-dev).
For channel=stable, any -suffix is stripped.

Only anchors whose product= matches --product AND whose channel= matches
--channel are rewritten. Other anchors are left untouched.

Usage:
    python release_doc_sync.py --tag vX.Y.Z --product ingero --channel stable [--dry-run]
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
VERSION_TOKEN = re.compile(
    r"(?P<vprefix>v?)(?P<ver>\d+\.\d+\.\d+(?:-[\w.]+)?)"
)

SCAN_EXTS = {".md", ".yaml", ".yml"}
SKIP_DIRS = {".git", "node_modules", "target", "dist", "build", "bin", "vendor"}
# Historical / changelog files are excluded from rewrites.
EXCLUDE_PATHS = {"CHANGELOG.md"}
# Max lines to scan after an anchor for a version token. Handles the common
# case of anchor-above-multi-line-shell-command where the target version is a
# few lines into the command (docker build --build-arg VERSION=...).
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
    """Return the anchor patterns to try for a given file.

    Markdown files accept both HTML-comment and `#`-comment anchors; the
    latter appears inside fenced code blocks where HTML comments render as
    visible text. YAML files only use `#` comments.
    """
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


def target_version(channel: str, tag: str) -> str:
    """Normalize tag to the version string written into files."""
    ver = tag.lstrip("v")
    base = ver.split("-", 1)[0]
    return f"{base}-dev" if channel == "dev" else base


def rewrite_file(
    path: Path,
    anchor_res: list[re.Pattern[str]],
    product: str,
    channel: str,
    target_ver: str,
    dry_run: bool,
) -> list[tuple[str, int, str, str]]:
    text = path.read_text(encoding="utf-8")
    # splitlines(True) preserves line endings so a round-trip write is faithful.
    lines = text.splitlines(keepends=True)
    changes: list[tuple[str, int, str, str]] = []

    i = 0
    while i < len(lines):
        m = find_anchor(lines[i], anchor_res)
        if not m:
            i += 1
            continue
        attrs = parse_attrs(m.group("attrs"))
        if attrs is None:
            i += 1
            continue
        if attrs.get("product") != product or attrs.get("channel") != channel:
            i += 1
            continue
        anchor_id = m.group("id")
        # Find the target line: first line within MAX_LOOKAHEAD that contains
        # a version token. Skips blanks, code-fence delimiters, and non-version
        # content (e.g. `docker build ...` above the `--build-arg VERSION=...`).
        j = i + 1
        end = min(i + 1 + MAX_LOOKAHEAD, len(lines))
        vm = None
        while j < end:
            vm = VERSION_TOKEN.search(lines[j])
            if vm:
                break
            j += 1
        if vm is None:
            i += 1
            continue
        old_token = vm.group(0)
        preserve_v = bool(vm.group("vprefix"))
        new_token = ("v" if preserve_v else "") + target_ver
        if old_token != new_token:
            lines[j] = lines[j][: vm.start()] + new_token + lines[j][vm.end():]
            changes.append((anchor_id, j + 1, old_token, new_token))
        i = j + 1

    if changes and not dry_run:
        path.write_text("".join(lines), encoding="utf-8")
    return changes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Target tag, e.g. v0.9.2")
    ap.add_argument("--product", required=True, choices=sorted(VALID_PRODUCTS))
    ap.add_argument("--channel", required=True, choices=sorted(VALID_CHANNELS))
    ap.add_argument("--root", default=".", help="Repo root (default: cwd)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target_ver = target_version(args.channel, args.tag)
    root = Path(args.root).resolve()

    all_changes: list[tuple[Path, str, int, str, str]] = []
    for f in iter_files(root):
        changes = rewrite_file(
            f, pick_anchor_res(f), args.product, args.channel, target_ver, args.dry_run
        )
        for c in changes:
            all_changes.append((f, *c))

    header = (
        f"product={args.product} channel={args.channel} "
        f"tag={args.tag} target={target_ver} dry_run={args.dry_run}"
    )
    if not all_changes:
        print(f"No updates required ({header}).")
        return 0

    verb = "Would rewrite" if args.dry_run else "Rewrote"
    print(f"{verb} {len(all_changes)} pin(s) ({header}):")
    for fp, aid, ln, old, new in all_changes:
        rel = fp.relative_to(root).as_posix()
        print(f"  {rel}:{ln}  [{aid}]  {old} -> {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
