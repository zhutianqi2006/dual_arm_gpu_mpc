#!/usr/bin/env python3
"""Bulk-update std/an_std in examples YAML files.

Edits all *.yaml and *.yml under examples/.
Replaces ONLY top-level key lines matching:
  ^\s*std:\s*...
  ^\s*an_std:\s*...

Does NOT touch log_std.

Usage:
  python3 scripts/update_examples_yaml_std.py --write
  python3 scripts/update_examples_yaml_std.py  # dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

STD_VALUE = "1.732050808"
AN_STD_VALUE = "0.774596669"

RE_STD = re.compile(r"^(?P<indent>\s*)std\s*:\s*.*$", re.MULTILINE)
RE_AN_STD = re.compile(r"^(?P<indent>\s*)an_std\s*:\s*.*$", re.MULTILINE)


def update_text(text: str) -> tuple[str, int, int]:
    n_std = 0
    n_an = 0

    def repl_std(m: re.Match[str]) -> str:
        nonlocal n_std
        n_std += 1
        return f"{m.group('indent')}std: {STD_VALUE}"

    def repl_an(m: re.Match[str]) -> str:
        nonlocal n_an
        n_an += 1
        return f"{m.group('indent')}an_std: {AN_STD_VALUE}"

    text = RE_STD.sub(repl_std, text)
    text = RE_AN_STD.sub(repl_an, text)
    return text, n_std, n_an


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="examples", help="Root folder to scan (default: examples)")
    ap.add_argument("--write", action="store_true", help="Write changes (default: dry-run)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    files = sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml")))

    changed = []
    total_std = 0
    total_an = 0

    for path in files:
        try:
            old = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            old = path.read_text(encoding="utf-8", errors="ignore")

        new, n_std, n_an = update_text(old)
        total_std += n_std
        total_an += n_an

        if new != old:
            changed.append((path, n_std, n_an))
            if args.write:
                path.write_text(new, encoding="utf-8")

    mode = "WROTE" if args.write else "DRY-RUN"
    print(f"{mode}: {len(changed)}/{len(files)} files changed")
    print(f"Replaced std lines: {total_std}")
    print(f"Replaced an_std lines: {total_an}")
    for p, n_std, n_an in changed:
        print(f"- {p} (std:{n_std}, an_std:{n_an})")


if __name__ == "__main__":
    main()
