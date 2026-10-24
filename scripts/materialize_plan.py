"""Write every ```-fenced block whose first line is `# file: <path>` from a plan to disk.

    python scripts/materialize_plan.py docs/superpowers/plans/<plan>.md [--root .] [--dry-run]

One block per file (a second block for the same path overwrites, with a warning). Paths are
relative to --root. The marker line is stripped; an otherwise empty block creates an empty file.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

FENCE = re.compile(r"^```[A-Za-z0-9_+-]*[ \t]*$")
MARK = re.compile(r"^#\s*file:\s*(?P<path>\S.*?)\s*$")


def extract(text: str) -> dict:
    blocks, cur, path = {}, None, None
    for line in text.splitlines():
        if cur is None:
            if FENCE.match(line):
                cur, path = [], None
            continue
        if line.strip() == "```":
            if path is not None:
                if path in blocks:
                    print(f"[warn] {path}: second block overwrites the first")
                blocks[path] = "\n".join(cur).rstrip("\n") + ("\n" if cur else "")
            cur, path = None, None
            continue
        if not cur and path is None:
            m = MARK.match(line)
            if m:
                path = m.group("path")
                continue
        cur.append(line)
    return blocks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    blocks = extract(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(args.root)
    for rel, content in blocks.items():
        target = root / rel
        print(f"{'would write' if args.dry_run else 'write'} {rel} ({len(content.splitlines())} lines)")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
    print(f"{len(blocks)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
