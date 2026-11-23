"""Verify every file path referenced in this project's markdown actually exists.

PowerShell proved unfit for this: `Get-ChildItem -Filter` matches Windows 8.3
short names (false positives), and a naive `` ` ``-span regex pairs straight
through ``` fences because three backticks is an odd count, which silently
misaligns every inline span after the first code block. This does it properly.

    python check_md_paths.py [root]
"""
import os
import re
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
EXTS = (".py", ".ipynb", ".xml", ".md", ".txt", ".ps1", ".yml", ".png", ".mp4")

# Index of every real file: its basename, and every path-suffix of its real path.
basenames, suffixes = set(), set()
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".ipynb_checkpoints"}]
    for fn in filenames:
        full = os.path.relpath(os.path.join(dirpath, fn), ROOT).replace("\\", "/")
        basenames.add(fn)
        parts = full.split("/")
        for i in range(len(parts)):
            suffixes.add("/".join(parts[i:]))

FENCE = re.compile(r"^(```|~~~)", re.M)


def inline_spans(text):
    """Yield inline code spans, ignoring fenced blocks entirely."""
    out, in_fence, buf = [], False, []
    for line in text.splitlines(keepends=True):
        if FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        buf.append(line)
    for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", "".join(buf)):
        out.append(m.group(1).strip())
    return out


problems, checked = [], 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
    for fn in sorted(filenames):
        if not fn.endswith(".md"):
            continue
        doc = os.path.relpath(os.path.join(dirpath, fn), ROOT).replace("\\", "/")
        if doc.startswith("repos/"):
            continue
        with open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace") as fh:
            for tok in inline_spans(fh.read()):
                tok = tok.strip(".,;:!?()")
                if not tok.endswith(EXTS):
                    continue
                if any(c in tok for c in "*{,}"):        # brace/glob shorthand
                    continue
                if tok.startswith(("http://", "https://")):
                    continue
                if " " in tok:                       # CLI snippet, not a path
                    continue
                # Files that live in the installed mujoco package or the conda
                # environment, not in this repository.
                if os.path.basename(tok) in {"native_viewer.py", "usercustomize.py",
                                             "sitecustomize.py"}:
                    continue
                checked += 1
                norm = tok.replace("\\", "/").lstrip("./")
                ok = norm in suffixes or os.path.basename(norm) in basenames
                if not ok:
                    problems.append((doc, tok))

print(f"checked {checked} path references")
if problems:
    print("MISSING:")
    for doc, tok in problems:
        print(f"  {doc}: {tok}")
    sys.exit(1)
print("OK: every referenced path resolves")
