r"""Execute the tutorial notebooks and report exactly where each one fails.

nbconvert's own error output is not machine-readable in a shell — the useful
line (which cell, what exception) gets buried in ANSI-coloured tracebacks, which
is why a first pass over these notebooks produced "exit=1 (see log)" seven
times over. This drives nbclient directly so the exception object is available,
and prints one line per notebook with the failing cell's source and the error.

Notebooks are copied to a scratch directory first: repos/ holds unmodified
upstream clones, and --inplace execution rewrites the .ipynb file.

    python playground/nb_check.py [dir ...]        # default: the tutorial dir
    KEEP_NB=0 python playground/nb_check.py        # delete copies afterwards
"""

import copy
import os
import pathlib
import shutil
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

DEFAULT_DIRS = ["repos/MuJoCo-Tutorial/tutorial"]
SCRATCH = pathlib.Path(os.environ.get("NB_SCRATCH", pathlib.Path.home() / "AppData" / "Local" / "Temp" / "opencode" / "nbcheck"))
TIMEOUT = int(os.environ.get("NB_TIMEOUT", "300"))


def run_one(nb_path, dest):
    """Execute one notebook from a scratch copy. Returns (status, detail)."""
    target = dest / nb_path.name
    shutil.copy2(nb_path, target)
    nb = nbformat.read(target, as_version=4)
    # Pin the kernel to *this* interpreter. Resolving "python3" by name picks up
    # whatever kernelspec is registered globally, which is frequently a different
    # (base) environment - and then a notebook fails for reasons that have
    # nothing to do with the notebook.
    kernel_spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "mujoco-learning check",
        "language": "python",
    }
    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name=os.environ.get("NB_KERNEL", "python3"),
        kernel_spec=kernel_spec,
        resources={"metadata": {"path": str(dest)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        # exc.ename/evalue are the real exception; the traceback text is noise.
        ename = getattr(exc, "ename", None) or "CellExecutionError"
        evalue = (getattr(exc, "evalue", "") or "").replace("\n", " ")[:110]
        idx = _failing_cell_index(nb)
        src = ""
        if idx is not None and idx < len(nb.cells):
            src = nb.cells[idx].source.strip().splitlines()[0][:60]
        nbformat.write(nb, target)
        return "FAIL", f"cell {idx} [{src}] -> {ename}: {evalue}"
    except Exception as exc:  # noqa: BLE001 - kernel startup, missing kernel, etc
        return "FAIL", f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}"
    nbformat.write(nb, target)
    return "OK", f"{len(nb.cells)} cells"


def _failing_cell_index(nb):
    """Index of the first cell whose outputs record an error."""
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                return i
    return None


def main():
    roots = sys.argv[1:] or DEFAULT_DIRS
    SCRATCH.mkdir(parents=True, exist_ok=True)

    results = []
    for root in roots:
        root_path = pathlib.Path(root)
        if not root_path.is_dir():
            print(f"  skipping (no such dir): {root}")
            continue
        for nb_path in sorted(root_path.glob("*.ipynb")):
            if ".ipynb_checkpoints" in nb_path.parts:
                continue
            name = nb_path.stem
            dest = SCRATCH / name
            dest.mkdir(parents=True, exist_ok=True)
            print(f"  running {name} ...", flush=True)
            status, detail = run_one(nb_path, dest)
            results.append((name, status, detail))
            print(f"  {name:32s} {status:5s} {detail}", flush=True)

    print()
    ok = [r for r in results if r[1] == "OK"]
    bad = [r for r in results if r[1] != "OK"]
    print(f"{len(ok)} execute cleanly, {len(bad)} fail")
    for name, _, detail in bad:
        print(f"  {name}: {detail}")
    if not os.environ.get("KEEP_NB", "1") == "1":
        shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
