"""Report which learning stages the current Python environment can run.

Run it against whichever interpreter you are about to use:

    python scripts/env_check.py

Exit code is 0 when the Stage 1 core is present, so it can be used as a gate.
The point of this file is that the upstream repositories do not say what a
given example needs, and their requirements files pin MuJoCo versions that
would downgrade a working install. Import names are checked rather than
package metadata because the two differ (opencv-python -> cv2).
"""

import importlib
import sys

# name -> (import name, minimum stage that needs it)
PACKAGES = [
    ("mujoco", "mujoco", 1),
    ("numpy", "numpy", 1),
    ("gymnasium", "gymnasium", 2),
    ("scipy", "scipy", 2),
    ("matplotlib", "matplotlib", 2),
    ("mediapy", "mediapy", 2),
    ("jupyterlab", "jupyterlab", 1),
    ("ipywidgets", "ipywidgets", 1),
    ("imageio-ffmpeg", "imageio_ffmpeg", 2),
    ("ffmpeg (CLI)", "shutil:ffmpeg", 2),
    ("torch", "torch", 4),
    ("stable-baselines3", "stable_baselines3", 4),
    ("pandas", "pandas", 4),
    ("seaborn", "seaborn", 4),
    ("transforms3d", "transforms3d", 4),
    ("ikpy", "ikpy", 4),
    ("opencv-python", "cv2", 5),
    ("networkx", "networkx", 5),
    ("shapely", "shapely", 5),
    ("osqp", "osqp", 3),
    ("jax", "jax", 6),
]

STAGE_NAMES = {
    1: "Stage 1  core: MJCF, MjModel/MjData, mj_step",
    2: "Stage 2  control + rendering (PD, LQR, IK, video, Gymnasium)",
    3: "Stage 3  manipulation and MPC",
    4: "Stage 4  reinforcement learning",
    5: "Stage 5  mobile robots and perception",
    6: "Stage 6  MJX / large-scale parallel sim",
}


def resolve(spec):
    """Return (ok, detail) for an import spec, or a shutil lookup for CLIs."""
    if spec.startswith("shutil:"):
        import shutil

        found = shutil.which(spec.split(":", 1)[1])
        return (bool(found), found or "not on PATH")
    try:
        mod = importlib.import_module(spec)
        return (True, getattr(mod, "__version__", ""))
    except Exception as exc:  # noqa: BLE001 - any import failure is the answer
        return (False, type(exc).__name__)


def main():
    print(f"python {sys.version.split()[0]}  {sys.executable}\n")
    missing_by_stage = {}
    for pkg, spec, stage in PACKAGES:
        ok, detail = resolve(spec)
        mark = "  ok " if ok else "MISSING"
        print(f"  {mark}  {pkg:20s} {detail}")
        if not ok:
            missing_by_stage.setdefault(stage, []).append(pkg)

    print()
    for stage in sorted(STAGE_NAMES):
        need = missing_by_stage.get(stage, [])
        status = "ready" if not need else f"blocked by: {', '.join(need)}"
        print(f"  {STAGE_NAMES[stage]}\n      -> {status}")

    core_ok = not missing_by_stage.get(1, [])
    print(f"\n{'CORE OK' if core_ok else 'CORE MISSING - install mujoco and numpy'}")
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
