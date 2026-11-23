"""Environment doctor for the MuJoCo Learning Hub container.

    docker compose run --rm sim python docker/doctor.py
    python playground/../docker/doctor.py        # works on the Windows side too

Checks only what actually matters for the tutorials, and says why something
failed instead of dumping a traceback. Exit code is non-zero if a core check
failed; optional libraries are reported but never fail the run.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ("mujoco", "numpy")
OPTIONAL = (
    "gymnasium",
    "scipy",
    "matplotlib",
    "mediapy",
    "ipywidgets",
    "jupyterlab",
    "torch",
    "stable_baselines3",
    "ikpy",
    "jax",
    "cv2",
    "networkx",
    "shapely",
    "osqp",
)

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark = "ok  " if ok else "FAIL"
    print(f"[{mark}] {name:32s} {detail}")


def check_python() -> None:
    v = sys.version_info
    supported = (3, 11) <= (v.major, v.minor) <= (3, 13)
    record(
        "python",
        supported,
        f"{platform.python_version()} on {platform.system()}-{platform.machine()}"
        + ("" if supported else " -- tutorials target 3.11-3.13"),
    )


def check_mujoco():
    try:
        import mujoco

        record("mujoco import", True, mujoco.__version__)
        return mujoco
    except Exception as exc:
        record("mujoco import", False, f"{type(exc).__name__}: {exc}")
        return None


def check_physics(mujoco) -> None:
    xml = """
<mujoco>
  <worldbody>
    <body><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
  </worldbody>
</mujoco>
"""
    try:
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        for _ in range(200):
            mujoco.mj_step(m, d)
        z = float(d.qpos[2])
        record("physics (mj_step)", z < 0, f"sphere fell to z={z:.4f}")
    except Exception as exc:
        record("physics (mj_step)", False, f"{type(exc).__name__}: {exc}")


def check_rendering(mujoco) -> None:
    gl = os.environ.get("MUJOCO_GL", "<unset>")
    xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <body><geom type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>
"""
    try:
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, height=64, width=64)
        r.update_scene(d)
        frame = r.render()
        r.close()
        record("offscreen render", frame.shape == (64, 64, 3),
               f"MUJOCO_GL={gl} -> {frame.shape}")
    except Exception as exc:
        record("offscreen render", False,
               f"MUJOCO_GL={gl}: {type(exc).__name__}: {exc}")


def check_writable() -> None:
    probe = ROOT / ".doctor-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        ok = probe.read_text(encoding="utf-8") == "ok"
        probe.unlink()
        record("workspace writable", ok, str(ROOT))
    except Exception as exc:
        record("workspace writable", False, f"{type(exc).__name__}: {exc}")


def check_repos() -> None:
    repos_dir = ROOT / "repos"
    if not repos_dir.is_dir():
        record("repos/ present", False, str(repos_dir))
        return
    for repo in sorted(p for p in repos_dir.iterdir() if p.is_dir()):
        try:
            sha = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=20,
            )
            if sha.returncode == 0:
                record(f"repo {repo.name}", True, sha.stdout.strip())
            else:
                record(f"repo {repo.name}", False,
                       sha.stderr.strip().splitlines()[0] if sha.stderr else "git failed")
        except Exception as exc:
            record(f"repo {repo.name}", False, f"{type(exc).__name__}: {exc}")


def check_optional() -> None:
    # MJX is a submodule of mujoco, so it needs importlib rather than __import__.
    try:
        import importlib

        importlib.import_module("mujoco.mjx")
        record("  mujoco.mjx", True, "available")
    except Exception:
        record("  mujoco.mjx", True, "not installed (optional; in mjx-shell only)")

    for name in OPTIONAL:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "?")
            extra = ""
            if name == "torch":
                try:
                    extra = " cuda=%s" % mod.cuda.is_available()
                except Exception:
                    extra = " cuda=?"
            if name == "jax":
                try:
                    extra = " devices=%s" % [d.device_kind for d in mod.devices()]
                except Exception:
                    extra = " devices=?"
            record(f"  {name}", True, str(ver) + extra)
        except ImportError:
            record(f"  {name}", True, "not installed (optional)")


def check_gpu() -> None:
    """Report the GPU path honestly. Meaningful only inside Linux containers."""
    if platform.system() == "Windows":
        record("gpu", True, "not checked on Windows (hardware GL works natively)")
        return

    import ctypes

    try:
        lib = ctypes.CDLL("libcuda.so.1")
        count = ctypes.c_int(0)
        lib.cuInit(0)
        lib.cuDeviceGetCount(ctypes.byref(count))
        if count.value > 0:
            record("gpu", True, f"libcuda visible, {count.value} CUDA device(s)")
            return
        detail = "libcuda.so.1 loads but reports 0 devices"
    except OSError:
        # This is the real test. /usr/local/nvidia existing proves nothing --
        # the nvidia runtime can mount scaffolding while exposing no adapter.
        detail = "libcuda.so.1 not found -> CPU only"

    if os.environ.get("MUJOCO_GL", "") == "egl":
        detail += "; MUJOCO_GL=egl is succeeding via Mesa llvmpipe = SOFTWARE, not the GPU"
    record("gpu", False, detail + ". Cause: host driver 452.56 predates CUDA-on-WSL "
                                  "(needs R470+). See notes/0002-gpu-enablement-checklist.md")


def main() -> int:
    print(f"MuJoCo Learning Hub -- environment check\nroot: {ROOT}\n")
    check_python()
    mujoco = check_mujoco()
    if mujoco is not None:
        check_physics(mujoco)
        check_rendering(mujoco)
    check_writable()
    check_repos()
    check_optional()
    check_gpu()

    hard = [r for r in results if not r[1] and not r[0].startswith("  ")]
    # GPU absence is expected on this host, so it is informational only.
    hard = [r for r in hard if r[0] != "gpu"]
    print(
        f"\n{len(results)} checks, {len(hard)} failed"
        + ("  (gpu absent is expected in-container here)"
           if any(r[0] == 'gpu' and not r[1] for r in results) else "")
    )
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
