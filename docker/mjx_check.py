"""MJX / GPU readiness check.

Run after working through notes/0002-gpu-enablement-checklist.md:

    docker compose -f docker-compose.yml -f docker-compose.gpu.yml \\
      run --rm mjx-shell python docker/mjx_check.py

Every check is device-aware: a step that succeeds on CPU does NOT count as a
pass here. Silent fallback to CPU is precisely what this script exists to catch,
so the only way to get "GPU PATH READY" is for compute to actually land on the
GPU.
"""

from __future__ import annotations

import os
import sys

ok = True


def report(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok = ok and passed
    print(f"[{'ok  ' if passed else 'FAIL'}] {label:24s} {detail}")


def is_gpu(device) -> bool:
    """True if this jax device is a GPU.

    Do NOT test device_kind for 'gpu'/'cuda'/'nvidia': device_kind is the
    *product* name, and a genuinely working P1000 reports 'Quadro P1000', which
    matches none of those. This made the gate fail a healthy GPU. `platform` is
    the reliable discriminator; the class name is the fallback for older jax.
    """
    platform = str(getattr(device, "platform", "") or "").lower()
    if platform:
        return platform in ("cuda", "gpu")
    name = type(device).__name__.lower()
    return "cuda" in name or "gpu" in name


def all_on_gpu(devices) -> bool:
    return bool(devices) and all(is_gpu(d) for d in devices)


def device_label(device) -> str:
    return str(getattr(device, "device_kind", "") or type(device).__name__)


def main() -> int:
    print(f"MUJOCO_GL = {os.environ.get('MUJOCO_GL', '<unset>')}\n")

    # --- 1. Does the container even have a CUDA driver to talk to? -----------
    try:
        import ctypes

        lib = ctypes.CDLL("libcuda.so.1")
        count = ctypes.c_int(0)
        lib.cuInit(0)
        lib.cuDeviceGetCount(ctypes.byref(count))
        report("libcuda.so.1", count.value > 0,
               f"{count.value} CUDA device(s) visible")
    except Exception as exc:
        report("libcuda.so.1", False,
               f"{type(exc).__name__}: {exc}".split("\n")[0])
        print("       -> driver not exposing a GPU to this container. Go back to")
        print("         Gate 3 in the checklist; nothing below can pass without it.")

    # --- 2. What does JAX actually see? -------------------------------------
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        report("jax import", False, str(exc))
        print("\nGPU PATH NOT READY")
        return 1

    devices = jax.devices()
    kinds = [device_label(d) for d in devices]
    gpu_devices = [d for d in devices if is_gpu(d)]
    report("jax.devices()", bool(gpu_devices), f"{devices} kinds={kinds}")
    if not gpu_devices:
        print("       ^ JAX is on CPU. Needs the cuda12 extra (never cuda13 --")
        print("         that one requires compute capability 7.5, this GPU is 6.1).")

    if not gpu_devices:
        print("\nGPU PATH NOT READY")
        return 1

    # --- 3. Real compute, placed on the GPU ---------------------------------
    try:
        target = gpu_devices[0]
        with jax.default_device(target):
            x = jnp.ones((2048, 2048))
            y = float((x @ x).sum())
        placed = {device_label(d) for d in x.devices()}
        report("gpu matmul", y > 0 and all_on_gpu(x.devices()),
               f"sum={y:.0f} on {sorted(placed)}")
    except Exception as exc:
        report("gpu matmul", False, f"{type(exc).__name__}: {exc}")

    # --- 4. MJX stepping on the GPU -----------------------------------------
    # Deliberately avoids mjx.get_data()/get_data_into(): that API has churned
    # between releases. Reading qpos off the MJX pytree is version-stable.
    try:
        import mujoco
        import mujoco.mjx as mjx

        xml = """
<mujoco>
  <worldbody>
    <body><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
  </worldbody>
</mujoco>
"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        with jax.default_device(target):
            mjx_model = mjx.put_model(m)
            mjx_data = mjx.put_data(m, d)
            for _ in range(200):
                mjx_data = mjx.step(mjx_model, mjx_data)
            qpos = jnp.asarray(mjx_data.qpos)
        z = float(qpos[2])
        placed = {device_label(dd) for dd in qpos.devices()}
        report("mjx.step on device", z < 0 and all_on_gpu(qpos.devices()),
               f"free-fall z={z:.4f} on {sorted(placed)}")
    except Exception as exc:
        report("mjx.step on device", False,
               f"{type(exc).__name__}: {str(exc).splitlines()[0]}")

    print("\n" + ("GPU PATH READY" if ok else "GPU PATH NOT READY"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
