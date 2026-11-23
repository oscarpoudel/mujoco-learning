"""Bisect the MJX GPU failure: what makes mjx.step loadable?

Symptom: `mjx.step` on the P1000 dies with
`Failed to load in-memory CUBIN ... CUDA_ERROR_NOT_SUPPORTED`.

Measured so far, one variant per process (a poisoned CUDA context invalidates
later variants in the same process):

    A mjx.step standalone, 200 iters ....... FAIL
    B same, inside jax.default_device ...... FAIL
    C 2048x2048 matmul first, then step .... FAIL (the matmul itself succeeds)
    D B + C combined ....................... FAIL
    E mjx.forward first, then step ......... (running)
    F mjx.kinematics first, then step ...... (running)
    G mjx.forward first, then one step ..... (running)

So the shape of the caller does not matter: `default_device` and a preceding
matmul are both innocent. What does matter is which MJX functions get compiled
before `step` -- _mjx_probe.py, which calls kinematics and forward first, passes
every stage including step.

    for v in A B C D E F G:
      docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
        run --rm -e VARIANT=$v mjx-shell python playground/_bisect.py
"""
import os

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

XML = """
<mujoco>
  <worldbody>
    <body><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
  </worldbody>
</mujoco>
"""
VARIANT = os.environ.get("VARIANT", "A")


def steps(n, warm=None):
    """Warm=None compiles mjx.step standalone. warm='forward'/'kinematics'
    compiles that sub-function first -- MJX shares compiled kernels between step
    and its parts, so this changes what has to be loaded later."""
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mm = mjx.put_model(m)
    md = mjx.put_data(m, d)
    if warm == "forward":
        mjx.forward(mm, md)
    elif warm == "kinematics":
        mjx.kinematics(mm, md)
    for _ in range(n):
        md = mjx.step(mm, md)
    return float(jnp.asarray(md.qpos)[2])


def matmul():
    x = jnp.ones((2048, 2048))
    return float((x @ x).sum())


try:
    if VARIANT == "A":                      # mjx only, no context manager
        print(f"A mjx-only: z={steps(200):.4f}")
    elif VARIANT == "B":                    # mjx inside default_device context
        with jax.default_device(jax.devices()[0]):
            print(f"B mjx-in-default_device: z={steps(200):.4f}")
    elif VARIANT == "C":                    # matmul first, then mjx, no context
        print(f"C matmul={matmul():.0f}")
        print(f"C mjx-after-matmul: z={steps(200):.4f}")
    elif VARIANT == "D":                    # exactly mjx_check.py's shape
        with jax.default_device(jax.devices()[0]):
            print(f"D matmul={matmul():.0f}")
            print(f"D mjx: z={steps(200):.4f}")
    elif VARIANT == "E":                    # forward compiled first, then step
        print(f"E forward-warmup then 200 steps: z={steps(200, warm='forward'):.4f}")
    elif VARIANT == "F":                    # kinematics compiled first, then step
        print(f"F kinematics-warmup then 200 steps: z={steps(200, warm='kinematics'):.4f}")
    elif VARIANT == "G":                    # warm-up + a single step: is count involved?
        print(f"G forward-warmup then 1 step: z={steps(1, warm='forward'):.4f}")
    print(f"VARIANT {VARIANT}: PASS")
except Exception as exc:
    print(f"VARIANT {VARIANT}: FAIL -> {type(exc).__name__}: {str(exc).splitlines()[0][:130]}")
