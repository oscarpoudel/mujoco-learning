"""Confirm the MuJoCo 4.0.1 replacements for the APIs two examples still use.

Checks, rather than asserts, what the documented fixes should be:

  1. data.qM was removed. What carries the mass matrix now?
  2. mj_fullM's signature, and whether dense data.M needs it at all.
  3. impedance_control.py: how many actuators vs DOF does its model actually
     have? (It computes a 3-vector and writes it into ctrl, which has size 2.)

    python playground/_api_check.py
"""

import mujoco

print(f"mujoco {mujoco.__version__}\n")

XML = """
<mujoco>
  <worldbody>
    <body><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)

print("mass-matrix attributes present on MjData:")
for name in ("qM", "qLD", "M", "qM_full"):
    print(f"  data.{name:8s} {'yes' if hasattr(d, name) else 'no'}")

print("\nmj_fullM signature:", mujoco.mj_fullM.__doc__.splitlines()[0]
      if mujoco.mj_fullM.__doc__ else "(undocumented)")

# Dense M is already filled by mj_step/mj_forward? Verify by comparing against a
# freshly-computed fullM so the docs can say which one to read.
import numpy as np

mujoco.mj_forward(m, d)
dense = np.array(d.M).copy()
buf = np.zeros((m.nv, m.nv))       # dst must be 2-D (nv, nv), not flat
try:
    mujoco.mj_fullM(m, d, buf)      # 4.x order: model, data, dst
    same = np.allclose(buf, dense.reshape(m.nv, m.nv)) if dense.size == m.nv * m.nv else None
    print(f"mj_fullM(model, data, dst) -> ok; dst is 2-D {(m.nv, m.nv)}")
except Exception as exc:
    print("mj_fullM(model, data, dst) ->", type(exc).__name__, str(exc)[:80])
print(f"data.M is documented as the dense mass matrix; observed size "
      f"{dense.size} vs nv*nv = {m.nv * m.nv}")

print(f"\nfree sphere: nv={m.nv}, nu={m.nu}, M shape={dense.shape}")
print("symmetric:", bool(np.allclose(dense, dense.T)))

# --- impedance_control.py's model -------------------------------------------
import pathlib
import re

src = pathlib.Path("repos/MuJoCo-Tutorial/examples/impedance_control.py").read_text(encoding="utf-8", errors="replace")
block = re.search(r'<mujoco.*?</mujoco>', src, re.S)
if block:
    try:
        im = mujoco.MjModel.from_xml_string(block.group(0))
        print(f"\nimpedance_control model: nv={im.nv} dof, nu={im.nu} actuators")
        for i in range(im.nu):
            print(f"  actuator {i}: {mujoco.mj_id2name(im, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")
        print("  -> the example computes a 3-vector for ctrl of size", im.nu)
    except Exception as exc:
        print("\ninline XML parse failed:", str(exc)[:120])
