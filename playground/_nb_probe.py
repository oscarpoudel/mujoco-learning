"""Two notebook failures, checked rather than explained away.

06_xml_modeling  cell 8: "unrecognized element 'composite'"
    Is <composite> valid MJCF in 4.0.1 at all, or does this notebook use a form
    the parser rejects? A composite constraint needs Warp in some configurations.

07_mjx  cell 2: "cannot import name 'mjx' from 'mujoco'"
    Is mjx a separate optional install, or does it need jax present?

    python playground/_nb_probe.py
"""

import importlib
import subprocess
import sys

import mujoco

print(f"mujoco {mujoco.__version__}\n")

# --- 1. is <composite> accepted by the parser? ------------------------------
MINIMAL_COMPOSITE = """
<mujoco>
  <worldbody>
    <body><freejoint/><geom type="sphere" size=".1"/></body>
    <composite type="grid" count="3 3 1" spacing="0.05" offset="0 1 0">
      <joint kind="main" damping=".01"/>
      <geom type="box" size=".02 .02 .02" mass=".1"/>
    </composite>
  </worldbody>
</mujoco>
"""
try:
    mujoco.MjModel.from_xml_string(MINIMAL_COMPOSITE)
    print("<composite type=grid>  -> parses fine, so the notebook's own form is "
          "the problem")
except Exception as exc:
    print(f"<composite type=grid>  -> rejected: {str(exc).splitlines()[0][:100]}")

# --- 2. what does importing mjx actually need? ------------------------------
for name in ("jax", "mujoco.mjx"):
    try:
        mod = importlib.import_module(name)
        print(f"import {name:12s} -> ok {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"import {name:12s} -> {type(exc).__name__}: {str(exc)[:90]}")

# Does mjx ship as a module file at all?
import pathlib

pkg = pathlib.Path(mujoco.__file__).parent
print("\nmjx present in the wheel:", any(p.name.startswith("mjx") for p in pkg.iterdir()))
print("names starting with mjx:",
      [p.name for p in pkg.iterdir() if p.name.lower().startswith("mjx")][:6])

# What the wheel thinks it provides as extras.
try:
    import importlib.metadata as md

    meta = md.metadata("mujoco")
    print("\nRequires-Dist:")
    for line in md.requires("mujoco") or []:
        print("  " + line)
except Exception as exc:
    print("metadata read failed:", exc)
