"""Does mjx.step mutate in place, and did my bisect harness break itself?

Suspect: `md = mjx.step(mm, md)`. If mjx.step mutates data in place and returns
None, then the second loop iteration calls mjx.step(mm, None) -- JAX treats None
as an empty pytree and compiles a *different* function, which is what fails to
load. The bisect harness would then have been testing a self-inflicted bug.

One variant per process:

    P  in-place calls, print return type and qpos   (correct API usage)
    Q  assignment form, 2 iterations                 (what _bisect.py did)
    R  in-place calls, 200 iterations                (does count matter?)

    docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
      run --rm -e VARIANT=P mjx-shell python playground/_mjx_seq.py
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
VARIANT = os.environ.get("VARIANT", "P")

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
mm = mjx.put_model(m)
md = mjx.put_data(m, d)

try:
    if VARIANT == "P":
        r = mjx.step(mm, md)
        print(f"P  mjx.step returned: {type(r).__name__}")
        print(f"P  md.qpos[2] after in-place step: {float(jnp.asarray(md.qpos)[2]):.6f}")
        r2 = mjx.step(mm, md)
        print(f"P  second in-place step returned: {type(r2).__name__}")
        print(f"P  md.qpos[2] after 2 steps: {float(jnp.asarray(md.qpos)[2]):.6f}")
    elif VARIANT == "Q":
        md = mjx.step(mm, md)
        print(f"Q  after iter1, md is now: {type(md).__name__}")
        md = mjx.step(mm, md)
        print(f"Q  after iter2, md is now: {type(md).__name__}")
    elif VARIANT == "R":
        for i in range(200):
            mjx.step(mm, md)
        print(f"R  200 in-place steps, z={float(jnp.asarray(md.qpos)[2]):.4f}")
    print(f"VARIANT {VARIANT}: PASS")
except Exception as exc:
    print(f"VARIANT {VARIANT}: FAIL -> {type(exc).__name__}: {str(exc).splitlines()[0][:130]}")
