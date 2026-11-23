"""Does MJX actually step on this GPU? Forced-sync version.

Earlier "MJX works on the GPU" evidence was vacuous, twice over:

1. mjx.step is FUNCTIONAL -- it returns a new Data. `mjx.step(mm, md)` with the
   result discarded changes nothing (verified: 200 such calls left a free sphere
   at z=0.0000). "It didn't raise" was really "it didn't run".
2. JAX executes asynchronously. A jitted call that nothing ever reads may not
   have been waited on, so a kernel-load failure can go unobserved. Here every
   result is consumed with float(), which forces a device-to-host copy and makes
   any error surface at the iteration where it happened.

Read z after the steps: a free sphere must be around -0.79 after 200 steps of
the default 2 ms timestep. z == 0.0 means the sim did not run.

    docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
      run --rm -e N=20 mjx-shell python playground/_mjx_sync.py
"""
import os

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

N = int(os.environ.get("N", "200"))
XML = """
<mujoco>
  <worldbody>
    <body><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
  </worldbody>
</mujoco>
"""

# Control: an op we already know works here (cuBLAS matmul + XLA reduce).
try:
    ctrl = float((jnp.ones((512, 512)) @ jnp.ones((512, 512))).sum())
    print(f"control consumed matmul = {ctrl:.0f}  (expect 134217728)")
except Exception as exc:
    print(f"control FAILED -> {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
mm = mjx.put_model(m)
md = mjx.put_data(m, d)

print(f"N={N}")
try:
    for i in range(N):
        md = mjx.step(mm, md)
    z = float(jnp.asarray(md.qpos)[2])          # forces device sync
    print(f"RESULT z={z:.6f}  ->  {'REAL PHYSICS' if z < -1e-6 else 'NO-OP, sim did not advance'}")
except Exception as exc:
    print(f"RESULT FAIL -> {type(exc).__name__}: {str(exc).splitlines()[0][:130]}")
