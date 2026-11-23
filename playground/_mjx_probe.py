"""Which MJX stage fails on this GPU?

mjx_check.py fails at mjx.step with a CUBIN load error, while standalone jit of
squeeze/scan/while_loop all work (see _jit_probe.py). This walks MJX one call at
a time to find the boundary -- "upload works, first real step fails" is a very
different answer from "nothing of MJX works on this card".

    docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
      run --rm mjx-shell python playground/_mjx_probe.py
"""
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


def probe(name, fn):
    try:
        detail = fn()
        print(f"  [ok  ] {name:26s} -> {detail}")
        return True
    except Exception as exc:
        print(f"  [FAIL] {name:26s} -> {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
        return False


print("device:", jax.devices()[0])
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)

mjm = None
probe("put_model", lambda: (globals().__setitem__("mjm", mjx.put_model(m)), "uploaded")[1])
mjd = None
probe("put_data", lambda: (globals().__setitem__("mjd", mjx.put_data(m, d)), "uploaded")[1])

probe("kinematics", lambda: "ok" if mjx.kinematics(mjm, mjd) is not None else "None")
probe("forward", lambda: "ok" if mjx.forward(mjm, mjd) is not None else "None")
probe("step x1", lambda: "ok" if mjx.step(mjm, mjd) is not None else "None")


def scan_steps():
    def body(c, _):
        return mjx.step(mjm, c), None
    out, _ = jax.lax.scan(body, mjd, None, length=10)
    return "ok"


probe("step x10 via scan", scan_steps)

# The same sequence pinned to the CPU backend: if this passes, the MJX *code* is
# fine and the fault is specific to the GPU codegen/driver path.
with jax.default_device(jax.devices("cpu")[0]):
    cpu_m = mjx.put_model(m)
    cpu_d = mjx.step(cpu_m, mjx.put_data(m, d))
    probe("step on CPU backend", lambda: f"z={float(jnp.asarray(cpu_d.qpos)[2]):.4f}")
