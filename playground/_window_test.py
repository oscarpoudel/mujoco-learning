"""Does mujoco.viewer open a window under 4.0.1 on this machine?

Run with the env's python. Prints a stage line, then stays in the viewer until
the window is closed (or KEEP_SECS elapses, after which it exits cleanly).
KEEP_SECS=0 means run forever until closed.

    python _window_test.py pendulum
"""

import os
import sys
import time

import mujoco
import mujoco.viewer  # noqa: F401 - not auto-imported by `import mujoco`

WHICH = sys.argv[1] if len(sys.argv) > 1 else "pendulum"
KEEP = int(os.environ.get("KEEP_SECS", "20"))

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="0 0 .1"/>
    <body pos="0 0 0.05">
      <freejoint/>
      <geom type="sphere" size=".1" rgba=".6 .3 .8 1"/>
    </body>
  </worldbody>
</mujoco>
"""

print(f"mujoco {mujoco.__version__}: building model", flush=True)
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
print("model built; calling mujoco.viewer.launch_passive", flush=True)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("VIEWER OPEN - window is up", flush=True)
    start = time.time()
    while viewer.is_running():
        step = data.time
        mujoco.mj_step(model, data)
        viewer.sync()
        if KEEP and time.time() - start > KEEP:
            print(f"VIEWER OK - survived {KEEP}s, {int(step/0.002)} steps, "
                  f"closing cleanly", flush=True)
            break
print("viewer closed, exiting", flush=True)
