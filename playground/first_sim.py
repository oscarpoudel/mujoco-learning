"""Sanity check + Stage 1 playground for the MuJoCo Learning Hub.

Run this first. If it prints SIM OK, your install is healthy and you can trust
the tutorials. It depends only on `mujoco` and `numpy` -- no gymnasium, no torch.

    conda activate robotics_muj
    python playground/first_sim.py

It also demonstrates the four ideas every one of the three tutorials assumes you
already have, using a cart-pole:

  1. MJCF text          -> mujoco.MjModel   (the compiled, read-only physics spec)
  2. simulation state   -> mujoco.MjData    (qpos, qvel, ctrl, ... mutable)
  3. stepping           -> mujoco.mj_step(model, data)
  4. named access       -> model.joint("hinge").qposadr[0] instead of magic indices

Pass --viewer to open the interactive window instead of running headless.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

# --- the model ---------------------------------------------------------------
# A cart on a rail with an unactuated pole. Geoms are set non-colliding so the
# pole swings freely; the only contact-free degree of freedom is the slider.
CARTPOLE = """
<mujoco model="cartpole">
  <option timestep="0.002"/>

  <default>
    <geom contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="rail" type="box" size="1.2 .05 .05" pos="0 0 .05"
          rgba=".3 .6 .9 1"/>

    <body name="cart" pos="0 0 .1">
      <joint name="slider" type="slide" axis="1 0 0" damping=".05"/>
      <geom name="cart_box" type="box" size=".1 .1 .1" mass="1"/>

      <body name="pole" pos="0 0 .1">
        <joint name="hinge" type="hinge" axis="0 1 0" damping=".001"/>
        <geom name="pole_capsule" type="capsule" fromto="0 0 0 0 0 .6"
              size=".03" mass=".1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="push" joint="slider" gear="1" ctrllimited="true"
           ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""

# Proportional-derivative gains on the pole angle, targeting upright (= pi).
# The cart accelerates toward whichever way the pole is leaning, which is what
# catches it. These are hand-tuned; try changing KP to see what breaks first.
KP, KD = 90.0, 9.0
UPRIGHT = np.pi


def balance(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Write a PD balancing torque into data.ctrl. Called before every step."""
    h = model.joint("hinge")
    q = data.qpos[h.qposadr[0]]
    qd = data.qvel[h.dofadr[0]]

    err = UPRIGHT - q
    data.ctrl[0] = KP * err - KD * qd


def render_png(model: mujoco.MjModel, data: mujoco.MjData, path: str) -> bool:
    """Offscreen-render the current state. Returns False if no GL backend."""
    try:
        renderer = mujoco.Renderer(model, height=240, width=320)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=mujoco.MjvCamera())
        frame = renderer.render()
        renderer.close()
    except Exception as exc:  # no GL context, headless server, etc.
        print(f"  render skipped: {type(exc).__name__}: {exc}")
        return False

    # mediapy/PIL are optional; a PPM needs no dependency at all.
    try:
        import mediapy

        mediapy.write_video(path.replace(".png", ".mp4"), [frame], fps=1)
        print(f"  wrote {path.replace('.png', '.mp4')}")
    except ImportError:
        with open(path, "wb") as f:
            f.write(b"P6\n320 240\n255\n")
            f.write(frame.tobytes())
        print(f"  wrote {path}  (open in any browser or image viewer)")
    return True


def main() -> int:
    print(f"mujoco {mujoco.__version__}")

    model = mujoco.MjModel.from_xml_string(CARTPOLE)
    data = mujoco.MjData(model)

    print(
        f"\nmodel: {model.nq} qpos, {model.nv} qvel, "
        f"{model.nu} actuators, {model.ngeom} geoms"
    )
    print(f"  joint 'hinge' -> qpos[{model.joint('hinge').qposadr[0]}]")
    print(f"  actuator 'push' -> ctrl[0], range "
          f"{model.actuator_ctrlrange[0]}")

    if "--viewer" in sys.argv:
        # Blocks until you close the window. Balance is live while you drag.
        # (from-import, not `import mujoco.viewer`, which would rebind `mujoco`
        #  as a local name for the whole function.)
        from mujoco import viewer

        mujoco.viewer.launch(model, data)
        return 0

    # Start 10 degrees off vertical, like a real initial-condition error.
    data.qpos[model.joint("hinge").qposadr[0]] = UPRIGHT - np.deg2rad(10)
    mujoco.mj_forward(model, data)

    duration, dt = 4.0, model.opt.timestep
    steps = int(duration / dt)
    log = np.zeros((steps, 2))
    max_ctrl = 0.0

    for i in range(steps):
        balance(model, data)
        mujoco.mj_step(model, data)
        max_ctrl = max(max_ctrl, abs(float(data.ctrl[0])))
        log[i, 0] = data.qpos[model.joint("hinge").qposadr[0]]
        log[i, 1] = data.qpos[model.joint("slider").qposadr[0]]

    err_deg = np.rad2deg(np.abs(log[:, 0] - UPRIGHT))
    print(f"\nbalanced {duration:.1f}s in {steps} steps")
    print(f"  pole error : {err_deg[0]:6.2f} deg at start"
          f" -> {err_deg[-1]:5.3f} deg at end")
    print(f"  cart travel: {log[0, 1]:+.3f} m -> {log[-1, 1]:+.3f} m")
    print(f"  max |ctrl| : {max_ctrl:.2f} N of the {abs(model.actuator_ctrlrange[0][1]):.0f} N limit")

    ok = err_deg[-1] < 1.0
    print("\nSIM " + ("OK" if ok else "DID NOT CONVERGE -- try retuning KP/KD"))

    # Note the cart position above: it ends up roughly half a metre off where it
    # started, and it will keep creeping. That is not a bug -- balance() only
    # closes the loop on the pole angle, so nothing ever asks the cart to return
    # home. It is a standing example of underactuation: one actuator, two
    # objectives. Exercise: add a Kp*(0 - cart_x) term and watch the drift go.
    print("hint: the cart drifts because only the pole angle is controlled. "
          "Add cart position feedback to fix it.")

    print("\nrendering final frame:")
    render_png(model, data, "playground/first_sim.png")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
