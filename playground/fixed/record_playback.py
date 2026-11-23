"""Windows fix of MuJoCo-Tutorial's record_playback.py (MIT, Manan Tayal).

Upstream fails on Windows with:

    FileNotFoundError: [Errno 2] No such file or directory: '/tmp/demo_trajectory.npz'

Cause: a hardcoded POSIX temp path. The fix is ``tempfile.gettempdir()``, which
gives the platform's temp directory on Windows, macOS and Linux alike. No other
line changed.

    python playground\\watch.py playground\\fixed\\record_playback.py
"""

import os
import tempfile

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>

    <body name="arm" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1" rgba="0.3 0.6 0.9 1"/>
      <body pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.015" mass="0.7" rgba="0.4 0.8 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-10 10"/>
    <motor joint="j2" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def record_trajectory(model, data, duration=5.0):
    """Record a trajectory with sinusoidal control inputs."""
    dt = model.opt.timestep
    n_steps = int(duration / dt)

    trajectory = {
        "time": np.zeros(n_steps),
        "qpos": np.zeros((n_steps, model.nq)),
        "qvel": np.zeros((n_steps, model.nv)),
        "ctrl": np.zeros((n_steps, model.nu)),
    }

    mujoco.mj_resetData(model, data)

    for i in range(n_steps):
        t = i * dt
        data.ctrl[0] = 5.0 * np.sin(2.0 * t)
        data.ctrl[1] = 3.0 * np.cos(3.0 * t)

        trajectory["time"][i] = t
        trajectory["qpos"][i] = data.qpos.copy()
        trajectory["qvel"][i] = data.qvel.copy()
        trajectory["ctrl"][i] = data.ctrl.copy()

        mujoco.mj_step(model, data)

    return trajectory


def save_trajectory(trajectory, filename):
    np.savez_compressed(filename, **trajectory)
    print(f"Saved trajectory to {filename} ({trajectory['time'].shape[0]} steps)")


def load_trajectory(filename):
    arr = np.load(filename)
    trajectory = {key: arr[key] for key in arr.files}
    print(f"Loaded trajectory: {trajectory['time'].shape[0]} steps, "
          f"{trajectory['time'][-1]:.1f}s duration")
    return trajectory


def playback_trajectory(model, trajectory):
    """Replay a recorded trajectory in the viewer."""
    data = mujoco.MjData(model)
    n_steps = len(trajectory["time"])
    print(f"Playing back {n_steps} steps...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(n_steps):
            if not viewer.is_running():
                break
            data.qpos[:] = trajectory["qpos"][i]
            data.qvel[:] = trajectory["qvel"][i]
            data.ctrl[:] = trajectory["ctrl"][i]
            mujoco.mj_forward(model, data)
            viewer.sync()

    print("Playback complete.")


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # tempfile, not '/tmp/...' - see module docstring.
    path = os.path.join(tempfile.gettempdir(), "demo_trajectory.npz")

    print("=== Recording trajectory ===")
    traj = record_trajectory(model, data, duration=5.0)

    print("=== Saving ===")
    save_trajectory(traj, path)

    print("=== Loading back ===")
    loaded = load_trajectory(path)

    print("\n=== Playing back recorded trajectory ===")
    playback_trajectory(model, loaded)


if __name__ == "__main__":
    main()
