"""MuJoCo 4.x fix of MuJoCo-Tutorial's impedance_control.py (MIT, Manan Tayal).

Upstream fails on MuJoCo 4.0.1 with:

    data.ctrl[:] = np.clip(tau, -100, 100)
    ValueError: could not broadcast input array from shape (3,) into shape (2,)

Cause: the model has three degrees of freedom -- the two arm hinges *plus* the
obstacle's slide joint -- but only two actuators. ``tau`` comes out of
``jacp.T @ F + qfrc_bias`` at nv=3, and cannot be written into ctrl of size nu=2.
This is a bug in the example, not a version incompatibility: it never worked.

Fix: address the actuated joints by their true dof index instead of assuming the
first ``nu`` rows are the arm. ``actuator_trnid[i, 0]`` is the driven joint id
(-1 for joint-reference-free actuators) and ``jnt_dofadr`` maps that to a dof.
The unfiltered vector is kept, which is the honest picture: the third component
is torque the model has no actuator for.

Run it through the pacing wrapper to actually watch it:

    python playground\\watch.py playground\\fixed\\impedance_control.py
"""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>

    <!-- Obstacle the arm will push against -->
    <body name="obstacle" pos="0.5 0.0 0.55">
      <geom type="sphere" size="0.05" rgba="1 0.2 0.2 0.8" mass="0.5"
            contype="1" conaffinity="1"/>
      <joint type="slide" axis="1 0 0" damping="5" stiffness="100"/>
    </body>

    <!-- 2-link arm -->
    <body name="link1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" mass="2"
            contype="0" conaffinity="0" rgba="0.3 0.5 0.8 1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1.5"
              contype="1" conaffinity="1" rgba="0.3 0.8 0.4 1"/>
        <site name="ee" pos="0.3 0 0" size="0.025" rgba="0 0 1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-100 100"/>
    <motor joint="j2" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""


def actuator_dof_indices(model):
    """Dof index each actuator drives, in ctrl order.

    Order matters here: the obstacle is declared before the arm, so its slide
    joint is dof 0 and the arm joints are dofs 1 and 2. Slicing tau[:nu] would
    silently command the obstacle with the arm's first joint.
    """
    dofs = []
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        if joint_id < 0:
            raise NotImplementedError(
                f"actuator {i} has no joint reference; resolve its dof address "
                "via actuator_trntype/actuator_trnid before indexing tau")
        dofs.append(model.jnt_dofadr[joint_id])
    return np.array(dofs)


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    nv = model.nv
    ctrl_dofs = actuator_dof_indices(model)

    # Desired equilibrium position (into the obstacle)
    x_des = np.array([0.55, 0.0, 0.5])

    # Impedance parameters (spring-damper behavior)
    # Low stiffness = compliant; high stiffness = stiff
    K_imp = np.diag([100.0, 100.0, 100.0])  # stiffness (N/m)
    D_imp = np.diag([20.0, 20.0, 20.0])     # damping (Ns/m)

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    print("Impedance control demo:")
    print(f"  Stiffness: {np.diag(K_imp)} N/m")
    print(f"  Damping:   {np.diag(D_imp)} Ns/m")
    print(f"  nv={nv} dofs, nu={model.nu} actuators -> commanding dofs "
          f"{ctrl_dofs.tolist()}")
    print("  The arm pushes toward the obstacle but yields on contact.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 20.0:
            mujoco.mj_forward(model, data)

            # Current EE state
            x = data.site_xpos[ee_id].copy()
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
            dx = jacp @ data.qvel

            # Impedance law: F = K*(x_des - x) - D*dx
            # This makes the EE behave like a mass-spring-damper
            F = K_imp @ (x_des - x) - D_imp @ dx

            # Gravity compensation + impedance torque
            tau = jacp.T @ F + data.qfrc_bias
            data.ctrl[:] = np.clip(tau[ctrl_dofs], -100, 100)

            mujoco.mj_step(model, data)
            viewer.sync()

    print(f"  final EE x = {data.site_xpos[ee_id]}, obstacle x = "
          f"{data.qpos[0]:.4f} m")


if __name__ == "__main__":
    main()
