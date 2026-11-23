# MuJoCo Learning Hub

A structured study environment for [MuJoCo](https://mujoco.readthedocs.org/)
built around three independent community tutorials. The upstream repositories
are kept as unmodified clones rather than merged, so every file, commit and
attribution stays intact. What this project adds is original material that sits
on top of them: a cross-repository topic index, a staged learning path,
containerised environments, and verified notes on version compatibility.

<img src="playground/first_sim.png" alt="Cart-pole rendered offscreen" width="320">

*Output of `playground/first_sim.py`, the install sanity check: a cart-pole
under PD control, rendered offscreen.*

---

## Contents

- [Acknowledgements](#acknowledgements)
- [Layout](#layout)
- [Setup](#setup) — [local](#local-install) · [upstream tutorials](#fetching-the-upstream-tutorials) · [Docker](#docker) · [GPU](#gpu-acceleration)
- [Learning path](#learning-path)
- [Topic index](#topic-index)
- [Compatibility notes](#compatibility-notes)
- [Known issues](#known-issues)
- [Reference](#reference)
- [License](#license)

---

## Acknowledgements

All tutorials, code and robot models referenced here belong to their respective
authors, who have my thanks. Please support the original projects with a star.

| Repository | Author | Scope | License |
|---|---|---|---|
| [**MuJoCo-Tutorial**](https://github.com/tayalmanan28/MuJoCo-Tutorial) | [Manan Tayal](https://github.com/tayalmanan28) | Broad modern introduction: MJCF modelling, `MjModel`/`MjData`, rendering, contact, PD/LQR/impedance/operational-space control, IK, Gymnasium, MJX. 13 notebooks, 18 examples. | MIT |
| [**mujoco-robotics-usage**](https://github.com/joonhyung-lee/mujoco-robotics-usage) | [Joonhyung Lee](https://github.com/joonhyung-lee) | Robotics algorithms on real platforms (UR5e, Franka Panda, Husky): IK variants, trajectory planning, interaction MPC, SLAM with RRT\* and MPPI navigation, point clouds, SAM/OpenVLA-style perception. | None stated |
| [**Tutorial_Mujoco_RL**](https://github.com/gimmyy/Tutorial_Mujoco_RL) | [gimmyy](https://github.com/gimmyy) | Reinforcement learning from first principles: custom Gymnasium environments, REINFORCE on an inverted pendulum, PPO for Panda obstacle avoidance and ORCA-hand cube reorientation with Stable-Baselines3. | None stated |

Two of the three publish no repository licence, which materially affects what you
may do with them. See [LICENSING.md](LICENSING.md) for the details.

MuJoCo is developed by [Google DeepMind](https://github.com/google-deepmind/mujoco)
under Apache 2.0. Several robot models used across these repositories come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

---

## Layout

```
mujoco-learning/
├── README.md              This file
├── DOCKER.md              Container design, verified behaviour, caveats
├── LICENSING.md           Per-repository licence status and its consequences
├── docker-compose.yml     studio / learn / sim / shell / rl-shell / mjx-shell
├── docker-compose.gpu.yml NVIDIA runtime override (opt-in)
├── .dockerignore          .gitignore
├── docker/
│   ├── Dockerfile         Headless image with three targets: base, rl, mjx
│   ├── doctor.py          Environment report; runs in-container or on Windows
│   └── mjx_check.py       GPU readiness gate; fails on silent CPU fallback
├── repos/                 Upstream clones, unmodified (not committed)
├── playground/            Exercises, sanity checks and benchmarks
│   ├── first_sim.py
│   ├── render_bench.py
│   └── models/ball.xml
├── scripts/
│   ├── studio.ps1         Studio launcher that avoids the 4.0.1 viewer bug
│   ├── gpu-gates.ps1      Runs the GPU enablement checks in order
│   └── check_md_paths.py  Verifies every file path referenced in the docs
└── notes/
    ├── 0001-mujoco4-studio-native-viewer.md
    └── 0002-gpu-enablement-checklist.md
```

---

## Setup

Requirements: Python 3.11+ (3.13 tested), `pip`, and optionally conda and
Docker. Roughly 1 GB of disk is needed for the upstream tutorials.

### Local install

```powershell
conda create -n robotics_muj python=3.13 -y
conda activate robotics_muj
pip install mujoco
python playground\first_sim.py     # prints SIM OK when the install is healthy
```

Add tutorial dependencies as you reach the relevant stage — the RL and
perception stacks are heavy:

```powershell
# Stages 1-3
pip install "gymnasium[mujoco]" scipy matplotlib mediapy ipywidgets

# Stage 4: reinforcement learning
pip install "stable-baselines3[extra]" torch pandas seaborn transforms3d ikpy

# Stage 5: perception and planning
pip install opencv-python networkx shapely osqp "imageio[ffmpeg]"
```

**Do not install the requirements files inside `repos/` unmodified.** They pin
older releases and will downgrade a working install:

```
mujoco-robotics-usage/code/manipulator/requirements_mpc.txt
  mujoco==3.3.7        # downgrades 4.0.1
```

Copy the file, remove the `mujoco` pin, and install from the copy. Because
`gymnasium[mujoco]` and `dm_control` declare their own MuJoCo constraints,
confirm the version after every install:

```powershell
python -c "import mujoco; print(mujoco.__version__)"
```

### Fetching the upstream tutorials

`repos/` is not committed: it holds three separate git histories, and two of
them are unlicensed. Clone them yourself — about 820 MB total:

```powershell
git clone --depth 1 https://github.com/tayalmanan28/MuJoCo-Tutorial        repos/MuJoCo-Tutorial
git clone --depth 1 https://github.com/joonhyung-lee/mujoco-robotics-usage repos/mujoco-robotics-usage
git clone --depth 1 https://github.com/gimmyy/Tutorial_Mujoco_RL           repos/Tutorial_Mujoco_RL
```

Update all three with:

```powershell
git -C repos/MuJoCo-Tutorial pull
git -C repos/mujoco-robotics-usage pull
git -C repos/Tutorial_Mujoco_RL pull
```

### Docker

A container with the full stack, for avoiding changes to a local Python
environment. Physics output matches the native install exactly.

```powershell
docker compose build
docker compose run --rm sim python docker/doctor.py   # 24-check environment report
docker compose up studio                              # http://localhost:9080/
docker compose up learn                               # http://localhost:8888
```

The project is bind-mounted at `/workspace`, so edits take effect without
rebuilding. [DOCKER.md](DOCKER.md) covers image targets, verified behaviour and
the two significant limitations: in-container rendering is software-only, and
there is no X11 display, so visualisation is served to a browser.

### GPU acceleration

Optional, and currently Windows + WSL2 specific. CUDA works in containers once
the host driver is on the correct branch; hardware GL does not, regardless of
driver. `scripts/gpu-gates.ps1` runs the checks in dependency order and reports
the specific fix at the first failure:

```powershell
.\scripts\gpu-gates.ps1 -IncludeMjx
```

Choosing the driver is the one non-obvious step: recent branches have dropped
support for older GPU architectures, so the newest driver is not necessarily the
correct one. [notes/0002-gpu-enablement-checklist.md](notes/0002-gpu-enablement-checklist.md)
records how the right branch was identified, the JAX memory settings required on
a 4 GB card, and the container-GL limitation with its measurements.

---

## Learning path

The three repositories overlap substantially. Working one stage at a time across
all three is faster than finishing them in sequence, because each exposes the
same concept at a different depth.

| Stage | Focus | Primary material |
|---|---|---|
| 1 | Core mental model: MJCF, `MjModel` vs `MjData`, `mj_step` vs `mj_forward` | `MuJoCo-Tutorial/tutorial/00_what_is_mujoco.ipynb` … `06_xml_modeling.ipynb`, then `playground/first_sim.py` |
| 2 | Classical control: PD, LQR, IK | `MuJoCo-Tutorial/examples/{pendulum_control,lqr_control,inverse_kinematics}.py`, then `mujoco-robotics-usage` IK notebooks |
| 3 | Contact-rich manipulation, MPC | `mujoco-robotics-usage/code/manipulator/` — `demo_table_transfer.py`, `interaction_mpc.py`; read `TABLE_TRANSFER.md` and `INTERACTION_MPC.md` first |
| 4 | Reinforcement learning | `Tutorial_Mujoco_RL` in its stated order: UR5e → REINFORCE → PPO → high-DOF hand; then `src/orca_sim/` for environment structure |
| 5 | Mobile robots and perception | `mujoco-robotics-usage/code/mobile/` (differential drive → Pure Pursuit → Stanley → RRT → MPPI → LiDAR SLAM → RRT\*), `code/point-cloud/` |
| 6 | Scale and sim-to-real | `MuJoCo-Tutorial` MJX material, `examples/domain_randomization.py` |

---

## Topic index

Concepts covered by more than one repository, usually at different depths. The
pairs are worth reading together.

| Concept | Intro | Deeper |
|---|---|---|
| MJCF modelling | `MuJoCo-Tutorial/tutorial/06_xml_modeling.ipynb` | Robot trees in `mujoco-robotics-usage/asset/{ur5e,panda,husky,husky_ur5e,snapbot}`, `Tutorial_Mujoco_RL/*/model/` |
| Inverse kinematics | `MuJoCo-Tutorial/examples/inverse_kinematics.py` (damped least squares) | `mujoco-robotics-usage/code/manipulator/demo_ur_solve_ik{,_augmented,_nullspace_projection,_repelling,_rrt}.ipynb` |
| Inverse dynamics | `MuJoCo-Tutorial/tutorial/04_control.ipynb` | `mujoco-robotics-usage/code/manipulator/demo_panda_inverse_dynamics.ipynb` |
| Trajectory generation | `MuJoCo-Tutorial/examples/trajectory_tracking.py` | `demo_ur_trajectory_planning.ipynb`, `demo_{ur,}_trapezoidal_spline.ipynb` (quintic, min-jerk, trapezoidal, S-spline) |
| Impedance / admittance | `MuJoCo-Tutorial/examples/impedance_control.py` | `mujoco-robotics-usage/code/manipulator/interaction_mpc.py` |
| Operational space | `MuJoCo-Tutorial/examples/operational_space_control.py` | `code/mobile_manipulator/demo_husky_02_whole_body_control.ipynb` |
| MPC | — | `code/manipulator/{demo_interaction_mpc,demo_table_transfer}.py` (needs `osqp`, `qpsolvers`) |
| LQR | `MuJoCo-Tutorial/examples/lqr_control.py` | — |
| Legged / FSM locomotion | `MuJoCo-Tutorial/examples/{locomotion_controller,hopper}.py` | — |
| Path planning | `code/mobile/demo_husky_04_RRT.ipynb` | `demo_husky_05_model_predictive_path_integral.ipynb`, `demo_husky_10_rrt_w_avoidance.ipynb` |
| SLAM | `code/mobile/demo_husky_06_slam.ipynb` | `demo_husky_07_slam_w_rrt.ipynb` |
| Pursuit / Stanley control | `demo_husky_03_pure_pursuit.ipynb` | `demo_husky_09_stanley.ipynb` |
| Point clouds | `code/point-cloud/demo_pcd.ipynb` | `code/point-cloud/demo_voxelize_pcd.ipynb`, `demo_voxel_grid.ipynb`, `code/miscellaneous/demo_voxel_value_map.ipynb` |
| Segmentation | `code/miscellaneous/demo_sam.ipynb` | `demo_owl-vit+sam.ipynb`, `demo_owl-vit+sam-2.ipynb` |
| Custom Gymnasium environment | `Tutorial_Mujoco_RL/2tutorial_InvertedPendulum/mujoco_reinforce.py` | `4tutorial_orca_cube_orientation/src/orca_sim/` (envs, registry, task_envs) |
| REINFORCE | `2tutorial_InvertedPendulum/mujoco_reinforce_disp.py` | — |
| PPO | `3tutorial_panda_obstacle/rl_panda_obstacle_high_profile.py` | `4tutorial_orca_cube_orientation/ppo_policy_orientation.py` |
| Policy validation | `4tutorial_orca_cube_orientation/random_policy_orientation.py` — evaluate a random policy before trusting a trained one | `MuJoCo-Tutorial/tutorial/10_troubleshooting.ipynb` |
| Domain randomisation | `MuJoCo-Tutorial/examples/domain_randomization.py` | — |
| MJX / parallel simulation | `MuJoCo-Tutorial/tutorial/07_mjx.ipynb` | `MuJoCo-Tutorial/examples/mjx_parallel.py` |
| Offscreen rendering and video | `MuJoCo-Tutorial/examples/offscreen_rendering.py` | `MuJoCo-Tutorial/scripts/record_gifs.py`, `mujoco-robotics-usage/mobile/render_*_gif.py` |
| Recording and replay | `MuJoCo-Tutorial/examples/record_playback.py` | — |

---

## Compatibility notes

The three tutorials were written against different MuJoCo generations. Status
against MuJoCo 4.0.1 / Python 3.13:

| Repository | Written for | Status |
|---|---|---|
| `MuJoCo-Tutorial` | `mujoco>=3.2` | Works. Current API throughout. |
| `mujoco-robotics-usage` | Python 3.8, MuJoCo 2.3.2 per its README; `requirements_mpc.txt` pins 3.3.7 | Mixed. The older notebooks are genuinely 2.x-era and the main blocker is `mujoco-python-viewer` (last release 0.1.4, Nov 2023, built for MuJoCo 2.x). The MPC and table-transfer scripts are much more recent and likely need only the viewer call replaced with `mujoco.viewer` or `scripts/studio.ps1`. Treat as reference material to port rather than execute. |
| `Tutorial_Mujoco_RL` | Windows, Python 3.11 | Expected to work; not exercised here. Dependencies are `gymnasium`, `stable-baselines3`, `torch`, `ikpy`. Check `ikpy` first — least maintained of the four and the most likely to lack 3.13 wheels. |

---

## Known issues

**MuJoCo Studio's native window crashes on 4.0.1** with
`ValueError: Target not found:`. This is a defect in the shipped
`native_viewer.py`, not a GPU or driver problem — reproduced across graphics
backends and driver branches. `scripts/studio.ps1` selects a working mode by
default; `--gfx=web` streams to a browser instead. Analysis and both
workarounds: [notes/0001](notes/0001-mujoco4-studio-native-viewer.md).

**In-container rendering is software-only.** Docker Desktop's WSL2 GPU support
injects CUDA but no GL implementation, so `MUJOCO_GL=egl` falls through to Mesa
llvmpipe. `playground/render_bench.py` measures the difference, which is within
5% of `osmesa` — along with the methodology trap that makes naive benchmarks
unable to detect it at all. Details in
[notes/0002](notes/0002-gpu-enablement-checklist.md).

**JAX preallocates too much VRAM by default on small cards** and fails with
`CUDA_ERROR_OUT_OF_MEMORY` before executing anything. `docker-compose.gpu.yml`
sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.35` on the MJX service.

---

## Reference

- [MuJoCo documentation](https://mujoco.readthedocs.org/) · [XML reference](https://mujoco.readthedocs.org/en/stable/XMLreference.html)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — production robot models
- [dm_control](https://github.com/google-deepmind/dm_control) — DeepMind's control suite
- [Gymnasium MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/)
- [MJX documentation](https://mujoco.readthedocs.org/en/stable/mjx.html) and [paper](https://arxiv.org/abs/2307.00282)
- [robosuite](https://robosuite.ai/) — manipulation benchmarks on MuJoCo

---

## License

Original files in this repository (this README, `notes/`, `playground/`,
`scripts/`, `docker/`, and the compose files) are MIT-licensed — see
[LICENSE](LICENSE). The three tutorials remain in their own repositories under
their own terms and are not distributed here; [LICENSING.md](LICENSING.md)
records what each permits. MuJoCo is Apache 2.0 by Google DeepMind.
