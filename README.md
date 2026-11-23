# MuJoCo Learning Hub

A single place to study MuJoCo by working through several independent community
tutorials side by side. The three upstream repositories below are **not** merged
or modified — they are cloned as-is into
[`repos/`](#fetching-the-upstream-tutorials) so every file,
commit and attribution stays intact. Everything written here is original: setup
notes, a topic index that maps the same concept across all three codebases, and
a playground for my own experiments.

**Environment:** conda env `robotics_muj` · Python 3.13.18 · MuJoCo 4.0.1 · NVIDIA Quadro P1000

<img src="playground/first_sim.png" alt="Cart-pole rendered headless" width="480">

*The final frame of `playground/first_sim.py` — the install sanity check. A
cart-pole starts tipped at 9.99° and PD control settles it to 0.004°; rendered
offscreen at 320×240 with `MUJOCO_GL=osmesa`. If this runs, your install is
trustworthy.*

---

## Acknowledgements

All credit for the tutorials, code and robot models in `repos/` belongs to their
respective authors. I did not write any of it; I am only collecting and studying
it. Please support them with a star on the original projects.

| Repository | Author | What it teaches | License |
|---|---|---|---|
| [**MuJoCo-Tutorial**](https://github.com/tayalmanan28/MuJoCo-Tutorial) | [Manan Tayal](https://github.com/tayalmanan28) | Modern, broad introduction: MJCF modelling, `MjModel`/`MjData`, rendering, contact, PD/LQR/impedance/operational-space control, IK, Gymnasium, MJX. 13 notebooks + 18 examples. | **MIT** ✅ |
| [**mujoco-robotics-usage**](https://github.com/joonhyung-lee/mujoco-robotics-usage) | [Joonhyung Lee](https://github.com/joonhyung-lee) | Robotics algorithms on real robots (UR5e, Franka Panda, Husky): IK variants, trajectory planning, interaction MPC, SLAM + RRT\* + MPPI navigation, point clouds, SAM/OpenVLA-style perception. | **None stated** ⚠️ |
| [**Tutorial_Mujoco_RL**](https://github.com/gimmyy/Tutorial_Mujoco_RL) | [gimmyy](https://github.com/gimmyy) | Reinforcement learning from zero: custom Gymnasium envs, REINFORCE on an inverted pendulum, PPO for Panda obstacle avoidance and ORCA-hand cube reorientation with Stable-Baselines3. | **None stated** ⚠️ |

Those two unlicensed repositories are handled carefully — see
[LICENSING.md](LICENSING.md). In short: fine to clone and read privately to
learn from, but not to copy code out of or redistribute.

MuJoCo itself is developed by [Google DeepMind](https://github.com/google-deepmind/mujoco)
under the Apache 2.0 license. Several robot models shared between these repos
come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

---

## Layout

```
mujoco-learning/
├── README.md              This file
├── DOCKER.md              Container setup: what works, what doesn't, and why
├── LICENSING.md           Per-repo license status and what it permits
├── docker-compose.yml     studio / learn / sim / shell / rl-shell / mjx-shell
├── docker-compose.gpu.yml GPU override — inactive until the driver is updated
├── .dockerignore          .gitignore (repos/ is not committed)
├── docker/
│   ├── Dockerfile         headless CPU image, 3 targets: base, rl, mjx
│   ├── doctor.py          environment check; runs in-container or on Windows
│   └── mjx_check.py       GPU readiness gate; fails loudly on CPU fallback
├── repos/                 Upstream clones, unmodified (git remotes intact)
│   ├── MuJoCo-Tutorial/          1.2 MB
│   ├── mujoco-robotics-usage/    461 MB  (heavy GIF/model assets)
│   └── Tutorial_Mujoco_RL/       356 MB  (hand/robot meshes)
├── playground/            My own scratch space — safe to edit
│   ├── first_sim.py
│   └── models/ball.xml
├── scripts/
│   ├── studio.ps1         MuJoCo Studio launcher that avoids the 4.0.1 crash
│   └── gpu-gates.ps1      Runs the GPU enablement gates, stops at the first failure
└── notes/                 Findings worth keeping
    ├── 0001-mujoco4-studio-native-viewer.md
    └── 0002-gpu-enablement-checklist.md
```

### Fetching the upstream tutorials

`repos/` is **not committed** — it holds three upstream git repositories with
their own histories, and two of them publish no license. Fetch them yourself
(~818 MB total, `--depth 1` keeps it to the working trees; `git clone` creates
`repos/` for you):

```powershell
git clone --depth 1 https://github.com/tayalmanan28/MuJoCo-Tutorial        repos/MuJoCo-Tutorial
git clone --depth 1 https://github.com/joonhyung-lee/mujoco-robotics-usage repos/mujoco-robotics-usage
git clone --depth 1 https://github.com/gimmyy/Tutorial_Mujoco_RL           repos/Tutorial_Mujoco_RL
```

Later, update all three with:

```powershell
git -C repos/MuJoCo-Tutorial pull
git -C repos/mujoco-robotics-usage pull
git -C repos/Tutorial_Mujoco_RL pull
```

---

## Windows setup

```powershell
conda create -n robotics_muj python=3.13 -y
conda activate robotics_muj
pip install mujoco                 # 4.0.1 at time of writing
python playground\first_sim.py     # must print the cart-pole settling
```

The visualiser in MuJoCo 4.0.1 crashes on this machine for reasons documented in
[notes/0001](notes/0001-mujoco4-studio-native-viewer.md); use either

```powershell
.\scripts\studio.ps1 playground\models\ball.xml      # native window, software GL
```

or `--gfx=web` and open the printed `http://localhost:8081/`. Docker is covered
in [DOCKER.md](DOCKER.md).

---

## Recommended learning path

The three repos overlap a lot, so the fastest route is to take one stage from
each rather than finishing one repo before starting another.

**Stage 1 — Core mental model.** `MuJoCo-Tutorial/tutorial/00_what_is_mujoco.ipynb`
through `06_xml_modeling.ipynb`. Get comfortable with MJCF, `MjModel` vs
`MjData`, and `mj_step` vs `mj_forward`. Run `playground/first_sim.py` to
confirm your install works before believing anything else.

**Stage 2 — Classical control.** Start with `MuJoCo-Tutorial` for the simple
versions (`pendulum_control.py`, `lqr_control.py`, `inverse_kinematics.py`),
then go deeper with `mujoco-robotics-usage`, which is the strongest source here
for IK variants and trajectory planning on a real 6-DOF arm.

**Stage 3 — Contact-rich manipulation.** `mujoco-robotics-usage/code/manipulator/`:
`demo_table_transfer.py` and `interaction_mpc.py` are the highlights — MPC that
recovers from contact and estimates end-effector wrench from joint torque. Read
`TABLE_TRANSFER.md` and `INTERACTION_MPC.md` in that folder first; they are
genuinely good explanations.

**Stage 4 — RL.** `Tutorial_Mujoco_RL` is the gentlest on-ramp because it is
explicitly ordered by difficulty: UR5e control → REINFORCE → PPO with a custom
reward → PPO on a high-DOF hand. Then read its `src/orca_sim/` package to see
how a real Gymnasium environment is structured, and cross-reference with
`MuJoCo-Tutorial/tutorial/08_gymnasium.ipynb`.

**Stage 5 — Mobile & perception.** `mujoco-robotics-usage/code/mobile/` is a
complete Husky curriculum: differential-drive kinematics → Pure Pursuit →
Stanley → RRT → MPPI → 2D LiDAR SLAM → RRT\* with dynamic obstacle avoidance.
`code/point-cloud/` covers projection, voxel grids and ICP.

**Stage 6 — Scale.** `MuJoCo-Tutorial` for MJX (GPU-parallel sim via JAX) and
`examples/domain_randomization.py` for sim-to-real habits.

---

## Topic index

The same concept appears in more than one repo, at different depths. Cross-
reading these pairs is where the actual learning happens.

| Concept | Intro | Deeper |
|---|---|---|
| MJCF modelling | `MuJoCo-Tutorial/tutorial/06_xml_modeling.ipynb` | Robot trees in `mujoco-robotics-usage/asset/{ur5e,panda,husky,husky_ur5e,snapbot}` and `Tutorial_Mujoco_RL/*/model/` |
| Inverse kinematics | `MuJoCo-Tutorial/examples/inverse_kinematics.py` (damped least squares) | `mujoco-robotics-usage/code/manipulator/demo_ur_solve_ik{,_augmented,_nullspace_projection,_repelling,_rrt}.ipynb` |
| Inverse dynamics | `MuJoCo-Tutorial/tutorial/04_control.ipynb` | `mujoco-robotics-usage/code/manipulator/demo_panda_inverse_dynamics.ipynb` |
| Trajectory generation | `MuJoCo-Tutorial/examples/trajectory_tracking.py` | `mujoco-robotics-usage/code/manipulator/demo_ur_trajectory_planning.ipynb`, `demo_{ur,}_trapezoidal_spline.ipynb` (quintic, min-jerk, trapezoidal, S-spline) |
| Impedance / admittance | `MuJoCo-Tutorial/examples/impedance_control.py` | `mujoco-robotics-usage/code/manipulator/interaction_mpc.py` |
| Operational space | `MuJoCo-Tutorial/examples/operational_space_control.py` | `mujoco-robotics-usage/code/mobile_manipulator/demo_husky_02_whole_body_control.ipynb` |
| MPC | — | `mujoco-robotics-usage/code/manipulator/{demo_interaction_mpc,demo_table_transfer}.py` (+ `requirements_mpc.txt`: osqp, qpsolvers) |
| LQR | `MuJoCo-Tutorial/examples/lqr_control.py` | — |
| Legged / FSM locomotion | `MuJoCo-Tutorial/examples/locomotion_controller.py`, `hopper.py` | — |
| Path planning | `mujoco-robotics-usage/code/mobile/demo_husky_04_RRT.ipynb` | `demo_husky_05_model_predictive_path_integral.ipynb`, `demo_husky_10_rrt_w_avoidance.ipynb` |
| SLAM | `mujoco-robotics-usage/code/mobile/demo_husky_06_slam.ipynb` | `demo_husky_07_slam_w_rrt.ipynb` |
| Pursuit / Stanley control | `demo_husky_03_pure_pursuit.ipynb` | `demo_husky_09_stanley.ipynb` |
| Point clouds | `mujoco-robotics-usage/code/point-cloud/demo_pcd.ipynb` | `demo_colored_pcd.ipynb`, `demo_voxelize_pcd.ipynb`, `demo_voxel_value_map.ipynb` |
| Segmentation | `mujoco-robotics-usage/code/miscellaneous/demo_sam.ipynb` | `demo_owl-vit+sam.ipynb`, `demo_owl-vit+sam-2.ipynb` |
| Custom Gymnasium env | `Tutorial_Mujoco_RL/2tutorial_InvertedPendulum/mujoco_reinforce.py` | `Tutorial_Mujoco_RL/4tutorial_orca_cube_orientation/src/orca_sim/` (envs, registry, task_envs, versions) |
| REINFORCE | `Tutorial_Mujoco_RL/2tutorial_InvertedPendulum/mujoco_reinforce_disp.py` | — |
| PPO | `Tutorial_Mujoco_RL/3tutorial_panda_obstacle/rl_panda_obstacle_high_profile.py` | `4tutorial_orca_cube_orientation/ppo_policy_orientation.py` |
| Policy sanity-checking | `4tutorial_orca_cube_orientation/random_policy_orientation.py` (run a random policy *before* trusting a trained one) | `MuJoCo-Tutorial/tutorial/10_troubleshooting.ipynb` |
| Domain randomization | `MuJoCo-Tutorial/examples/domain_randomization.py` | — |
| MJX / GPU-parallel | `MuJoCo-Tutorial/tutorial/07_mjx.ipynb` | `MuJoCo-Tutorial/examples/mjx_parallel.py` |
| Offscreen render / video | `MuJoCo-Tutorial/examples/offscreen_rendering.py` | `MuJoCo-Tutorial/scripts/record_gifs.py`, `mujoco-robotics-usage/mobile/render_*_gif.py` |
| Record & replay | `MuJoCo-Tutorial/examples/record_playback.py` | — |

---

## Setup

The `robotics_muj` env already has a working MuJoCo 4.0.1 (verified: import,
`mj_step`, and offscreen rendering all pass). Add the tutorial dependencies as
you need them rather than all at once — the RL and perception stacks are heavy.

```powershell
conda activate robotics_muj

# Stage 1-3: everything MuJoCo-Tutorial needs
python -m pip install "gymnasium[mujoco]" scipy matplotlib mediapy ipywidgets

# Stage 4: RL
python -m pip install "stable-baselines3[extra]" torch pandas seaborn transforms3d

# Stage 5: perception / planning (large)
python -m pip install opencv-python networkx shapely osqp "imageio[ffmpeg]"
```

**Do not blindly run the requirements files inside `repos/`.** I checked them and
they will clobber your MuJoCo:

```powershell
# mujoco-robotics-usage/code/manipulator/requirements_mpc.txt contains:
#   mujoco==3.3.7      <-- would downgrade your 4.0.1
```

If you need it, either drop the `mujoco` line from a local copy, or install into
a throwaway env with `python -m pip install -r requirements_mpc.txt --target ...`.
After any of the installs above, re-check that you still have what you expect:

```powershell
python -c "import mujoco; print(mujoco.__version__)"   # want 4.0.1
```

### Compatibility warnings

These three repos were written against different MuJoCo generations. Expect
friction; the notes below are what I found when checking them.

| Repo | Written for | Status on MuJoCo 4.0.1 / Python 3.13 |
|---|---|---|
| `MuJoCo-Tutorial` | `mujoco>=3.2` | **Should work.** Modern API, current viewer. |
| `mujoco-robotics-usage` | README says Ubuntu 20.04, Python 3.8.10, **MuJoCo 2.3.2** — but `requirements_mpc.txt` pins **3.3.7** | **Mixed.** The older notebooks really are 2.x-era. The main blocker for running anything here is `mujoco_viewer` (`mujoco-python-viewer`, last release 0.1.4 in Nov 2023, built for MuJoCo 2.x); the MPC/table-transfer scripts are much more recent and may only need that viewer call swapped for `mujoco.viewer` or `scripts/studio.ps1`. Treat it as a reference to read and port rather than execute. If you must run it unmodified, make a separate `python=3.8` + `mujoco=2.3.2` env instead of downgrading `robotics_muj`. |
| `Tutorial_Mujoco_RL` | Windows 10, Python 3.11 | **Probably works**, untested here. Uses `gymnasium`, `stable-baselines3`, `torch`, and `ikpy` (in `1tutorial_ur5e/main.py`). Check `ikpy` first — it is the least-maintained dependency of the four and the most likely to lack 3.13 wheels. |

Also note that `gymnasium[mujoco]` and `dm_control` pin their own MuJoCo
versions. Install them, then re-check with `python -c "import mujoco; print(mujoco.__version__)"`
so a resolver doesn't quietly swap your 4.0.1 out from under you.

### MuJoCo Studio on this machine

Studio's native desktop window **crashes on 4.0.1** with
`ValueError: Target not found:` — a bug in the shipped `native_viewer.py`, not
a GPU or driver problem. Full analysis and the two working workarounds are in
[`notes/0001-mujoco4-studio-native-viewer.md`](notes/0001-mujoco4-studio-native-viewer.md).
Use the launcher rather than typing the flag:

```powershell
.\scripts\studio.ps1 repos\Tutorial_Mujoco_RL\1tutorial_ur5e\model\universal_robots_ur5e\scene.xml
```

### Option B: Docker (no local Python install needed)

If you would rather not touch your Python environment at all, the container has
the whole stack. It is verified working on this machine — physics matches the
Windows env byte for byte, and Studio is served to your browser.

```powershell
docker compose build
docker compose run --rm sim python docker/doctor.py   # verify
docker compose up studio                              # http://localhost:9080/
docker compose up learn                               # http://localhost:8888
```

The project is bind-mounted at `/workspace`, so edits are live — no rebuilds.

Two honest caveats, both explained in [DOCKER.md](DOCKER.md): the container is
**CPU-only** (this host exposes no CUDA to WSL, so `--gpus all` fails), and it
has **no X11**, so all visualisation goes through the browser. That second
limitation is what makes Docker a good fit here, because `--gfx=web` never
touches the buggy native-viewer code path.

Want the GPU working? [notes/0002-gpu-enablement-checklist.md](notes/0002-gpu-enablement-checklist.md)
is a gated step-by-step. Headline: the blocker is driver **452.56**, which
predates CUDA-on-WSL (introduced in R470), and **R580 is the last driver branch
that supports this Pascal Quadro P1000** — so installing "the latest" (R590+)
would leave the card unsupported. `docker-compose.gpu.yml` is already written
and validated for when the driver lands.

---

## Reference

- [MuJoCo documentation](https://mujoco.readthedocs.org/) and [XML reference](https://mujoco.readthedocs.org/en/stable/XMLreference.html)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — production robot models (Franka, UR5e, Unitree, …)
- [dm_control](https://github.com/google-deepmind/dm_control) — DeepMind's control suite
- [Gymnasium MuJoCo envs](https://gymnasium.farama.org/environments/mujoco/) — HalfCheetah, Ant, Humanoid benchmarks
- [MJX docs](https://mujoco.readthedocs.org/en/stable/mjx.html) and [paper](https://arxiv.org/abs/2307.00282)
- [robosuite](https://robosuite.ai/) — manipulation benchmarks on MuJoCo

---

## License

The files I wrote (README, `notes/`, `playground/`, `scripts/`, `docker/`,
compose files) are MIT — see [LICENSE](LICENSE). The three tutorials live in
their own repositories under their own terms and are **not** published here;
[LICENSING.md](LICENSING.md) records what each of them permits, including the
fact that two of them state no license at all. MuJoCo is Apache 2.0 by
Google DeepMind.
