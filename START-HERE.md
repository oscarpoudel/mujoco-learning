# Start here

The path from an empty checkout to having run real MuJoCo code, in the order
that produces understanding fastest. Every command below was executed on this
machine against MuJoCo 4.0.1 / Python 3.13, and the output shown is the output
it produced. Anything that does not work is listed in
[What does not work](#what-does-not-work) with its cause, so you do not spend an
evening debugging a tutorial instead of learning from it.

Estimated time: about an hour for steps 0–3.

---

## 0. One-time setup

```powershell
conda create -n robotics_muj python=3.13 -y
conda activate robotics_muj
pip install mujoco "gymnasium[mujoco]" scipy matplotlib mediapy ipywidgets jupyterlab imageio-ffmpeg
python scripts\setup_ffmpeg_shim.ps1
git clone --depth 1 https://github.com/tayalmanan28/MuJoCo-Tutorial repos/MuJoCo-Tutorial
```

`setup_ffmpeg_shim.ps1` is not optional if you want video output. `mediapy`
shells out to an `ffmpeg` binary, `conda install ffmpeg` yields one that dies at
startup on Windows, and the resulting error message ("Program 'ffmpeg' is not
found", or `Error writing 'x.mp4': ` with nothing after the colon) points
anywhere but at the real cause. See
[Known issues in README](README.md#known-issues).

The other two upstream tutorials are optional at this stage —
[README](README.md#fetching-the-upstream-tutorials) has their clone commands.

---

## 1. Confirm the install

Two checks. The first tells you what the environment *can* do:

```powershell
python scripts\env_check.py
```

```
python 3.13.18  C:\...\envs\robotics_muj\python.exe

  ok   mujoco               4.0.1
  ok   numpy                2.5.5
  ok   gymnasium            1.4.0
  ...
  Stage 1  core: MJCF, MjModel/MjData, mj_step
      -> ready
CORE OK
```

The second proves the physics engine actually runs and that PID control
converges — `playground/first_sim.py`:

```powershell
python playground\first_sim.py
```

```
mujoco 4.0.1
model: 2 qpos, 2 qvel, 1 actuators, 3 geoms
  joint 'hinge' -> qpos[1]
  actuator 'push' -> ctrl[0], range [-20.  20.]

balanced 4.0s in 2000 steps
  pole error :   9.99 deg at start -> 0.004 deg at end
  cart travel: +0.000 m -> -0.488 m
  max |ctrl| : 15.71 N of the 20 N limit

SIM OK
```

The cart ends 0.49 m from where it started because only the pole angle is fed
back. That is your first exercise: add cart-position feedback and watch the
drift go away.

---

## 2. Learn the core model (notebooks 00 → 05)

Read these in order. They are the mental model everything else assumes: what an
MJCF file is, why `MjModel` is immutable and `MjData` is not, and what
`mj_step` does versus `mj_forward`.

```powershell
jupyter lab repos\MuJoCo-Tutorial\tutorial
```

| Notebook | Executed end to end |
|---|---|
| `quickstart.ipynb` | yes |
| `00_what_is_mujoco.ipynb` | yes (26 cells) |
| `01_basics.ipynb` | yes |
| `02_rendering.ipynb` | yes |
| `03_simulation.ipynb` | yes |
| `04_control.ipynb` | yes (21 cells) |
| `05_contact.ipynb` | yes |
| `06_xml_modeling.ipynb` | **no** — one cell uses `<composite>`, see below |
| `07_mjx.ipynb` | **no** — needs the separate `mujoco-mjx` package |
| `08_gymnasium.ipynb` | yes |
| `09_sensors_and_estimation.ipynb` | **no** — one cell's `motor_force` sensor, see below |
| `10_troubleshooting.ipynb` | yes (22 cells) |

"Executed end to end" means every cell ran without error, verified by
`playground/nb_check.py` — not by opening them and skimming. Note that
`02_rendering` and `04_control` only pass *after* the ffmpeg shim exists; before
it, they failed for reasons that looked like rendering bugs.

Run the check yourself (notebooks are copied to a scratch directory first,
because `repos/` holds unmodified upstream clones):

```powershell
python playground\nb_check.py
```

---

## 3. Watch things move

Run `examples/pendulum_control.py` and you will see a window appear and vanish
in a fraction of a second. Nothing crashed: the loop condition is
`data.time < 10.0`, and simulated time is not wall-clock time. MuJoCo steps a
simple scene ~40× faster than real time, so ten seconds of simulation finishes
in about 2.9 s and the viewer closes with it. `mujoco.viewer.launch_passive`
offers no `real_time` option, so there is nothing to set in the example itself.

`playground/watch.py` fixes it without editing the upstream file — it wraps
`mj_step` to pace against the wall clock and keeps the window open after the
example's loop exits:

```powershell
python playground\watch.py repos\MuJoCo-Tutorial\examples\pendulum_control.py
```

Measured: 10 s of simulated time now takes 11.3 s of wall time, and
`WATCH_SPEED=0.5` doubles it again.

```powershell
$env:WATCH_SPEED = "0.25"    # quarter speed, for watching contact events
$env:WATCH_HOLD  = "0"       # exit immediately when the example finishes
```

---

## 4. Worked examples

All verified against 4.0.1. "Watch" means run it through `watch.py`.

| Example | Result | What it teaches |
|---|---|---|
| `pendulum_control` | pass | PD control to an angle setpoint |
| `double_pendulum` | pass | Underactuated chaos, no controller |
| `projectile` | pass | Ballistics; ends at x = 8.49 m |
| `contact_forces` | pass | Contact forces settle at box height 0.6199 |
| `inverse_kinematics` | pass | Damped least squares |
| `lqr_control` | pass | Linearising a model, computing gain K |
| `trajectory_tracking` | pass | Following a reference trajectory |
| `hopper` | pass | Gait + state machine on one leg |
| `locomotion_controller` | pass | Walking to a target x |
| `pick_and_place` | pass | A 7-phase manipulation sequence |
| `gymnasium_env` | pass | MuJoCo behind a Gymnasium API; ~950 steps/episode |
| `offscreen_rendering` | pass | Headless video (91 frames) — needs ffmpeg |
| `domain_randomization` | pass | Randomised gravity/mass per episode — needs ffmpeg |
| `impedance_control` | **fails** | → use `playground/fixed/impedance_control.py` |
| `operational_space_control` | **fails** | → use `playground/fixed/operational_space_control.py` |
| `record_playback` | **fails** | → use `playground/fixed/record_playback.py` |
| `menagerie_robot` | no-op | Exits 0 after printing a hint; needs `mujoco_menagerie` cloned |
| `mjx_parallel` | container | Needs `mujoco-mjx`; works on the CPU backend, see below |

Suggested order for the control examples, which builds on itself:
`pendulum_control` → `lqr_control` → `inverse_kinematics` →
`fixed/impedance_control` → `fixed/operational_space_control`.

The `fixed/` copies are the upstream files with the minimum change needed to run
on 4.x, each documenting its own diff:

- `impedance_control`: writes a length-`nv` torque into a length-`nu` `ctrl`.
  The scene has three degrees of freedom — the obstacle's slide joint is dof 0,
  the arm joints are dofs 1 and 2 — so the fix maps each actuator to the dof it
  actually drives rather than taking the first two rows.
- `operational_space_control`: `data.qM` was removed in MuJoCo 3.0 and
  `mj_fullM` now takes the data object directly,
  `mj_fullM(model, data, dst)` with `dst` a 2-D `(nv, nv)` array. It converges
  to 0.0 mm end-effector error.
- `record_playback`: hardcoded `/tmp/...` path.

---

## 5. Two ways to look at a model

**In-browser Studio** — no local GUI code involved, and the mode that works
reliably on 4.0.1:

```powershell
docker compose up studio        # http://localhost:9080/
```

**Native viewer** — works from Python (`launch_passive` opened a window and
completed 315,924 steps in 15 s when tested). Studio's *own* native window does
not work on 4.0.1; see
[notes/0001](notes/0001-mujoco4-studio-native-viewer.md), and use
`scripts\studio.ps1` or `--gfx=web`.

---

## What does not work

Four things in the core path fail on 4.0.1. Each was reproduced, and the cause
is a fact about the library, not about your setup.

**`<composite>` is rejected by the XML parser.** `06_xml_modeling.ipynb` cell 8
builds a rope from a composite primitive and dies with
`Schema violation: unrecognized element Element 'composite'`. This is not the
notebook's syntax: a minimal canonical `<composite type="grid">` is rejected by
4.0.1 as well (`playground/_nb_probe.py` demonstrates it). Skip that cell; the
rest of the notebook is fine.

**`mujoco.mjx` is a separate package.** `07_mjx.ipynb` fails on
`cannot import name 'mjx' from 'mujoco'`. The `mujoco` wheel ships no `mjx`
module at all — MJX lives in `mujoco-mjx`, version-pinned to the engine. The
container's `mjx-shell` target has it installed; locally, `pip install
mujoco-mjx` matching your `mujoco` version.

**A `motor_force` sensor in `09_sensors_and_estimation.ipynb` cell 4 fails** with
`unrecognized name 'motor' of sensorized object`. The sensor references an
actuator by a name the model does not define.

**MJX runs on CPU, not on the GPU here.** With the CUDA backend `mjx.step`
cannot load its kernel at any step count, while ordinary CUDA compute succeeds
in the same process. Forcing the CPU backend gives correct physics — a free
sphere falls to `z=-0.7887` after 200 steps, and
`repos/MuJoCo-Tutorial/examples/mjx_parallel.py` runs 1000 environments × 100
steps:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run -e "JAX_PLATFORMS=cpu" --rm mjx-shell python repos/MuJoCo-Tutorial/examples/mjx_parallel.py
```

```
Simulating 1000 environments in parallel...
Trajectory shape: (1000, 100, 3)  (batch, timesteps, xyz)
```

The `Failed to import warp: No module named 'warp'` lines in that run are not
errors — MJX probes for an optional backend and proceeds. Full investigation,
including the two ways a JAX test can pass while computing nothing:
[notes/0002](notes/0002-gpu-enablement-checklist.md).

---

## Where to go next

The staged path across all three repositories is in
[README — Learning path](README.md#learning-path). The short version: finish
notebooks 00–05 and the control examples above, then pick by interest —
`Tutorial_Mujoco_RL` for reinforcement learning, `mujoco-robotics-usage` for
manipulation and mobile-robot algorithms, `MuJoCo-Tutorial`'s MJX material for
large-scale parallel simulation.

Before starting a new stage, re-run `python scripts\env_check.py`: it names the
exact missing packages rather than leaving you to discover them mid-notebook.
