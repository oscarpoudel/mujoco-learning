# Docker

Everything here runs **headless on CPU**, and that is a deliberate design
forced by this machine, not laziness. The short version:

```powershell
cd path\to\mujoco-learning                                  # wherever you cloned it
docker compose build                                     # once
docker compose run --rm sim python docker/doctor.py       # verify the container
docker compose up studio                                  # -> http://localhost:9080/
docker compose up learn                                   # -> http://localhost:8888
docker compose run --rm shell                             # bash, with repos mounted
```

The project is **bind-mounted** at `/workspace`, so editing a file in VS Code on
Windows takes effect immediately — you never rebuild to iterate. The image itself
copies none of the project; it only installs the toolchain.

---

## What works, verified on this machine

Checked by actually running it, not by reading docs:

| Thing | In container | Evidence |
|---|---|---|
| `import mujoco` | ✅ 4.0.1 | `doctor.py` |
| Physics (`mj_step`) | ✅ identical to Windows | sphere falls to `z=-0.7887` in **both** the container and `robotics_muj` |
| Cart-pole playground | ✅ converges 9.99° → 0.004° | `docker compose run --rm sim` |
| Offscreen rendering | ✅ via `MUJOCO_GL=osmesa` (Mesa software GL) | `(64, 64, 3)` frame, no GPU |
| Video export | ✅ ffmpeg baked in | wrote `playground/first_sim.mp4` back to the Windows mount |
| Studio in browser | ✅ **`http://localhost:9080/`** | HTTP 200, 25,860 bytes, `title='MuJoCo Web Viewer'` |
| Write back to `/mnt/c` | ✅ | container runs as uid 1000, files land on the Windows side |
| `git pull` the repos in-container | ✅ | all three report HEADs: `7cb080d`, `2137f53`, `d6f8710` |
| GPU / CUDA | ❌ **not available** | see below |
| Native windows (`mujoco.viewer`, Studio's own window) | ❌ no X11 | see below |

---

## Why there is no GPU, and no window

Three separate dead ends, all confirmed on this host:

**1. The nvidia runtime is registered but has nothing to attach to.**
`docker info` lists an `nvidia` runtime, so the toolkit is installed. But:

```
$ docker run --rm --gpus all ubuntu:24.04 nvidia-smi
nvidia-container-cli: initialization error: WSL environment detected
                    but no adapters were found
```

**2. The Windows NVIDIA driver doesn't expose CUDA to WSL at all.**

```
$ wsl -d Ubuntu -- nvidia-smi
bash: line 1: nvidia-smi: command not found
$ ls /usr/lib/wsl/lib
libd3d12.so  libd3d12core.so  libdxcore.so       # no libcuda.so.1
```

**3. There is no X server reachable from the Docker VM.**

```
$ docker run -v /tmp/.X11-unix:/tmp/.X11-unix ubuntu:24.04 ls /tmp/.X11-unix
total 4      # empty
```

WSLg 1.0.79 exists in your `Ubuntu` distro, but Docker Desktop runs containers in
its own `docker-desktop` VM, which does not inherit the WSLg socket.

### Consequence: use the browser as your display

`--gfx=web` is the right answer for Docker, and it has a bonus: it also dodges
the [4.0.1 native-viewer bug](notes/0001-mujoco4-studio-native-viewer.md)
because it never touches `native_viewer.py` or Filament server-side. Rendering
happens in your browser via WASM, so no GPU is involved anywhere.

If you want a real native window, run it on the Windows side:

```powershell
.\scripts\studio.ps1 playground\models\ball.xml          # --gfx=opengl_software
```

### Fixing the GPU later

There is a full gated checklist in
[notes/0002-gpu-enablement-checklist.md](notes/0002-gpu-enablement-checklist.md).
The two facts that make it non-generic:

- The blocker is driver **452.56 (CUDA 11.0, Oct 2020)**. CUDA-on-WSL was only
  introduced in driver branch **R470**, so this driver has no WSL GPU component
  to misconfigure.
- **Do not install the latest driver.** **R580 is the last branch supporting
  Maxwell/Pascal/Volta**; R590+ dropped them, and this is a Pascal Quadro P1000.

`docker-compose.gpu.yml` is already written and validated, so once the driver
lands it is `-f docker-compose.yml -f docker-compose.gpu.yml` and nothing else.
One caveat learned the hard way: under `runtime: nvidia` the container starts
**successfully with no GPU**, so verify with
`run --rm sim python docker/doctor.py` and look for
`libcuda visible` rather than trusting a clean startup.

Verified in `mjx-shell`: `mujoco-mjx 4.0.1` (version-matched to the engine),
`mujoco.mjx` imports, `jax 0.10.2 devices=['cpu']`. On startup MJX prints

```
Failed to import warp: No module named 'warp'
Failed to import mujoco_warp: No module named 'warp'
```

That is **harmless** — `mujoco_warp` is an optional alternative backend. MJX
itself runs fine on the CPU. Ignore it.

---

## Services

| Service | What it is | Command |
|---|---|---|
| `studio` | MuJoCo Studio in browser, hot-reload on | `docker compose up studio` → `:9080` |
| `learn` | JupyterLab for the tutorial notebooks | `docker compose up learn` → `:8888` |
| `sim` | one-shot smoke test, exits when done | `docker compose run --rm sim` |
| `shell` | bash with the base image | `docker compose run --rm shell` |
| `rl-shell` | + stable-baselines3 2.9.2, **CPU-only** torch 2.15, ikpy 4.5 (SB3's `[extra]` also brings opencv and networkx) | `docker compose run --rm rl-shell` |
| `mjx-shell` | + JAX (CPU backend) | `docker compose run --rm mjx-shell` |

Verified in-container: `torch 2.15.0+cpu cuda=False`, `stable_baselines3 2.9.2`,
`ikpy 4.5.0`, `cv2 5.1.0`, `networkx 3.6.1`. The MPC demos in
`mujoco-robotics-usage` still need two more that nothing else pulls in:

```bash
docker compose run --rm rl-shell pip install osqp shapely
```

(Persistent across runs only if you commit the container or add them to the
`rl` stage of the Dockerfile and rebuild — a `run` container is discarded.)

`rl-shell` and `mjx-shell` are separate build targets so the base image stays
small. Torch is pulled from the CPU wheel index on purpose — the default PyPI
wheel would add ~2.5 GB of CUDA this machine cannot use.

### Loading a specific robot into Studio

```powershell
# UR5e from the RL tutorial
docker compose run --rm -e MODEL=repos/Tutorial_Mujoco_RL/1tutorial_ur5e/model/universal_robots_ur5e/scene.xml studio

# Franka Panda
docker compose run --rm -e MODEL=repos/Tutorial_Mujoco_RL/3tutorial_panda_obstacle/model/franka_emika_panda/scene.xml studio

# Husky with a lidar, from the robotics repo (note: no scene.xml here, the
# models in that folder are named after what's bolted on)
docker compose run --rm -e MODEL=repos/mujoco-robotics-usage/asset/husky/husky_w_lidar.xml studio
```

`MODEL` is relative to `/workspace`. Hot reload is on, so editing the XML in VS
Code makes the browser page reload itself.

---

## Two things that will bite you

**1. `/mnt/c` is slow.** The project sits on a Windows drive (`drvfs`), which is
orders of magnitude slower for many-small-file access than ext4. The `repos/`
tree is 818 MB, and Jupyter plus Python imports hammer exactly that pattern. If
things feel sluggish, put the code in WSL-native storage:

```bash
# from inside the Ubuntu distro
mkdir -p ~/mujoco-learning && cp -r "/mnt/c/Users/Your Name/Projects/mujoco-learning/." ~/mujoco-learning/
cd ~/mujoco-learning && docker compose up studio
```

**2. Quote Windows paths that contain spaces** (mine does). Docker Compose
handles it, but anything you write by hand must be quoted. Prefer running
compose from inside the project directory so `.` does the work.

---

## Housekeeping

```powershell
docker compose down                    # stop containers
docker compose build --pull --no-cache # full rebuild after Dockerfile changes
docker system df                       # what the images are costing
docker compose run --rm sim python docker/doctor.py    # re-verify anytime
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm mjx-shell python docker/mjx_check.py
```

`mjx_check.py` is the GPU readiness gate; it deliberately fails on CPU-only
compute so it can never report a false pass. See
[notes/0002-gpu-enablement-checklist.md](notes/0002-gpu-enablement-checklist.md).

After a driver update, `scripts/gpu-gates.ps1` walks the whole chain (Windows
driver → WSL → Docker → container `libcuda` → MJX on device), stops at the first
gate that fails, and prints the specific fix. `-From 3` resumes at a given gate;
`-IncludeMjx` adds the slow JAX check.

`doctor.py` runs on **both** the Windows side and in the container and prints
the same check list, so it is the fastest way to compare the two environments:

```powershell
& "$HOME\miniconda3\envs\robotics_muj\python.exe" docker\doctor.py
```

The Studio web viewer has **no authentication** — it says so in its own startup
banner. Docker Desktop publishes it on `localhost` only, which is fine, but
don't add `--network host` or firewall-forward 9080 on a campus network.
