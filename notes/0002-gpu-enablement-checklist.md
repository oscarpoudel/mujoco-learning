# 0002 — Enabling GPU support (checklist)

**Target:** Quadro P1000 (Pascal, compute capability 6.1, 4 GB) on build 26300,
Docker Desktop 29.1.3 + WSL 3.0.5 / WSLg 1.0.79.

**Do these in order.** Every step has a gate: do not continue until the previous
gate passes, or you will be debugging three changes at once.

`scripts/gpu-gates.ps1` runs these gates for you and stops at the first failure
with the fix to apply. It is read-only — nothing in it installs or changes
anything. Use it after each step to confirm before moving on:

```powershell
.\scripts\gpu-gates.ps1              # from the top
.\scripts\gpu-gates.ps1 -From 3      # resume at a gate after fixing something
.\scripts\gpu-gates.ps1 -IncludeMjx  # also check JAX/MJX on the device (slow)
```

Today it reports `[FAIL] gate 1` and stops, which is correct.

---

## Root cause, established by measurement

Your driver is **452.56, CUDA 11.0, dated 2020-10-18**:

```
$ nvidia-smi
| NVIDIA-SMI 452.56    Driver Version: 452.56    CUDA Version: 11.0 |
|   0  Quadro P1000  WDDM | 00000000:01:00.0  On |  2573MiB / 4096MiB |
```

**CUDA-on-WSL was introduced in driver branch R470 (August 2021).** R452 predates
it entirely, so the WSL GPU component is not broken or misconfigured — it does not
exist. That is why `/usr/lib/wsl/lib` contains only `libd3d12.so`,
`libd3d12core.so`, `libdxcore.so`, and no `libcuda.so.1`.

Everything downstream fails from this one cause:

| Layer | Symptom |
|---|---|
| Windows driver | 452.56, no WSL adapter support |
| WSL distro | `nvidia-smi: command not found`, no `libcuda.so.1` |
| Docker `--gpus all` | `nvidia-container-cli: WSL environment detected but no adapters were found` |
| Docker `runtime: nvidia` | Container **starts** but has no `libcuda` — fails silently |
| JAX | `A CUDA-enabled jaxlib is not installed. Falling back to cpu.` |
| MuJoCo `MUJOCO_GL=egl` | Succeeds, but via **Mesa llvmpipe (software)** — not the GPU |

⚠️ That `runtime: nvidia` row matters: the container comes up fine and looks
healthy while having no GPU. Never trust "the container started". `doctor.py`
loads `libcuda.so.1` with `ctypes` for exactly this reason; `/usr/local/nvidia`
existing proves nothing.

---

## Gate 0 — Before touching anything (5 min)

Checked on this machine, so you don't have to wonder:

- [x] **You have the rights to do this.** `whoami /groups` shows
      `BUILTIN\Administrators ... Group used for deny only` — that is a local
      admin account running with a UAC-filtered token, i.e. elevation is
      available and a driver install will just prompt. The machine is in
      `WORKGROUP`, not domain-joined, so there is no group policy blocking it.
      Just right-click the installer → **Run as administrator**.
- [ ] **Rollback plan.** Do **not** plan on reinstalling 452.56 — every usual
      `download.nvidia.com` path for it returns 404 today (checked). Two better
      options: Device Manager → Display adapters → Properties → Driver →
      **Roll Back Driver**, or reinstall an earlier 58x build, since the whole
      R580 branch supports this card. There is no need to go back to a 2020
      driver: R580 fixes what is broken here and is the branch NVIDIA maintains
      for Pascal.
- [ ] **Note free VRAM.** 2573 MiB of 4096 MiB is already consumed by the
      desktop. Practical headroom for GPU compute is ~1.4 GB, which constrains
      MJX batch sizes. Close browsers before benchmarking.
- [ ] Confirm nothing is running: `docker compose ps -a` (should print nothing).

---

## Gate 1 — Install the right driver (the one real decision)

**R580 is the last NVIDIA driver branch that supports Maxwell, Pascal and
Volta.** Straight from this driver's own release notes, §2.1.4.2:

> "NVIDIA RTX Enterprise Driver Release 580 will be the last branch to support
> Maxwell, Pascal, and Volta GPUs."

NVIDIA's
[branch history](https://www.nvidia.com/en-us/drivers/rtx-enterprise-and-quadro-driver-branch-history)
shows R580 is still the maintained Production Branch (security updates through
October 2028), with newer branches R595 and R620 alongside it — and **those
dropped this card**. So: **do not install the newest driver.**

### The exact driver to install

**582.99 = Release 580 Update 12 (21-Oct-2026)** — the newest build that still
supports Pascal.

```
https://us.download.nvidia.com/Windows/Quadro_Certified/582.99/582.99-quadro-rtx-desktop-notebook-win10-win11-64bit-international-dch-whql.exe
```

Verified just now: the URL returns HTTP 200, **689.8 MB**, Quadro/RTX Enterprise
combined desktop + notebook package, Windows 10/11 64-bit. I also extracted the
text of the matching release notes and confirmed the supported-products list
names your card under Pascal:

```
... NVIDIA Quadro P2200  NVIDIA Quadro P2000  NVIDIA Quadro P1000  NVIDIA Quadro P600 ...
```

Details page, if you prefer clicking to a link:
<https://www.nvidia.com/en-us/drivers/details/282544/>

### Install

- [ ] Save your work and expect the screen to flicker for a minute or so.
- [ ] Run the installer → **Custom** → tick **Perform a clean installation**.
- [ ] If you drive updates from **NVIDIA App**: its Game Ready / Studio channels
      are R595+ and **will drop this card**. Switch it to the
      **RTX Enterprise / Quadro — Production Branch (R580)** driver type first,
      or just run the installer above and disable auto-update afterwards.
- [ ] If the install fails or R452 refuses to be replaced: DDU (Display Driver
      Uninstaller) in Safe Mode, then reinstall.
- [ ] Reboot.

**Gate:**

```powershell
.\scripts\gpu-gates.ps1        # or just: nvidia-smi
```

`nvidia-smi` must print `Driver Version: 582.99` (any 58x is fine) and
`CUDA Version: 12.x`. If it still says 452.56, the install did not take — stop
and fix that before continuing.

---

## Gate 2 — Make WSL see the GPU

WSL mounts the driver into the distro at start, so a running distro keeps the old
one. This is the step people skip.

- [ ] `wsl --shutdown`
- [ ] `wsl --update` (optional — you are on 3.0.5.0, which is fine)
- [ ] Start Ubuntu again.

**Gate:**

```bash
nvidia-smi
ls /usr/lib/wsl/lib | grep -i cuda
```

`nvidia-smi` must list the Quadro P1000, and the second command must show
`libcuda.so.1` (usually also `libcublas`, `libcudnn` when the CUDA runtime
packages are included).

If `nvidia-smi` is still missing, check that no WSL instance is alive:

```powershell
wsl --list --verbose      # every STATE must be Stopped
wsl --shutdown
```

---

## Gate 3 — Make Docker see the GPU

Docker Desktop runs its own `docker-desktop` WSL VM, which also has to pick up
the new driver.

- [ ] Quit Docker Desktop **fully** (tray → Quit, not just close windows).
- [ ] `wsl --shutdown`, then start Docker Desktop.

**Gate:**

```powershell
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

Must print the P1000 table. This is the same command that failed before with
`no adapters were found`.

If it still fails:

- Docker Desktop → Settings → Resources → **WSL Integration** → enable your
  `Ubuntu` distro.
- `docker info --format '{{.Runtimes}}'` should still list `nvidia`.
- Update Docker Desktop (29.1.3 is current-ish; the NVIDIA Container Toolkit it
  bundles must be ≥ v2.6.0 / libnvidia-container ≥ 1.5.1 for current CUDA).
- Confirm the driver is actually loaded on Windows: Device Manager → Display
  adapters → NVIDIA Quadro P1000 → no ⚠️, driver version 580.xx.

---

## Gate 4 — Confirm the container has CUDA

The override file `docker-compose.gpu.yml` already exists in this project and is
validated (`docker compose -f docker-compose.yml -f docker-compose.gpu.yml config`
passes). It sets `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES=all`,
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`, and `MUJOCO_GL=egl` on
every service. The `graphics` capability is what makes GL/EGL work, not just CUDA.

**Gate:**

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm sim python docker/doctor.py
```

The `gpu` line must read **`libcuda visible, 1 CUDA device(s)`** instead of
`libcuda.so.1 not found -> CPU only`. Everything else should stay green.

---

## Gate 5 — Actually use it for MJX

The `mjx` image target currently installs **plain `jax`**, which is CPU-only —
that is why you see `Falling back to cpu`. One line to change, and the extra
matters:

| Extra | Min driver | Min compute capability | P1000 (6.1)? |
|---|---|---|---|
| `jax[cuda13]` | 580 | **7.5 (Turing)** | ❌ **not supported** |
| `jax[cuda12]` | 525 | 5.2 (Maxwell) | ✅ works |
| plain `jax` | — | — | CPU only |

So use `cuda12`. Never `cuda13` — it would install successfully and then refuse
to use the GPU.

In `docker/Dockerfile`, in the `mjx` stage, change:

```dockerfile
RUN pip install mujoco-mjx jax jaxlib \
```

to:

```dockerfile
RUN pip install "mujoco-mjx==4.0.1" "jax[cuda12]" \
```

(`mujoco-mjx` must stay pinned to the engine version, currently 4.0.1. The
cuda12 wheels add roughly 600 MB, which is why this is opt-in and not the
default.)

Then rebuild and verify:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build mjx-shell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm mjx-shell python docker/mjx_check.py
```

**Gate** — `docker/mjx_check.py` checks four things in order, each of which can
fail independently:

```
[ok  ] libcuda.so.1               1 CUDA device(s) visible
[ok  ] jax.devices()              [CudaDevice(id=0)] kinds=['Quadro P1000']
[ok  ] gpu matmul                 sum=... on Quadro P1000
[ok  ] mjx.step on device         sphere fell to z=-0.7887

GPU PATH READY
```

If it prints `GPU PATH NOT READY`, the line that failed tells you which layer is
still broken.

---

## Gate 6 — Hardware rendering (optional)

With the driver fixed, `MUJOCO_GL=egl` in the gpu override becomes real hardware
EGL rather than llvmpipe software. Confirm by comparing render timing:

```powershell
docker compose run --rm sim python docker/doctor.py                                    # osmesa, software
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm sim python docker/doctor.py   # egl, hardware
```

Both should pass; the egl run should be visibly faster. Note this does **not**
fix Studio's native window — that is the separate
[4.0.1 code bug](0001-mujoco4-studio-native-viewer.md), and no driver fixes it.
Keep using `--gfx=web`.

---

## Rollback

Worst case is milder than it sounds: if a newly installed driver fails to
initialise, Windows falls back to the basic display driver at next boot rather
than leaving you with a black screen, and the installer itself will offer to
revert if the swap fails verification.

1. Device Manager → Display adapters → NVIDIA Quadro P1000 → Properties → Driver
   → **Roll Back Driver**.
2. If that is greyed out: DDU in Safe Mode, then install an **earlier R580 build**
   (any 58x supports this card). Don't try to hunt down 452.56 — its download
   links are dead.
3. `wsl --shutdown`, restart Docker Desktop.
4. Nothing in this project needs reverting except, if you applied it, the one
   Dockerfile line in the `mjx` stage.

---

## What you get, and the honest limits

| | |
|---|---|
| **Gained** | GPU MJX and GPU-side JAX training; hardware EGL for offscreen rendering; real CUDA in containers for anything else |
| **Not fixed** | Studio's native window (separate 4.0.1 bug); the `/mnt/c` slowness (that's the Windows filesystem, not the GPU) |
| **VRAM** | 4 GB total, ~2.5 GB already used by the desktop. MJX batch sizes must stay small — `nvidia-smi` inside the container while a run is going tells you the truth fast |
| **WSL overhead** | Roughly 80–95% of bare-metal CUDA throughput. No GPUDirect, no multi-GPU NCCL. Fine for learning and single-GPU RL, not for large training |
| **Deprecation** | Pascal is being wound down: CUDA 13 already excludes it, and JAX/MJX will follow. The working combo is pinned — **R580 + jax[cuda12]** — so avoid `pip install -U jax` and driver "upgrades" past 580.x once it works |

## If the driver update is not possible

Nothing here is wasted. MJX still works on CPU in `mjx-shell`, physics is
identical (verified: `z=-0.7887` on every path), and the only real loss is
throughput. For a GPU when you need one, Colab's free tier runs the same MJX
code, and `MuJoCo-Tutorial` has a Colab button on every notebook.
