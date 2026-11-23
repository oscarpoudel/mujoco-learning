# 0001 — MuJoCo 4.0.1 Studio: native viewer crashes with `Target not found:`

**Date:** 2026-11-22
**Affects:** `mujoco-studio` 4.0.1, Windows 11, Python 3.13
**Status:** upstream bug, worked around locally — not a driver or hardware problem

## Symptom

Any attempt to open MuJoCo Studio in its native window mode dies on the first
frame:

```
File ".../mujoco/studio/native_viewer.py", line 214, in sync
    self._renderer.render([scene_view, ux_view])
File ".../mujoco/rendering/filament/renderer.py", line 200, in render
    raise ValueError(f"Target not found: {view.target}")
ValueError: Target not found:
```

Note the empty target name after the colon. The process then unwinds through
`viewer_handle.run_sim_loop` and exits.

## Root cause

In `mujoco/studio/native_viewer.py`, `sync()` allocates the output buffer
conditionally:

```python
pixels = None
if _viewer.render_to_pixels(self._window.GetGraphicsMode()):
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    self._renderer.target('out', pixels)      # <-- registers the 'out' target
else:
    width = int(width * scale)
    height = int(height * scale)
```

Both views then name their target with the same expression:

```python
scene_view = renderer.View(scene='scn', target='out' if pixels is not None else '', ...)
ux_view    = renderer.View(scene='ux',  target='out' if pixels is not None else '', ...)
```

In hardware graphics modes `render_to_pixels()` returns false, so `pixels` stays
`None`, the `'out'` target is never registered, and both views are built with
`target=''`. `Renderer.render()` looks the name up in `self._reads`, misses, and
raises. There is no code path that ever registers a window-backed target.

## Evidence that it is the code path, not the GPU

- `--gfx=opengl` (the default) and `--gfx=vulkan` fail with the **identical**
  Python-level traceback, so no graphics API is implicated.
- `--gfx=opengl_software` takes the `render_to_pixels()` branch, so the target
  *is* registered, and it runs indefinitely with a real top-level window
  (`MainWindowHandle` present).
- `--gfx=web` never touches `native_viewer.py` and runs fine.
- The classic OpenGL renderer is healthy: `mujoco.Renderer` produced a correct
  64×64×3 offscreen buffer, and `python -m mujoco.viewer` opened without error.
- 4.0.1 is the newest release on PyPI, so there is no newer wheel to move to.

Hardware: NVIDIA Quadro P1000, driver 27.21.14.5256 (Sep 2020). Old driver, but
irrelevant — the failure happens before any driver call. Worth updating anyway.

## Workarounds

**1. Software OpenGL, native window** — real desktop window, avoids the bug:

```powershell
python -m mujoco.studio.viewer --model=my.xml --gfx=opengl_software
```

**2. Web viewer** — bypasses Filament and desktop windowing entirely, serves a
browser UI:

```powershell
python -m mujoco.studio.viewer --model=my.xml --gfx=web
# → http://localhost:8081/
```

⚠️ Web mode prints its own warning: no authentication or encryption on the port.
On a shared or campus network, don't leave it running unattended and don't
expose the port beyond `localhost`.

`scripts/studio.ps1` in this folder wraps workaround 1 and takes the graphics
mode as a flag.

## Also worth remembering

- MuJoCo 4.x ships **no `mujoco` CLI binary**. `mujoco simulate model.xml` from
  the 2.x/3.x C++ distribution does not exist. The only console script installed
  is `mujoco-studio.exe`. For the classic viewer use `python -m mujoco.viewer`.
- MuJoCo 4 split the Python package into `mujoco` + `mujoco-core` +
  `mujoco-studio` (+ `mujoco-studio-wasm`), all pinned to the same version.
  A partial install breaks imports. `__init__.py` raises a helpful message and
  suggests `pip install --force-reinstall --no-deps mujoco mujoco-core mujoco-studio`.
- The physics engine itself is fine: `mj_step` and offscreen rendering both
  verified on this machine in the `robotics_muj` env.

## Reproduction

```powershell
conda activate robotics_muj
python -m mujoco.studio.viewer --model=repos\Tutorial_Mujoco_RL\1tutorial_ur5e\model\universal_robots_ur5e\scene.xml
```

Crashes within ~2 seconds. Add `--gfx=opengl_software` and it stays up.
