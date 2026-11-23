"""Offscreen render benchmark: software (osmesa) vs "hardware" (egl).

    # software
    docker compose run --rm sim python playground/render_bench.py

    # with the GPU override (notes/0002 gate 4)
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
      run --rm sim python playground/render_bench.py

Options: --size 1920x1080 --frames 60 --geoms 600

Result on this machine (Quadro P1000, driver 582.99, Docker Desktop/WSL2):
**egl is not faster, because it is not the GPU.** With 600 overlapping spheres
the two agree to within 5% (2.0 vs 1.9 fps at 1280x720), and no
libEGL_nvidia.so.0 exists inside the container at all. Docker Desktop's WSL2 GPU
support injects CUDA only; container GL stays Mesa llvmpipe. Full write-up and
the ldconfig evidence: notes/0002-gpu-enablement-checklist.md, Gate 6.

The methodology trap is the instructive part. At one geom, both backends are
bound by fixed per-frame overhead (context sync + framebuffer readback), so
throughput *rises* with frame size -- 5.4 -> 18.9 Mpix/s -- which no rasteriser
can do. A benchmark in that regime cannot tell a GPU from a CPU. Use --geoms to
make rasterisation the cost you are actually measuring.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import mujoco
import numpy as np

# Overlapping spheres spread across the frame. {bodies} is filled by build_xml().
XML_TEMPLATE = """
<mujoco>
  <visual>
    <global offwidth="{w}" offheight="{h}"/>
  </visual>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1"/>
    <geom type="plane" size="30 30 .1" rgba=".8 .8 .8 1"/>
    {bodies}
  </worldbody>
</mujoco>
"""

W, H, FRAMES, GEOMS = 640, 480, 30, 1


def parse_args(argv: list[str]) -> tuple[int, int, int]:
    """--size 1920x1080 --frames 60 --geoms 600"""
    w, h, n = W, H, FRAMES
    i = 0
    while i < len(argv):
        if argv[i] == "--size" and i + 1 < len(argv):
            w, h = (int(v) for v in argv[i + 1].lower().split("x"))
            i += 2
        elif argv[i] == "--frames" and i + 1 < len(argv):
            n = int(argv[i + 1])
            i += 2
        elif argv[i] == "--geoms" and i + 1 < len(argv):
            globals()["GEOMS"] = int(argv[i + 1])
            i += 2
        else:
            i += 1
    return w, h, n


def build_xml(w: int, h: int) -> str:
    """Fill the frame with overlapping spheres so rasterisation, not readback,
    dominates the measurement. With --geoms 1 a 1920x1080 frame is cheap for a
    CPU rasteriser and tells you nothing; with 600 it is not."""
    bodies = []
    side = int(math.ceil(math.sqrt(GEOMS)))
    for k in range(GEOMS):
        x = (k % side) * 0.28 - side * 0.14
        y = (k // side) * 0.28 - side * 0.14
        bodies.append(f'<body pos="{x:.2f} {y:.2f} 0.15">'
                      f'<geom type="sphere" size="0.22" rgba=".3 .6 .9 1"/></body>')
    return XML_TEMPLATE.format(w=w, h=h, bodies="\n    ".join(bodies))


def main() -> int:
    w, h, frames = parse_args(sys.argv[1:])
    backend = os.environ.get("MUJOCO_GL", "<unset>")
    print(f"MUJOCO_GL={backend}  platform={platform.system()}  "
          f"{w}x{h} x {frames} frames x {GEOMS} geoms")

    m = mujoco.MjModel.from_xml_string(build_xml(w, h))
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 14.0, 45, -25

    # Warm up first: the first frame pays for context creation and shader
    # compilation, which would otherwise dominate the measurement.
    renderer.update_scene(d, camera=cam)
    renderer.render()

    t0 = time.perf_counter()
    for _ in range(frames):
        mujoco.mj_step(m, d)
        renderer.update_scene(d, camera=cam)
        frame = renderer.render()
    dt = time.perf_counter() - t0

    renderer.close()
    print(f"  {dt:.3f}s  ->  {frames / dt:6.1f} fps   frame dtype={frame.dtype} shape={frame.shape}")
    print(f"  non-black pixels: {int((frame.astype(np.int32).sum(axis=2) > 30).sum())} / {w * h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
