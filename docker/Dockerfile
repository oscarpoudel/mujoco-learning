# syntax=docker/dockerfile:1
#
# MuJoCo Learning Hub -- headless CPU container.
#
# WHY HEADLESS: on this host there is no GPU path into Docker at all. The nvidia
# runtime is registered, but nvidia-container-cli reports "WSL environment
# detected but no adapters were found", and the Windows NVIDIA driver exposes
# only /usr/lib/wsl/lib/libd3d12.so to WSL -- no libcuda, no nvidia-smi. The
# Docker Desktop VM also has an empty /tmp/.X11-unix, so there is no X11 either.
#
# So rendering is done two ways, and neither needs a GPU:
#   1. MuJoCo Studio --gfx=web  -> served on a port, rendered in your BROWSER
#   2. MUJOCO_GL=osmesa         -> offscreen frames via software Mesa (llvmpipe)
#
# Native windows (mujoco.viewer, Studio's native window) CANNOT appear from this
# container. Run those on the Windows side; see ../scripts/studio.ps1.
#
# BUILD TARGETS
#   base  physics + jupyter + gymnasium          (~1.5 GB)
#   rl    + stable-baselines3, torch (CPU), ikpy (~2.5 GB)
#   mjx   + jax[cuda12] for MJX                  (~+600 MB over CPU jax)
#
# GPU: pass the nvidia runtime (see docker-compose.gpu.yml). Without it the
# container still builds and runs, and MJX quietly uses the CPU.
#
# docker build -f docker/Dockerfile --target base -t mujoco-learning .

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Debian 12. Bookworm has current-enough Mesa for llvmpipe offscreen GL.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    # Software GL. Switch to egl only if a real GPU is ever passed through.
    MUJOCO_GL=osmesa \
    # No display for matplotlib inside the container.
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1

# Runtime libraries:
#   libosmesa6, libgl1, libegl1, libgles2  -> MuJoCo offscreen rendering
#   libglfw3 + the X libs                   -> GLFW (present, but no display here)
#   libgomp1, libglib2.0-0                  -> numeric/GLib deps
#   ffmpeg                                    -> mediapy video export
#   git, curl, ca-certificates, procps, vim   -> usable shell + git-pull the repos
#   tini                                      -> correct PID 1 signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
      libosmesa6 \
      libgl1 \
      libegl1 \
      libgles2 \
      libglu1-mesa \
      libglfw3 \
      libgomp1 \
      libglib2.0-0 \
      libx11-6 \
      libxinerama1 \
      libxrandr2 \
      libxcursor1 \
      libxi6 \
      libxxf86vm1 \
      ffmpeg \
      git \
      git-lfs \
      curl \
      wget \
      ca-certificates \
      procps \
      less \
      vim-tiny \
      tini \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Python stack. Kept explicit rather than `-r requirements.txt` so the layer
# cache survives edits to the tutorial repos, which are bind-mounted anyway.
RUN pip install \
      mujoco \
      "gymnasium[mujoco]" \
      numpy \
      scipy \
      matplotlib \
      mediapy \
      ipywidgets \
      jupyterlab \
    && python -c "import mujoco; print('built with mujoco', mujoco.__version__)"

# Non-root user. UID 1000 matches the default WSL/Linux user; override with
# --build-arg UID=$(id -u) if your mount needs it. Note that files written back
# to /mnt/c land on a Windows (drvfs) mount, where ownership is decided by the
# mount options rather than by chown.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} mujoco \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash mujoco \
    && mkdir -p /workspace \
    && chown -R mujoco:mujoco /workspace

USER mujoco
WORKDIR /workspace

# Pin the git identity so `git pull` / commits inside the container don't fail.
# Override these with -e GIT_USER_NAME / -e GIT_USER_EMAIL or a global config.
RUN git config --global --add safe.directory '*' \
    && git config --global user.name  "${GIT_USER_NAME:-MuJoCo Learning}" \
    && git config --global user.email "${GIT_USER_EMAIL:-mujoco@localhost}"

# Studio's web viewer (9080-9082) and JupyterLab (8888).
EXPOSE 9080-9082 8888

# tini reaps zombies and forwards SIGINT so Ctrl+C closes the viewer cleanly.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

# Cheap liveness probe only; the real smoke test is `docker compose run --rm sim`.
# Kept quote-free on purpose -- nested quotes in shell-form CMD parse badly.
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
  CMD python -c "import mujoco; print(mujoco.__version__)"

# ---------------------------------------------------------------------------
# RL layer: Stable-Baselines3 + CPU-only torch + the extras Tutorial_Mujoco_RL
# needs. Torch comes from the CPU wheel index on purpose -- the default PyPI
# wheel drags in ~2.5 GB of CUDA that this host cannot use.
FROM base AS rl

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "stable-baselines3[extra]" ikpy transforms3d pandas seaborn \
    && python -c "import torch, stable_baselines3; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

CMD ["bash"]

# ---------------------------------------------------------------------------
# MJX layer: JAX with the CUDA 12 backend.
#
# The extra MUST be cuda12, never cuda13. The cuda13 wheels require compute
# capability 7.5 (Turing); this host's Quadro P1000 is 6.1, so cuda13 installs
# cleanly and then silently refuses to use the GPU. cuda12 wants driver R525+,
# verified here against 582.99 (Release 580 Update 12). Costs ~600 MB over the
# CPU wheel.
#
# To go back to CPU-only (smaller image, no driver needed): "jax" instead of
# "jax[cuda12]". Physics is identical either way.
FROM base AS mjx

# mujoco-mjx is the actual MJX package -- jax/jaxlib alone do NOT give you
# `import mujoco.mjx`. Version is pinned to match the engine already installed.
# Only importability is checked at build time: a build container has no GPU, so
# jax.devices() here would always print CPU. Device visibility is verified at
# run time by docker/mjx_check.py.
RUN pip install "mujoco-mjx==4.0.1" "jax[cuda12]" \
    && python -c "import jax, mujoco.mjx; print('jax', jax.__version__, '+ mjx import ok')"

CMD ["bash"]
