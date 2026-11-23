r"""Run any MuJoCo example at real time, and keep the window open when it ends.

Two things make the upstream examples hard to learn from on a fast machine:

1. **They run about 40x faster than real time.** ``mujoco.viewer.launch_passive``
   has no ``real_time`` option, and the examples loop
   ``while viewer.is_running() and data.time < 10.0``. Simulated time is not
   wall-clock time, so ten seconds of sim finishes in a quarter second: the
   window flashes up and closes. Nothing is wrong, you just missed it.
2. **They close the window the instant the loop ends**, so there is no chance to
   look at the final state or inspect it in the UI.

This wraps both without editing the example: it patches ``mujoco.mj_step`` to
sleep so wall-clock time tracks simulated time, and wraps the viewer context
manager to hold the window open after the script's loop exits.

    python playground\watch.py repos\MuJoCo-Tutorial\examples\pendulum_control.py

Environment variables:

    WATCH_SPEED   wall-clock multiplier. 1 = real time, 0.25 = quarter speed.
    WATCH_HOLD    0 to exit as soon as the example finishes (default: hold).
    WATCH_MAX     seconds of wall time before this runner gives up (default 0,
                  no limit). Useful in non-interactive sessions.

Original examples are Manan Tayal's, from MuJoCo-Tutorial (MIT). This file does
not modify them; it only re-runs them with the two patches applied.
"""

import os
import runpy
import sys
import time

import mujoco
import mujoco.viewer

SPEED = float(os.environ.get("WATCH_SPEED", "1.0"))
HOLD = os.environ.get("WATCH_HOLD", "1") not in ("0", "", "false", "no")
MAX_SECS = float(os.environ.get("WATCH_MAX", "0"))

if len(sys.argv) < 2:
    sys.exit("usage: python playground/watch.py <example.py> [args...]")

target, *rest = sys.argv[1:]
if not os.path.exists(target):
    sys.exit(f"no such script: {target}")

# --- 1. pace mj_step so simulated time tracks wall time ----------------------
_real_step = mujoco.mj_step
_clock = {}


def _paced_step(model, data):
    _real_step(model, data)
    now = time.perf_counter()
    if "t0" not in _clock:
        _clock["t0"] = now
        _clock["sim0"] = data.time
        return
    due = _clock["t0"] + (data.time - _clock["sim0"]) / SPEED
    delay = due - now
    if delay > 0:
        time.sleep(delay)
    elif delay < -2.0:
        # Fell more than 2 s behind (window drag, a long contact solve, a break):
        # re-anchor instead of spinning to catch up.
        _clock["t0"], _clock["sim0"] = now, data.time


mujoco.mj_step = _paced_step

# --- 2. keep the window open after the example's loop exits ------------------
_real_launch = mujoco.viewer.launch_passive
_started = time.perf_counter()


class _HeldHandle:
    """Delegates to the real viewer Handle, delaying context exit."""

    def __init__(self, handle):
        self._handle = handle

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def is_running(self):
        if not self._handle.is_running():
            return False
        if MAX_SECS and time.perf_counter() - _started > MAX_SECS:
            print("[watch] WATCH_MAX reached, closing", flush=True)
            return False
        return True


class _HeldContext:
    def __init__(self, ctx):
        self._ctx = ctx
        self._handle = None

    def __enter__(self):
        self._handle = self._ctx.__enter__()
        return _HeldHandle(self._handle) if self._handle is not None else None

    def __exit__(self, *exc):
        if HOLD and self._handle is not None and self._handle.is_running():
            print("[watch] example finished - window held open, close it to exit",
                  flush=True)
            while self._handle.is_running():
                time.sleep(0.1)
        return self._ctx.__exit__(*exc)


def _held_launch(model, data, **kwargs):
    return _HeldContext(_real_launch(model, data, **kwargs))


mujoco.viewer.launch_passive = _held_launch

print(f"[watch] {target}  speed={SPEED}x  hold={HOLD}", flush=True)
sys.argv = [target, *rest]
runpy.run_path(target, run_name="__main__")
print(f"[watch] done after {time.perf_counter() - _started:.1f}s wall", flush=True)
