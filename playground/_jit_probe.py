"""Isolate whether XLA-compiled kernels can load on this GPU at all.

The matmul in mjx_check.py goes through cuBLAS (precompiled vendor library), so
it succeeding proves less than it looks like. MJX fails on `jit_squeeze`, which
is XLA codegen. This probes the difference.
"""
import jax
import jax.numpy as jnp

print("device:", jax.devices()[0], "platform:", getattr(jax.devices()[0], "platform", "?"))


def probe(name, fn):
    try:
        out = fn()
        print(f"  [ok  ] {name:34s} -> {out}")
    except Exception as exc:
        msg = str(exc).splitlines()[0][:120]
        print(f"  [FAIL] {name:34s} -> {type(exc).__name__}: {msg}")


x = jnp.ones((4, 4))
probe("elementwise (jit add)", lambda: float(jax.jit(lambda a: a + 1.0)(x).sum()))
probe("reduce (jit sum axis)", lambda: float(jax.jit(lambda a: a.sum(axis=0))(x)[0]))
probe("squeeze (the MJX failure)", lambda: float(jax.jit(lambda a: a.squeeze())(jnp.ones((1, 3))).sum()))
probe("transpose+matmul fused", lambda: float(jax.jit(lambda a: (a.T @ a).sum())(x)))
probe("scan (control-flow codegen)",
      lambda: float(jax.lax.scan(lambda c, _: (c + 1.0, None), 0.0, None, length=4)[0]))
probe("while_loop", lambda: float(jax.lax.while_loop(lambda c: c < 4.0, lambda c: c + 1.0, 0.0)))
