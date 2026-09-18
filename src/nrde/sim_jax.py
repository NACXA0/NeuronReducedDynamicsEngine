"""Optional JAX Euler for offline f-I scans (vmap over I, lax.scan over time).

Not used for online GPU SpMV. Guarded import: core does not require JAX.
"""

from __future__ import annotations

import numpy as np

from nrde.sim_kernels import PackedModel


def jax_available() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def firing_rates_jax(
    packed: PackedModel,
    I_grid: np.ndarray,
    t_total: float,
    dt: float,
    window: float,
) -> np.ndarray:
    """Batch firing rates (Hz). Requires the optional ``jax`` extra."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    n = int(np.round(t_total / dt))
    I_grid = np.ascontiguousarray(I_grid, dtype=np.float64)
    if n <= 0 or I_grid.size == 0:
        return np.zeros(I_grid.size, dtype=np.float64)

    t_discard = max(0.0, float(t_total) - float(window))
    window_s = max(float(window) / 1000.0, 1e-12)
    model_id = int(packed.model_id)
    p = jnp.asarray(packed.p, dtype=jnp.float64)
    y0 = jnp.zeros(4, dtype=jnp.float64).at[: packed.n_state].set(jnp.asarray(packed.y0, dtype=jnp.float64))
    vth = jnp.float64(packed.vth)
    t_ref = jnp.float64(packed.t_ref)
    hybrid = bool(packed.hybrid)
    vreset = jnp.float64(packed.vreset)
    dt64 = jnp.float64(dt)
    t_discard64 = jnp.float64(t_discard)

    def rhs(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        if model_id == 0:
            dv = (-p[1] * (y[0] - p[2]) + I) / p[0]
            return jnp.array([dv, 0.0, 0.0, 0.0], dtype=jnp.float64)
        if model_id == 1:
            x = jnp.clip((y[0] - p[3]) / p[4], -20.0, 20.0)
            exp_term = p[1] * p[4] * jnp.exp(x)
            dv = (-p[1] * (y[0] - p[2]) + exp_term + I) / p[0]
            return jnp.array([dv, 0.0, 0.0, 0.0], dtype=jnp.float64)
        if model_id == 2:
            x = jnp.clip((y[0] - p[3]) / p[4], -20.0, 20.0)
            exp_term = p[1] * p[4] * jnp.exp(x)
            dv = (-p[1] * (y[0] - p[2]) + exp_term - y[1] + I) / p[0]
            dw = (p[5] * (y[0] - p[2]) - y[1]) / p[6]
            return jnp.array([dv, dw, 0.0, 0.0], dtype=jnp.float64)
        if model_id == 3:
            dv = 0.04 * y[0] * y[0] + 5.0 * y[0] + 140.0 - y[1] + I
            du = p[0] * (p[1] * y[0] - y[1])
            return jnp.array([dv, du, 0.0, 0.0], dtype=jnp.float64)
        v, m, h, n_g = y[0], y[1], y[2], y[3]
        i_na = p[1] * (m**3) * h * (v - p[4])
        i_k = p[2] * (n_g**4) * (v - p[5])
        i_l = p[3] * (v - p[6])
        dv = (I - i_na - i_k - i_l) / p[0]
        d = v + 40.0
        am = jnp.where(jnp.abs(d) < 1e-6, 1.0, 0.1 * d / (1.0 - jnp.exp(-d / 10.0)))
        bm = 4.0 * jnp.exp(-(v + 65.0) / 18.0)
        ah = 0.07 * jnp.exp(-(v + 65.0) / 20.0)
        bh = 1.0 / (1.0 + jnp.exp(-(v + 35.0) / 10.0))
        d2 = v + 55.0
        an = jnp.where(jnp.abs(d2) < 1e-6, 0.1, 0.01 * d2 / (1.0 - jnp.exp(-d2 / 10.0)))
        bn = 0.125 * jnp.exp(-(v + 65.0) / 80.0)
        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n_g) - bn * n_g
        return jnp.array([dv, dm, dh, dn], dtype=jnp.float64)

    def reset(y: jnp.ndarray) -> jnp.ndarray:
        if model_id == 2:
            return y.at[0].set(p[9]).at[1].add(p[7])
        if model_id == 3:
            return y.at[0].set(p[2]).at[1].add(p[3])
        if model_id == 4:
            return y
        return y.at[0].set(vreset)

    def one_I(I: jnp.ndarray) -> jnp.ndarray:
        def body(carry, k):
            y, v_prev, ref_left, n_spk = carry
            t = k.astype(jnp.float64) * dt64
            in_ref = jnp.bool_(hybrid) & (ref_left > 0.0)
            dy = rhs(y, I)
            y_e = y + dt64 * dy
            y_ref = y_e.at[0].set(vreset)
            ref_left_ref = jnp.maximum(jnp.float64(0.0), ref_left - dt64)
            v = y_e[0]
            spiked = (v_prev < vth) & (vth <= v)
            y_act = jnp.where(spiked & jnp.bool_(hybrid), reset(y_e), y_e)
            v_act = y_act[0]
            ref_left_act = jnp.where(spiked & jnp.bool_(hybrid), t_ref, jnp.float64(0.0))
            n_spk_act = n_spk + jnp.where(spiked & (t >= t_discard64), jnp.float64(1.0), jnp.float64(0.0))
            y_out = jnp.where(in_ref, y_ref, y_act)
            v_prev_out = jnp.where(in_ref, y_ref[0], v_act)
            ref_left_out = jnp.where(in_ref, ref_left_ref, ref_left_act)
            n_spk_out = jnp.where(in_ref, n_spk, n_spk_act)
            return (y_out, v_prev_out, ref_left_out, n_spk_out), None

        init = (y0, y0[0], jnp.float64(0.0), jnp.float64(0.0))
        carry, _ = lax.scan(body, init, jnp.arange(n, dtype=jnp.int32))
        n_spk = carry[3]
        return n_spk / jnp.float64(window_s)

    rates = jax.jit(jax.vmap(one_I))(jnp.asarray(I_grid, dtype=jnp.float64))
    return np.asarray(rates, dtype=np.float64)
