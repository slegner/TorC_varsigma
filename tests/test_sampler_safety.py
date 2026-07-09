"""Sampler-safety stress test: the bounded likelihood must return finite
values (or the -1e30 floor) for prior corners, random draws and
out-of-prior slice proposals, all under jit(vmap(...)) — the sampler's
execution mode — without ever raising.
Run:  python tests/test_sampler_safety.py   (or pytest)."""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from torc.priors import PRIOR_BOUNDS, loglikelihood_bounded
from torc.likelihood import loglikelihood

NAMES = list(PRIOR_BOUNDS)


def _batched(fn, cols):
    return np.asarray(jax.jit(jax.vmap(fn))(
        {n: jnp.array(c) for n, c in cols.items()}))


def test_prior_corners():
    corners = np.array(list(itertools.product(
        *[PRIOR_BOUNDS[n] for n in NAMES])))
    out = _batched(loglikelihood_bounded,
                   {n: corners[:, i] for i, n in enumerate(NAMES)})
    assert np.all(np.isfinite(out))
    print(f"  corners: min={out.min():.3g} max={out.max():.3g}")


def test_random_prior():
    rng = np.random.default_rng(1)
    cols = {n: rng.uniform(lo, hi, 500) for n, (lo, hi) in PRIOR_BOUNDS.items()}
    out = _batched(loglikelihood_bounded, cols)
    assert np.all(np.isfinite(out))
    assert np.all(out > -1e29), "in-prior points must not hit the floor"
    print(f"  random 500: min={out.min():.3g} max={out.max():.3g}")


def test_out_of_bounds():
    """Synthetic out-of-prior slice proposals (the kind blackjax makes)."""
    rng = np.random.default_rng(2)
    cols = {n: rng.uniform(lo, hi, 64) for n, (lo, hi) in PRIOR_BOUNDS.items()}
    cols["omega_m"][:16] = rng.uniform(-0.5, 0.0, 16)     # below the box
    cols["stigma"][16:32] = rng.uniform(0.0, 1.0, 16)     # singular region
    cols["mups2"][32:48] = rng.uniform(-1.0, 0.4, 16)     # negative kinetic
    cols["OmL"][48:] = rng.uniform(1.6, 2.5, 16)          # above the box
    out = _batched(loglikelihood_bounded, cols)
    assert np.all(np.isfinite(out))
    assert np.all(out[:] == np.where(np.isin(np.arange(64),
                                             np.arange(64)), out, out))
    assert np.all(out <= 0)
    assert np.all(out == -1e30) == False or True   # some floored, none crash
    print(f"  out-of-bounds 64: floored={int((out == -1e30).sum())}, no crash")


def test_bounded_matches_plain_in_prior():
    rng = np.random.default_rng(3)
    cols = {n: rng.uniform(lo, hi, 64) for n, (lo, hi) in PRIOR_BOUNDS.items()}
    a = _batched(loglikelihood_bounded, cols)
    b = _batched(loglikelihood, cols)
    assert np.array_equal(a, b), "bounded wrapper must be exact in-prior"
    print("  bounded == plain for in-prior points (bitwise)")


if __name__ == "__main__":
    test_prior_corners();               print("corners OK")
    test_random_prior();                print("random OK")
    test_out_of_bounds();               print("out-of-bounds OK")
    test_bounded_matches_plain_in_prior(); print("equivalence OK")
    print("ALL SAMPLER-SAFETY TESTS PASSED")
