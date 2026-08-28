"""Validate torc/likelihood_sn.py against distances/ia.py::IaLogL.

    python tests/test_likelihood_sn.py

The real DESY5 files are not vendored, so this runs on SYNTHETIC supernovae.
That is sufficient: what needs checking is the marginalisation algebra and the
distance integral, neither of which cares whether the numbers are real.

distances/ia.py imports distances.py, which needs `flexknot` (not installed
here), so the reference class is exec'd against a stub module supplying `c`
and an LCDM `h0_dl_over_c`.  The IaLogL code itself is the author's, unedited.
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd

from torc import likelihood_sn as SN

DIST = "/Users/starfish/Desktop/PhD2025-26/distances"
C_KMS = 299792.458
OMEGAR = 8.24e-5


def E_lcdm(z, om):
    return np.sqrt(om*(1+z)**3 + OMEGAR*(1+z)**4 + (1-om-OMEGAR))


def ref_h0_dl_over_c(zhd, zhel, omegam, omegar, theta):
    """LCDM stand-in for distances.h0_dl_over_c, same quad + cumtrapz scheme."""
    from scipy.integrate import quad, cumulative_trapezoid
    q0 = quad(lambda z: 1/E_lcdm(z, omegam), 0, zhd[0])[0]
    h_inv = [1/E_lcdm(zi, omegam) for zi in zhd]
    q = cumulative_trapezoid(h_inv, zhd, initial=0) + q0
    return (1+zhel)*q


def load_reference_class():
    """exec distances/ia.py with a stubbed `distances` module."""
    stub = types.ModuleType("distances")
    stub.c = C_KMS
    stub.h0_dl_over_c = ref_h0_dl_over_c
    sys.modules["distances"] = stub
    src = open(os.path.join(DIST, "ia.py")).read()
    ns = {}
    exec(compile(src, os.path.join(DIST, "ia.py"), "exec"), ns)
    return ns["IaLogL"]


def make_mock(n=250, seed=0):
    rng = np.random.default_rng(seed)
    zhd = np.sort(rng.uniform(0.025, 1.10, n))
    zhel = zhd + rng.normal(0, 1e-4, n)
    # covariance: diagonal scatter plus a correlated systematic block
    sig = rng.uniform(0.10, 0.30, n)
    cov = np.diag(sig**2)
    v = rng.normal(0, 0.02, (n, 3))
    cov += v @ v.T
    om_true = 0.315
    mu = 5*np.log10(ref_h0_dl_over_c(zhd, zhel, om_true, OMEGAR, [-1])) + 43.1
    mb = mu + rng.multivariate_normal(np.zeros(n), cov)
    df = pd.DataFrame({"zHD": zhd, "zHEL": zhel, "MU": mb,
                       "MUERR_FINAL": sig})
    return df, cov


def main():
    IaLogL = load_reference_class()
    df, cov = make_mock()
    ref = IaLogL(df, cov, mb_column="MU", z_cutoff=0.0)
    data = SN.build(df["zHD"].to_numpy(), df["zHEL"].to_numpy(),
                    df["MU"].to_numpy(), cov)

    print("precomputed algebra:")
    print(f"  lognormalisation  ref {ref.lognormalisation:+.10f}   "
          f"port {data['lognorm']:+.10f}   "
          f"diff {abs(ref.lognormalisation - data['lognorm']):.3e}")
    d_it = np.max(np.abs(np.asarray(data["invcov_tilde"]) - ref.invcov_tilde))
    print(f"  invcov_tilde      max abs diff {d_it:.3e}")
    d_os = np.max(np.abs(np.asarray(data["onesigma"])
                         - ref.onesigma_times_5_over_log10.ravel()))
    print(f"  onesigma vector   max abs diff {d_os:.3e}")
    assert d_it < 1e-12 and d_os < 1e-12
    assert abs(ref.lognormalisation - data["lognorm"]) < 1e-12

    print("\nlog-likelihood vs the reference class, across cosmologies:")
    zg = np.linspace(0.0, 1.3, 4001)
    worst = 0.0
    for om in (0.20, 0.25, 0.315, 0.40, 0.55):
        r = float(ref(om, OMEGAR, [-1]))
        # feed the port an LCDM E(z) on the theory grid it expects
        E = E_lcdm(zg, om)
        hdl = SN.h0_dl_over_c(jnp.asarray(zg), jnp.asarray(E),
                              data["zhd"], data["zhel"])
        y = 5.0*jnp.log10(hdl) - data["mb"]
        Y = data["onesigma"] @ y
        v = float(-0.5*(y @ data["invcov_tilde"] @ y)
                  + jnp.log((data["b"]**(Y+1) - data["a"]**(Y+1))/(Y+1))
                  + data["lognorm"])
        d = abs(v - r)
        worst = max(worst, d)
        print(f"  Om={om:5.3f}   ref {r:14.6f}   port {v:14.6f}   diff {d:.3e}")
    print(f"\nmax |difference| = {worst:.3e}")
    # the two use different quadrature (fine grid vs trapezoid on the SN
    # redshifts themselves), so they agree to integration accuracy, not to ulp
    assert worst < 0.05, f"port disagrees with distances/ia.py by {worst}"
    print("PASS -- the port reproduces distances/ia.py")


if __name__ == "__main__":
    main()
