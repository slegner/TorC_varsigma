"""Type Ia supernova likelihoods in JAX (Pantheon+ and DESY5), for TorC.

A port of distances/ia.py::IaLogL onto torc.theory_lam's E(z), so supernovae
can be combined with the DESI DR2 BAO likelihood in the blackjax
nested-sampling pipeline.

WHAT IS MARGINALISED
--------------------
Supernovae constrain the SHAPE of the distance-redshift relation, not its
normalisation: the absolute magnitude M is degenerate with H0.  ia.py handles
this analytically and this port reproduces it exactly:

  * the offset direction is projected out of the inverse covariance
        invcov_tilde = invcov - invcov 1 1^T invcov / (1^T invcov 1)
    which is marginalisation over M with a flat prior;
  * H0 is then integrated over a UNIFORM prior on [h0min, h0max] = [20, 100]
    km/s/Mpc, giving the (b^(Y+1) - a^(Y+1))/(Y+1) factor.

So this likelihood adds NO new sampled parameters.  It depends on the
cosmology only through E(z) = H(z)/H(z=0), which is exactly what
torc.theory_lam.rhopsol_jax returns -- no h0rd, no H0.

DATA
----
Pantheon+ (vendored in data/PantheonPlus, or $PANTHEONPLUS_DIR):
    Pantheon+SH0ES.dat            (space separated; zHD, zHEL, m_b_corr, ...)
    Pantheon+SH0ES_STAT+SYS.cov   (flattened STAT+SYS covariance, 1 header line)

DESY5 ($DESY5_DIR; not vendored):
    DES-SN5YR_HD.csv     (columns zHD, zHEL, MU, MUERR_FINAL)
    covsys_000.txt       (flattened SYSTEMATIC covariance, 1 header line)

The two differ in ways that matter and are handled separately below:
    Pantheon+  m_b_corr, already sorted by zHD, STAT already in the
               covariance, z_cutoff = 0.023 (peculiar velocities)
    DESY5      MU, NOT sorted by zHD (data and covariance both permuted),
               MUERR_FINAL**2 added to the covariance diagonal, no cutoff
Both follow distances/pantheonplus.py and distances/des5y.py exactly.
"""
import gzip
import os

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .theory_lam import rhopsol_jax
from .theory import OMEGA_R_DEFAULT

OMEGAR = OMEGA_R_DEFAULT
H0MIN, H0MAX = 20.0, 100.0
C_KMS = 299792.458

_SEARCH_PP = [
    os.environ.get("PANTHEONPLUS_DIR"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                 "PantheonPlus"),
    "/Users/starfish/Desktop/PhD2025-26/clik_installs/desi/data/sn_data/PantheonPlus",
]

_SEARCH = [
    os.environ.get("DESY5_DIR"),
    # DES-SN5YR public release (v1.2 layout)
    "/Users/starfish/Desktop/PhD2025-26/distances/DES-SN5YR-1.2/4_DISTANCES_COVMAT",
    "/Users/starfish/Desktop/PhD2025-26/clik_installs/desi/data/sn_data/DESY5",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "DESY5"),
]

# the systematic-only covariance, under either release's name.  ORDER MATTERS
# only in that the first match wins; both are the same object.
_DESY5_COV_NAMES = ["covsys_000.txt", "STAT+SYS.txt.gz", "STAT+SYS.txt"]


def find_data_dir():
    for c in _SEARCH:
        if c and os.path.isfile(os.path.join(c, "DES-SN5YR_HD.csv")):
            return c
    raise FileNotFoundError(
        "DESY5 data not found.  Set $DESY5_DIR to the directory containing "
        "DES-SN5YR_HD.csv and covsys_000.txt (see distances/des5y.py).  "
        f"Looked in: {[c for c in _SEARCH if c]}")


def find_pantheonplus_dir():
    for c in _SEARCH_PP:
        if c and os.path.isfile(os.path.join(c, "Pantheon+SH0ES.dat")):
            return c
    raise FileNotFoundError(
        "Pantheon+ data not found.  Set $PANTHEONPLUS_DIR, or place "
        "Pantheon+SH0ES.dat and Pantheon+SH0ES_STAT+SYS.cov in "
        f"data/PantheonPlus.  Looked in: {[c for c in _SEARCH_PP if c]}")


def load_pantheonplus(path=None, z_cutoff=0.023):
    """Load Pantheon+, following distances/pantheonplus.py.

    z_cutoff = 0.023 is that script's value: below it peculiar velocities
    dominate the redshift and the distance-redshift relation is unreliable.
    Pantheon+ is already sorted by zHD and its STAT+SYS covariance already
    contains the statistical part, so -- unlike DESY5 -- nothing is permuted
    and nothing is added to the diagonal.
    """
    import pandas as pd
    path = path or find_pantheonplus_dir()
    df = pd.read_table(os.path.join(path, "Pantheon+SH0ES.dat"), sep=" ",
                       engine="python")
    cov = np.loadtxt(os.path.join(path, "Pantheon+SH0ES_STAT+SYS.cov"),
                     skiprows=1)
    cov = cov.reshape([-1, int(np.sqrt(len(cov)))])

    mask = df["zHD"].to_numpy() > z_cutoff
    return build(df["zHD"].to_numpy()[mask], df["zHEL"].to_numpy()[mask],
                 df["m_b_corr"].to_numpy()[mask], cov[mask, :][:, mask])


def _load_cov_txt(path):
    """Read a SNANA-style flattened covariance: one header line with N, then
    N*N values.  Transparently handles .gz."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        n = int(fh.readline())
        v = np.loadtxt(fh)
    if v.size != n * n:
        raise ValueError(f"{path}: header says {n} but found {v.size} values")
    return v.reshape(n, n)


def load_desy5(path=None, z_cutoff=0.0):
    """Load DES-SN5YR (DESY5), following distances/des5y.py exactly.

    COVARIANCE CONVENTION -- the subtle part.  The release ships
    STATONLY.txt.gz and STAT+SYS.txt.gz, and those names describe the
    ANALYSIS VARIANT, not the file contents: both are SYSTEMATIC-ONLY
    matrices, to which the statistical variance MUERR_FINAL**2 must be added
    on the diagonal.  Verified on v1.2:

        STATONLY.txt.gz  is identically zero (no systematics at all)
        STAT+SYS.txt.gz  has a diagonal only ~6% of MUERR_FINAL**2, and is
                         NOT positive definite on its own (min eigenvalue
                         -6.5e-4); adding diag(MUERR**2) makes it PD (+2.2e-3)

    So it is the same object as the older release's covsys_000.txt, which is
    what distances/des5y.py reads.  Using it as though it already contained
    the statistical part would both double-count nothing and hand a
    non-positive-definite matrix to the inverse.

    SORTING: DES5Y is not ordered by redshift, so the data AND the covariance
    are both permuted into ascending zHD (see the warning in the zenodo
    README).
    """
    import pandas as pd
    path = path or find_data_dir()
    df = pd.read_table(os.path.join(path, "DES-SN5YR_HD.csv"), sep=",",
                       engine="python")
    covfile = next((os.path.join(path, n) for n in _DESY5_COV_NAMES
                    if os.path.isfile(os.path.join(path, n))), None)
    if covfile is None:
        raise FileNotFoundError(
            f"no systematic covariance in {path}; expected one of "
            f"{_DESY5_COV_NAMES}")
    cov = _load_cov_txt(covfile)

    idx = np.argsort(df["zHD"].to_numpy())
    delta = df["MUERR_FINAL"].to_numpy()
    np.fill_diagonal(cov, delta**2 + cov.diagonal())   # statistical part
    cov = cov[idx, :][:, idx]
    df = df.iloc[idx]

    mask = df["zHD"].to_numpy() > z_cutoff
    return build(df["zHD"].to_numpy()[mask], df["zHEL"].to_numpy()[mask],
                 df["MU"].to_numpy()[mask], cov[mask, :][:, mask])


def build(zhd, zhel, mb, cov):
    """Precompute the projected inverse covariance and normalisation.

    Separated from file loading so the port can be validated against
    distances/ia.py on synthetic data when the real files are unavailable.
    """
    one = np.ones(len(cov))[:, None]
    invcov = np.linalg.inv(cov)
    invcov_tilde = (invcov
                    - invcov @ one @ one.T @ invcov / (one.T @ invcov @ one))
    lognormalisation = 0.5 * (
        np.log(2*np.pi) - np.linalg.slogdet(2*np.pi*cov)[1]
        - np.log((one.T @ invcov @ one).squeeze())
    ) + np.log(C_KMS / (1e-5 * (H0MAX - H0MIN)))
    return dict(
        zhd=jnp.asarray(zhd), zhel=jnp.asarray(zhel), mb=jnp.asarray(mb),
        invcov_tilde=jnp.asarray(invcov_tilde),
        onesigma=jnp.asarray((one.T @ invcov_tilde * 5 / np.log(10)).ravel()),
        lognorm=float(lognormalisation),
        a=1e-5 * H0MIN / C_KMS, b=1e-5 * H0MAX / C_KMS,
    )


def _cumtrapz(y, x):
    dx = jnp.diff(x)
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * dx * (y[1:] + y[:-1]))])


def h0_dl_over_c(z_th, E_th, zhd, zhel, n_grid=2001):
    """(1 + zhel) * int_0^zhd dz/E(z), i.e. H0 d_L / c -- dimensionless.

    Integrated on a fine uniform grid rather than on the (very unevenly
    spaced) supernova redshifts themselves; distances/distances.py uses
    quad to the first point plus a trapezoid over the data points, which is
    the less accurate of the two.
    """
    zmax = jnp.max(zhd)
    zg = jnp.linspace(0.0, zmax, n_grid)
    invE = 1.0 / jnp.interp(zg, z_th, E_th)
    q = jnp.interp(zhd, zg, _cumtrapz(invE, zg))
    return (1.0 + zhel) * q


def make_loglikelihood(data):
    """Return logL(params) for a preloaded DESY5 `data` dict."""

    def loglikelihood(params):
        z_th, E_th, _ = rhopsol_jax(
            params["omega_m"], OMEGAR, params.get("OmL", 0.0),
            params["stigma"], params.get("mups2", 4.0/3.0),
            params.get("lam", 0.0))
        hdl = h0_dl_over_c(z_th, E_th, data["zhd"], data["zhel"])
        y = 5.0 * jnp.log10(hdl) - data["mb"]
        Y = data["onesigma"] @ y
        chi2_term = -0.5 * (y @ data["invcov_tilde"] @ y)
        h0_term = jnp.log((data["b"] ** (Y + 1) - data["a"] ** (Y + 1))
                          / (Y + 1))
        out = chi2_term + h0_term + data["lognorm"]
        return jnp.where(jnp.isfinite(out), out, -1e30)

    return loglikelihood
