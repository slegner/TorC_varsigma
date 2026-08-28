"""Why the lambda term cannot replace Omega_Lambda.

In the a^4 source bracket of the field equation,

    3*OmL*(1-4*s)*s^2                              <- Omega_Lambda: no varpi
  + cp^4*lam*(-1+4*s^2)                            <- lambda: varpi^4
  + cp^2*s*(-7*lam - 3*OmL + 4*lam*s + 12*OmL*s^2) <- mixed: varpi^2

Omega_Lambda carries a varpi-INDEPENDENT floor; every lambda term is
multiplied by varpi^2 or varpi^4.  Since varpi decays, the lambda source
decays with it -- it is strongest in the early universe and weakest today,
the opposite of what late-time acceleration needs.

    python scripts/plot_why_lam_fails.py   ->  plots/why_lam_fails.pdf
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                         N_EARLY, N_LATE, Z_SPLIT, OMEGA_R_DEFAULT,
                         _SOLVER, _CTRL)
from torc import theory_lam as TL

OMR, MU = OMEGA_R_DEFAULT, 4.0/3.0
STIGMA = 1.73          # the DESY5 posterior value
OM = 0.30

plt.rcParams.update({
    "text.usetex": True, "font.family": "cmtt", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 11, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
})


def trajectory(om, OmL, st, lam):
    tau = small(om, OMR, st, MU)
    x0 = float(jnp.log(aseries(tau, om, OMR, OmL, st, MU)))
    y0 = (cpseries(tau, om, OMR, OmL, st, MU),
          aseriesp(tau, om, OMR, OmL, st, MU),
          cpseriesp(tau, om, OMR, OmL, st, MU))
    xs = jnp.concatenate([
        jnp.linspace(x0, -np.log(1+Z_SPLIT), N_EARLY, endpoint=False),
        jnp.linspace(-np.log(1+Z_SPLIT), 0.0, N_LATE)])
    sol = diffrax.diffeqsolve(
        TL._TERM_LNA, _SOLVER, t0=x0, t1=0.0, dt0=None, y0=y0,
        args=(om, OMR, OmL, st, MU, lam), stepsize_controller=_CTRL,
        saveat=diffrax.SaveAt(ts=xs), max_steps=4096, throw=False,
        adjoint=diffrax.ForwardMode())
    return np.exp(np.asarray(xs)), np.asarray(sol.ys[0])


def sources(cp, st, OmL, lam):
    oml = 3*OmL*(1-4*st)*st**2 + cp**2*st*(-3*OmL + 12*OmL*st**2)
    lm = cp**4*lam*(-1+4*st**2) + cp**2*st*(-7*lam + 4*lam*st)
    return oml, lm


a_l, cp_l = trajectory(OM, 0.0, STIGMA, 1.0)       # lambda only
a_o, cp_o = trajectory(OM, 0.7, STIGMA, 0.0)       # Omega_Lambda only

_, src_l = sources(cp_l, STIGMA, 0.0, 1.0)
src_o, _ = sources(cp_o, STIGMA, 0.7, 0.0)

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.2, 6.4), sharex=True)

ax0.plot(a_l, cp_l, color="#FE6100", lw=1.6, label=r"$\lambda$ only ($\Omega_\Lambda=0$)")
ax0.plot(a_o, cp_o, color="#648FFF", lw=1.6, label=r"$\Omega_\Lambda$ only ($\lambda=0$)")
ax0.set_ylabel(r"$\varpi(a)$")
ax0.set_xscale("log")
ax0.legend()
ax0.set_title(r"$\varsigma=%.2f$, $\Omega_\mathrm{m}=%.2f$" % (STIGMA, OM),
              fontsize=9)

ax1.plot(a_l, np.abs(src_l)/np.abs(src_l[0]), color="#FE6100", lw=1.6,
         label=r"$\lambda$ source $\propto \varpi^2,\varpi^4$")
ax1.plot(a_o, np.abs(src_o)/np.abs(src_o[0]), color="#648FFF", lw=1.6,
         label=r"$\Omega_\Lambda$ source (has a $\varpi$-free floor)")
ax1.axhline(1.0, color="k", ls=":", lw=1)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"$a$")
ax1.set_ylabel(r"$|$source$|$ relative to $a=10^{-3}$")
ax1.legend()

note = (r"$\varpi$ decays, so the $\lambda$ source decays with it: it is "
        r"strongest early and weakest today," "\n"
        r"the opposite of what late-time acceleration requires.  "
        r"$\Omega_\Lambda$ instead grows \textit{and} stabilises $\varpi$.")
fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=7.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "..", "plots", f"why_lam_fails.{ext}"),
                bbox_inches="tight")
print("saved plots/why_lam_fails.{pdf,png}")
for nm, a, s in (("lambda only", a_l, src_l), ("OmL only", a_o, src_o)):
    j0 = int(np.argmin(np.abs(a-1e-3))); j1 = int(np.argmin(np.abs(a-1.0)))
    print(f"  {nm:12s}: |source| a=1e-3 -> a=1 changes by "
          f"x{abs(s[j1])/abs(s[j0]):.3f}")
