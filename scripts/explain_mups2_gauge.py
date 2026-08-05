"""WHY does mups2 have no effect on distances?  The derivation, checked.

tests/test_theory.py::test_mups2_gauge shows E(z) is independent of mups2 to
1e-9.  That is a measurement.  This script explains it, and checks every step
of the explanation numerically.

--------------------------------------------------------------------------
THE CLAIM

mups2 (mu below) is a choice of units for the field's clock.  Changing it
rescales conformal time,

    tau = sqrt(mu) * sigma,

and NOTHING ELSE.  In the variable sigma the equations, the initial
conditions and the trajectory contain no mu at all.  H picks up one overall
factor 1/sqrt(mu); E(z) = H/H(a=1) is a ratio of two H's, so it cancels
exactly.  mu is therefore not a parameter of the background theory --- it is
a gauge, in the same sense that a0 and h are.

--------------------------------------------------------------------------
STEP 1 -- mu enters the field equations in exactly one pattern

Read dda / ddcp in torc/theory.py and group the numerator by what carries a
factor of mu.  Every term quadratic in the conformal-time derivatives
(da^2, da*dcp, dcp^2) carries mu; the matter/radiation/Lambda source terms
(the OmL and omegar pieces) do not.  The whole denominator carries mu.  So
both right-hand sides have the form

    f = Q(a, cp, da, dcp)  +  P(a, cp) / mu

with Q homogeneous of degree 2 in (da, dcp), because mu cancels between
those terms and the denominator, and survives only on the sources.

That structure is equivalent to the one-line identity

    f(a, L*da, cp, L*dcp, mu/L^2) == L^2 * f(a, da, cp, dcp, mu)      (*)

for any L: the left side gives L^2 Q + (L^2/mu) P.  Section 1 below tests
(*) on random points.  It holds to ~5e-15, so the grouping is exact, not
approximate.

Now set L = sqrt(mu) in (*).  With sigma = tau/sqrt(mu) we have
d/dsigma = sqrt(mu) d/dtau, so a' = sqrt(mu)*da and a'' = mu*dda, and (*)
reads

    a'' = f(a, a', cp, cp', mu = 1).

The mu-system in tau IS the mu=1 system in sigma.  mu has been absorbed.

--------------------------------------------------------------------------
STEP 2 -- the initial conditions respect the same rescaling

An absorbed parameter can still reappear through the ICs.  It does not here.
small() returns tau_ini proportional to sqrt(mu), so sigma_ini is mu-free.
And in the series solutions every term of order n scales as tau^n / mu^(n/2)
-- check the powers in aseries/cpseries: mu^(1/2), mu, mu^(3/2), mu^2 against
tau, tau^2, tau^3, tau^4 -- which is (tau/sqrt(mu))^n = sigma^n.  So
a(sigma_ini) and cp(sigma_ini) are mu-free, while the derivative series
aseriesp/cpseriesp carry exactly one factor 1/sqrt(mu).

Section 2 tests this: same sigma_ini, same a, same cp, same sqrt(mu)*a'.

--------------------------------------------------------------------------
STEP 3 -- therefore the trajectory is mu-free, and so is E(z)

Same equations, same ICs => a(sigma) and cp(sigma) are literally the same
functions for every mu.  Section 3 integrates in tau at several mu and shows
the curves collapse when plotted against sigma.

Finally the observable.  In theory.py, H is proportional to (da/dtau)/a^2,
and da/dtau = a'(sigma)/sqrt(mu).  So

    H(mu) = H(mu=1) / sqrt(mu)

at EVERY epoch -- one common factor, not an epoch-dependent one.  E is
normalised as E = da*exp(-2x) / da[-1], a ratio of the same quantity at two
epochs, so the 1/sqrt(mu) cancels identically:

    E(z) = a'(sigma) e^{-2x} / a'(0)     -- no mu anywhere.

Note this holds whether or not H0 is sampled: H0 is rescaled by the same
factor as every other H.  mu is unobservable in ANY ratio of expansion
rates, which is all that background distances are.

--------------------------------------------------------------------------
WHAT WOULD MAKE IT PHYSICAL

mu is a clock rate, so it is invisible to observables built only from the
background trajectory in a-space.  It becomes physical the moment something
else in the problem carries a time or length scale that does NOT co-rescale
with tau -- a comoving wavenumber k in the perturbation equations being the
obvious case.  That is why priors.py keeps SAMPLE_MUPS2 as a switch and why
mups2 stays in the function signatures, rather than being deleted.

Run:
    python scripts/explain_mups2_gauge.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import diffrax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torc.theory import (dda, ddcp, small, aseries, cpseries, aseriesp,
                         cpseriesp, _TERM_TAU, rhopsol_jax)

PLOT_DIR = os.path.join(HERE, "..", "plots")

OM, OR, OL, ST = 0.3, 8.24e-5, 0.7, 1.5      # the _SAFE point
MUS = [0.5, 1.0, 2.0, 3.0]
COLS = ["#DC267F", "#785EF0", "#009E73", "#FFB000"]

plt.rcParams.update({
    "text.usetex": True, "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
})


def rule(t):
    print("\n" + t + "\n" + "-" * len(t))


# =========================================================================
rule("1. mu enters only as  f = Q(a,cp,da,dcp) + P(a,cp)/mu,  Q quadratic")
print("   test:  f(a, L*da, cp, L*dcp, mu/L^2)  ==  L^2 * f(a, da, cp, dcp, mu)")
rng = np.random.default_rng(0)
for f in (dda, ddcp):
    worst = 0.0
    for _ in range(2000):
        a, da = float(np.exp(rng.uniform(-8, 0))), rng.uniform(0.1, 3)
        cp, dcp = rng.uniform(0.1, 0.9), rng.uniform(-3, 3)
        mu, L = rng.uniform(0.4, 3), rng.uniform(0.4, 3)
        lhs = f(a, L*da, cp, L*dcp, OM, OR, OL, ST, mu/L**2)
        rhs = L**2 * f(a, da, cp, dcp, OM, OR, OL, ST, mu)
        worst = max(worst, abs(lhs - rhs) / max(abs(rhs), 1e-300))
    print(f"   {f.__name__:5s}  max rel dev over 2000 random points: {worst:.2e}")
print("   => with L = sqrt(mu):  a'' = f(a, a', cp, cp', mu=1),  sigma = tau/sqrt(mu)")

# =========================================================================
rule("2. the initial conditions carry the same rescaling (nothing sneaks back)")
print(f"   {'mu':>5s} {'tau_ini':>14s} {'sigma_ini':>18s} {'a(sig)':>18s}"
      f" {'cp(sig)':>16s} {'sqrt(mu)*da/dtau':>18s}")
ic = []
for mu in MUS:
    t = small(OM, OR, ST, mu)
    ar = (t, OM, OR, OL, ST, mu)
    row = (t / np.sqrt(mu), float(aseries(*ar)), float(cpseries(*ar)),
           float(aseriesp(*ar)) * np.sqrt(mu))
    ic.append(row)
    print(f"   {mu:5.1f} {t:14.6e} {row[0]:18.12e} {row[1]:18.12e}"
          f" {row[2]:16.12f} {row[3]:18.12f}")
ic = np.array(ic)
print("   max spread across mu (should be ~1e-16, i.e. identical):")
for j, n in enumerate(["sigma_ini", "a", "cp", "sqrt(mu)*a'"]):
    print(f"     {n:14s} {np.ptp(ic[:, j])/abs(ic[:, j]).mean():.2e}")

# =========================================================================
rule("3. so the trajectory in sigma is mu-free -- integrate in tau and check")
# a -> inf at FINITE conformal time (Lambda domination => de Sitter, whose
# conformal future infinity is at finite tau).  At this parameter point that
# singularity sits just past sigma = 0.030, so stop at 0.024, which reaches
# a = 1.65 -- comfortably past today (a = 1) without approaching the blow-up.
SIG_END = 0.024
NS = 600
sig = np.linspace(0.0, SIG_END, NS)
traj = []
for mu in MUS:
    t0 = float(small(OM, OR, ST, mu))
    ar = (t0, OM, OR, OL, ST, mu)
    y0 = (float(aseries(*ar)), float(cpseries(*ar)),
          float(aseriesp(*ar)), float(cpseriesp(*ar)))
    ts = np.sqrt(mu) * np.maximum(sig, t0 / np.sqrt(mu))   # tau = sqrt(mu)*sigma
    s = diffrax.diffeqsolve(
        _TERM_TAU, diffrax.Dopri8(), t0=ts[0], t1=ts[-1], dt0=None, y0=y0,
        args=(OM, OR, OL, ST, mu),
        stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-12),
        saveat=diffrax.SaveAt(ts=ts), max_steps=100000)
    a, cp, da, dcp = [np.asarray(v) for v in s.ys]
    traj.append(dict(mu=mu, tau=ts, a=a, cp=cp, da=da,
                     da_sig=da * np.sqrt(mu)))     # a' = sqrt(mu) * da/dtau
ref = traj[MUS.index(1.0)]
_h3 = "max|a'(sig)/a'_1 - 1|"
print(f"   {'mu':>5s} {'max|da(tau)/da_1 - 1|':>24s} {'max|a(sig)/a_1 - 1|':>22s}"
      f" {_h3:>24s}")
for tr in traj:
    print(f"   {tr['mu']:5.1f} {np.max(np.abs(tr['da']/ref['da'] - 1)):24.2e}"
          f" {np.max(np.abs(tr['a']/ref['a'] - 1)):22.2e}"
          f" {np.max(np.abs(tr['da_sig']/ref['da_sig'] - 1)):24.2e}")
print("   col 1 is LARGE: da/dtau genuinely differs, by exactly 1/sqrt(mu).")
print("   cols 2-3 are at solver tolerance: in sigma the trajectory is identical.")
print("\n   the 1/sqrt(mu) in H, made explicit:")
for tr in traj:
    r = np.median(tr["da"] / ref["da"])
    print(f"     mu={tr['mu']:4.1f}  median da/dtau ratio = {r:.12f}"
          f"   1/sqrt(mu) = {1/np.sqrt(tr['mu']):.12f}")

# =========================================================================
rule("4. E(z) is a RATIO of H's, so the common 1/sqrt(mu) cancels exactly")
Es = []
for mu in MUS:
    z, E, _ = rhopsol_jax(OM, OR, OL, ST, mu)
    Es.append(np.asarray(E))
z = np.asarray(z)
E1 = Es[MUS.index(1.0)]
print(f"   {'mu':>5s} {'max |E/E(mu=1) - 1|':>24s}")
for mu, E in zip(MUS, Es):
    print(f"   {mu:5.1f} {np.max(np.abs(E/E1 - 1)):24.2e}")
print("   -> mu is a gauge: unobservable in any ratio of expansion rates,")
print("      which is all a background distance measurement ever is.")

# =========================================================================
fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.0))

ax = axs[0]
for tr, c in zip(traj, COLS):
    ax.plot(tr["tau"], tr["a"], color=c, lw=1.5,
            label=rf"$\mu_{{\upsilon 2}}={tr['mu']}$")
ax.set_xlabel(r"conformal time $\tau$")
ax.set_ylabel(r"$a$")
ax.set_title(r"(a) in $\tau$: the curves genuinely differ", fontsize=9.5)
ax.axhline(1.0, color="k", ls=":", lw=0.9)
ax.text(0.002, 1.04, r"today, $a=1$", fontsize=7)
ax.legend(); ax.grid(alpha=0.3, lw=0.5); ax.set_ylim(0, 1.8)

ax = axs[1]
for tr, c in zip(traj, COLS):
    ax.plot(sig, tr["a"], color=c, lw=3.0 if tr["mu"] == 0.5 else 1.2,
            alpha=0.85, label=rf"$\mu_{{\upsilon 2}}={tr['mu']}$")
ax.set_xlabel(r"$\sigma = \tau/\sqrt{\mu_{\upsilon 2}}$")
ax.set_ylabel(r"$a$")
ax.set_title(r"(b) in $\sigma$: exact collapse", fontsize=9.5)
ax.axhline(1.0, color="k", ls=":", lw=0.9)
ax.legend(); ax.grid(alpha=0.3, lw=0.5); ax.set_ylim(0, 1.8)
worst_a = max(np.max(np.abs(tr["a"]/ref["a"] - 1)) for tr in traj)
ax.text(0.04, 0.94, rf"$\max|\Delta a/a| = {worst_a:.0e}$" "\n(solver tolerance)",
        transform=ax.transAxes, va="top", fontsize=7.5)

ax = axs[2]
for mu, E, c in zip(MUS, Es, COLS):
    ax.plot(1.0 + z, np.abs(E/E1 - 1) + 1e-18, color=c, lw=1.3,
            label=rf"$\mu_{{\upsilon 2}}={mu}$")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim(1.0, 1e9)
ax.axvspan(1.0, 1.0 + 2.7, color="0.85", zorder=0)
ax.text(1.9, 3e-18, r"DESI BAO", fontsize=7, ha="center")
ax.set_xlabel(r"$1+z$")
ax.set_ylabel(r"$|E(z)/E(z)|_{\mu_{\upsilon 2}=1} - 1|$")
ax.set_title(r"(c) the observable: $E(z)$ is untouched", fontsize=9.5)
ax.set_ylim(1e-18, 1e-6)
ax.axhline(1e-9, color="k", ls=":", lw=0.9)
ax.text(0.5, 1.4e-9, r"$10^{-9}$: the tolerance of \texttt{test\_mups2\_gauge}",
        transform=ax.get_yaxis_transform(), fontsize=7)
ax.legend(loc="lower right"); ax.grid(alpha=0.3, lw=0.5)

fig.suptitle(r"Why $\mu_{\upsilon 2}$ does not affect distances: it rescales "
             r"the field's clock, $\tau=\sqrt{\mu_{\upsilon 2}}\,\sigma$, and "
             r"nothing else", fontsize=11, y=1.0)
fig.text(0.5, -0.06,
         r"In the field equations $\mu_{\upsilon 2}$ multiplies every term "
         r"quadratic in $\tau$-derivatives but none of the matter/radiation/"
         r"$\Lambda$ sources, so it is absorbed entirely by $\tau=\sqrt{\mu_"
         r"{\upsilon 2}}\sigma$; the series initial conditions scale the same "
         r"way ($\tau_\mathrm{ini}\propto\sqrt{\mu_{\upsilon 2}}$, term "
         r"$n\propto\tau^n\mu_{\upsilon 2}^{-n/2}$)."
         "\n"
         r"The trajectory $a(\sigma)$ is therefore \emph{identical} for every "
         r"$\mu_{\upsilon 2}$, and $H\propto a^{-2}\,\mathrm{d}a/\mathrm{d}\tau$ "
         r"picks up one common factor $\mu_{\upsilon 2}^{-1/2}$ at every epoch. "
         r"$E=H/H(a{=}1)$ is a ratio, so it cancels identically --- for any "
         r"$H_0$, sampled or not."
         "\n"
         r"$\mu_{\upsilon 2}$ is a gauge, like $a_0$ and $h$.  It becomes "
         r"physical only where a scale that does \emph{not} co-rescale with "
         r"$\tau$ enters --- e.g. a comoving $k$ at perturbation level.",
         ha="center", va="top", fontsize=7.4)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
out = os.path.join(PLOT_DIR, "mups2_gauge")
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, bbox_inches="tight",
                dpi=300 if ext == "png" else None)
plt.close(fig)
print(f"\nsaved plots/mups2_gauge.{{pdf,png}}")
