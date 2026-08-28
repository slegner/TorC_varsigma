"""
H(z) and calcresult = H0*asol/(a0*hsol) for the Stigma background, varpi_0 = 1.
Uses the original aseries/cpseries seeds and the original hstop normalisation:
  pass 1 with a0=1 -> find where da = 100h -> a0 := a there
  pass 2 reseeded with that a0  -> asol = 1 today, calcresult -> 1 today.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from stigma_series import aseries, cpseries, aseriesp, cpseriesp, small, omega_r
from stigma_rhs import coupdiff

h, omega_m, OmL, mups2 = 0.67, 0.31, 0.69, 4/3
H0, OmR = 100*h, omega_r(h)
print(f"h={h}  Om_m={omega_m}  Om_L={OmL}  Om_r={OmR:.4e}  H0={H0:.2f}  mups2={mups2:.4f}")

def seeds(t0, stigma, a0):
    return [aseries (t0, omega_m, h, OmL, a0, stigma, mups2),
            cpseries (t0, omega_m, h, OmL, a0, stigma, mups2),
            aseriesp (t0, omega_m, h, OmL, a0, stigma, mups2),
            cpseriesp(t0, omega_m, h, OmL, a0, stigma, mups2)]

def hstop(a0):
    def ev(t, y, *a): return y[2] - H0*a0
    ev.terminal, ev.direction = True, 1
    return ev

def wall(stigma):
    def ev(t, y, *a): return y[1]**2 - stigma
    ev.terminal = True
    return ev

def integrate(stigma, a0, tmax, stop_today, amax=None):
    t0 = small(omega_m, h, stigma, mups2)
    y0 = seeds(t0, stigma, a0)
    ev = [wall(stigma)] + ([hstop(a0)] if stop_today else [])
    if amax is not None:
        def big(t, y, *a): return y[0] - amax
        big.terminal = True; ev.append(big)
    sol = solve_ivp(coupdiff, [t0, tmax], y0, args=(omega_m, h, OmL, a0, stigma, mups2),
                    events=ev, rtol=1e-9, atol=1e-30, method="LSODA")
    return t0, y0, sol

results = {}
for stigma in (0.9, 1.2, 1.5):
    # ---- pass 1: a0 = 1, find today (da = H0) ----
    t0, y0, s1 = integrate(stigma, 1.0, 60.0, stop_today=True)
    hit_wall = len(s1.t_events[0]) > 0
    got_today = len(s1.t_events) > 1 and len(s1.t_events[1]) > 0
    print(f"\nstigma={stigma}:  tau_ini={t0:.3e}  varpi_0={y0[1]:.8f}")
    print(f"  pass1 ended tau={s1.t[-1]:.4e}  a={s1.y[0,-1]:.4e}  varpi^2={s1.y[1,-1]**2:.6f}"
          f"   wall={hit_wall}  today={got_today}")
    if not got_today:
        print(f"  -> never reaches da=H0 (halts at the wall first). No normalisation possible.")
        results[stigma] = dict(halted=True, a=s1.y[0], cp=s1.y[1], a0=1.0)
        continue
    a0 = s1.y_events[1][0][0]
    print(f"  a0 = {a0:.6e}")

    # ---- pass 2: reseed with a0, run past today ----
    t0, y0, s2 = integrate(stigma, a0, 60.0, stop_today=False, amax=3.0)
    asol, cpsol, hsol = s2.y[0], s2.y[1], s2.y[2]
    hubble     = (a0*hsol)/asol**2
    z          = 1.0/asol - 1.0
    calcresult = H0*asol/(a0*hsol)
    i0 = np.argmin(np.abs(asol - 1.0))
    print(f"  pass2 a in [{asol.min():.3e}, {asol.max():.3e}]"
          f"   H(a=1)={hubble[i0]:.4f}  calcresult(a=1)={calcresult[i0]:.6f}")
    results[stigma] = dict(halted=False, a=asol, cp=cpsol, H=hubble, z=z,
                           calc=calcresult, a0=a0)

# ------------------------------ LCDM ------------------------------
def H_lcdm(a):  return H0*np.sqrt(OmR/a**4 + omega_m/a**3 + OmL)
def calc_lcdm(a): return H0/(a*H_lcdm(a))
agrid = np.logspace(-6, np.log10(3), 900)
print(f"\nLCDM: calcresult(a=1) = {calc_lcdm(1.0):.6f}   (should be 1)")

# ------------------------------ plot ------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.7))
cols = {0.9: "#A32E4A", 1.2: "#0E6B5F", 1.5: "#4338A8"}

ax[0].loglog(agrid, calc_lcdm(agrid), "k--", lw=1.7, label=r"$\Lambda$CDM", zorder=5)
for st, r in results.items():
    if r["halted"]:
        continue
    m = r["a"] > 0
    ax[0].loglog(r["a"][m], r["calc"][m], "-", color=cols[st], lw=1.9, label=rf"$\varsigma={st}$")
    ax[1].semilogx(r["a"][m], r["calc"][m]/calc_lcdm(r["a"][m]) - 1.0, "-",
                   color=cols[st], lw=1.9, label=rf"$\varsigma={st}$")
ax[0].axvline(1, color="0.75", lw=.8); ax[0].axhline(1, color="0.75", lw=.8)
ax[0].set_xlabel(r"$a$"); ax[0].set_ylabel(r"calcresult $=H_0\,a/(a_0\,a')$")
ax[0].set_title(r"calcresult vs scale factor")

ax[1].axhline(0, color="k", ls="--", lw=1.4)
ax[1].axvline(1, color="0.75", lw=.8)
ax[1].set_xlabel(r"$a$"); ax[1].set_ylabel(r"calcresult$/$calcresult$_{\Lambda\rm CDM}-1$")
ax[1].set_title("Fractional deviation from $\Lambda$CDM")

for st, r in results.items():
    m = r["a"] > 0
    lab = rf"$\varsigma={st}$" + ("  (halts)" if r["halted"] else "")
    ax[2].loglog(r["a"][m], r["cp"][m]**2, "-", color=cols[st], lw=1.9, label=lab)
    ax[2].axhline(st, color=cols[st], ls=":", lw=1.2)
ax[2].text(2e-9, 0.93, r"$\varpi^2=\varsigma$ walls (dotted)", fontsize=8.5, color="0.35")
ax[2].axvline(1, color="0.75", lw=.8)
ax[2].set_xlabel(r"$a$"); ax[2].set_ylabel(r"$\varpi^2$")
ax[2].set_title(r"$\varpi^2$ and the singular surface")

for k in (0, 1, 2):
    ax[k].legend(frameon=False, fontsize=9); ax[k].grid(alpha=.25, which="both")
fig.suptitle(r"Stigma background,  $\varpi_0=1$", y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig("hubble_stigma.png", dpi=150, bbox_inches="tight")
print("saved hubble_stigma.png")
