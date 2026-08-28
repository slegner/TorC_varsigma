import os, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torc.likelihood import model_vector, DATA, COV_INV
from anesthetic import NestedSamples

D, Ci = jnp.asarray(DATA), jnp.asarray(COV_INV)
DCD = float(np.asarray(D) @ np.asarray(Ci) @ np.asarray(D))
ST = 1.3797

def chi2(om, ol):                       # h0rd profiled ANALYTICALLY (pred ~ 1/h0rd)
    p = model_vector(om, 10000.0, ST, 4/3, ol)
    pCp, pCD = p @ Ci @ p, p @ Ci @ D
    return DCD - pCD**2/pCp

N = 90
oms = np.linspace(0.01, 0.99, N)
ols = np.linspace(0.10, 1.50, N)
OM, OL = np.meshgrid(oms, ols, indexing="ij")
Z = np.asarray(jax.vmap(chi2)(jnp.asarray(OM.ravel()), jnp.asarray(OL.ravel())))
Z = Z.reshape(N, N)
Z = np.where(np.isfinite(Z), Z, np.nan)
dchi2 = Z - np.nanmin(Z)

d = np.load("chains/torc_ns_julia_1000_exponential_lam2.npz")
s = NestedSamples({k: d[k] for k in ["omega_m","h0rd","stigma","OmL"]},
                  logL=d["logL"], logL_birth=d["logL_birth"])
w = np.asarray(s.get_weights()); M = np.asarray(s["omega_m"]); L = np.asarray(s["OmL"])
slope = np.average(L/M, weights=w)

plt.rcParams.update({"text.usetex": True, "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300})
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.9), sharex=True, sharey=True)

ax = axs[0]
lv = [1, 4, 9, 25, 100, 400]
cf = ax.contourf(OM, OL, dchi2, levels=[0]+lv, cmap="viridis_r", extend="max")
ax.contour(OM, OL, dchi2, levels=[1, 4, 9], colors="w", linewidths=[1.6,1.1,0.8])
plt.colorbar(cf, ax=ax, label=r"$\Delta\chi^2$ (profiled over $H_0r_\mathrm{d}$)")
x = np.linspace(0, 0.99, 10)
ax.plot(x, slope*x, "r--", lw=1.4, label=rf"$\Omega_\Lambda={slope:.2f}\,\Omega_\mathrm{{m}}$")
ax.annotate("", xy=(0.62, slope*0.62), xytext=(0.16, slope*0.16),
            arrowprops=dict(arrowstyle="<->", color="r", lw=1.8))
ax.text(0.20, slope*0.20+0.085, r"\textbf{along}: $\Delta\chi^2<1$", color="r",
        fontsize=8.5, rotation=62, ha="center", va="bottom")
ax.annotate("", xy=(0.30, 0.50), xytext=(0.30, 1.45),
            arrowprops=dict(arrowstyle="<->", color="k", lw=1.8))
ax.text(0.325, 0.60, r"\textbf{across}: $\Delta\chi^2\sim200$", fontsize=8.5)
ax.set_title(r"(a) the \emph{likelihood} is a wedge", fontsize=10)
ax.legend(loc="upper left", fontsize=8)

ax = axs[1]
ax.scatter(M, L, s=2.2, c="#009E73", alpha=np.clip(w/w.max()*3, 0, 1), rasterized=True)
ax.contour(OM, OL, dchi2, levels=[1, 4, 9], colors="k",
           linewidths=[1.4, 1.0, 0.7], linestyles=["-", "--", ":"])
ax.plot(x, slope*x, "r--", lw=1.4)
ax.axhline(1.5, color="grey", ls="-.", lw=1.6)
ax.text(0.97, 1.46, r"$\Omega_\Lambda$ prior ceiling: the flat top is the PRIOR, not the data",
        fontsize=8, va="top", ha="right", color="dimgrey",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
ax.set_title(r"(b) so the \emph{posterior} is a wedge", fontsize=10)

for ax in axs:
    ax.set_xlabel(r"$\Omega_\mathrm{m}$"); ax.set_xlim(0.01, 0.99); ax.set_ylim(0.10, 1.50)
axs[0].set_ylabel(r"$\Omega_\Lambda$")
fig.suptitle(r"Why the $(\Omega_\mathrm{m},\Omega_\Lambda)$ posterior is a wedge "
             r"pointing at the origin", fontsize=11.5, y=0.99)
fig.text(0.5, -0.04,
  r"Scaling \emph{both} $\Omega$'s together rescales $H$ by one common factor at every "
  r"epoch, which $E(z)=H/H(a{=}1)$ divides out --- so the fit is flat along "
  r"$\Omega_\Lambda\propto\Omega_\mathrm{m}$ ($\Delta\chi^2<1$ over a factor 7.5 in "
  r"$\Omega_\mathrm{m}$)."
  "\n"
  r"Changing \emph{one} of them moves $\Omega_\Lambda/\Omega_\mathrm{m}$, hence the "
  r"epoch when $\Lambda$ overtakes matter ($z_\Lambda\approx0.54$, right where DESI "
  r"observes) --- so the fit collapses across the wedge ($\Delta\chi^2\sim200$)."
  "\n"
  r"The valley is broken only by $\Omega_r a^{-4}$, which is $<0.1\%$ of the source "
  r"at $z<2.6$.  The wedge therefore runs until the \emph{prior} stops it, not the data.",
  ha="center", va="top", fontsize=7.6)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(f"plots/wedge_explained.{ext}", bbox_inches="tight",
                dpi=300 if ext == "png" else None)
print(f"slope Om_L/Om_m = {slope:.3f}")
print("saved plots/wedge_explained.{pdf,png}")
