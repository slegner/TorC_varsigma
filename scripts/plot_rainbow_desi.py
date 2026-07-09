"""
Rainbow plots (style of TorC_test_rainbow_cpr1_allparams.py) for the noH0
anchoring: a0 = 1 pinned, "today" := a = 1, E normalised by H(a=1) — the
convention of noH0/theory_jax_noH0.py.  All five parameters swept over
their full sampling priors; w_DE = P_DE/rho_DE from the field solution.

Run from anywhere (paths are absolute).  Writes
plots/TorC_rainbow_noH0_<param>.{pdf,png}.
"""
import sys, os
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
from pandas import read_csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

TORC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, TORC)
from torc.scipy_reference import (aseries, cpseries, aseriesp, cpseriesp, coupdiff,
                                   small, omega_r, P_deriveda0, rho_deriveda0)

h      = 0.7
omegar = omega_r(h)
c_kms  = 299792.458

STIGMA_FID = 1.0001
MUPS2_FID  = 4/3

SWEEPS = {
    "stigma":  np.unique(np.concatenate([np.linspace(1.0001, 1.3, 6),
                                         np.linspace(1.3, 3.0, 10)])),
    "mups2":   np.linspace(0.5, 3.0, 15),
    "OmL":     np.linspace(0.1, 1.5, 15),
    "omega_m": np.linspace(0.01, 0.99, 15),
    "h0rd":    np.linspace(3650.0, 18250.0, 15),
}
CBAR_LABELS = {
    "stigma":  r'$\varsigma$',
    "mups2":   r'$-\upsilon_2$',
    "OmL":     r'$\Omega_\Lambda$',
    "omega_m": r'$\Omega_\mathrm{m}$',
    "h0rd":    r'$H_0 r_\mathrm{d}$ [km/s]',
}

CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom_cmap", ["#785EF0", "#DC267F", "#FFB000"], N=256)

# --- DESI DR2 data ------------------------------------------------------------
_dd = os.path.join(TORC, "..", "desidr2")
data      = read_csv(os.path.join(_dd, "desidr2_mean.txt"),
                     header=None, index_col=None, sep=r"\s+", comment="#")
cov       = np.loadtxt(os.path.join(_dd, "desidr2_cov.txt"))
zs        = data.iloc[:, 0].to_numpy()
mean_data = data.iloc[:, 1].to_numpy()
types     = data.iloc[:, 2].to_numpy()
cov_inv   = np.linalg.inv(cov)

# --- LCDM reference fit ---------------------------------------------------------
def E(z, omegam):
    return np.sqrt(omegam*(1+z)**3 + omegar*(1+z)**4 + (1 - omegam - omegar))

def dm_lcdm(z, h0rd, omegam):
    return c_kms/h0rd*quad(lambda zp: 1/E(zp, omegam), 0, z)[0]

def dh_lcdm(z, h0rd, omegam):
    return c_kms/h0rd/E(z, omegam)

def dv_lcdm(z, h0rd, omegam):
    return (z*dm_lcdm(z, h0rd, omegam)**2*dh_lcdm(z, h0rd, omegam))**(1/3)

def chi2_lcdm(params):
    h0rd, omegam = params
    if omegam <= 0 or omegam >= 1 or h0rd <= 0:
        return 1e10
    fns = {"DM_over_rs": dm_lcdm, "DH_over_rs": dh_lcdm, "DV_over_rs": dv_lcdm}
    pred = [fns[t](z, h0rd, omegam) for z, t in zip(zs, types)]
    return (mean_data-pred) @ cov_inv @ (mean_data-pred)

result = minimize(chi2_lcdm, x0=[10145, 0.298], method='Nelder-Mead',
                  options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 10000})
h0rd_bf, omegam_bf = result.x
print(f"ΛCDM best fit: H0rd = {h0rd_bf:.1f} km/s,  Ωm = {omegam_bf:.4f}")

FID = dict(stigma=STIGMA_FID, mups2=MUPS2_FID,
           omega_m=omegam_bf, OmL=1 - omegam_bf - omegar, h0rd=h0rd_bf)

def lcdm_dv_over_rdz23(z):
    return dv_lcdm(z, h0rd_bf, omegam_bf) / z**(2/3)

def lcdm_ratio(z):
    return E(z, omegam_bf)*quad(lambda zp: 1/E(zp, omegam_bf), 0, z)[0]/z

# --- DESI data residuals --------------------------------------------------------
np.random.seed(42)
samples = multivariate_normal(mean_data, cov).rvs(10_000)
dmidx, dhidx, dvidx = (types == t for t in
                       ("DM_over_rs", "DH_over_rs", "DV_over_rs"))
dm_s, dh_s, dv_s = samples[:, dmidx], samples[:, dhidx], samples[:, dvidx]
zm, zv_bgs = zs[dmidx], zs[dvidx]

dv_derived = (zm*dm_s**2*dh_s)**(1/3)
zv_all = np.concatenate([zv_bgs, zm])
dv_all = np.hstack([dv_s, dv_derived])
o = np.argsort(zv_all); zv_all, dv_all = zv_all[o], dv_all[:, o]

dv_ratio_mean = (dv_all/zv_all**(2/3)).mean(axis=0)
dv_ratio_std  = (dv_all/zv_all**(2/3)).std(axis=0)
dv_residuals  = dv_ratio_mean - np.array([lcdm_dv_over_rdz23(z) for z in zv_all])

dm_ratio      = dm_s/(zm*dh_s)
dm_ratio_mean = dm_ratio.mean(axis=0)
dm_ratio_std  = dm_ratio.std(axis=0)
dm_residuals  = dm_ratio_mean - np.array([lcdm_ratio(z) for z in zm])

# --- theory grid ----------------------------------------------------------------
a_plot = np.linspace(0.001, 0.99, 500)
z_plot = 1/a_plot - 1

# --- theory runner: noH0 anchoring (a0 = 1, stop at a = 1, E := H/H(a=1)) -------
def run_theory(stigma, mups2, omega_m, OmL, h0rd):
    a0 = 1
    tauini = small(omega_m, h, stigma, mups2)
    y0 = [aseries(tauini, omega_m, h, OmL, a0, stigma, mups2),
          cpseries(tauini, omega_m, h, OmL, a0, stigma, mups2),
          aseriesp(tauini, omega_m, h, OmL, a0, stigma, mups2),
          cpseriesp(tauini, omega_m, h, OmL, a0, stigma, mups2)]

    def astop(t, y): return y[0] - 1.0
    astop.terminal = True
    rhs = lambda t, y: coupdiff(y, t, omega_m, h, OmL, a0, stigma, mups2)

    sol = solve_ivp(rhs, [tauini, np.inf], y0, events=astop,
                    rtol=1e-8, atol=1e-8)
    tsol  = interp1d(sol.y[0], sol.t, bounds_error=False)
    avals = np.logspace(np.log10(sol.y[0][0]), 0, 10000)
    tvals = tsol(avals); tvals[0] = tauini
    sol2 = solve_ivp(rhs, [tauini, np.inf], y0, t_eval=tvals,
                     events=astop, rtol=1e-8, atol=1e-8)

    asol, varpisol, hsol, dvarpisol = sol2.y
    aend, vpend, hend, dvpend = sol2.y_events[0][0]
    if aend > asol[-1]:
        asol      = np.append(asol,      aend)
        hsol      = np.append(hsol,      hend)
        varpisol  = np.append(varpisol,  vpend)
        dvarpisol = np.append(dvarpisol, dvpend)

    # normalise by H at a = 1 (NOT by 100h): today := a = 1
    H0_eff  = hend / aend**2
    z_arr   = (1/asol - 1)[::-1]
    E_arr   = (hsol/asol**2 / H0_eff)[::-1]
    h_interp = interp1d(z_arr, E_arr, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')

    def theory_dv(z):
        integ, _ = quad(lambda zp: 1/float(h_interp(zp)), 0, z)
        dm = c_kms/h0rd*integ
        dh = c_kms/h0rd/float(h_interp(z))
        return (z*dm**2*dh)**(1/3)/z**(2/3)

    def theory_dm_ratio(z):
        integ, _ = quad(lambda zp: 1/float(h_interp(zp)), 0, z)
        return float(h_interp(z))*integ/z

    dv_res = np.array([theory_dv(z)       - lcdm_dv_over_rdz23(z) for z in z_plot])
    dm_res = np.array([theory_dm_ratio(z) - lcdm_ratio(z)          for z in z_plot])

    rhoderiv = rho_deriveda0(asol, hsol, varpisol, dvarpisol,
                             omega_m, h, OmL, a0, stigma, mups2)
    P_DE   = P_deriveda0(asol, hsol, varpisol, dvarpisol,
                         omega_m, h, OmL, a0, stigma, mups2)
    w_de_arr = (P_DE/rhoderiv)[::-1]
    w_interp = interp1d(z_arr, w_de_arr, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    return dv_res, dm_res, w_interp(z_plot)


def make_rainbow(param_name, param_values):
    norm = Normalize(vmin=param_values.min(), vmax=param_values.max())
    sm   = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    ax_dv, ax_dm, ax_w = axes
    for ax in axes:
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)

    labels_dv = ["BGS", "LRG", "LRG", "LRG+ELG", "ELG", "QSO", r"Ly-$\alpha$"]
    labels_dm = ["LRG", "LRG", "LRG+ELG", "ELG", "QSO", r"Ly-$\alpha$"]
    dv_colors = [f"C{i}" for i in range(7)]
    dm_colors = dv_colors[1:]          # same tracer, same colour in both panels

    for a, res, err, label, color in zip(1/(1+zv_all), dv_residuals,
                                         dv_ratio_std, labels_dv, dv_colors):
        ax_dv.errorbar(a, res, yerr=err, marker='+', markersize=7,
                       linewidth=1.2, color=color, label=label, capsize=0,
                       zorder=5)
    for a, res, err, label, color in zip(1/(1+zm), dm_residuals,
                                         dm_ratio_std, labels_dm, dm_colors):
        ax_dm.errorbar(a, res, yerr=err, marker='+', markersize=7,
                       linewidth=1.2, color=color, label=label, capsize=0,
                       zorder=5)
    for w in [-1, 0, 1/3]:
        ax_w.axhline(w, color='grey', linestyle=':', linewidth=0.8)

    def theory_for(val):
        p = dict(FID); p[param_name] = val
        return run_theory(p["stigma"], p["mups2"], p["omega_m"], p["OmL"],
                          p["h0rd"])

    print(f"[{param_name}] fiducial + {len(param_values)} curves", flush=True)
    try:
        fid = theory_for(FID[param_name])
        for ax, curve in zip(axes, fid):
            ax.plot(a_plot, curve, color='k', linestyle='--', linewidth=1.2,
                    label='Fiducial')
    except Exception as e:
        print(f"  Fiducial SKIPPED: {e}")

    for val in param_values:
        print(f"  {param_name}={val:.4f}", flush=True)
        try:
            curves = theory_for(val)
            color = CUSTOM_CMAP(norm(val))
            for ax, curve in zip(axes, curves):
                ax.plot(a_plot, curve, color=color, linewidth=0.9)
        except Exception as e:
            print(f"    SKIPPED: {e}")

    fixed_y = param_name != "h0rd"
    ax_dv.set(ylabel=r'$D_\mathrm{V}/(r_d z^{2/3}) - (D_\mathrm{V}/(r_d z^{2/3}))_{\Lambda\mathrm{CDM}}$',
              **(dict(ylim=(-1.5, 1.5)) if fixed_y else {}))
    ax_dm.set(ylabel=r'$D_\mathrm{M}/(z D_H) - (D_\mathrm{M}/(z D_H))_{\Lambda\mathrm{CDM}}$',
              **(dict(ylim=(-0.25, 0.25)) if fixed_y else {}))
    ax_w.set( ylabel=r'$w_\mathrm{DE} = P_\mathrm{DE}/\rho_\mathrm{DE}$',
              **(dict(ylim=(-2, 0.5)) if fixed_y else {}), xlabel=r'$a$')
    ax_dv.set_xlim(0, 1)
    ax_dv.legend(fontsize=7)
    ax_dm.legend(fontsize=7)

    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='vertical',
                        fraction=0.02, pad=0.02)
    cbar.set_label(CBAR_LABELS[param_name])

    out = os.path.join(TORC, "plots", f"TorC_rainbow_noH0_{param_name}")
    fig.tight_layout()
    fig.savefig(out + ".pdf", bbox_inches='tight')
    fig.savefig(out + ".png", bbox_inches='tight', dpi=200)
    print(f"Saved {out}.pdf", flush=True)
    plt.close()


if __name__ == "__main__":
    for pname, values in SWEEPS.items():
        make_rainbow(pname, values)
