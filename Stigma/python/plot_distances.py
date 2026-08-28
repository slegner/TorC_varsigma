import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from pandas import read_csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 2. LCDM REFERENCE (fit to DESI DR2) ──────────────────────────────────────

# Load DESI DR2 data
data = read_csv("../../../desidr2/desidr2_mean.txt",
                header=None, index_col=None, sep=r"\s+", comment="#")
cov = np.loadtxt("../../../desidr2/desidr2_cov.txt")

zs = data.iloc[:, 0].to_numpy()
mean_data = data.iloc[:, 1].to_numpy()
types = data.iloc[:, 2].to_numpy()
cov_inv = np.linalg.inv(cov)

def E_lcdm(z, omegam):
    return np.sqrt(omegam*(1+z)**3 + omegar*(1+z)**4 + (1-omegam-omegar))

def dm_lcdm(z, h0rd, omegam):
    integral, _ = quad(lambda zp: 1/E_lcdm(zp, omegam), 0, z)
    return c_kms / h0rd * integral

def dh_lcdm(z, h0rd, omegam):
    return c_kms / h0rd / E_lcdm(z, omegam)

def dv_lcdm(z, h0rd, omegam):
    return (z * dm_lcdm(z, h0rd, omegam)**2 * dh_lcdm(z, h0rd, omegam))**(1/3)

def chi2_lcdm(params):
    h0rd, omegam = params
    if omegam <= 0 or omegam >= 1 or h0rd <= 0:
        return 1e10
    fns = {"DM_over_rs": dm_lcdm, "DH_over_rs": dh_lcdm, "DV_over_rs": dv_lcdm}
    pred = [fns[t](z, h0rd, omegam) for z, t in zip(zs, types)]
    return (mean_data - pred) @ cov_inv @ (mean_data - pred)

result = minimize(chi2_lcdm, x0=[10145, 0.298], method='Nelder-Mead',
                  options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 10000})
h0rd_bf, omegam_bf = result.x
print(f"ΛCDM best fit: H0rd = {h0rd_bf:.1f} km/s,  Ωm = {omegam_bf:.4f}")

def lcdm_ratio(z):
    integral, _ = quad(lambda zp: 1/E_lcdm(zp, omegam_bf), 0, z)
    return E_lcdm(z, omegam_bf) * integral / z


# ── 3. DESI DR2 DATA POINTS ───────────────────────────────────────────────────

np.random.seed(42)
samples = multivariate_normal(mean_data, cov).rvs(10_000)

dmidx = types == "DM_over_rs"
dhidx = types == "DH_over_rs"
dm_s = samples[:, dmidx]
dh_s = samples[:, dhidx]
zm = zs[dmidx]

ratio = dm_s / (zm * dh_s)
ratio_mean = ratio.mean(axis=0)
ratio_std = ratio.std(axis=0)
residuals = ratio_mean - np.array([lcdm_ratio(z) for z in zm])

labels = ["LRG", "LRG", "LRG+ELG", "ELG", "QSO", r"Ly-$\alpha$"]
colors = ['C6', 'C4', 'C3', 'C3', 'C2', 'C1']
a_vals = 1 / (1 + zm)


z_arr, H_arr = rhopsol(cpr, omega_m, h, OmL, stigma, mups2)
h_interp = interp1d(z_arr, H_arr / H_arr[0], kind='cubic')

def theory_ratio(z):
    integral, _ = quad(lambda zp: 1/float(h_interp(zp)), 0, z)
    return float(h_interp(z)) * integral / z


# ── 5. PLOT ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 5))
ax.axhline(0, color='k', linestyle='--')

for a, res, err, label, color in zip(a_vals, residuals, ratio_std, labels, colors):
    ax.errorbar(a, res, yerr=err, marker='+', markersize=8, linewidth=1.5,
                color=color, label=label, capsize=0)

a_plot = np.linspace(0.01, 0.99, 200)
z_plot = 1 / a_plot - 1
theory_res = np.array([theory_ratio(z) - lcdm_ratio(z) for z in z_plot])
ax.plot(a_plot, theory_res, color='k', linewidth=1.5, label='My theory')

ax.set(
    xlabel=r'$a$',
    ylabel=r'$D_\mathrm{M}/(zD_H) - (D_\mathrm{M}/(zD_H))_{\Lambda\mathrm{CDM}}$',
    xlim=(0, 1),
    ylim=(-0.25, 0.25),
    title='DESI DR2',
)
ax.legend(fontsize='small')
fig.tight_layout()

