import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import camb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from equations_double import rhopsol, constraint

# --- Baseline cosmological params ---
h       = 0.67
ombh2   = 0.02242
omch2   = 0.11933
mnu     = 0.06
nnu     = 3.046
tau     = 0.0544
As      = 2.1e-9
ns      = 0.965
H0      = h * 100
omega_m = 0.3
LMAX    = 2500
OmL     = 0.7
cpr     = 0.99

# --- Baseline Stigma params ---
STIGMA_FID = 1.0
MUPS2_FID  = 4.0 / 3.0

SCALE_FACTORS_A = np.logspace(-6, 0, 1000)
Z_VALUES        = 1.0 / SCALE_FACTORS_A - 1.0

# --- Style (matches reference) ---
plt.rcParams.update({
    "text.usetex": True, "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 11, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 1.0,
})

CMAP_COLORS = ["#785EF0", "#DC267F", "#FFB000"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_cmap", CMAP_COLORS, N=256)


def compute_hr(asol, rho_DE, P_DE):
    """H0/(aH(a)) computed via CAMB background — identical to original."""
    pars = camb.CAMBparams()
    pars.set_dark_energy_rho_p_a(a_p=asol, a_rho=asol, rho=rho_DE, p=P_DE)
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, nnu=nnu,
                       omk=0, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_for_lmax(LMAX, lens_potential_accuracy=0)
    bg = camb.get_background(pars)
    return H0 / (SCALE_FACTORS_A * bg.hubble_parameter(Z_VALUES))


def constraint_ratio(asol, hsol, varpisol, a0, stigma, mups2):
    """
    |C| / |T_hd| where T_hd is the H² term in the (0,0) Friedmann constraint.
    Should sit at the level of rtol (~1e-8) everywhere.
    """
    T_hd = (3.0 * asol * a0 / (4.0 * stigma)) * \
           (hsol**2 * mups2 * (-varpisol**2 + 4.0 * stigma)) / asol**2
    return T_hd   # caller divides cval by this


def make_rainbow(param_values, param_name, cbar_label, fid_label,
                 fixed_stigma, fixed_mups2, filename):

    norm = Normalize(vmin=param_values.min(), vmax=param_values.max())
    sm   = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm)
    sm.set_array([])

    # --- Baseline run (dashed black) ---
    print(f"Running fiducial: stigma={fixed_stigma}, mups2={fixed_mups2}")
    asol_fid, varpisol_fid, hsol_fid, dvarpisol_fid, a0_fid, P_fid, rho_fid, _ = rhopsol(
        cpr, omega_m, h, OmL, fixed_stigma, fixed_mups2)
    hr_fid = compute_hr(asol_fid, rho_fid, P_fid)

    # --- Loop over parameter values ---
    run_data = []
    for val in param_values:
        stigma = val if param_name == 'stigma' else fixed_stigma
        mups2  = val if param_name == 'mups2'  else fixed_mups2
        print(f"  {param_name}={val:.4f}  stigma={stigma:.4f}  mups2={mups2:.4f}")
        try:
            asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE, cval = rhopsol(
                cpr, omega_m, h, OmL, stigma, mups2)
            hr = compute_hr(asol, rho_DE, P_DE)
            run_data.append({
                'val': val, 'asol': asol, 'varpisol': varpisol,
                'hsol': hsol, 'dvarpisol': dvarpisol, 'a0': a0,
                'hr': hr, 'cval': cval, 'stigma': stigma, 'mups2': mups2,
            })
        except Exception as e:
            print(f"    SKIPPED: {e}")

    # --- Figure ---
    fig_width_in  = (510 * 0.99) / 72.27
    fig_height_in = fig_width_in * (6 / 8) / 2.4

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fig_width_in, fig_height_in),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.30},
    )

    # Left: H0/(aH(a)) — identical to original
    for d in run_data:
        ax1.plot(SCALE_FACTORS_A, d['hr'], color=CUSTOM_CMAP(norm(d['val'])))
    ax1.plot(SCALE_FACTORS_A, hr_fid, '--', color='black', label=fid_label, zorder=5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e-6, 1)
    ax1.set_ylim(1e-4, 2)
    ax1.legend(loc='upper left')
    ax1.set_xlabel(r'$a$')
    ax1.set_ylabel(r'$H_0/(aH(a))$')

    # Inset: residual vs fiducial
    axins = ax1.inset_axes([0.53, 0.04, 0.44, 0.36])
    for d in run_data:
        axins.plot(SCALE_FACTORS_A, d['hr'] - hr_fid,
                   color=CUSTOM_CMAP(norm(d['val'])))
    axins.axhline(0, c='k', ls='--', lw=0.8)
    axins.set_xlim(0, 1)
    axins.tick_params(labelsize=6)
    axins.set_xlabel(r'$a$', fontsize=7)
    axins.set_ylabel(r'$\Delta(H_0/aH)$', fontsize=7)

    # Right: varpisol vs asol
    for d in run_data:
        ax2.plot(d['asol'], d['varpisol'], color=CUSTOM_CMAP(norm(d['val'])))
    ax2.plot(asol_fid, varpisol_fid, '--', color='black', label=fid_label, zorder=5)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$a$')
    ax2.set_ylabel(r'$\varpi$')
    ax2.legend(loc='best')

    cbar = fig.colorbar(sm, ax=ax2, orientation='vertical',
                        fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)

    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

    return run_data, asol_fid, varpisol_fid, hsol_fid, dvarpisol_fid, a0_fid


def plot_constraint_diagnostic(run_data, asol_fid, varpisol_fid, hsol_fid,
                                dvarpisol_fid, a0_fid,
                                param_name, cbar_label, fid_label,
                                fixed_stigma, fixed_mups2, filename):
    """
    Plot |C| / |T_hd| vs a.
    T_hd is the highest-derivative (H²) term in the (0,0) Friedmann constraint.
    The ratio should stay at the level of rtol (~1e-8) everywhere.
    """
    norm = Normalize(vmin=min(d['val'] for d in run_data),
                     vmax=max(d['val'] for d in run_data))
    sm   = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm)
    sm.set_array([])

    fig_width_in  = (510 * 0.99) / 72.27
    fig_height_in = fig_width_in * (6 / 8) / 2.4

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    # Fiducial (need to run rhopsol again for cval)
    _, _, _, _, _, _, _, cval_fid = rhopsol(
        cpr, omega_m, h, OmL, fixed_stigma, fixed_mups2)
    T_hd_fid = constraint_ratio(asol_fid, hsol_fid, varpisol_fid,
                                  a0_fid, fixed_stigma, fixed_mups2)
    ax.plot(asol_fid, np.abs(cval_fid) / np.abs(T_hd_fid), '--', color='black',
            label=fid_label, zorder=5)

    for d in run_data:
        T_hd  = constraint_ratio(d['asol'], d['hsol'], d['varpisol'],
                                  d['a0'], d['stigma'], d['mups2'])
        ratio = np.abs(d['cval']) / np.abs(T_hd)
        ax.plot(d['asol'], ratio, color=CUSTOM_CMAP(norm(d['val'])))

    ax.axhline(1e-8, ls=':', color='grey', lw=0.8, label=r'$\mathtt{rtol}=10^{-8}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$a$')
    ax.set_ylabel(r'$|\mathcal{C}|\,/\,|T_{\mathrm{hd}}|$')
    ax.set_title(r'Constraint residual: $(0,0)$ Friedmann / $H^2$ term')
    ax.legend(loc='upper right')

    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


# ======================================================================
# Rainbow 1: vary stigma, fix mups2 = 4/3
# ======================================================================
stigma_values = np.unique(np.concatenate([
    np.linspace(1.001, 1.3, 12),
    np.linspace(1.3,   2.0,  8),
]))

run_data_s, asol_fid, vp_fid, hs_fid, dvp_fid, a0_fid = make_rainbow(
    param_values  = stigma_values,
    param_name    = 'stigma',
    cbar_label    = r'$\varsigma$',
    fid_label     = r'Baseline ($\varsigma=1,\;m\upsilon_2=4/3$)',
    fixed_stigma  = STIGMA_FID,
    fixed_mups2   = MUPS2_FID,
    filename      = 'rainbow_stigma.pdf',
)

plot_constraint_diagnostic(
    run_data_s, asol_fid, vp_fid, hs_fid, dvp_fid, a0_fid,
    param_name    = 'stigma',
    cbar_label    = r'$\varsigma$',
    fid_label     = r'Baseline ($\varsigma=1,\;m\upsilon_2=4/3$)',
    fixed_stigma  = STIGMA_FID,
    fixed_mups2   = MUPS2_FID,
    filename      = 'constraint_diagnostic_stigma.pdf',
)

# ======================================================================
# Rainbow 2: vary mups2, fix stigma = 1
# ======================================================================
mups2_values = np.linspace(2.0/3.0, 1.8, 10)

run_data_m, asol_fid, vp_fid, hs_fid, dvp_fid, a0_fid = make_rainbow(
    param_values  = mups2_values,
    param_name    = 'mups2',
    cbar_label    = r'$m\upsilon_2$',
    fid_label     = r'Baseline ($\varsigma=1,\;m\upsilon_2=4/3$)',
    fixed_stigma  = STIGMA_FID,
    fixed_mups2   = MUPS2_FID,
    filename      = 'rainbow_mups2.pdf',
)

plot_constraint_diagnostic(
    run_data_m, asol_fid, vp_fid, hs_fid, dvp_fid, a0_fid,
    param_name    = 'mups2',
    cbar_label    = r'$m\upsilon_2$',
    fid_label     = r'Baseline ($\varsigma=1,\;m\upsilon_2=4/3$)',
    fixed_stigma  = STIGMA_FID,
    fixed_mups2   = MUPS2_FID,
    filename      = 'constraint_diagnostic_mups2.pdf',
)
