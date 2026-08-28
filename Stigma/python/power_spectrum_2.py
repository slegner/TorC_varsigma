"""
Power spectrum analysis for Stigma Q-torsion theory using CAMB.

This module generates CMB power spectra for different varsigma and upsilon2 parameter values
using the modified cosmology from the Stigma theory equations.

Refactored for efficiency and simplicity by eliminating redundant computations
and consolidating repetitive plotting logic.
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
from equations import omega_r, rhopsol
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Global Settings & Cache ---

# Enable LaTeX rendering for plots
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Global cache to store results of expensive computations
COMPUTATION_CACHE = {}

# --- Cosmological Parameters (Constants) ---
H0_value = 67.35987729255982
ombh2 = 0.022373199185278664
omch2 = 0.1199937689496612
mnu = 0.06
tau = 0.05436897788272714
As = 2.099999031911216e-9
ns = 0.9648887143375743
cpr = 0.8

# Derived parameters
h = H0_value / 100
omega_m = (ombh2 + omch2) / (h**2)
OmL = 0.685


def compute_power_spectrum(stigma, ups2):
    """
    Core function to compute CMB power spectrum and related data for given parameters.
    
    This function is the primary computational engine. The results are cached
    to avoid re-computation.
    
    Returns:
        dict: A dictionary containing all computed results, or None on failure.
    """
    try:
        # Get Stigma theory solutions
        asol, _, hsol, _, _, P_DE, rho_DE = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
        
        # Set up CAMB parameters
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0_value, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=tau)
        pars.set_dark_energy_rho_p_a(a_p=asol, a_rho=asol, p=P_DE, rho=rho_DE)
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        pars.InitPower.set_params(As=As, ns=ns, r=0)
        
        # Get results from CAMB
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        totCL_TT = powers['total'][:, 0]
        ls = np.arange(len(totCL_TT))
        
        # Calculate Hubble parameter ratio check
        zlist = 1 / asol - 1
        hresult = H0_value / (results.hubble_parameter(zlist) / (1 + zlist))
        
        print(f"✓ Success for σ={stigma:.3f}, υ₂={ups2:.3f}")
        
        return {
            "ls": ls, "cl_tt": totCL_TT, "hresult": hresult,
            "zlist": zlist, "hsol": hsol, "success": True
        }
        
    except Exception as e:
        print(f"✗ Failed for σ={stigma:.3f}, υ₂={ups2:.3f}: {e}")
        return {"success": False}


def get_or_compute_data(stigma, ups2):
    """
    A wrapper function that retrieves computed data from cache or runs the computation.
    """
    param_key = (stigma, ups2)
    if param_key not in COMPUTATION_CACHE:
        # If not in cache, compute it and store the result
        COMPUTATION_CACHE[param_key] = compute_power_spectrum(stigma, ups2)
    return COMPUTATION_CACHE[param_key]


def plot_spectra_comparison(param_to_vary, param_values, fixed_param_value, cmap_name, output_filename):
    """
    A generalized function to plot CMB power spectra comparisons.
    
    Args:
        param_to_vary (str): The name of the parameter to vary ('varsigma' or 'upsilon2').
        param_values (list): A list of values for the parameter being varied.
        fixed_param_value (float): The value of the other parameter, held constant.
        cmap_name (str): The name of the matplotlib colormap to use.
        output_filename (str): The filename for the saved plot.
    """
    fig, PSlin = plt.subplots(figsize=(12, 8))
    my_cmap = plt.get_cmap(cmap_name)
    colors = my_cmap(np.linspace(0, 1, len(param_values)))

    # --- Setup split log/linear axes ---
    divider = make_axes_locatable(PSlin)
    PSlog = divider.append_axes("left", size="80%", pad=0, sharey=PSlin)
    PSlog.set_xscale('log')
    PSlin.spines['left'].set_visible(False)
    PSlog.spines['right'].set_visible(False)
    PSlin.yaxis.set_ticks_position('right')

    # --- Plotting Loop ---
    for i, val in enumerate(param_values):
        stigma = val if param_to_vary == 'varsigma' else fixed_param_value
        ups2 = val if param_to_vary == 'upsilon2' else fixed_param_value
        
        data = get_or_compute_data(stigma, ups2)
        if data["success"]:
            label = f"$\\varsigma = {val:.2f}$" if param_to_vary == 'varsigma' else f"$\\upsilon_2 = {val:.3f}$"
            PSlin.plot(data['ls'], data['cl_tt'], color=colors[i], label=label)
            PSlog.plot(data['ls'], data['cl_tt'], color=colors[i])

    # --- Plot Reference Case ---
    ref_data = get_or_compute_data(1.0, -4/3)
    if ref_data["success"]:
        PSlin.plot(ref_data['ls'], ref_data['cl_tt'], c='k', ls='--', linewidth=2, label='Reference')
        PSlog.plot(ref_data['ls'], ref_data['cl_tt'], c='k', ls='--', linewidth=2)

    # --- Final Customization ---
    PSlin.set_xlim(30, 2500)
    PSlog.set_xlim(2, 30)
    PSlin.axvline(x=30, c="grey", ls="--")
    PSlog.set_ylabel("$D_\\ell (\\mu K^2)$")
    PSlin.set_xlabel(r'$\ell$')
    PSlin.xaxis.set_label_coords(0.25, -0.1)
    PSlog.yaxis.set_label_coords(-0.2, 0.5)

    title_varsigma = f"{fixed_param_value:.2f}" if param_to_vary == 'upsilon2' else "\\text{varied}"
    title_upsilon2 = f"{fixed_param_value:.3f}" if param_to_vary == 'varsigma' else "\\text{varied}"
    PSlin.set_title(f'CMB TT Power Spectrum: $\\varsigma = {title_varsigma}$, $\\upsilon_2 = {title_upsilon2}$')
    
    PSlin.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()


def plot_hubble_comparison(output_filename):
    """
    Generalized function to plot Hubble parameter evolution (H(z)) and the
    Hubble ratio (hresult) side-by-side.
    """
    stigma_values = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    ups2_values = [-2.5, -2.0, -1.5, -4/3, -1.0, -0.5]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    cmap1, cmap2 = plt.cm.viridis, plt.cm.plasma
    
    # Plot H(z) vs z
    ax1, ax2 = axes[0, 0], axes[0, 1]
    for i, stigma in enumerate(stigma_values):
        data = get_or_compute_data(stigma, -4/3)
        if data["success"]:
            ax1.plot(data['zlist'], data['hsol'], color=cmap1(i / len(stigma_values)), label=f'$\\varsigma = {stigma:.2f}$')
            
    for i, ups2 in enumerate(ups2_values):
        data = get_or_compute_data(1.0, ups2)
        if data["success"]:
            ax2.plot(data['zlist'], data['hsol'], color=cmap2(i / len(ups2_values)), label=f'$\\upsilon_2 = {ups2:.3f}$')

    # Plot hresult vs z
    ax3, ax4 = axes[1, 0], axes[1, 1]
    for i, stigma in enumerate(stigma_values):
        data = get_or_compute_data(stigma, -4/3)
        if data["success"]:
            ax3.plot(data['zlist'], data['hresult'], color=cmap1(i / len(stigma_values)), label=f'$\\varsigma = {stigma:.2f}$')

    for i, ups2 in enumerate(ups2_values):
        data = get_or_compute_data(1.0, ups2)
        if data["success"]:
            ax4.plot(data['zlist'], data['hresult'], color=cmap2(i / len(ups2_values)), label=f'$\\upsilon_2 = {ups2:.3f}$')

    # --- Formatting ---
    ax1.set(title='Hubble Parameter Evolution ($\\upsilon_2 = -4/3$)', ylabel='$H(z)$ [km/s/Mpc]')
    ax2.set(title='Hubble Parameter Evolution ($\\varsigma = 1.0$)')
    ax3.set(title='Hubble Parameter Ratio ($\\upsilon_2 = -4/3$)', xlabel='Redshift $z$', ylabel='$H_0/H(z) \\cdot (1+z)$')
    ax4.set(title='Hubble Parameter Ratio ($\\varsigma = 1.0$)', xlabel='Redshift $z$')

    for ax in axes.flatten():
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()


def plot_comprehensive_parameter_survey(output_filename):
    """Create a comprehensive survey plot showing parameter space."""
    stigma_grid = [0.9, 0.95, 1.0, 1.05, 1.1]
    ups2_grid = [-2.0, -1.5, -4/3, -1.0, -0.5]
    
    fig, axes = plt.subplots(len(ups2_grid), len(stigma_grid), figsize=(20, 16), sharex=True, sharey=True)
    fig.suptitle('CMB Power Spectrum Parameter Survey: $\\varsigma$ vs $\\upsilon_2$', fontsize=16)
    
    for i, ups2 in enumerate(ups2_grid):
        for j, stigma in enumerate(stigma_grid):
            ax = axes[i, j]
            data = get_or_compute_data(stigma, ups2)
            
            if data["success"]:
                ax.plot(data['ls'], data['cl_tt'], 'b-', linewidth=1)
                ax.set_xscale('log')
            else:
                ax.text(0.5, 0.5, 'Failed', ha='center', va='center', transform=ax.transAxes)
            
            # Formatting
            if i == 0: ax.set_title(f'$\\varsigma={stigma:.2f}$')
            if j == 0: ax.set_ylabel(f'$\\upsilon_2={ups2:.3f}$\n$D_\\ell$')
            if i == len(ups2_grid) - 1: ax.set_xlabel('$\\ell$')
            ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run all analyses."""
    print("Starting Stigma Theory Power Spectrum Analysis...")
    print("=" * 50)
    
    # --- Run Analyses ---
    print("\n1. Computing power spectra for different σ values...")
    plot_spectra_comparison(
        param_to_vary='varsigma',
        param_values=[0.8, 0.85, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.15, 1.2],
        fixed_param_value=-4/3,
        cmap_name='viridis',
        output_filename='cmb_power_spectrum_varsigma_comparison.pdf'
    )
    
    print("\n2. Computing power spectra for different υ₂ values...")
    plot_spectra_comparison(
        param_to_vary='upsilon2',
        param_values=[-2.5, -2.0, -1.8, -1.5, -4/3, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2],
        fixed_param_value=1.0,
        cmap_name='plasma',
        output_filename='cmb_power_spectrum_upsilon2_comparison.pdf'
    )
    
    print("\n3. Computing Hubble parameter comparisons...")
    plot_hubble_comparison('hubble_comparison.pdf')
    
    print("\n4. Creating comprehensive parameter survey...")
    plot_comprehensive_parameter_survey('parameter_survey_grid.pdf')
    
    print("\nAnalysis complete! Check the generated PDF files.")


if __name__ == "__main__":
    main()
