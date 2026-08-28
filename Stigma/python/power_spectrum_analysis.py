"""
Power spectrum analysis for Stigma Q-torsion theory using CAMB.

This module generates CMB power spectra for different varsigma and upsilon2 parameter values
using the modified cosmology from the Stigma theory equations.
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
from equations import omega_r, rhopsol, StigmaConstraintError
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Cosmological parameters
H0_value = 67.35987729255982
ombh2 = 0.022373199185278664
omch2 = 0.1199937689496612
mnu = 0.06
tau = 0.05436897788272714
As = 2.099999031911216e-9
ns = 0.9648887143375743

# Derived parameters
h = H0_value / 100
omega_m = (ombh2 + omch2) / (h**2)
OmL = 0.685
cpr = 0.8


def compute_power_spectrum(stigma, ups2, plot_label=None):
    """
    Compute CMB power spectrum for given stigma and ups2 values.
    
    Args:
        stigma: Stigma parameter value (displayed as varsigma)
        ups2: Upsilon2 parameter value (displayed as upsilon_2)
        plot_label: Label for plotting (optional)
        
    Returns:
        tuple: (ls, totCL_TT, pars, results) - multipoles, power spectrum, CAMB params, results
    """
    try:
        # Get Stigma theory solutions
        asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
        
        # Set up CAMB parameters
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0_value, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=tau)
        
        # Set dark energy from Stigma theory
        pars.set_dark_energy_rho_p_a(a_p=asol, a_rho=asol, p=P_DE, rho=rho_DE)
        
        # Set up power spectrum calculation
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        pars.InitPower.set_params(As=As, ns=ns, r=0)
        
        # Get results
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        totCL = powers['total']
        ls = np.arange(totCL.shape[0])
        
        # Hubble parameter check
        zlist = 1/asol - 1
        hresult = H0_value / (results.hubble_parameter(zlist) / (1 + zlist))
        
        print(f"✓ Success for $\\varsigma$={stigma:.3f}, $\\upsilon_2$={ups2:.3f}")
        
        return ls, totCL[:, 0], pars, results, hresult
        
    except StigmaConstraintError as e:
        print(f"✗ Constraint violation for $\\varsigma$={stigma:.3f}, $\\upsilon_2$={ups2:.3f}: {e}")
        return None, None, None, None, None
    except Exception as e:
        print(f"✗ Failed for $\\varsigma$={stigma:.3f}, $\\upsilon_2$={ups2:.3f}: {e}")
        return None, None, None, None, None


def plot_power_spectra_varsigma_comparison():
    """Plot varpisol for different varsigma values."""

    # Stigma values to test (displayed as varsigma) - expanded range
    stigma_list = [0.85, 0.9, 0.92, 0.95, 0.97, 1.0, 1.02, 1.05, 1.07, 1.1, 1.15]
    ups2_fixed = -4/3

    # Figure settings
    my_cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define color iterator
    color = iter(my_cmap(np.linspace(0, 1, len(stigma_list))))

    for stigma in stigma_list:
        c = next(color)
        try:
            asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE = rhopsol(cpr, omega_m, h, OmL, stigma, ups2_fixed)
            ax.plot(asol, varpisol, color=c, label=f'$\\varsigma = {stigma:.2f}$')
            print(f"✓ $\\varsigma$={stigma:.3f}")
        except StigmaConstraintError as e:
            print(f"✗ Constraint violation for $\\varsigma$={stigma:.3f}: {e}")
        except Exception as e:
            print(f"✗ Failed for $\\varsigma$={stigma:.3f}: {e}")

    ax.set_xscale("log")
    ax.set_xlabel('$a$')
    ax.set_ylabel('$\\varpi$')
    ax.set_title(f'Scalar field $\\varpi(a)$: $\\upsilon_2 = {ups2_fixed:.3f}$')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('Plots/varpisol_varsigma_comparison.pdf', bbox_inches='tight')
    plt.show()


def plot_power_spectra_upsilon2_comparison():
    """Plot varpisol for different upsilon2 values."""

    # ups2 values to test (displayed as upsilon_2) - expanded range
    ups2_list = [-2.5, -2.0, -1.8, -1.5, -4/3, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2]
    stigma_fixed = 1.0

    # Figure settings
    my_cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define color iterator
    color = iter(my_cmap(np.linspace(0, 1, len(ups2_list))))

    for ups2 in ups2_list:
        c = next(color)
        try:
            asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE = rhopsol(cpr, omega_m, h, OmL, stigma_fixed, ups2)
            ax.plot(asol, varpisol, color=c, label=f'$\\upsilon_2 = {ups2:.3f}$')
            print(f"✓ $\\upsilon_2$={ups2:.3f}")
        except StigmaConstraintError as e:
            print(f"✗ Constraint violation for $\\upsilon_2$={ups2:.3f}: {e}")
        except Exception as e:
            print(f"✗ Failed for $\\upsilon_2$={ups2:.3f}: {e}")

    ax.set_xscale("log")
    ax.set_xlabel('$a$')
    ax.set_ylabel('$\\varpi$')
    ax.set_title(f'Scalar field $\\varpi(a)$: $\\varsigma = {stigma_fixed:.1f}$')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('Plots/varpisol_upsilon2_comparison.pdf', bbox_inches='tight')
    plt.show()


def plot_hubble_results():
    """Plot hresult with hubble ratio overlaid as dashed lines."""
    
    # Parameter ranges
    stigma_values = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    ups2_values = [-2.5, -2.0, -1.5, -4/3, -1.0, -0.5]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Different stigma values
    for i, stigma in enumerate(stigma_values):
        try:
            ls, totCL_TT, pars, results, hresult = compute_power_spectrum(stigma, -4/3)
            if hresult is not None:
                asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE = rhopsol(cpr, omega_m, h, OmL, stigma, -4/3)
                hubble_ratio = asol * h * 100 / hsol / a0
                
                color = plt.cm.viridis(i / len(stigma_values))
                # Plot hresult as solid line
                ax1.plot(asol, hresult, color=color, label=f'$\\varsigma = {stigma:.2f}$ (hresult)', linewidth=2)
                # Plot hubble ratio as black dashed line
                ax1.plot(asol, hubble_ratio, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
            
        except Exception as e:
            print(f"Failed for $\\varsigma$={stigma}: {e}")
    
    ax1.set_xlabel('Scale Factor $a$')
    ax1.set_ylabel('Value')
    ax1.set_title('hresult (solid) vs Hubble Ratio (dashed) ($\\upsilon_2 = -4/3$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Different ups2 values
    for i, ups2 in enumerate(ups2_values):
        try:
            ls, totCL_TT, pars, results, hresult = compute_power_spectrum(1.0, ups2)
            if hresult is not None:
                asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE = rhopsol(cpr, omega_m, h, OmL, 1.0, ups2)
                hubble_ratio = asol * h * 100 / hsol / a0
                
                color = plt.cm.plasma(i / len(ups2_values))
                # Plot hresult as solid line
                ax2.plot(asol, hresult, color=color, label=f'$\\upsilon_2 = {ups2:.3f}$ (hresult)', linewidth=2)
                # Plot hubble ratio as black dashed line
                ax2.plot(asol, hubble_ratio, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
            
        except Exception as e:
            print(f"Failed for $\\upsilon_2$={ups2}: {e}")
    
    ax2.set_xlabel('Scale Factor $a$')
    ax2.set_ylabel('Value')
    ax2.set_title('hresult (solid) vs Hubble Ratio (dashed) ($\\varsigma = 1.0$)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('Plots/hubble_results.pdf', bbox_inches='tight')
    plt.show()






def main():
    """Main function to run all analyses."""
    print("Starting Stigma Theory Power Spectrum Analysis...")
    print("=" * 50)
    
    print("\n1. Computing power spectra for different $\\varsigma$ values...")
    plot_power_spectra_varsigma_comparison()
    
    print("\n2. Computing power spectra for different $\\upsilon_2$ values...")
    plot_power_spectra_upsilon2_comparison()
    
    print("\n3. Computing Hubble results...")
    plot_hubble_results()
    
    print("\nAnalysis complete! Check the generated PDF files in Plots/:")
    print("- Plots/varpisol_varsigma_comparison.pdf")
    print("- Plots/varpisol_upsilon2_comparison.pdf")
    print("- Plots/hubble_results.pdf")


if __name__ == "__main__":
    main()
