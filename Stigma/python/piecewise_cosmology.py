from equationsa0 import omega_r, rhopsol
import matplotlib.pyplot as plt
import numpy as np

def gi_ratio(wi, stigma, ups2, plus_solution=False):
    """
    Calculate gi ratio from the gisolution expressions
    """
    if plus_solution:
        return (4*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma + 3*stigma*wi)**2 + (1 + 3*wi)*(((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2))))/(stigma*ups2*(1 + 2*stigma + 6*(-1 + stigma)*wi)*(-7 - 3*wi + 4*stigma*(1 + 3*wi)))
    else:
        return (4*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma + 3*stigma*wi)**2 - (1 + 3*wi)*(((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2))))/(stigma*ups2*(1 + 2*stigma + 6*(-1 + stigma)*wi)*(-7 - 3*wi + 4*stigma*(1 + 3*wi)))

def find_epoch_transitions(asol, omega_m_a, omega_r_a, omega_lambda_a):
    """
    Find scale factors where epoch transitions occur
    """
    # Matter-radiation equality: Ω_m = Ω_r
    mr_diff = np.abs(omega_m_a - omega_r_a)
    a_mr_idx = np.argmin(mr_diff)
    a_mr_equal = asol[a_mr_idx]
    
    # Matter-dark energy equality: Ω_m = Ω_Λ  
    ml_diff = np.abs(omega_m_a - omega_lambda_a)
    a_ml_idx = np.argmin(ml_diff)
    a_ml_equal = asol[a_ml_idx]
    
    print(f"Matter-radiation equality at a = {a_mr_equal:.6f}")
    print(f"Matter-dark energy equality at a = {a_ml_equal:.6f}")
    
    return a_mr_equal, a_ml_equal

def construct_piecewise_gi_cosmology(asol, a_mr_equal, a_ml_equal, stigma, ups2, plus_solution=False):
    """
    Construct piecewise gi cosmology with different wi in each epoch
    """
    # Define equation of state for each epoch
    wi_radiation = 1/3    # Radiation dominated
    wi_matter = 0         # Matter dominated  
    wi_dark_energy = -1   # Dark energy dominated
    
    # Calculate gi for each epoch
    gi_radiation = gi_ratio(wi_radiation, stigma, ups2, plus_solution)
    gi_matter = gi_ratio(wi_matter, stigma, ups2, plus_solution) 
    gi_dark_energy = gi_ratio(wi_dark_energy, stigma, ups2, plus_solution)
    
    print(f"gi values:")
    print(f"  Radiation (w=1/3): gi = {gi_radiation:.4f}")
    print(f"  Matter (w=0): gi = {gi_matter:.4f}")
    print(f"  Dark Energy (w=-1): gi = {gi_dark_energy:.4f}")
    
    # Construct piecewise gi array
    gi_piecewise = np.zeros_like(asol)
    wi_piecewise = np.zeros_like(asol)
    
    # Assign gi values based on epoch
    radiation_mask = asol < a_mr_equal
    matter_mask = (asol >= a_mr_equal) & (asol < a_ml_equal)
    de_mask = asol >= a_ml_equal
    
    gi_piecewise[radiation_mask] = gi_radiation
    gi_piecewise[matter_mask] = gi_matter
    gi_piecewise[de_mask] = gi_dark_energy
    
    wi_piecewise[radiation_mask] = wi_radiation
    wi_piecewise[matter_mask] = wi_matter  
    wi_piecewise[de_mask] = wi_dark_energy
    
    return gi_piecewise, wi_piecewise

def calculate_piecewise_hubble(asol, gi_piecewise, omega_m, omega_r_val, OmL, h, kappa=1.0):
    """
    Calculate Hubble parameter for piecewise gi cosmology
    """
    # Total density at each scale factor
    rho_r = omega_r_val / asol**4
    rho_m = omega_m / asol**3  
    rho_lambda = OmL * np.ones_like(asol)
    rho_total = rho_r + rho_m + rho_lambda
    
    # Modified Friedmann: H² = (gi * κ * ρ_total) / 3
    H_piecewise_squared = (gi_piecewise * kappa * rho_total) / 3
    H_piecewise = np.sqrt(H_piecewise_squared) * (h * 100)  # Convert to same units
    
    return H_piecewise

def compare_full_vs_piecewise():
    """
    Main comparison function
    """
    # Parameters
    cpr = 0.8
    omega_m = 0.3
    h = 0.7
    stigma = 0.99
    ups2 = -4/3
    OmL = 1 - omega_m - omega_r(h)
    omega_r_val = omega_r(h)
    
    print(f"Parameters: σ={stigma}, υ₂={ups2}, ϖ₀={cpr}")
    print(f"Ω_m={omega_m}, Ω_r={omega_r_val:.6f}, Ω_Λ={OmL:.6f}")
    
    # Get full Stigma solution
    print("\nRunning full Stigma theory...")
    asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
    
    # Calculate density fractions
    denominator = (a0 * hsol)**2 / (h * 100)**2
    omega_m_a = (omega_m * asol) / denominator
    omega_r_a = (omega_r_val) / denominator
    omega_lambda_a = (OmL * asol**4) / denominator
    omega_extra = 1 - (omega_m_a + omega_r_a + omega_lambda_a)
    
    # Find epoch transitions
    print("\nFinding epoch transitions...")
    a_mr_equal, a_ml_equal = find_epoch_transitions(asol, omega_m_a, omega_r_a, omega_lambda_a)
    
    # Construct piecewise cosmologies (both gi solutions)
    print("\nConstructing piecewise gi cosmologies...")
    gi_minus, wi_minus = construct_piecewise_gi_cosmology(asol, a_mr_equal, a_ml_equal, 
                                                         stigma, ups2, plus_solution=False)
    gi_plus, wi_plus = construct_piecewise_gi_cosmology(asol, a_mr_equal, a_ml_equal, 
                                                       stigma, ups2, plus_solution=True)
    
    # Calculate piecewise Hubble parameters
    H_minus = calculate_piecewise_hubble(asol, gi_minus, omega_m, omega_r_val, OmL, h)
    H_plus = calculate_piecewise_hubble(asol, gi_plus, omega_m, omega_r_val, OmL, h)
    
    # Standard ΛCDM for comparison
    rho_standard = omega_r_val/asol**4 + omega_m/asol**3 + OmL
    H_standard = np.sqrt(rho_standard / 3) * (h * 100)
    
    # Create comprehensive comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Hubble parameters
    ax1.loglog(asol, hsol, 'k-', linewidth=3, label='Full Stigma Theory')
    ax1.loglog(asol, H_minus, 'b--', linewidth=2, label='Piecewise gi (minus)')
    ax1.loglog(asol, H_plus, 'r--', linewidth=2, label='Piecewise gi (plus)')
    ax1.loglog(asol, H_standard, 'g:', linewidth=2, alpha=0.7, label='Standard ΛCDM')
    
    # Mark epoch transitions
    ax1.axvline(x=a_mr_equal, color='orange', linestyle=':', alpha=0.7, label='M-R equality')
    ax1.axvline(x=a_ml_equal, color='purple', linestyle=':', alpha=0.7, label='M-DE equality')
    
    ax1.set_xlabel('Scale Factor (a)')
    ax1.set_ylabel('Hubble Parameter H(a)')
    ax1.set_title('Hubble Parameter Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fractional differences
    diff_minus = (hsol - H_minus) / hsol
    diff_plus = (hsol - H_plus) / hsol
    diff_standard = (hsol - H_standard) / hsol
    
    ax2.semilogx(asol, diff_minus, 'b-', linewidth=2, label='(Full - Piecewise minus)/Full')
    ax2.semilogx(asol, diff_plus, 'r-', linewidth=2, label='(Full - Piecewise plus)/Full') 
    ax2.semilogx(asol, diff_standard, 'g-', linewidth=1, alpha=0.7, label='(Full - ΛCDM)/Full')
    
    ax2.axvline(x=a_mr_equal, color='orange', linestyle=':', alpha=0.7)
    ax2.axvline(x=a_ml_equal, color='purple', linestyle=':', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax2.set_xlabel('Scale Factor (a)')
    ax2.set_ylabel('Fractional Difference')
    ax2.set_title('Relative Errors in Hubble Parameter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: gi evolution
    ax3.semilogx(asol, gi_minus, 'b-', linewidth=2, label='gi (minus solution)')
    ax3.semilogx(asol, gi_plus, 'r-', linewidth=2, label='gi (plus solution)')
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='GR limit (gi=1)')
    
    ax3.axvline(x=a_mr_equal, color='orange', linestyle=':', alpha=0.7)
    ax3.axvline(x=a_ml_equal, color='purple', linestyle=':', alpha=0.7)
    
    ax3.set_xlabel('Scale Factor (a)')
    ax3.set_ylabel('Effective gi Parameter')
    ax3.set_title('Piecewise gi Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Equation of state
    ax4.semilogx(asol, wi_minus, 'b-', linewidth=2, label='w (minus solution)')
    ax4.semilogx(asol, wi_plus, 'r-', linewidth=2, label='w (plus solution)')
    
    # Reference lines
    ax4.axhline(y=1/3, linestyle='--', alpha=0.7, label='Radiation (w=1/3)')
    ax4.axhline(y=0, linestyle='--', alpha=0.7, label='Matter (w=0)')
    ax4.axhline(y=-1, linestyle='--', alpha=0.7, label='Dark Energy (w=-1)')
    
    ax4.axvline(x=a_mr_equal, color='orange', linestyle=':', alpha=0.7)
    ax4.axvline(x=a_ml_equal, color='purple', linestyle=':', alpha=0.7)
    
    ax4.set_xlabel('Scale Factor (a)')
    ax4.set_ylabel('Equation of State Parameter (w)')
    ax4.set_title('Piecewise Equation of State')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-1.5, 0.5)
    
    plt.tight_layout()
    plt.savefig('piecewise_gi_cosmology_comparison.pdf', bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"RMS fractional error (minus solution): {np.sqrt(np.mean(diff_minus**2)):.4f}")
    print(f"RMS fractional error (plus solution): {np.sqrt(np.mean(diff_plus**2)):.4f}")
    print(f"RMS fractional error (ΛCDM): {np.sqrt(np.mean(diff_standard**2)):.4f}")
    
    max_error_minus = np.max(np.abs(diff_minus))
    max_error_plus = np.max(np.abs(diff_plus))
    print(f"Maximum fractional error (minus): {max_error_minus:.4f}")
    print(f"Maximum fractional error (plus): {max_error_plus:.4f}")
    
    return asol, hsol, H_minus, H_plus, H_standard

def analyze_epoch_structure():
    """
    Detailed analysis of the epoch structure
    """
    # Same parameters as main analysis
    cpr = 0.8
    omega_m = 0.3
    h = 0.7
    stigma = 0.99
    ups2 = -4/3
    OmL = 1 - omega_m - omega_r(h)
    
    # Get solution
    asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
    
    # Calculate effective w
    log_a = np.log(asol)
    log_H = np.log(hsol)
    dlogH_dloga = np.gradient(log_H, log_a)
    w_eff = -1 - (2/3) * dlogH_dloga
    
    plt.figure(figsize=(12, 6))
    
    # Plot effective equation of state
    plt.subplot(1, 2, 1)
    plt.semilogx(asol, w_eff, 'b-', linewidth=2, label='$w_{eff}$ (Full Stigma)')
    
    # Mark standard values
    plt.axhline(y=1/3, linestyle='--', alpha=0.7, color='orange', label='Radiation (w=1/3)')
    plt.axhline(y=0, linestyle='--', alpha=0.7, color='green', label='Matter (w=0)')
    plt.axhline(y=-1/3, linestyle='--', alpha=0.7, color='red', label='Acceleration threshold')
    plt.axhline(y=-1, linestyle='--', alpha=0.7, color='purple', label='Dark Energy (w=-1)')
    
    plt.xlabel('Scale Factor (a)')
    plt.ylabel('Effective Equation of State ($w_{eff}$)')
    plt.title('Effective w from Full Theory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-2, 1)
    
    # Plot density evolution
    plt.subplot(1, 2, 2)
    denominator = (a0 * hsol)**2 / (h * 100)**2
    omega_m_a = (omega_m * asol) / denominator
    omega_r_a = (omega_r(h)) / denominator
    omega_lambda_a = (OmL * asol**4) / denominator
    omega_extra = 1 - (omega_m_a + omega_r_a + omega_lambda_a)
    
    plt.loglog(asol, omega_r_a, label='Radiation ($\\Omega_r$)')
    plt.loglog(asol, omega_m_a, label='Matter ($\\Omega_m$)')
    plt.loglog(asol, omega_lambda_a, label='Dark Energy ($\\Omega_\\Lambda$)')
    plt.loglog(asol, np.abs(omega_extra), label='|Extra| (Stigma)')
    
    plt.xlabel('Scale Factor (a)')
    plt.ylabel('Fractional Density')
    plt.title('Density Component Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('epoch_structure_analysis.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Piecewise gi Cosmology Analysis")
    print("=" * 40)
    
    # Main comparison
    asol, hsol, H_minus, H_plus, H_standard = compare_full_vs_piecewise()
    
    # Detailed epoch analysis
    print("\nAnalyzing epoch structure...")
    analyze_epoch_structure()