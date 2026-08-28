from equationsa0 import omega_r, rhopsol
import matplotlib.pyplot as plt
import numpy as np

def basic_density_analysis():
    """
    Run rhopsol and analyze density evolution to find epoch transitions
    """
    # Parameters
    cpr = 0.8
    omega_m = 0.3
    h = 0.7
    stigma = 0.99
    ups2 = -4/3
    OmL = 1 - omega_m - omega_r(h)
    
    print(f"Parameters: σ={stigma}, υ₂={ups2}, ϖ₀={cpr}")
    print(f"Ω_m={omega_m}, Ω_r={omega_r(h):.6f}, Ω_Λ={OmL:.6f}")
    
    # Run rhopsol
    print("\nRunning full Stigma theory...")
    asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
    
    # Calculate density evolution
    denominator = (a0 * hsol)**2 / (h * 100)**2
    
    # Fractional density for each component
    omega_m_a = (omega_m * asol) / denominator
    omega_r_a = (omega_r(h)) / denominator
    omega_lambda_a = (OmL * asol**4) / denominator
    omega_extra = 1 - (omega_m_a + omega_r_a + omega_lambda_a)
    
    # Find epoch transitions
    # Matter-radiation equality from standard relation
    a_mr_standard = omega_r(h) / omega_m
    print(f"\nStandard matter-radiation equality at a = {a_mr_standard:.6f}")
    
    # Find actual matter-radiation equality in the solution
    mr_diff = np.abs(omega_m_a - omega_r_a)
    a_mr_idx = np.argmin(mr_diff)
    a_mr_actual = asol[a_mr_idx]
    print(f"Actual matter-radiation equality at a = {a_mr_actual:.6f}")
    
    # Find matter-dark energy equality
    ml_diff = np.abs(omega_m_a - omega_lambda_a)
    a_ml_idx = np.argmin(ml_diff)
    a_ml_actual = asol[a_ml_idx]
    print(f"Matter-dark energy equality at a = {a_ml_actual:.6f}")
    
    # Plot density evolution
    plt.figure(figsize=(10, 6))
    
    # Plot each component
    plt.plot(asol, omega_r_a, label='Radiation ($\\Omega_r$)')
    plt.plot(asol, omega_m_a, label='Matter ($\\Omega_m$)')
    plt.plot(asol, omega_lambda_a, label='Dark Energy ($\\Omega_\\Lambda$)')
    plt.plot(asol, omega_extra, label='Extra (Stigma)')
    
    # Formatting the plot
    plt.xscale('log')
    plt.xlabel('Scale Factor (a)')
    plt.ylabel('Fractional Energy Density (Ω)')
    plt.title('Evolution of the Contents of the Universe')
    plt.grid(True, which="both", ls="--")
    plt.ylim(-0.1, 1.1)
    plt.xlim(asol.min(), asol.max())
    
    # Add vertical lines for important epochs
    plt.axvline(x=a_mr_standard, linestyle='--', color='orange', alpha=0.7, 
                label=f'M-R Equality (standard): a={a_mr_standard:.4f}')
    plt.axvline(x=a_mr_actual, linestyle=':', color='orange', alpha=0.7,
                label=f'M-R Equality (actual): a={a_mr_actual:.4f}')
    plt.axvline(x=a_ml_actual, linestyle='--', color='purple', alpha=0.7,
                label=f'M-DE Equality: a={a_ml_actual:.4f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('basic_density_evolution.pdf', bbox_inches='tight')
    plt.show()
    
    # Print final values
    print(f"\nFinal density values (at a={asol[-1]:.4f}):")
    print(f"Ω_m = {omega_m_a[-1]:.4f}")
    print(f"Ω_r = {omega_r_a[-1]:.6f}")
    print(f"Ω_Λ = {omega_lambda_a[-1]:.4f}")
    print(f"Ω_extra = {omega_extra[-1]:.4f}")
    print(f"Sum = {omega_m_a[-1] + omega_r_a[-1] + omega_lambda_a[-1] + omega_extra[-1]:.4f}")
    
    return asol, omega_m_a, omega_r_a, omega_lambda_a, omega_extra, a_mr_actual, a_ml_actual

if __name__ == "__main__":
    asol, omega_m_a, omega_r_a, omega_lambda_a, omega_extra, a_mr, a_ml = basic_density_analysis()