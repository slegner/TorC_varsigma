from equationsa0 import omega_r, rhopsol
import matplotlib.pyplot as plt
import numpy as np

def plot_density_evolution(cpr=0.8, omega_m=0.3, h=0.7, stigma=0.99, ups2=-4/3):
    """
    Plot the evolution of density components in the universe with a0 = 1.
    """
    OmL = 1 - omega_m - omega_r(h)
    
    # Get solutions using modified function
    asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
    
    # Calculate denominator (note: a0 = 1)
    denominator = (a0 * hsol)**2 / (h * 100)**2
    
    # Fractional density for each component
    omega_m_a = (omega_m * asol) / denominator
    omega_r_a = (omega_r(h)) / denominator  
    omega_lambda_a = (OmL * asol**4) / denominator
    omega_extra = 1 - (omega_m_a + omega_r_a + omega_lambda_a)
    
    plt.figure(figsize=(10, 6))
    
    # Plot each component
    plt.plot(asol, omega_r_a, label='Radiation ($\\Omega_r$)', linewidth=2)
    plt.plot(asol, omega_m_a, label='Matter ($\\Omega_m$)', linewidth=2)
    plt.plot(asol, omega_lambda_a, label='Dark Energy ($\\Omega_\\Lambda$)', linewidth=2)
    plt.plot(asol, omega_extra, label='Extra (Stigma)', linewidth=2, linestyle='--')
    
    # Formatting the plot
    plt.xscale('log')
    plt.xlabel('Scale Factor (a)')
    plt.ylabel('Fractional Energy Density (Ω)')
    plt.title(f'Evolution of Universe Contents ($\\sigma$={stigma}, $\\upsilon_2$={ups2}, $\\varpi_0$={cpr})')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.xlim(asol.min(), asol.max())
    
    # Add vertical lines for important epochs
    matter_rad_equality = omega_r(h) / omega_m
    plt.axvline(x=matter_rad_equality, linestyle='--', color='gray', alpha=0.7, 
                label=f'Matter-Radiation Equality (a={matter_rad_equality:.4f})')
    
    # Add today marker (a = 1)
    plt.axvline(x=1.0, linestyle=':', color='red', alpha=0.7, label='Today (a = 1)')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Print some diagnostic information
    print(f"Parameters: σ={stigma}, υ₂={ups2}, ϖ₀={cpr}")
    print(f"Matter-radiation equality at a = {matter_rad_equality:.6f}")
    print(f"Final omega values at a=1: Ωₘ={omega_m_a[-1]:.4f}, Ωᵣ={omega_r_a[-1]:.4f}, ΩΛ={omega_lambda_a[-1]:.4f}, Extra={omega_extra[-1]:.4f}")
    print(f"Sum = {omega_m_a[-1] + omega_r_a[-1] + omega_lambda_a[-1] + omega_extra[-1]:.4f}")
    
    plt.savefig(f'density_evolution_sigma{stigma}_ups{ups2:.2f}_varpi{cpr}_a0_1.pdf', bbox_inches='tight')
    plt.show()
    
    return asol, omega_m_a, omega_r_a, omega_lambda_a, omega_extra


def compare_density_evolution():
    """
    Compare density evolution for different parameter sets.
    """
    # Parameters to compare
    param_sets = [
        {"stigma": 0.85, "ups2": -4/3, "label": "σ=0.85, υ₂=-4/3"},
        {"stigma": 0.95, "ups2": -4/3, "label": "σ=0.95, υ₂=-4/3"}, 
        {"stigma": 1.05, "ups2": -4/3, "label": "σ=1.05, υ₂=-4/3"},
        {"stigma": 0.99, "ups2": -1.0, "label": "σ=0.99, υ₂=-1.0"},
        {"stigma": 0.99, "ups2": -2.0, "label": "σ=0.99, υ₂=-2.0"}
    ]
    
    cpr = 0.8
    omega_m = 0.3
    h = 0.7
    OmL = 1 - omega_m - omega_r(h)
    
    plt.figure(figsize=(12, 8))
    
    for i, params in enumerate(param_sets):
        try:
            stigma = params["stigma"]
            ups2 = params["ups2"]
            label = params["label"]
            
            asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
            
            # Calculate omega_extra
            denominator = (a0 * hsol)**2 / (h * 100)**2
            omega_m_a = (omega_m * asol) / denominator
            omega_r_a = (omega_r(h)) / denominator
            omega_lambda_a = (OmL * asol**4) / denominator
            omega_extra = 1 - (omega_m_a + omega_r_a + omega_lambda_a)
            
            color = plt.cm.tab10(i)
            plt.plot(asol, omega_extra, label=f'Extra: {label}', color=color, linewidth=2)
            
        except Exception as e:
            print(f"Failed for {params}: {e}")
    
    plt.xscale('log')
    plt.xlabel('Scale Factor (a)')
    plt.ylabel('Extra Density Component (Ω_extra)')
    plt.title('Comparison of Extra Density Component')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.axvline(x=1.0, linestyle=':', color='red', alpha=0.7, label='Today (a = 1)')
    plt.axhline(y=0, linestyle='-', color='black', alpha=0.3)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('density_extra_comparison_a0_1.pdf', bbox_inches='tight')
    plt.show()


def detailed_density_breakdown():
    """
    Create detailed breakdown showing all components and their sum.
    """
    cpr = 0.8
    omega_m = 0.3
    h = 0.7
    stigma = 0.99
    ups2 = -4/3
    OmL = 1 - omega_m - omega_r(h)
    
    # Get solutions
    asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
    
    # Calculate all components
    denominator = (a0 * hsol)**2 / (h * 100)**2
    omega_m_a = (omega_m * asol) / denominator
    omega_r_a = (omega_r(h)) / denominator
    omega_lambda_a = (OmL * asol**4) / denominator
    omega_extra = 1 - (omega_m_a + omega_r_a + omega_lambda_a)
    omega_total = omega_m_a + omega_r_a + omega_lambda_a + omega_extra
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top plot: Individual components
    ax1.plot(asol, omega_r_a, label='Radiation (Ωᵣ)', linewidth=2)
    ax1.plot(asol, omega_m_a, label='Matter (Ωₘ)', linewidth=2)
    ax1.plot(asol, omega_lambda_a, label='Dark Energy (ΩΛ)', linewidth=2)
    ax1.plot(asol, omega_extra, label='Extra (Stigma)', linewidth=2, linestyle='--')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Scale Factor (a)')
    ax1.set_ylabel('Fractional Energy Density (Ω)')
    ax1.set_title('Individual Density Components')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend()
    ax1.axvline(x=1.0, linestyle=':', color='red', alpha=0.7)
    
    # Bottom plot: Total and constraint check
    ax2.plot(asol, omega_total, label='Total Ω', linewidth=3, color='black')
    ax2.axhline(y=1, linestyle='--', color='red', alpha=0.7, label='Ω = 1 (constraint)')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Scale Factor (a)')
    ax2.set_ylabel('Total Density Parameter')
    ax2.set_title('Friedmann Constraint Check: Ω_total = 1?')
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend()
    ax2.axvline(x=1.0, linestyle=':', color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('detailed_density_breakdown_a0_1.pdf', bbox_inches='tight')
    plt.show()
    
    # Print final values
    print(f"\nFinal values at a = 1:")
    print(f"Ωₘ = {omega_m_a[-1]:.6f}")
    print(f"Ωᵣ = {omega_r_a[-1]:.6f}")
    print(f"ΩΛ = {omega_lambda_a[-1]:.6f}")
    print(f"Ω_extra = {omega_extra[-1]:.6f}")
    print(f"Ω_total = {omega_total[-1]:.6f}")
    print(f"Deviation from 1: {abs(omega_total[-1] - 1):.6f}")


if __name__ == "__main__":
    # Plot standard density evolution for single parameter set
    plot_density_evolution()