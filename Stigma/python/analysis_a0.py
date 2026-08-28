from equationsa0 import omega_r, rhopsol
import matplotlib.pyplot as plt
import numpy as np

# Parameters
a0 = 1
cpr = 0.8  
omega_m = 0.3
h = 0.7
stigma = 0.99
ups2 = -4/3
OmL = 1 - omega_m - omega_r(h)

# Get solutions using modified function (no P_DE, rho_DE)
asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)

# Calculate denominator
denominator = (hsol)**2 / (h*100)**2

# Fractional density for each component
omega_m_a = (omega_m * asol) / denominator
omega_r_a = (omega_r(h)) / denominator
omega_lambda_a = (OmL * asol**4) / denominator
omega_extra = 1 - (omega_m_a + omega_r_a + omega_lambda_a)

print("Sample omega_m_a values:", omega_m_a[:5])


def find_valid_stigma_values(varpi_0, n_points=10):
    """Find stigma values that satisfy the constraints."""
    # Use specific values around 1.0
    stigma_candidates = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    
    # Check constraints
    constraint1 = stigma_candidates > varpi_0**2
    constraint2 = stigma_candidates * (5 + 4*stigma_candidates*(-2 + varpi_0**2)) < varpi_0**2
    
    valid_mask = constraint1 & constraint2
    valid_stigmas = stigma_candidates[valid_mask]
    
    print(f"Testing $\\varsigma$ values: {stigma_candidates}")
    print(f"Valid $\\varsigma$ values: {valid_stigmas}")
    print(f"$\\varpi_0$ = {varpi_0}, $\\varpi_0^2$ = {varpi_0**2}")
    
    return valid_stigmas


def plot_hubble_horizon():
    """Plot Hubble horizon for different stigma values with a0=1."""
    # Parameters
    cpr = 0.8
    omega_m = 0.3
    h = 0.7
    ups2 = -4/3
    OmL = 1 - omega_m - omega_r(h)
    
    stigma_values = find_valid_stigma_values(cpr, n_points=10)
    
    if len(stigma_values) == 0:
        print("No valid stigma values found.")
        return
    
    plt.figure(figsize=(10, 6))
    
    for i, stigma in enumerate(stigma_values):
        try:
            asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
            
            # Hubble horizon: a / H with a0 = 1
            y_values = asol / (hsol / (h * 100))
            
            # Use viridis colormap for better distinction
            color = plt.cm.viridis(i / len(stigma_values))
            plt.plot(asol, y_values, label=f'$\\varsigma$ = {stigma:.3f}', color=color)
            
        except Exception as e:
            print(f"Failed for $\\varsigma$ = {stigma}: {e}")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Scale Factor (a)')
    plt.ylabel('Hubble Horizon (a/H)')
    plt.title(f'Hubble Horizon for Different $\\varsigma$ Values ($\\varpi_0$ = {cpr}, $a_0$ = 1)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('hubble_horizon_varsigma_a0_1.pdf', bbox_inches='tight')
    plt.show()


def plot_constraint():
    """Plot the constraint equation for different stigma values with a0=1."""
    from equationsa0 import constraint
    
    # Parameters
    cpr = 0.8
    omega_m = 0.3
    h = 0.7
    ups2 = -4/3
    OmL = 1 - omega_m - omega_r(h)
    
    stigma_values = find_valid_stigma_values(cpr, n_points=10)
    
    if len(stigma_values) == 0:
        print("No valid stigma values found.")
        return
    
    plt.figure(figsize=(10, 6))
    
    for i, stigma in enumerate(stigma_values):
        try:
            asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
            
            # Calculate constraint equation
            constraint_values = []
            for j in range(len(asol)):
                c_val = constraint(asol[j], hsol[j], varpisol[j], dvarpisol[j], 
                                 omega_m, h, OmL, a0, stigma, ups2)
                constraint_values.append(c_val)
            
            # Use viridis colormap for better distinction
            color = plt.cm.viridis(i / len(stigma_values))
            plt.plot(asol, constraint_values, label=f'$\\varsigma$ = {stigma:.3f}', color=color)
            
        except Exception as e:
            print(f"Failed for $\\varsigma$ = {stigma}: {e}")
    
    plt.xscale('log')
    plt.xlabel('Scale Factor (a)')
    plt.ylabel('Constraint Equation')
    plt.title(f'Constraint Equation for Different $\\varsigma$ Values ($\\varpi_0$ = {cpr}, $a_0$ = 1)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.savefig('constraint_equation_varsigma_a0_1.pdf', bbox_inches='tight')
    plt.show()


def plot_hubble_horizon_ups2():
    """Plot Hubble horizon for different upsilon2 values with a0=1."""
    # Parameters
    cpr = 0.8
    omega_m = 0.3
    h = 0.7
    stigma = 1.0
    OmL = 1 - omega_m - omega_r(h)
    
    # Different upsilon2 values (more values)
    ups2_values = [-2.5, -2.0, -1.8, -1.5, -4/3, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2]
    
    plt.figure(figsize=(10, 6))
    
    for i, ups2 in enumerate(ups2_values):
        try:
            asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
            
            # Hubble horizon: a / H with a0 = 1
            y_values = asol / (hsol / (h * 100))
            
            # Use viridis colormap for better distinction
            color = plt.cm.viridis(i / len(ups2_values))
            plt.plot(asol, y_values, label=f'$\\upsilon_2$ = {ups2:.2f}', color=color)
            
        except Exception as e:
            print(f"Failed for $\\upsilon_2$ = {ups2}: {e}")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Scale Factor (a)')
    plt.ylabel('Hubble Horizon (a/H)')
    plt.title(f'Hubble Horizon for Different $\\upsilon_2$ Values ($\\varpi_0$ = {cpr}, $\\varsigma$ = {stigma}, $a_0$ = 1)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('hubble_horizon_upsilon2_a0_1.pdf', bbox_inches='tight')
    plt.show()


def plot_scale_factor_evolution():
    """Plot scale factor evolution for different parameter values."""
    # Parameters
    cpr = 0.8
    omega_m = 0.3
    h = 0.7
    ups2 = -4/3
    OmL = 1 - omega_m - omega_r(h)
    
    stigma_values = find_valid_stigma_values(cpr, n_points=5)
    
    if len(stigma_values) == 0:
        print("No valid stigma values found.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot scale factor vs conformal time
    plt.subplot(2, 2, 1)
    for i, stigma in enumerate(stigma_values[:5]):  # Limit to 5 for clarity
        try:
            asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
            
            # Calculate conformal time (approximate)
            tau = np.linspace(0, 1, len(asol))  # Placeholder
            
            color = plt.cm.viridis(i / len(stigma_values))
            plt.plot(tau, asol, label=f'$\\varsigma$ = {stigma:.3f}', color=color)
            
        except Exception as e:
            print(f"Failed for $\\varsigma$ = {stigma}: {e}")
    
    plt.xlabel('Conformal Time $\\tau$')
    plt.ylabel('Scale Factor a($\\tau$)')
    plt.title('Scale Factor Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Hubble parameter
    plt.subplot(2, 2, 2)
    for i, stigma in enumerate(stigma_values[:5]):
        try:
            asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
            
            color = plt.cm.viridis(i / len(stigma_values))
            plt.plot(asol, hsol, label=f'$\\varsigma$ = {stigma:.3f}', color=color)
            
        except Exception as e:
            print(f"Failed for $\\varsigma$ = {stigma}: {e}")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Scale Factor a')
    plt.ylabel('Hubble Parameter H')
    plt.title('Hubble Parameter vs Scale Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot varpi evolution
    plt.subplot(2, 2, 3)
    for i, stigma in enumerate(stigma_values[:5]):
        try:
            asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
            
            color = plt.cm.viridis(i / len(stigma_values))
            plt.plot(asol, varpisol, label=f'$\\varsigma$ = {stigma:.3f}', color=color)
            
        except Exception as e:
            print(f"Failed for $\\varsigma$ = {stigma}: {e}")
    
    plt.xscale('log')
    plt.xlabel('Scale Factor a')
    plt.ylabel('$\\varpi$ field')
    plt.title('Varpi Field Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot varpi derivative
    plt.subplot(2, 2, 4)
    for i, stigma in enumerate(stigma_values[:5]):
        try:
            asol, varpisol, hsol, dvarpisol, a0 = rhopsol(cpr, omega_m, h, OmL, stigma, ups2)
            
            color = plt.cm.viridis(i / len(stigma_values))
            plt.plot(asol, dvarpisol, label=f'$\\varsigma$ = {stigma:.3f}', color=color)
            
        except Exception as e:
            print(f"Failed for $\\varsigma$ = {stigma}: {e}")
    
    plt.xscale('log')
    plt.xlabel('Scale Factor a')
    plt.ylabel('$\\dot{\\varpi}$')
    plt.title('Varpi Field Derivative')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scale_factor_evolution_a0_1.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_hubble_horizon()
    plot_constraint()
    plot_hubble_horizon_ups2()
    plot_scale_factor_evolution()