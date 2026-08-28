"""
Visualization of Stigma theory parameter constraints.

This module creates plots showing the allowed parameter space for the Stigma Q-torsion theory
based on the validity constraints:
1. stigma > varpi_0^2
2. stigma * (5 + 4*stigma*(-2 + varpi_0^2)) < varpi_0^2
"""

import numpy as np
import matplotlib.pyplot as plt
from equations import StigmaConstraintError

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def check_constraints(stigma, varpi_0):
    """
    Check if given stigma and varpi_0 values satisfy the constraints.
    
    Args:
        stigma: Stigma parameter value (can be array)
        varpi_0: Initial varpi value (can be array)
        
    Returns:
        bool or array: True where constraints are satisfied
    """
    constraint1 = stigma > varpi_0**2
    constraint2 = stigma * (5 + 4*stigma*(-2 + varpi_0**2)) < varpi_0**2
    
    return constraint1 & constraint2

def plot_constraint_regions():
    """Plot the allowed parameter regions in stigma-varpi_0 space."""
    
    # Create parameter grids
    varpi_0_range = np.linspace(0.1, 1.5, 1000)
    stigma_range = np.linspace(0.1, 2.0, 1000)
    
    VARPI_0, STIGMA = np.meshgrid(varpi_0_range, stigma_range)
    
    # Check constraints for all combinations
    valid_mask = check_constraints(STIGMA, VARPI_0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot allowed regions in green, forbidden in red
    ax.contourf(VARPI_0, STIGMA, valid_mask.astype(int), 
                levels=[0, 0.5, 1], colors=['red', 'lightgreen'], alpha=0.7)
    
    # Add constraint boundaries
    # Constraint 1: stigma = varpi_0^2
    varpi_0_fine = np.linspace(0.1, 1.5, 500)
    stigma_boundary1 = varpi_0_fine**2
    ax.plot(varpi_0_fine, stigma_boundary1, 'b-', linewidth=2, 
            label=r'$\varsigma = \varpi_0^2$ (Constraint 1)')
    
    # Constraint 2: stigma * (5 + 4*stigma*(-2 + varpi_0^2)) = varpi_0^2
    # This is more complex, so we'll solve it numerically for each varpi_0
    stigma_boundary2 = []
    varpi_0_boundary2 = []
    
    for v0 in varpi_0_fine:
        # Solve: stigma * (5 + 4*stigma*(-2 + v0^2)) = v0^2
        # This is a quadratic: 4*stigma^2*(-2 + v0^2) + 5*stigma - v0^2 = 0
        a = 4*(-2 + v0**2)
        b = 5
        c = -v0**2
        
        if a != 0:
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                stigma1 = (-b + np.sqrt(discriminant)) / (2*a)
                stigma2 = (-b - np.sqrt(discriminant)) / (2*a)
                
                # Take the physically relevant solution
                for s in [stigma1, stigma2]:
                    if s > 0 and s < 2.0:  # Reasonable range
                        stigma_boundary2.append(s)
                        varpi_0_boundary2.append(v0)
                        break
    
    if stigma_boundary2:
        ax.plot(varpi_0_boundary2, stigma_boundary2, 'orange', linewidth=2,
                label=r'$\varsigma (5 + 4\varsigma(-2 + \varpi_0^2)) = \varpi_0^2$ (Constraint 2)')
    
    # Mark some interesting parameter values
    test_points = [
        (0.8, 0.99, 'Typical values'),
        (0.8, 1.0, r'$\varsigma = 1.0$'),
        (0.8, 1.05, 'Edge case'),
    ]
    
    for varpi_0, stigma, label in test_points:
        is_valid = check_constraints(stigma, varpi_0)
        color = 'green' if is_valid else 'red'
        marker = 'o' if is_valid else 'x'
        ax.plot(varpi_0, stigma, marker, color=color, markersize=8, 
                markeredgecolor='black', markeredgewidth=1)
        ax.annotate(f'{label}\n({varpi_0}, {stigma})', 
                   xy=(varpi_0, stigma), xytext=(10, 10), 
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(r'$\varpi_0$ (Initial varpi value)', fontsize=14)
    ax.set_ylabel(r'$\varsigma$ (Stigma parameter)', fontsize=14)
    ax.set_title('Stigma Theory Parameter Constraints\n'
                 r'Green: Allowed, Red: Forbidden', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 1.5)
    ax.set_ylim(0.1, 2.0)
    
    plt.tight_layout()
    plt.savefig('Plots/stigma_constraint_regions.pdf', bbox_inches='tight')
    plt.show()

def plot_constraint_boundaries_detailed():
    """Create a detailed plot focusing on the constraint boundaries."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Focus on constraint 1
    varpi_0_range = np.linspace(0.5, 1.2, 500)
    stigma_boundary1 = varpi_0_range**2
    
    ax1.fill_between(varpi_0_range, stigma_boundary1, 2.0, 
                     alpha=0.3, color='green', label='Allowed by Constraint 1')
    ax1.fill_between(varpi_0_range, 0, stigma_boundary1, 
                     alpha=0.3, color='red', label='Forbidden by Constraint 1')
    ax1.plot(varpi_0_range, stigma_boundary1, 'b-', linewidth=3,
             label=r'$\varsigma = \varpi_0^2$')
    
    ax1.set_xlabel(r'$\varpi_0$', fontsize=14)
    ax1.set_ylabel(r'$\varsigma$', fontsize=14)
    ax1.set_title('Constraint 1: $\\varsigma > \\varpi_0^2$', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 1.2)
    ax1.set_ylim(0.2, 2.0)
    
    # Right plot: Combined constraints with specific varpi_0 values
    varpi_0_values = [0.6, 0.7, 0.8, 0.9, 1.0]
    stigma_range = np.linspace(0.1, 2.0, 1000)
    
    for i, v0 in enumerate(varpi_0_values):
        valid_mask = check_constraints(stigma_range, v0)
        color = plt.cm.viridis(i / len(varpi_0_values))
        
        # Plot valid ranges
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            # Find continuous segments
            segments = []
            start = valid_indices[0]
            for j in range(1, len(valid_indices)):
                if valid_indices[j] - valid_indices[j-1] > 1:
                    segments.append((start, valid_indices[j-1]))
                    start = valid_indices[j]
            segments.append((start, valid_indices[-1]))
            
            # Plot each segment
            for start_idx, end_idx in segments:
                ax2.fill_between([v0-0.02, v0+0.02], 
                                stigma_range[start_idx], stigma_range[end_idx],
                                color=color, alpha=0.7,
                                label=f'$\\varpi_0 = {v0}$' if segments.index((start_idx, end_idx)) == 0 else "")
    
    ax2.set_xlabel(r'$\varpi_0$', fontsize=14)
    ax2.set_ylabel(r'$\varsigma$', fontsize=14)
    ax2.set_title('Valid $\\varsigma$ ranges for different $\\varpi_0$ values', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.55, 1.05)
    ax2.set_ylim(0.3, 1.5)
    
    plt.tight_layout()
    plt.savefig('Plots/constraint_boundaries_detailed.pdf', bbox_inches='tight')
    plt.show()

def print_valid_ranges():
    """Print valid stigma ranges for common varpi_0 values."""
    
    print("Valid stigma ranges for common varpi_0 values:")
    print("=" * 50)
    
    varpi_0_values = [0.6, 0.7, 0.8, 0.9, 1.0]
    
    for v0 in varpi_0_values:
        # Find valid stigma range
        stigma_test = np.linspace(0.1, 2.0, 10000)
        valid_mask = check_constraints(stigma_test, v0)
        
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            stigma_min = stigma_test[valid_indices[0]]
            stigma_max = stigma_test[valid_indices[-1]]
            
            print(f"varpi_0 = {v0:.1f}: stigma ∈ ({stigma_min:.6f}, {stigma_max:.6f})")
            
            # Check if our typical value of 1.0 is valid
            is_valid_1 = check_constraints(1.0, v0)
            print(f"  → stigma = 1.0 is {'VALID' if is_valid_1 else 'INVALID'}")
        else:
            print(f"varpi_0 = {v0:.1f}: NO VALID STIGMA VALUES")
        
        print()

def main():
    """Main function to run all constraint visualizations."""
    print("Creating Stigma theory constraint visualizations...")
    print("=" * 50)
    
    print("\n1. Creating constraint region plot...")
    plot_constraint_regions()
    
    print("\n2. Creating detailed boundary analysis...")
    plot_constraint_boundaries_detailed()
    
    print("\n3. Printing valid parameter ranges...")
    print_valid_ranges()
    
    print("\nConstraint analysis complete! Check the generated PDF files:")
    print("- Plots/stigma_constraint_regions.pdf")
    print("- Plots/constraint_boundaries_detailed.pdf")

if __name__ == "__main__":
    main()