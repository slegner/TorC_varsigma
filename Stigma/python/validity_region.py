"""
Plot the validity region in (varsigma, varpi_0) parameter space.

The two constraints are:
  1. ς > ϖ₀²
  2. ς(5 + 4ς(-2 + ϖ₀²)) < ϖ₀²

The valid region is where BOTH hold simultaneously.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Plots")
os.makedirs(OUTDIR, exist_ok=True)

# Grid over (varsigma, varpi_0) from 0 to 2
varsigma = np.linspace(0, 2, 800)
varpi    = np.linspace(0, 2, 800)
S, P     = np.meshgrid(varsigma, varpi)

# Evaluate constraints
c1 = S > P**2                              # ς > ϖ₀²
c2 = S * (5 + 4*S*(-2 + P**2)) < P**2    # ς(5 + 4ς(-2 + ϖ₀²)) < ϖ₀²
valid = c1 & c2

fig, ax = plt.subplots(figsize=(7, 6))

# Shade the valid region
ax.contourf(S, P, valid.astype(float), levels=[0.5, 1.5], colors=["#4488CC"], alpha=0.4)

# Draw the boundary of each constraint
ax.contour(S, P, (S - P**2),                        levels=[0], colors=["#CC3333"], linewidths=1.8)
ax.contour(S, P, (S*(5 + 4*S*(-2 + P**2)) - P**2), levels=[0], colors=["#FF8800"], linewidths=1.8)

# Dummy lines for legend
ax.plot([], [], color="#4488CC", lw=8, alpha=0.4, label="Valid region")
ax.plot([], [], color="#CC3333", lw=1.8, label=r"$\varsigma = \varpi_0^2$")
ax.plot([], [], color="#FF8800", lw=1.8, label=r"$\varsigma(5 + 4\varsigma(-2+\varpi_0^2)) = \varpi_0^2$")

ax.set_xlabel(r"$\varsigma$",  fontsize=14)
ax.set_ylabel(r"$\varpi_0$",   fontsize=14)
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_title("Validity region in $(\\varsigma,\\, \\varpi_0)$ parameter space", fontsize=13)
ax.legend(fontsize=10, loc="upper right")
ax.grid(ls="--", alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "validity_region.pdf"), bbox_inches="tight")
print("Saved → validity_region.pdf")
plt.show()
