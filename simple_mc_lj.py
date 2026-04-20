"""
Monte Carlo simulation of a 3D Lennard-Jones fluid — minimal version
Exercise Sheet 2, Molecular Simulation S2026
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────────
N        = 100       # number of particles
L        = 6.0       # box side length
T        = 1.0       # temperature (LJ units)
max_disp = 0.5       # maximum displacement per move
n_steps  = 10_000    # total MC steps
rng      = np.random.default_rng(42)

rho = N / L**3
rc  = L / 2.0        # cutoff = half box length

# ── Initialise on a simple-cubic lattice ──────────────────────────────────────
n_side   = int(np.ceil(N**(1/3)))
spacing  = L / n_side
idx      = np.arange(N)
pos      = np.column_stack([
    (idx // n_side**2        ) % n_side,
    (idx // n_side           ) % n_side,
     idx                       % n_side,
]) * spacing + 0.5 * spacing   # shape (N, 3)

# ── LJ energy of the whole system ─────────────────────────────────────────────
def total_energy(pos):
    E = 0.0
    for i in range(N - 1):
        dr  = pos[i] - pos[i+1:]               # all pairs with j > i
        dr -= L * np.round(dr / L)             # minimum image
        r2  = np.sum(dr**2, axis=1)
        cut = r2 < rc**2
        ir6 = (1.0 / r2[cut])**3
        E  += np.sum(4.0 * ir6 * (ir6 - 1.0))
    return E

# ── Tail corrections ──────────────────────────────────────────────────────────
# These account for interactions beyond rc, assuming g(r) = 1 for r > rc.
utail = (8.0/3.0) * np.pi * rho * N * ((1.0/rc)**9 / 3.0 - (1.0/rc)**3)

# ── MC loop ───────────────────────────────────────────────────────────────────
E        = total_energy(pos) + utail
energies = []

for step in range(n_steps):
    # pick a random particle and propose a displacement
    i        = rng.integers(N)
    pos_new  = pos.copy()
    pos_new[i] += rng.uniform(-max_disp, max_disp, 3)
    pos_new[i] %= L                            # wrap into box

    # Metropolis acceptance
    E_new = total_energy(pos_new) + utail
    if E_new < E or rng.random() < np.exp(-(E_new - E) / T):
        pos = pos_new
        E   = E_new

    energies.append(E)


