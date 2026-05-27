"""
Lennard-Jones molecular dynamics NVT with Velocity Verlet integration and Nose-Hoover thermostat.
Reduced units (sigma = epsilon = mass = 1).
Exercise Sheet 3, Molecular Simulation S2026
"""

import numpy as np
from thermostats import VelocityScaling, NoseHooverNVT

### Some functions

def init_positions(N, L):
    """
    Generate initial positions for N LJ particles on a simple cubic lattice
    inside a cubic box of side L.
 
    The lattice has n_side = ceil(N**(1/3)) sites per dimension, giving
    n_side**3 slots. Only the first N slots are filled, so when N is not a
    perfect cube the last "row" is partially populated; the resulting
    configuration is slightly anisotropic but harmless after equilibration.
 
    N: Number of particles
    L: Box length
    """
    
    n_side = int(np.ceil(N ** (1/3)))
    spacing = L / n_side
    idx      = np.arange(N)
    pos      = np.column_stack([
         (idx // n_side**2        ) % n_side,
         (idx // n_side           ) % n_side,
         idx                       % n_side,
    ]) * spacing + 0.5 * spacing
    return pos


def init_velocities(N, temperature,rng):
	
    """
    Maxwell-Boltzmann velocities, zero net momentum.
    
    N: Number of particles
    temperature
    
    """
    vel = rng.normal(0.0, np.sqrt(temperature), size=(N, 3))
    vel -= vel.mean(axis=0) # COM velocity to zero
    # rescale to target temperature: <KE> = (3/2) N kT
    ke = 0.5 * np.sum(vel **2) # Kinetic energy
    target = 1.5 * N * temperature
    vel *= np.sqrt(target / ke)
    return vel


def lj_forces(pos, L, r_cut):
    """Compute LJ forces and potential energy with minimum image convention."""
    n = len(pos)
    forces = np.zeros_like(pos)
    energy = 0.0
    rcut2 = r_cut**2

    for i in range(n - 1):
        rij = pos[i + 1:] - pos[i]
        rij -= L * np.round(rij / L)        # nearest image distance
        r2 = np.sum(rij**2, axis=1)
        # Apply cut-off
        mask = r2 < rcut2
        r2 = r2[mask]
        rij = rij[mask]

        if len(r2) == 0:
            continue

        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2

        energy += np.sum(4.0 * (inv_r12 - inv_r6)) # Lennard-Jones Energy
        fmag = 48.0 * inv_r2 * (inv_r12 - 0.5 * inv_r6)   # LJ Forces magnitude
        fij = fmag[:, None] * rij # LJ forces

        forces[i] -= np.sum(fij, axis=0)
        idx = np.where(mask)[0] + (i + 1)
        np.add.at(forces, idx, fij)

    return forces, energy


def kinetic_energy(vel):
    return 0.5 * np.sum(vel**2)


def thermostat(vel, ke,equil = False):
    """
    This function calls the thermostat. If in equilibrium step, it uses simple velocity rescaling otherwise it uses the Nose-Hoover.
    """
    if equil:
        vel, ke = vel_scal.scale(vel,ke) 
        return vel, ke       
    else:
        vel, ke = nh_therm.scale(vel,ke)
        nh_ener = nh_therm.get_energy()
        return vel, ke, nh_ener
    

def verlet_integration(vel, pos, forces, dt, box_size, r_cut):
    """One Velocity Verlet step.
       It returns the updated positions and velocities.
       Complementary, the forces and potential energy of LJ potential are returned
    """
    vel += 0.5 * dt * forces
    pos += dt * vel
    pos -= box_size * np.floor(pos / box_size)             # PBC wrap
    forces, pe = lj_forces(pos, box_size, r_cut)
    vel += 0.5 * dt * forces
    return vel, pos, forces, pe


def write_xyz(fh, pos, step):
	
    """Append a frame in plain XYZ format."""
    
    fh.write(f"{len(pos)}\n") # Number of particles (This is always the header of .xyz files).
    fh.write(f'Step = {step}\n') # Identifier for the step. 
    for r in pos: 
        fh.write(f"X {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n") # Write the coordinates of each LJ particle, named 'X' here.


def write_energy(fh, step, ke, pe, T,nh_ener):
    """Append a line of step KE PE E_tot T to an energy log."""
    fh.write(f"{step:>8d} {ke:>14.6f} {pe:>14.6f} {ke + pe:>14.6f} {T:>10.6f} {nh_ener:>14.6f}\n")


def run(n_part=200,
        box_size=8.0,
        r_cut=4.0,
        temperature=1.0, 
        dt=0.005,
        eq_steps=None,
        prod_steps=2000,
        write_every=20, 
        traj_save=True,
        traj_file_name=None,
        ene_save=True, 
        ene_file_name=None, 
        thermostat= False,
        nh_timescale = 500,
        seed=42):
    
    # Safety check for the cutoff distance
    if r_cut >= 0.5 * box_size:
        raise ValueError(
            f"r_cut ({r_cut}) must be strictly less than L/2 ({0.5 * box_size}) "
            "for the minimum image convention to be valid."
        )

    # Initialize thermostate
    if thermostat:
        dof = 3*n_part -3
        # Initialize the velocity scaling
        vel_scal = VelocityScaling(temperature, dof)
        # Nose-Hoover
        timescale = nh_timescale*dt
        nh_therm = NoseHooverNVT(temperature, dof, timescale, dt, seed)
    if thermostat and eq_steps is None:
        raise ValueError('If you use the thermostat, you require an equilibration time')
    
    # Set the seed
    rng = np.random.default_rng(seed)
    
    #Initialize the positions on a cubic box.
    pos = init_positions(n_part,box_size)
    # Initialize the velocities following a Maxwell-Boltzmann dist. 
    vel = init_velocities(n_part, temperature,rng)
    
    #Calculate initial LJ forces and energies
    forces, pe = lj_forces(pos, box_size,r_cut)
    # Calculate the density of the system.
    density = n_part*(1/box_size**3)
    
    
    ### Check if an equilibrium step is required.
    if eq_steps is not None:
        print(f"Running initial equilibration simulation for {eq_steps} steps")
        print(f"# N={len(pos)}  box size={box_size:.4f}  rho={density}  T0={temperature}")
        print(f"# {'step':>6} {'KE':>12} {'PE':>12} {'E_tot':>12} {'T':>8}")
        
        #Main MD loop for the equilibriation step
        for step in range(eq_steps+1):
            vel, pos, forces, pe = verlet_integration(vel,pos, forces,dt,box_size,r_cut)
            ke = kinetic_energy(vel)
            
            if thermostat:
                vel, ke = vel_scal.scale(vel,ke) 
            
            # How often the energies are saved.
            if step % write_every == 0:
                T = 2.0 * ke / (3.0 * len(pos))
                print(f"{step:>6d} {ke:>12.4f} {pe:>12.4f} {ke + pe:>12.4f} {T:>8.4f}")

    # Check if the trajectory needs to be saved. 
    if traj_save:
        fh = open(traj_file_name if traj_file_name is not None else 'trajectory.xyz', "w")
    else:
        fh = None
     
    # Check if the energies need to be saved.
    if ene_save:
        fh_ene = open(ene_file_name if ene_file_name is not None else 'energy.dat', "w")
        fh_ene.write(f"# N={n_part}  box size={box_size:.4f}  rho={density:.4f}  T0={temperature}  dt={dt}\n")
        fh_ene.write(f"# {'step':>6} {'KE':>14} {'PE':>14} {'E_tot':>14} {'T':>10} {'NH_E':>14}\n")
    else:
        fh_ene = None
    
    print(f"Production simulation for {prod_steps} steps")
    
    print(f"# N={len(pos)}  box size={box_size:.4f}  rho={density}  T0={temperature}")
    print(f"# {'step':>6} {'KE':>12} {'PE':>12} {'E_tot':>12} {'T':>8} {'NH_E':>12}")
    
    # Main MD loop for production step.
    for step in range(prod_steps + 1):
        vel,pos,forces,pe = verlet_integration(vel,pos,forces,dt,box_size,r_cut)
        ke = kinetic_energy(vel)
        nh_ener = 0
        if thermostat:
            vel, ke = nh_therm.scale(vel,ke)
            nh_ener += nh_therm.get_energy()

        if step % write_every == 0:
           T = 2.0 * ke / (3.0 * len(pos))
           print(f"{step:>6d} {ke:>12.4f} {pe:>12.4f} {ke + pe:>12.4f} {T:>8.4f} {nh_ener:>8.4f}")
           if fh is not None:
              write_xyz(fh, pos, step)
              
           if fh_ene is not None:
               write_energy(fh_ene, step, ke, pe, T,nh_ener)


if __name__ == "__main__":

    # ── Parameters ────────────────────────────────────────────────────────────────
    n_part = 200 # Number of particles
    temperature = 1.0 # Temperature in LJ units
    dt = 0.005 #Time step
    box_size = 8.0 # box side length
    eq_steps = int(1e3) # Number of equilibrium steps. Set to None if you wish to skip this part
    prod_steps = int(2e3) # Number of production steps
    seed = 42 # Random seed
    write_every = 100 # Control every how many steps it saves energy and positions
    r_cut = 2.5 # Cutoff 
    traj_file_name = 'lj_traj_nve.xyz' # Name of the file to which the trajectory would be written
    ene_file_name = 'lj_energy_nve.dat' # Name of the file to which the energies would be written
    thermostat= False # Activate or deactive the Nose Hoover thermostat
    nh_timescale = 500 # Timescale for the NH thermostat. 
    
 
    # Run simulation
    run(n_part=n_part, box_size=box_size, r_cut=r_cut, temperature=temperature,
        dt=dt, eq_steps=eq_steps, prod_steps=prod_steps,
        write_every=write_every, traj_save=True,
        traj_file_name=traj_file_name,
        ene_save=True, ene_file_name=ene_file_name, thermostat=thermostat,
        nh_timescale=nh_timescale, seed=seed)
