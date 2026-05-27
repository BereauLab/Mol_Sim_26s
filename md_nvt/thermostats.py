import math
import random
import numpy as np
from typing import List, Tuple

class VelocityScaling:
    """
    Simple instantaneous velocity-rescaling thermostat.

    Scales all momenta by s = sqrt(T * g / (2K)) so that the
    instantaneous kinetic temperature exactly equals *temperature*.

    Equations
    ---------
    s = sqrt(T * g / (2 * K))
    p'_i = s * p_i
    """

    def __init__(self, temperature: float, degrees_of_freedom: int) -> None:
        self.temperature       = temperature
        self.degrees_of_freedom = degrees_of_freedom

    def scale(self, momenta: np.ndarray, kinetic_energy: float) -> Tuple[np.ndarray, float]:
        """
        Return rescaled momenta and updated kinetic energy.

        Parameters
        ----------
        momenta        : (N, 3) array of particle momenta
        kinetic_energy : current total kinetic energy

        Returns
        -------
        momenta        : rescaled (N, 3) array
        kinetic_energy : kinetic energy after rescaling
        """
        s = math.sqrt(self.temperature * self.degrees_of_freedom / (2.0 * kinetic_energy))
        momenta        = momenta * s
        kinetic_energy = kinetic_energy * s * s
        return momenta, kinetic_energy


class NoseHooverNVT:
    """
    Nose-Hoover NVT thermostat (single chain).

    Adds one extra degree of freedom (xi, eta) that acts as a heat bath,
    producing canonical (NVT) sampling in the long-time limit.

    Equations  (notation from thermostats.cpp)
    ------------------------------------------
    Q        = g * tau^2                          (thermostat mass)
    xi_dot   = (2K - g*T) / Q                    (thermostat force)
    xi(t+dt) = xi(t) + xi_dot * dt
    s        = exp(-xi * dt)                      (velocity scale factor)
    eta(t+dt)= eta(t) + xi * dt                  (position, for energy)
    p'_i     = s * p_i
    E_NH     = 0.5 * Q * xi^2 + g * T * eta      (conserved contribution)
    """

    def __init__(self, temperature: float, degrees_of_freedom: int,
                 timescale_parameter: float, time_step: float, seed: int = 12) -> None:
        self.temperature         = temperature
        self.degrees_of_freedom  = degrees_of_freedom
        self.timescale_parameter = timescale_parameter
        self.time_step           = time_step

        # Q = g * tau^2
        self.thermostat_mass = degrees_of_freedom * timescale_parameter ** 2

        # initialise thermostat velocity from Maxwell-Boltzmann
        rng = random.Random(seed)
        # Box-Muller for a single Gaussian sample
        u1, u2 = rng.random(), rng.random()
        gauss   = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        self.thermostat_velocity = gauss * math.sqrt(temperature / self.thermostat_mass)

        self.thermostat_force    = 0.0
        self.thermostat_position = 0.0

    def scale(self, momenta: np.ndarray, kinetic_energy: float) -> Tuple[np.ndarray, float]:
        """
        Apply one Nose-Hoover step and return updated momenta + kinetic energy.

        Parameters
        ----------
        momenta        : (N, 3) array
        kinetic_energy : current total KE

        Returns
        -------
        momenta        : updated (N, 3) array
        kinetic_energy : updated KE
        """
        dt = self.time_step

        # thermostat force
        self.thermostat_force = (
            (2.0 * kinetic_energy - self.degrees_of_freedom * self.temperature)
            / self.thermostat_mass
        )
        # integrate thermostat velocity
        self.thermostat_velocity += self.thermostat_force * dt

        # velocity scale factor  s = exp(-xi * dt)
        s = math.exp(-self.thermostat_velocity * dt)

        # integrate thermostat position
        self.thermostat_position += self.thermostat_velocity * dt

        # scale particle momenta
        momenta        = momenta * s
        kinetic_energy = kinetic_energy * s * s

        return momenta, kinetic_energy

    def get_energy(self) -> float:
        """
        Return the Nose-Hoover contribution to the conserved Hamiltonian.

        E_NH = 0.5 * Q * xi^2 + g * T * eta
        """
        return (0.5 * self.thermostat_mass * self.thermostat_velocity ** 2
                + self.degrees_of_freedom * self.temperature * self.thermostat_position)
