"""Fluid properties module - PVT and transport properties.

Supports water, oil, gas, and geothermal fluids.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FluidProperties:
    """Fluid properties for reservoir simulation.

    Parameters
    ----------
    fluid_type : str
        Type of fluid: 'water', 'oil', 'gas', 'geothermal'
    mu : float, optional
        Viscosity [Pa·s] (constant if provided)
    rho : float, optional
        Density [kg/m³] (constant if provided)
    cp : float, optional
        Specific heat capacity [J/(kg·K)]
    beta : float, optional
        Thermal expansion coefficient [1/K]
    c_fluid : float, optional
        Fluid compressibility [1/Pa]

    """

    fluid_type: str = "water"
    mu: float | None = None  # Viscosity [Pa·s]
    rho: float | None = None  # Density [kg/m³]
    cp: float = 4182  # Specific heat [J/(kg·K)] for water
    beta: float = 2.1e-4  # Thermal expansion [1/K] for water
    c_fluid: float = 4.4e-10  # Compressibility [1/Pa] for water

    # Reference conditions
    T_ref: float = 293.15  # Reference temperature [K]
    p_ref: float = 1e5  # Reference pressure [Pa]
    rho_ref: float | None = None  # Reference density

    def __post_init__(self):
        """Set default properties based on fluid type."""
        if self.fluid_type == "water":
            if self.mu is None:
                self.mu = 1e-3  # Water at 20°C
            if self.rho is None:
                self.rho = 998  # kg/m³ at 20°C
            if self.rho_ref is None:
                self.rho_ref = self.rho

        elif self.fluid_type == "oil":
            if self.mu is None:
                self.mu = 5e-3  # Light oil
            if self.rho is None:
                self.rho = 850
            if self.rho_ref is None:
                self.rho_ref = self.rho

        elif self.fluid_type == "gas":
            if self.mu is None:
                self.mu = 1.8e-5  # Air at 20°C
            if self.rho is None:
                self.rho = 1.2
            if self.rho_ref is None:
                self.rho_ref = self.rho

        elif self.fluid_type == "geothermal":
            # High-temperature water/steam
            if self.mu is None:
                self.mu = 1.5e-4  # Water at 250°C
            if self.rho is None:
                self.rho = 800  # Reduced density at high T
            if self.rho_ref is None:
                self.rho_ref = 1000
            self.cp = 4500  # Higher cp at high T
            self.beta = 5e-4  # Higher expansion

    def density(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray = None,
    ) -> float | np.ndarray:
        """Compute density as function of pressure and temperature.

        rho = rho_ref * exp(c_fluid * (p - p_ref) - beta * (T - T_ref))

        Parameters
        ----------
        pressure : float or ndarray
            Pressure [Pa]
        temperature : float or ndarray, optional
            Temperature [K] (uses T_ref if None)

        Returns
        -------
        rho : float or ndarray
            Density [kg/m³]

        """
        if temperature is None:
            temperature = self.T_ref

        dp = np.asarray(pressure) - self.p_ref
        dT = np.asarray(temperature) - self.T_ref

        rho = self.rho_ref * np.exp(self.c_fluid * dp - self.beta * dT)

        return rho

    def viscosity(
        self,
        temperature: float | np.ndarray = None,
    ) -> float | np.ndarray:
        """Compute viscosity as function of temperature.

        Uses simplified Andrade equation:
        mu = A * exp(B / T)

        Parameters
        ----------
        temperature : float or ndarray, optional
            Temperature [K]

        Returns
        -------
        mu : float or ndarray
            Viscosity [Pa·s]

        """
        if self.mu is not None and temperature is None:
            return self.mu

        if temperature is None:
            temperature = self.T_ref

        T = np.asarray(temperature)

        # Simplified temperature dependence
        # For water: mu decreases ~2% per °C
        mu_ref = self.mu if self.mu is not None else 1e-3
        mu = mu_ref * np.exp(0.02 * (self.T_ref - T))

        return mu

    def formation_volume_factor(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray = None,
    ) -> float | np.ndarray:
        """Compute formation volume factor B.

        B = V_reservoir / V_surface = rho_surface / rho_reservoir

        Parameters
        ----------
        pressure : float or ndarray
            Pressure [Pa]
        temperature : float or ndarray, optional
            Temperature [K]

        Returns
        -------
        B : float or ndarray
            Formation volume factor (dimensionless)

        """
        rho = self.density(pressure, temperature)
        return self.rho_ref / rho

    def total_compressibility(
        self,
        pressure: float | np.ndarray = None,
        temperature: float | np.ndarray = None,
    ) -> float | np.ndarray:
        """Compute total compressibility.

        c_t = c_fluid + c_rock (for single phase)

        Returns
        -------
        c_t : float or ndarray
            Total compressibility [1/Pa]

        """
        # Fluid compressibility only (rock added separately)
        return self.c_fluid
