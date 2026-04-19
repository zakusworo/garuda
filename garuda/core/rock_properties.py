"""
Rock properties module - Permeability, porosity, and rock mechanics.

Supports heterogeneous and anisotropic reservoirs.
"""

import numpy as np
from typing import Union, Dict
from dataclasses import dataclass


@dataclass
class RockProperties:
    """
    Rock properties for reservoir simulation.
    
    Parameters
    ----------
    porosity : float or ndarray
        Porosity (dimensionless, 0-1)
    permeability : float or ndarray
        Permeability [m²] or [Darcy]
    permeability_unit : str
        Unit of permeability: 'm2', 'darcy', 'md' (millidarcy)
    c_rock : float
        Rock compressibility [1/Pa]
    cp : float
        Rock heat capacity [J/(kg·K)]
    rho_rock : float
        Rock density [kg/m³]
    lambda_rock : float
        Rock thermal conductivity [W/(m·K)]
    """
    
    porosity: Union[float, np.ndarray] = 0.2
    permeability: Union[float, np.ndarray] = 1e-12  # 1 Darcy default
    permeability_unit: str = 'm2'
    c_rock: float = 1e-9  # Rock compressibility [1/Pa]
    cp: float = 840  # Heat capacity [J/(kg·K)] for sandstone
    rho_rock: float = 2650  # Density [kg/m³] for sandstone
    lambda_rock: float = 2.5  # Thermal conductivity [W/(m·K)]
    
    # Anisotropy ratios (kx:ky:kz)
    k_ratio: tuple = (1.0, 1.0, 0.1)  # Typical: kz << kx, ky
    
    def __post_init__(self):
        """Convert permeability to SI units and set up tensors."""
        # Convert to m²
        if self.permiability_unit == 'darcy':
            self.permiability_m2 = np.asarray(self.permiability) * 9.869233e-13
        elif self.permiability_unit == 'md':
            self.permiability_m2 = np.asarray(self.permiability) * 9.869233e-16
        else:  # 'm2'
            self.permiability_m2 = np.asarray(self.permiability)
        
        # Build permeability tensor
        self._build_perm_tensor()
    
    def _build_perm_tensor(self):
        """Build full permeability tensor for each cell."""
        perm = self.permiability_m2
        
        if np.isscalar(perm) or perm.ndim == 0:
            # Homogeneous isotropic
            perm_val = float(perm) * np.array(self.k_ratio)
            self.perm_tensor = np.zeros((3, 3))
            for i in range(3):
                self.perm_tensor[i, i] = perm_val[i]
            self.perm_tensor = np.expand_dims(self.perm_tensor, 0)
            
        elif perm.ndim == 1:
            # Heterogeneous isotropic (one value per cell)
            n_cells = len(perm)
            self.perm_tensor = np.zeros((n_cells, 3, 3))
            for i in range(3):
                self.perm_tensor[:, i, i] = perm * self.k_ratio[i]
                
        elif perm.ndim == 2 and perm.shape[1] == 3:
            # Anisotropic (kx, ky, kz per cell)
            n_cells = perm.shape[0]
            self.perm_tensor = np.zeros((n_cells, 3, 3))
            for i in range(3):
                self.perm_tensor[:, i, i] = perm[:, i]
                
        elif perm.ndim == 3:
            # Full tensor already provided
            self.perm_tensor = perm
        else:
            raise ValueError(f"Invalid permeability shape: {perm.shape}")
    
    def set_heterogeneous(
        self,
        porosity: np.ndarray,
        permeability: np.ndarray,
        permeability_unit: str = 'md',
    ):
        """
        Set heterogeneous properties.
        
        Parameters
        ----------
        porosity : ndarray
            Porosity per cell
        permeability : ndarray
            Permeability per cell
        permeability_unit : str
            Unit of permeability
        """
        self.porosity = porosity
        self.permiability = permeability
        self.permiability_unit = permeability_unit
        self.__post_init__()
    
    def set_channelized_permeability(
        self,
        nx: int, ny: int, nz: int,
        channel_orientation: str = 'x',
        channel_fraction: float = 0.3,
        k_channel: float = 1000,
        k_background: float = 10,
    ):
        """
        Generate channelized permeability field.
        
        Parameters
        ----------
        nx, ny, nz : int
            Grid dimensions
        channel_orientation : str
            Direction of channels: 'x', 'y', or 'z'
        channel_fraction : float
            Fraction of cells that are channels
        k_channel : float
            Channel permeability [md]
        k_background : float
            Background permeability [md]
        """
        np.random.seed(42)  # Reproducibility
        
        if channel_orientation == 'x':
            # Channels aligned in x-direction
            perm = np.zeros((nx, ny, nz))
            for j in range(ny):
                for k in range(nz):
                    is_channel = np.random.rand() < channel_fraction
                    perm[:, j, k] = k_channel if is_channel else k_background
                    
        elif channel_orientation == 'y':
            perm = np.zeros((nx, ny, nz))
            for i in range(nx):
                for k in range(nz):
                    is_channel = np.random.rand() < channel_fraction
                    perm[i, :, k] = k_channel if is_channel else k_background
                    
        else:  # 'z'
            perm = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    is_channel = np.random.rand() < channel_fraction
                    perm[i, j, :] = k_channel if is_channel else k_background
        
        self.set_heterogeneous(
            porosity=np.full((nx, ny, nz), self.porosity),
            permeability=perm.ravel(),
            permeability_unit='md',
        )
    
    def set_gaussian_permeability(
        self,
        nx: int, ny: int, nz: int,
        mean_logk: float = 3.0,
        std_logk: float = 1.0,
        correlation_length: float = 0.1,
    ):
        """
        Generate Gaussian random permeability field.
        
        Parameters
        ----------
        nx, ny, nz : int
            Grid dimensions
        mean_logk : float
            Mean of log10(k) [md]
        std_logk : float
            Standard deviation of log10(k)
        correlation_length : float
            Correlation length (fraction of domain)
        """
        from scipy import ndimage
        
        # Generate white noise in log-permeability
        logk = np.random.normal(mean_logk, std_logk, (nx, ny, nz))
        
        # Apply Gaussian filter for spatial correlation
        sigma = correlation_length * max(nx, ny, nz)
        logk_smooth = ndimage.gaussian_filter(logk, sigma=sigma)
        
        # Convert to permeability
        perm = 10 ** logk_smooth
        
        self.set_heterogeneous(
            porosity=np.full((nx, ny, nz), self.porosity),
            permeability=perm.ravel(),
            permeability_unit='md',
        )
    
    def total_compressibility(
        self,
        fluid_compressibility: float,
    ) -> Union[float, np.ndarray]:
        """
        Compute total compressibility (rock + fluid).
        
        c_t = c_rock + φ * c_fluid
        
        Parameters
        ----------
        fluid_compressibility : float
            Fluid compressibility [1/Pa]
        
        Returns
        -------
        c_t : float or ndarray
            Total compressibility [1/Pa]
        """
        phi = self.porosity
        return self.c_rock + phi * fluid_compressibility
    
    def heat_capacity_bulk(
        self,
        fluid_cp: float,
        fluid_rho: float,
    ) -> float:
        """
        Compute bulk heat capacity of rock-fluid system.
        
        (ρCp)_bulk = (1-φ) * ρ_rock * Cp_rock + φ * ρ_fluid * Cp_fluid
        
        Parameters
        ----------
        fluid_cp : float
            Fluid specific heat [J/(kg·K)]
        fluid_rho : float
            Fluid density [kg/m³]
        
        Returns
        -------
        rhoCp : float
            Bulk heat capacity [J/(m³·K)]
        """
        phi = self.porosity if np.isscalar(self.porosity) else np.mean(self.porosity)
        
        rhoCp = (
            (1 - phi) * self.rho_rock * self.cp
            + phi * fluid_rho * fluid_cp
        )
        
        return rhoCp
    
    def thermal_diffusivity(
        self,
        fluid_cp: float,
        fluid_rho: float,
    ) -> float:
        """
        Compute thermal diffusivity.
        
        alpha = lambda / (ρCp)_bulk
        
        Parameters
        ----------
        fluid_cp : float
            Fluid specific heat [J/(kg·K)]
        fluid_rho : float
            Fluid density [kg/m³]
        
        Returns
        -------
        alpha : float
            Thermal diffusivity [m²/s]
        """
        rhoCp = self.heat_capacity_bulk(fluid_cp, fluid_rho)
        return self.lambda_rock / rhoCp
