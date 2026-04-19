"""
Physics module - Governing equations for reservoir flow.

Single-phase and multiphase flow implementations.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class SinglePhaseFlow:
    """
    Single-phase flow in porous media.
    
    Governing equation:
        ∂(φρ)/∂t + ∇·(ρu) = q
    
    where u = -(k/μ)·(∇p - ρg∇z) (Darcy's law)
    
    Parameters
    ----------
    grid : Grid
        Computational grid
    fluid : FluidProperties
        Fluid properties
    rock : RockProperties
        Rock properties
    """
    
    grid: object  # Grid type
    fluid: object  # FluidProperties type
    rock: object  # RockProperties type
    
    # State variables
    pressure: np.ndarray = None
    saturation: np.ndarray = None  # Always 1.0 for single-phase
    temperature: np.ndarray = None  # For non-isothermal
    
    def __post_init__(self):
        """Initialize state variables."""
        num_cells = self.grid.num_cells
        self.pressure = np.full(num_cells, 1e5)  # Default 1 bar
        self.saturation = np.ones(num_cells)
        if self.temperature is None:
            self.temperature = np.full(num_cells, 293.15)  # Default 20°C
    
    def compute_accumulation(self) -> np.ndarray:
        """
        Compute accumulation term: ∂(φρ)/∂t
        
        Returns
        -------
        accumulation : ndarray
            Mass accumulation rate [kg/(m³·s)]
        """
        phi = self.rock.porosity
        rho = self.fluid.density(self.pressure, self.temperature)
        
        # For implicit time stepping, this is evaluated at new time level
        return phi * rho
    
    def compute_flux(self, solver) -> np.ndarray:
        """
        Compute total flux for each cell.
        
        Parameters
        ----------
        solver : TPFASolver
            TPFA flux solver
        
        Returns
        -------
        flux : ndarray
            Net flux for each cell [kg/s]
        """
        flux_data = solver.compute_flux(self.pressure)
        
        # Accumulate face fluxes to cells
        cell_flux = np.zeros(self.grid.num_cells)
        
        if self.grid.dim == 1:
            for i in range(self.grid.num_cells):
                cell_flux[i] = flux_data.flux[i] - flux_data.flux[i + 1]
        
        return cell_flux
    
    def step_implicit(
        self,
        dt: float,
        source_terms: np.ndarray,
        bc_type: str,
        bc_values: Optional[np.ndarray],
        solver,
        max_iter: int = 10,
        tol: float = 1e-8,
    ) -> Dict:
        """
        Implicit time step for single-phase flow.
        
        Parameters
        ----------
        dt : float
            Time step [s]
        source_terms : ndarray
            Source/sink terms [kg/s]
        bc_type : str
            Boundary condition type
        bc_values : ndarray, optional
            Boundary values
        solver : TPFASolver
            TPFA solver
        max_iter : int
            Maximum Newton iterations
        tol : float
            Convergence tolerance
        
        Returns
        -------
        result : dict
            Convergence info and final state
        """
        for iteration in range(max_iter):
            # Build residual: R = accumulation_new - accumulation_old + flux*dt - source*dt
            accumulation_new = self.compute_accumulation()
            
            # Compute flux at current pressure
            cell_flux = self.compute_flux(solver)
            
            # Residual
            residual = (
                self.grid.cell_volumes * (accumulation_new - self.prev_accumulation) / dt
                + cell_flux
                - source_terms
            )
            
            # Check convergence
            residual_norm = np.linalg.norm(residual) / np.linalg.norm(source_terms + 1e-10)
            
            if residual_norm < tol:
                break
            
            # TODO: Newton update (requires Jacobian)
            # For now, use pressure solve
            self.pressure = solver.solve(source_terms, bc_type, bc_values)
            
            # Update state
            self.prev_accumulation = accumulation_new.copy()
        
        return {
            'converged': residual_norm < tol,
            'iterations': iteration + 1,
            'residual_norm': residual_norm,
        }
