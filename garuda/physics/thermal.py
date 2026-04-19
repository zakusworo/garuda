"""
Thermal flow module - Non-isothermal reservoir simulation.

Essential for geothermal applications where temperature affects:
- Fluid properties (density, viscosity)
- Rock properties (thermal expansion)
- Heat transport (conduction + convection)
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from garuda.core.grid import Grid


@dataclass
class ThermalFlow:
    """
    Non-isothermal flow in porous media.
    
    Coupled equations:
    
    Mass:   ∂(φρ)/∂t + ∇·(ρu) = q
    Energy: ∂(ρCpT)/∂t + ∇·(ρCpTu) - ∇·(λ∇T) = Q_T
    
    where u = -(k/μ(T))·(∇p - ρ(T)g∇z)
    
    Parameters
    ----------
    grid : Grid
        Computational grid
    rock : RockProperties
        Rock properties (includes thermal conductivity)
    """
    
    grid: Grid
    rock: object  # RockProperties
    fluid: object  # FluidProperties
    
    # State variables
    pressure: np.ndarray = None
    temperature: np.ndarray = None
    
    def __post_init__(self):
        """Initialize state variables."""
        num_cells = self.grid.num_cells
        self.pressure = np.full(num_cells, 1e5)  # 1 bar
        self.temperature = np.full(num_cells, 293.15)  # 20°C
        
        # Previous time step for time derivatives
        self.p_prev = self.pressure.copy()
        self.T_prev = self.temperature.copy()
    
    def compute_energy_accumulation(self) -> np.ndarray:
        """
        Compute energy accumulation term: ∂(ρCpT)/∂t
        
        Returns
        -------
        accumulation : ndarray
            Energy accumulation [J/(m³·s)]
        """
        phi = self.rock.porosity if np.isscalar(self.rock.porosity) else np.mean(self.rock.porosity)
        
        # Fluid properties at current state
        rho = self.fluid.density(self.pressure, self.temperature)
        Cp = self.fluid.cp
        
        # Bulk heat capacity
        rhoCp_bulk = self.rock.heat_capacity_bulk(Cp, rho)
        
        # Energy density
        energy_density = rhoCp_bulk * self.temperature
        
        return energy_density
    
    def compute_heat_flux(
        self,
        mass_flux: np.ndarray,
    ) -> np.ndarray:
        """
        Compute heat flux including convection and conduction.
        
        q_T = ρCpT·u - λ∇T
        
        Parameters
        ----------
        mass_flux : ndarray
            Mass flux across faces [kg/(m²·s)]
        
        Returns
        -------
        heat_flux : ndarray
            Heat flux across faces [W/m²]
        """
        num_faces = self.grid.num_faces
        
        # Convective heat flux: ρCpT·u
        # Need face temperatures (upwind)
        T_face = self._interpolate_temperature_to_faces()
        rho_face = self.fluid.density(self.pressure.mean(), T_face)
        Cp = self.fluid.cp
        
        convective_flux = rho_face * Cp * T_face * mass_flux
        
        # Conductive heat flux: -λ∇T
        conductive_flux = self._compute_conductive_flux()
        
        return convective_flux + conductive_flux
    
    def _interpolate_temperature_to_faces(self) -> np.ndarray:
        """Interpolate cell temperatures to faces (upwind)."""
        num_faces = self.grid.num_faces
        T_face = np.zeros(num_faces)
        
        # Simple averaging for now (upwind would be better for stability)
        if self.grid.dim == 1:
            T_face[1:-1] = (self.temperature[:-1] + self.temperature[1:]) / 2
            T_face[0] = self.temperature[0]
            T_face[-1] = self.temperature[-1]
        
        return T_face
    
    def _compute_conductive_flux(self) -> np.ndarray:
        """Compute conductive heat flux: -λ∇T."""
        num_faces = self.grid.num_faces
        conductive_flux = np.zeros(num_faces)
        
        # Thermal conductivity (effective, including fluid)
        lambda_eff = self._effective_thermal_conductivity()
        
        if self.grid.dim == 1:
            dx = self.grid.dx if np.isscalar(self.grid.dx) else np.mean(self.grid.dx)
            
            # Interior faces
            for i in range(1, num_faces - 1):
                dT = self.temperature[i] - self.temperature[i - 1]
                conductive_flux[i] = -lambda_eff * dT / dx
            
            # Boundary faces (one-sided)
            conductive_flux[0] = -lambda_eff * (self.temperature[0] - self.T_prev[0]) / (dx / 2)
            conductive_flux[-1] = -lambda_eff * (self.temperature[-1] - self.T_prev[-1]) / (dx / 2)
        
        return conductive_flux
    
    def _effective_thermal_conductivity(self) -> float:
        """
        Compute effective thermal conductivity of rock-fluid system.
        
        lambda_eff = (1-φ)·lambda_rock + φ·lambda_fluid
        
        Returns
        -------
        lambda_eff : float
            Effective thermal conductivity [W/(m·K)]
        """
        phi = self.rock.porosity if np.isscalar(self.rock.porosity) else np.mean(self.rock.porosity)
        lambda_fluid = 0.6  # Water thermal conductivity [W/(m·K)]
        
        lambda_eff = (1 - phi) * self.rock.lambda_rock + phi * lambda_fluid
        
        return lambda_eff
    
    def build_energy_matrix(
        self,
        dt: float,
        mass_flux: np.ndarray,
        heat_sources: np.ndarray,
    ) -> Tuple[csr_matrix, np.ndarray]:
        """
        Build the energy equation linear system.
        
        (ρCp)^n+1 * T^n+1 / dt - ∇·(λ∇T^n+1) + ∇·(ρCpT u)^n+1 = Q_T + (ρCp)^n * T^n / dt
        
        Parameters
        ----------
        dt : float
            Time step [s]
        mass_flux : ndarray
            Mass flux field [kg/(m²·s)]
        heat_sources : ndarray
            Heat sources [W/m³]
        
        Returns
        -------
        A : csr_matrix
            System matrix
        b : ndarray
            Right-hand side vector
        """
        num_cells = self.grid.num_cells
        
        # Current energy accumulation
        energy_accum_new = self.compute_energy_accumulation()
        
        # Previous energy accumulation
        rhoCp_bulk_prev = self.rock.heat_capacity_bulk(
            self.fluid.cp,
            self.fluid.density(self.p_prev, self.T_prev)
        )
        energy_accum_old = rhoCp_bulk_prev * self.T_prev
        
        # Build matrix
        A = lil_matrix((num_cells, num_cells))
        b = np.zeros(num_cells)
        
        # Time derivative term: (ρCp T)^n+1 / dt
        rhoCp_bulk = self.rock.heat_capacity_bulk(
            self.fluid.cp,
            self.fluid.density(self.pressure, self.temperature)
        )
        
        for i in range(num_cells):
            A[i, i] += rhoCp_bulk[i] / dt
            b[i] += energy_accum_old[i] / dt
            b[i] += heat_sources[i]
        
        # Conduction term: -∇·(λ∇T)
        lambda_eff = self._effective_thermal_conductivity()
        
        if self.grid.dim == 1:
            dx = self.grid.dx if np.isscalar(self.grid.dx) else np.mean(self.grid.dx)
            A_cond = dx  # Face area (assume unit area in 1D)
            
            for i in range(num_cells):
                # Left face
                if i > 0:
                    T_cond = lambda_eff * A_cond / dx
                    A[i, i] += T_cond
                    A[i, i - 1] -= T_cond
                
                # Right face
                if i < num_cells - 1:
                    T_cond = lambda_eff * A_cond / dx
                    A[i, i] += T_cond
                    A[i, i + 1] -= T_cond
        
        # Convective term handled explicitly (or via upwind scheme)
        # TODO: Implement implicit convection
        
        return A.tocsr(), b
    
    def step_coupled(
        self,
        dt: float,
        source_terms: np.ndarray,
        heat_sources: np.ndarray,
        bc_type: str,
        bc_values: Optional[Dict],
        flow_solver,
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> Dict:
        """
        Fully coupled implicit time step for thermal flow.
        
        Uses sequential implicit method (SIM):
        1. Solve pressure equation (with fixed T)
        2. Solve energy equation (with updated fluxes)
        3. Update fluid properties
        4. Iterate until convergence
        
        Parameters
        ----------
        dt : float
            Time step [s]
        source_terms : ndarray
            Mass source/sink [kg/s]
        heat_sources : ndarray
            Heat source/sink [W/m³]
        bc_type : str
            Boundary condition type
        bc_values : dict, optional
            Boundary values {'pressure': ..., 'temperature': ...}
        flow_solver : TPFASolver
            TPFA flow solver
        max_iter : int
            Maximum coupling iterations
        tol : float
            Convergence tolerance
        
        Returns
        -------
        result : dict
            Convergence info and iteration count
        """
        converged = False
        
        for iteration in range(max_iter):
            # Store old values
            p_old = self.pressure.copy()
            T_old = self.temperature.copy()
            
            # Step 1: Solve pressure equation (with current T)
            # Update viscosity and density based on current T
            flow_solver.mu = self.fluid.viscosity(self.temperature)
            flow_solver.rho = self.fluid.density(self.pressure, self.temperature)
            flow_solver.transmissibilities = flow_solver._compute_transmissibilities()
            
            # Solve for pressure
            self.pressure = flow_solver.solve(source_terms, bc_type, bc_values.get('pressure') if bc_values else None)
            
            # Step 2: Compute mass flux
            flux_data = flow_solver.compute_flux(self.pressure)
            mass_flux = flux_data.flux
            
            # Step 3: Solve energy equation
            A, b = self.build_energy_matrix(dt, mass_flux, heat_sources)
            self.temperature = spsolve(A, b)
            
            # Step 4: Check convergence
            dp_norm = np.linalg.norm(self.pressure - p_old) / (np.linalg.norm(p_old) + 1e-10)
            dT_norm = np.linalg.norm(self.temperature - T_old) / (np.linalg.norm(T_old) + 1e-10)
            
            max_change = max(dp_norm, dT_norm)
            
            if max_change < tol:
                converged = True
                break
        
        # Update previous time step
        self.p_prev = self.pressure.copy()
        self.T_prev = self.temperature.copy()
        
        return {
            'converged': converged,
            'iterations': iteration + 1,
            'pressure_change': dp_norm,
            'temperature_change': dT_norm,
        }
    
    def compute_geothermal_gradient(
        self,
        surface_temp: float = 293.15,
        gradient: float = 0.03,  # 30°C/km typical
        depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Initialize temperature with geothermal gradient.
        
        T(z) = T_surface + gradient * z
        
        Parameters
        ----------
        surface_temp : float
            Surface temperature [K]
        gradient : float
            Geothermal gradient [K/m]
        depth : ndarray, optional
            Cell depths [m] (uses grid centroids if None)
        
        Returns
        -------
        T_initial : ndarray
            Initial temperature field [K]
        """
        if depth is None:
            depth = -self.grid.cell_centroids[:, 2]  # z is negative downward
        
        T_initial = surface_temp + gradient * depth
        
        self.temperature = T_initial
        self.T_prev = T_initial.copy()
        
        return T_initial
