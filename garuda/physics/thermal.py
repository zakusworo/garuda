"""Thermal flow module - Non-isothermal reservoir simulation.

Essential for geothermal applications where temperature affects:
- Fluid properties (density, viscosity)
- Rock properties (thermal expansion)
- Heat transport (conduction + convection)
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from garuda.core.grid import Grid


@dataclass
class ThermalFlow:
    """Non-isothermal flow in porous media.

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
        """Compute energy accumulation term: ∂(ρCpT)/∂t

        Returns
        -------
        accumulation : ndarray
            Energy accumulation [J/(m³·s)]

        """
        # Fluid properties at current state
        rho = self.fluid.density(self.pressure, self.temperature)
        Cp = self.fluid.cp

        # Bulk heat capacity (rock_properties.heat_capacity_bulk handles
        # heterogeneous porosity natively).
        rhoCp_bulk = self.rock.heat_capacity_bulk(Cp, rho)

        # Energy density
        energy_density = rhoCp_bulk * self.temperature

        return energy_density

    def compute_heat_flux(
        self,
        mass_flux: np.ndarray,
    ) -> np.ndarray:
        """Compute heat flux including convection and conduction.

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
        """Interpolate cell temperatures to faces (central averaging).

        Works for any grid dimension by walking the grid.face_cells map.
        Boundary faces inherit the interior cell's temperature.
        """
        num_faces = self.grid.num_faces
        T_face = np.zeros(num_faces)

        for f in range(num_faces):
            cL, cR = int(self.grid.face_cells[f, 0]), int(self.grid.face_cells[f, 1])
            if cL >= 0 and cR >= 0:
                T_face[f] = 0.5 * (self.temperature[cL] + self.temperature[cR])
            elif cL >= 0:
                T_face[f] = self.temperature[cL]
            elif cR >= 0:
                T_face[f] = self.temperature[cR]

        return T_face

    def _compute_conductive_flux(self) -> np.ndarray:
        """Compute conductive heat flux per face: q_f = -λ_eff (T_R - T_L) / d.

        Uses face_cells connectivity, so it works in 1D, 2D, and 3D.
        Boundary fluxes are returned as zero (zero-flux Neumann); apply
        Dirichlet BCs via build_energy_matrix for consistent treatment.
        """
        num_faces = self.grid.num_faces
        conductive_flux = np.zeros(num_faces)
        lambda_eff = self._effective_thermal_conductivity()  # per-cell

        for f in range(num_faces):
            cL, cR = int(self.grid.face_cells[f, 0]), int(self.grid.face_cells[f, 1])
            if cL >= 0 and cR >= 0:
                d = float(np.linalg.norm(self.grid.cell_centroids[cR] - self.grid.cell_centroids[cL]))
                if d > 0:
                    lam_face = self._harmonic_mean(float(lambda_eff[cL]), float(lambda_eff[cR]))
                    conductive_flux[f] = -lam_face * (self.temperature[cR] - self.temperature[cL]) / d

        return conductive_flux

    def _effective_thermal_conductivity(self) -> np.ndarray:
        """Effective thermal conductivity of rock+fluid system, per cell.

        lambda_eff = (1-φ)·lambda_rock + φ·lambda_fluid

        Returns
        -------
        lambda_eff : ndarray of shape (num_cells,)
            Effective thermal conductivity [W/(m·K)] per cell. Heterogeneous
            porosity propagates through; homogeneous porosity yields a
            uniform array.

        """
        phi = np.asarray(self.rock.porosity)
        if phi.ndim == 0:
            phi = np.full(self.grid.num_cells, float(phi))
        lambda_fluid = 0.6  # Water thermal conductivity [W/(m·K)]
        return (1.0 - phi) * self.rock.lambda_rock + phi * lambda_fluid

    @staticmethod
    def _harmonic_mean(a: float, b: float) -> float:
        """Harmonic mean used to combine cell-centred conductivities at a face."""
        s = a + b
        return 2.0 * a * b / s if s > 0 else 0.0

    def build_energy_matrix(
        self,
        dt: float,
        mass_flux: np.ndarray,
        heat_sources: np.ndarray,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Build the energy equation linear system.

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
        rhoCp_bulk_prev = self.rock.heat_capacity_bulk(self.fluid.cp, self.fluid.density(self.p_prev, self.T_prev))
        energy_accum_old = rhoCp_bulk_prev * self.T_prev

        # Build matrix
        A = lil_matrix((num_cells, num_cells))
        b = np.zeros(num_cells)

        # Time derivative term: (ρCp T)^n+1 / dt
        rhoCp_bulk = self.rock.heat_capacity_bulk(self.fluid.cp, self.fluid.density(self.pressure, self.temperature))

        for i in range(num_cells):
            A[i, i] += rhoCp_bulk[i] / dt
            b[i] += energy_accum_old[i] / dt
            b[i] += heat_sources[i]

        # Conduction term: -∇·(λ∇T) assembled per face
        #   T_f = λ_face * A_f / d_LR   (W/K)
        # with λ_face = harmonic mean of cell-centred λ on each side. Yields
        # a symmetric stencil; dimension-agnostic via face_cells.
        lambda_eff = self._effective_thermal_conductivity()  # per-cell

        for f in range(self.grid.num_faces):
            cL, cR = int(self.grid.face_cells[f, 0]), int(self.grid.face_cells[f, 1])
            if cL >= 0 and cR >= 0:
                d = float(np.linalg.norm(self.grid.cell_centroids[cR] - self.grid.cell_centroids[cL]))
                if d > 0:
                    area = float(self.grid.face_areas[f]) if self.grid.face_areas.size else 1.0
                    lam_face = self._harmonic_mean(float(lambda_eff[cL]), float(lambda_eff[cR]))
                    T_cond = lam_face * area / d
                    A[cL, cL] += T_cond
                    A[cR, cR] += T_cond
                    A[cL, cR] -= T_cond
                    A[cR, cL] -= T_cond
            # Boundary faces: zero-flux Neumann (no contribution).

        # Convective term handled explicitly (or via upwind scheme)
        # TODO: Implement implicit convection

        return A.tocsr(), b

    def step_coupled(
        self,
        dt: float,
        source_terms: np.ndarray,
        heat_sources: np.ndarray,
        bc_type: str,
        bc_values: dict | None,
        flow_solver,
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> dict:
        """Fully coupled implicit time step for thermal flow.

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
            # Use scalar values for solver parameters (gravity term needs scalar rho)
            flow_solver.mu = np.mean(self.fluid.viscosity(self.temperature))
            flow_solver.rho = np.mean(self.fluid.density(self.pressure, self.temperature))
            flow_solver.transmissibilities = flow_solver._compute_transmissibilities()

            # Solve for pressure
            self.pressure = flow_solver.solve(source_terms, bc_type, bc_values.get("pressure") if bc_values else None)

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
            "converged": converged,
            "iterations": iteration + 1,
            "pressure_change": dp_norm,
            "temperature_change": dT_norm,
        }

    def compute_geothermal_gradient(
        self,
        surface_temp: float = 293.15,
        gradient: float = 0.03,  # 30°C/km typical
        depth: np.ndarray | None = None,
    ) -> np.ndarray:
        """Initialize temperature with geothermal gradient.

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
            # Grid z grows from 0 upward; treat z as depth below the top of
            # the reservoir so temperature increases with z.
            depth = self.grid.cell_centroids[:, 2]

        T_initial = surface_temp + gradient * depth

        self.temperature = T_initial
        self.T_prev = T_initial.copy()

        return T_initial
