"""Physics module - Governing equations for reservoir flow.

Single-phase and multiphase flow implementations.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SinglePhaseFlow:
    """Single-phase flow in porous media.

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

    grid: Any
    fluid: Any
    rock: Any

    # State variables
    pressure: Any = field(default_factory=lambda: None)
    saturation: Any = field(default_factory=lambda: None)
    temperature: Any = field(default_factory=lambda: None)

    # Internal
    prev_accumulation: Any = field(default_factory=lambda: None, repr=False)

    def __post_init__(self):
        """Initialize state variables."""
        num_cells = self.grid.num_cells
        if self.pressure is None:
            self.pressure = np.full(num_cells, 1e5)  # Default 1 bar
        if self.saturation is None:
            self.saturation = np.ones(num_cells)
        if self.temperature is None:
            self.temperature = np.full(num_cells, 293.15)  # Default 20°C
        if self.prev_accumulation is None:
            self.prev_accumulation = np.zeros(num_cells)

    def compute_accumulation(self) -> np.ndarray:
        """Mass per unit volume at the current state: φ·ρ.

        Returned by itself (not divided by dt). The time-derivative
        ∂(φρ)/∂t is formed by the caller as
        ``(accumulation_new − accumulation_old) / dt``; see
        :meth:`step_implicit`.

        Returns
        -------
        accumulation : ndarray
            Mass density [kg/m³] per cell at the current pressure /
            temperature.

        """
        phi = self.rock.porosity
        rho = self.fluid.density(self.pressure, self.temperature)
        return phi * rho

    def compute_flux(self, solver) -> np.ndarray:
        """Compute net mass outflow per cell from TPFA face fluxes.

        Sign convention: ``cell_flux[i] = Σ_f q_f`` summed with the outward
        orientation, i.e. positive value means net mass leaving the cell.
        This is consistent with the residual

            V * ∂(φρ)/∂t + cell_flux − source = 0

        used in :meth:`step_implicit` and works in 1D, 2D, and 3D via the
        shared face_cells map.

        Parameters
        ----------
        solver : TPFASolver
            TPFA flux solver

        Returns
        -------
        flux : ndarray
            Net mass outflow per cell [kg/s]

        """
        flux_data = solver.compute_flux(self.pressure)
        cell_flux = np.zeros(self.grid.num_cells)

        # face flux is oriented L → R (compute_flux convention). For cell L
        # this is an outflow (+); for cell R it is an inflow (−).
        face_cells = self.grid.face_cells
        for f in range(self.grid.num_faces):
            cL, cR = int(face_cells[f, 0]), int(face_cells[f, 1])
            q = flux_data.flux[f]
            if cL >= 0:
                cell_flux[cL] += q
            if cR >= 0:
                cell_flux[cR] -= q

        return cell_flux

    def step_implicit(
        self,
        dt: float,
        source_terms: np.ndarray,
        bc_type: str,
        bc_values: np.ndarray | None,
        solver,
        max_iter: int = 10,
        tol: float = 1e-8,
    ) -> dict:
        """Implicit time step for single-phase flow.

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
                self.grid.cell_volumes * (accumulation_new - self.prev_accumulation) / dt + cell_flux - source_terms
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
            "converged": residual_norm < tol,
            "iterations": iteration + 1,
            "residual_norm": residual_norm,
        }
