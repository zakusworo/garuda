"""
TPFA Solver - Two-Point Flux Approximation for porous media flow.

Implements finite volume discretization with TPFA for structured and
unstructured grids. Supports single-phase and multiphase flow.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, cg
from numba import jit, prange
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import warnings

from garuda.core.grid import Grid


@dataclass
class TPFAFlux:
    """Container for TPFA flux computation results."""
    
    # Transmissibilities (faces,)
    transmissibilities: np.ndarray
    
    # Pressure gradient (faces,)
    pressure_gradient: np.ndarray
    
    # Flux across each face (faces,)
    flux: np.ndarray
    
    # Upstream cell for each face (for multiphase)
    upstream_cell: np.ndarray


class TPFASolver:
    """
    Two-Point Flux Approximation solver for porous media flow.
    
    The TPFA method computes fluxes across cell faces using:
    
        q = -T * (p_right - p_left + rho * g * (z_right - z_left))
    
    where T is the transmissibility combining geometry and permeability.
    
    Parameters
    ----------
    grid : Grid
        Computational grid
    mu : float or array-like
        Fluid viscosity [Pa·s]
    rho : float or array-like
        Fluid density [kg/m³]
    
    Examples
    --------
    >>> from garuda import StructuredGrid, TPFASolver
    >>> grid = StructuredGrid(nx=10, ny=10, nz=1, dx=100, dy=100, dz=10)
    >>> solver = TPFASolver(grid, mu=1e-3, rho=1000)
    >>> pressure = solver.solve(source_terms, bc)
    """
    
    def __init__(
        self,
        grid: Grid,
        mu: float = 1e-3,  # Water viscosity at 20°C [Pa·s]
        rho: float = 1000,  # Water density [kg/m³]
        g: float = 9.81,  # Gravity [m/s²]
    ):
        self.grid = grid
        self.mu = mu
        self.rho = rho
        self.g = g
        
        # Precompute transmissibilities
        self.transmissibilities = self._compute_transmissibilities()
    
    def _compute_transmissibilities(self) -> np.ndarray:
        """
        Compute face transmissibilities.
        
        For a face between cells L and R:
        
            T = 2 * A / (dL/kL + dR/kR) / mu
        
        where:
            A = face area
            dL, dR = distances from cell centroids to face
            kL, kR = permeabilities
            mu = viscosity
        
        Returns
        -------
        transmissibilities : ndarray
            Face transmissibilities [m³·s/kg]
        """
        num_faces = self.grid.num_faces
        T = np.zeros(num_faces)
        
        # Get permeability (assume isotropic for now)
        if hasattr(self.grid, 'permeability'):
            perm = self.grid.permeability
            k = np.array([perm[i, 0, 0] for i in range(self.grid.num_cells)])
        elif hasattr(self.grid, 'permiability'):
            perm = self.grid.permiability
            k = np.array([perm[i, 0, 0] for i in range(self.grid.num_cells)])
        else:
            k = np.ones(self.grid.num_cells)
        
        # Compute transmissibilities for each face direction
        if self.grid.dim == 1:
            T = self._compute_1d_transmissibilities(k)
        elif self.grid.dim == 2:
            T = self._compute_2d_transmissibilities(k)
        else:
            T = self._compute_3d_transmissibilities(k)
        
        return T
    
    def _compute_1d_transmissibilities(self, k: np.ndarray) -> np.ndarray:
        """Compute 1D transmissibilities."""
        nx = self.grid.nx
        dx = self.grid.dx if np.isscalar(self.grid.dx) else np.mean(self.grid.dx)
        A = float(np.mean(self.grid.dy) * np.mean(self.grid.dz)) if hasattr(self.grid, 'dy') else 1.0
        
        T = np.zeros(nx + 1)
        
        # Interior faces
        for i in range(1, nx):
            dL = dx / 2
            dR = dx / 2
            kL = k[i - 1]
            kR = k[i]
            T[i] = 2 * A / (dL / kL + dR / kR) / self.mu
        
        # Boundary faces (half-cell)
        T[0] = k[0] * A / (dx / 2) / self.mu
        T[-1] = k[-1] * A / (dx / 2) / self.mu
        
        return T
    
    def _compute_2d_transmissibilities(self, k: np.ndarray) -> np.ndarray:
        """Compute 2D transmissibilities."""
        nx, ny = self.grid.nx, self.grid.ny
        dx = self.grid.dx if np.isscalar(self.grid.dx) else np.mean(self.grid.dx)
        dy = self.grid.dy if np.isscalar(self.grid.dy) else np.mean(self.grid.dy)
        
        num_faces_x = (nx + 1) * ny
        num_faces_y = nx * (ny + 1)
        T = np.zeros(num_faces_x + num_faces_y)
        
        # X-faces (normal in x-direction)
        for j in range(ny):
            for i in range(nx + 1):
                face_idx = j * (nx + 1) + i
                A = dy * 1.0  # Assuming dz=1 for 2D
                
                if i == 0:  # Left boundary
                    dL = dx / 2
                    kL = k[self.grid.get_cell_index(0, j)]
                    T[face_idx] = kL * A / dL / self.mu
                elif i == nx:  # Right boundary
                    dR = dx / 2
                    kR = k[self.grid.get_cell_index(nx - 1, j)]
                    T[face_idx] = kR * A / dR / self.mu
                else:  # Interior
                    cell_L = self.grid.get_cell_index(i - 1, j)
                    cell_R = self.grid.get_cell_index(i, j)
                    dL = dx / 2
                    dR = dx / 2
                    kL = k[cell_L]
                    kR = k[cell_R]
                    T[face_idx] = 2 * A / (dL / kL + dR / kR) / self.mu
        
        # Y-faces (normal in y-direction)
        for j in range(ny + 1):
            for i in range(nx):
                face_idx = num_faces_x + i * (ny + 1) + j
                A = dx * 1.0
                
                if j == 0:  # Bottom boundary
                    dL = dy / 2
                    kL = k[self.grid.get_cell_index(i, 0)]
                    T[face_idx] = kL * A / dL / self.mu
                elif j == ny:  # Top boundary
                    dR = dy / 2
                    kR = k[self.grid.get_cell_index(i, ny - 1)]
                    T[face_idx] = kR * A / dR / self.mu
                else:  # Interior
                    cell_L = self.grid.get_cell_index(i, j - 1)
                    cell_R = self.grid.get_cell_index(i, j)
                    dL = dy / 2
                    dR = dy / 2
                    kL = k[cell_L]
                    kR = k[cell_R]
                    T[face_idx] = 2 * A / (dL / kL + dR / kR) / self.mu
        
        return T
    
    def _compute_3d_transmissibilities(self, k: np.ndarray) -> np.ndarray:
        """Compute 3D transmissibilities."""
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dx = self.grid.dx if np.isscalar(self.grid.dx) else np.mean(self.grid.dx)
        dy = self.grid.dy if np.isscalar(self.grid.dy) else np.mean(self.grid.dy)
        dz = self.grid.dz if np.isscalar(self.grid.dz) else np.mean(self.grid.dz)
        
        num_faces_x = (nx + 1) * ny * nz
        num_faces_y = nx * (ny + 1) * nz
        num_faces_z = nx * ny * (nz + 1)
        T = np.zeros(num_faces_x + num_faces_y + num_faces_z)
        
        # Use vectorized computation for speed
        T = self._compute_3d_transmissibilities_numba(
            k, nx, ny, nz, dx, dy, dz,
            num_faces_x, num_faces_y, num_faces_z, self.mu
        )
        
        return T
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _compute_3d_transmissibilities_numba(
        k: np.ndarray,
        nx: int, ny: int, nz: int,
        dx: float, dy: float, dz: float,
        num_faces_x: int, num_faces_y: int, num_faces_z: int,
        mu: float,
    ) -> np.ndarray:
        """Numba-accelerated 3D transmissibility computation."""
        T = np.zeros(num_faces_x + num_faces_y + num_faces_z)
        
        # X-faces
        for iz in prange(nz):
            for iy in range(ny):
                for ix in range(nx + 1):
                    face_idx = iz * (nx + 1) * ny + iy * (nx + 1) + ix
                    A = dy * dz
                    
                    if ix == 0:
                        cell_idx = iy * nx + iz * nx * ny
                        T[face_idx] = k[cell_idx] * A / (dx / 2) / mu
                    elif ix == nx:
                        cell_idx = (nx - 1) + iy * nx + iz * nx * ny
                        T[face_idx] = k[cell_idx] * A / (dx / 2) / mu
                    else:
                        cell_L = (ix - 1) + iy * nx + iz * nx * ny
                        cell_R = ix + iy * nx + iz * nx * ny
                        dL = dx / 2
                        dR = dx / 2
                        T[face_idx] = 2 * A / (dL / k[cell_L] + dR / k[cell_R]) / mu
        
        # Y-faces
        for iz in prange(nz):
            for iy in range(ny + 1):
                for ix in range(nx):
                    face_idx = num_faces_x + iz * nx * (ny + 1) + ix * (ny + 1) + iy
                    A = dx * dz
                    
                    if iy == 0:
                        cell_idx = ix + iz * nx * ny
                        T[face_idx] = k[cell_idx] * A / (dy / 2) / mu
                    elif iy == ny:
                        cell_idx = ix + (ny - 1) * nx + iz * nx * ny
                        T[face_idx] = k[cell_idx] * A / (dy / 2) / mu
                    else:
                        cell_L = ix + (iy - 1) * nx + iz * nx * ny
                        cell_R = ix + iy * nx + iz * nx * ny
                        dL = dy / 2
                        dR = dy / 2
                        T[face_idx] = 2 * A / (dL / k[cell_L] + dR / k[cell_R]) / mu
        
        # Z-faces
        for iz in range(nz + 1):
            for iy in prange(ny):
                for ix in range(nx):
                    face_idx = num_faces_x + num_faces_y + iz * nx * ny + iy * nx + ix
                    A = dx * dy
                    
                    if iz == 0:
                        cell_idx = ix + iy * nx
                        T[face_idx] = k[cell_idx] * A / (dz / 2) / mu
                    elif iz == nz:
                        cell_idx = ix + iy * nx + (nz - 1) * nx * ny
                        T[face_idx] = k[cell_idx] * A / (dz / 2) / mu
                    else:
                        cell_L = ix + iy * nx + (iz - 1) * nx * ny
                        cell_R = ix + iy * nx + iz * nx * ny
                        dL = dz / 2
                        dR = dz / 2
                        T[face_idx] = 2 * A / (dL / k[cell_L] + dR / k[cell_R]) / mu
        
        return T
    
    def compute_flux(
        self,
        pressure: np.ndarray,
        permeability: Optional[np.ndarray] = None,
    ) -> TPFAFlux:
        """
        Compute fluxes given pressure field.
        
        Parameters
        ----------
        pressure : ndarray
            Cell pressures [Pa]
        permeability : ndarray, optional
            Cell permeabilities (overrides grid permeability)
        
        Returns
        -------
        flux_data : TPFAFlux
            Container with transmissibilities, gradients, and fluxes
        """
        num_faces = self.grid.num_faces
        num_cells = self.grid.num_cells
        
        # Pressure at faces (from connected cells)
        p_face = np.zeros(num_faces)
        
        # Gravity term (elevation difference)
        dz = np.zeros(num_faces)
        
        # Compute pressure differences using face_cells connectivity
        for f in range(num_faces):
            cell_L, cell_R = self.grid.face_cells[f]
            
            if cell_L >= 0 and cell_R >= 0:
                # Interior face: average of two cells
                p_face[f] = (pressure[cell_L] + pressure[cell_R]) / 2
                # Elevation difference (using z-coordinate of centroids)
                dz[f] = self.grid.face_centroids[f, 2]  # z-coordinate of face
                
            elif cell_L >= 0:
                # Boundary face with only left cell
                p_face[f] = pressure[cell_L]
                dz[f] = self.grid.face_centroids[f, 2]
                
            elif cell_R >= 0:
                # Boundary face with only right cell
                p_face[f] = pressure[cell_R]
                dz[f] = self.grid.face_centroids[f, 2]
        
        # Compute pressure gradient across each face
        dp = np.zeros(num_faces)
        for f in range(num_faces):
            cell_L, cell_R = self.grid.face_cells[f]
            
            if cell_L >= 0 and cell_R >= 0:
                # Interior: p_R - p_L
                dp[f] = pressure[cell_R] - pressure[cell_L]
            elif cell_L >= 0:
                # Boundary: extrapolate from cell_L
                dp[f] = 0  # Will be handled by BC
            elif cell_R >= 0:
                # Boundary: extrapolate from cell_R
                dp[f] = 0
        
        # Flux = -T * (dp + rho * g * dz)
        # Note: dz here is the elevation change in the gravity term
        gravity_term = self.rho * self.g * dz
        flux = -self.transmissibilities * (dp + gravity_term)
        
        # Upstream direction (for multiphase): cell with higher pressure
        upstream = np.zeros(num_faces, dtype=int)
        for f in range(num_faces):
            cell_L, cell_R = self.grid.face_cells[f]
            if cell_L >= 0 and cell_R >= 0:
                upstream[f] = cell_L if pressure[cell_L] > pressure[cell_R] else cell_R
            elif cell_L >= 0:
                upstream[f] = cell_L
            elif cell_R >= 0:
                upstream[f] = cell_R
        
        return TPFAFlux(
            transmissibilities=self.transmissibilities.copy(),
            pressure_gradient=dp,
            flux=flux,
            upstream_cell=upstream,
        )
    
    def build_matrix(
        self,
        source_terms: np.ndarray,
        bc_type: str = 'dirichlet',
        bc_values: Optional[np.ndarray] = None,
    ) -> Tuple[csr_matrix, np.ndarray]:
        """
        Build the linear system A * p = b.
        
        Parameters
        ----------
        source_terms : ndarray
            Source/sink terms [kg/s] (positive = injection)
        bc_type : str
            Boundary condition type: 'dirichlet', 'neumann', or 'mixed'
        bc_values : ndarray, optional
            Boundary condition values
        
        Returns
        -------
        A : csr_matrix
            System matrix
        b : ndarray
            Right-hand side vector
        """
        num_cells = self.grid.num_cells
        
        # Build sparse matrix
        A = lil_matrix((num_cells, num_cells))
        b = source_terms.copy()
        
        # Assemble TPFA discretization using face connectivity
        # For each cell: sum of fluxes = source
        # Flux across face f: q_f = -T_f * (p_neighbor - p_cell + rho*g*dz)
        
        # Process each face and accumulate contributions to cells
        for f in range(self.grid.num_faces):
            cell_L, cell_R = self.grid.face_cells[f]
            T_f = self.transmissibilities[f]
            
            # Get face normal direction to determine which coordinate for gravity
            normal = self.grid.face_normals[f]
            
            if cell_L >= 0 and cell_R >= 0:
                # Interior face: connects two cells
                # Flux from L to R: q = T * (p_L - p_R + rho*g*(z_L - z_R))
                z_L = self.grid.cell_centroids[cell_L, 2]
                z_R = self.grid.cell_centroids[cell_R, 2]
                dz = z_L - z_R
                
                # Contribution to cell_L (outflow is negative)
                A[cell_L, cell_L] += T_f
                A[cell_L, cell_R] -= T_f
                b[cell_L] += T_f * self.rho * self.g * dz
                
                # Contribution to cell_R (inflow is positive)
                A[cell_R, cell_R] += T_f
                A[cell_R, cell_L] -= T_f
                b[cell_R] -= T_f * self.rho * self.g * dz
                
            elif cell_L >= 0:
                # Boundary face with cell on left
                if bc_type == 'dirichlet' and bc_values is not None:
                    # Dirichlet BC: use boundary pressure
                    # For boundary faces, we need to determine which BC value to use
                    # Simplified: use first value for all left/bottom boundaries, second for right/top
                    if np.any(normal < 0):  # Left/bottom/bottom boundary
                        p_bc = bc_values[0] if len(bc_values) > 0 else 0
                    else:  # Right/top/top boundary
                        p_bc = bc_values[1] if len(bc_values) > 1 else bc_values[0]
                    
                    A[cell_L, cell_L] += T_f
                    b[cell_L] += T_f * p_bc
                    
            elif cell_R >= 0:
                # Boundary face with cell on right
                if bc_type == 'dirichlet' and bc_values is not None:
                    if np.any(normal < 0):
                        p_bc = bc_values[0] if len(bc_values) > 0 else 0
                    else:
                        p_bc = bc_values[1] if len(bc_values) > 1 else bc_values[0]
                    
                    A[cell_R, cell_R] += T_f
                    b[cell_R] += T_f * p_bc
        
        return A.tocsr(), b
    
    def _build_1d_matrix(
        self,
        A: lil_matrix,
        b: np.ndarray,
        bc_type: str,
        bc_values: Optional[np.ndarray],
    ) -> Tuple[lil_matrix, np.ndarray]:
        """Build 1D system matrix."""
        nx = self.grid.nx
        T = self.transmissibilities
        
        # Interior cells
        for i in range(nx):
            # Left face
            if i == 0:
                if bc_type == 'dirichlet' and bc_values is not None:
                    # Dirichlet BC: p = bc_values[0]
                    A[i, i] += T[0]
                    b[i] += T[0] * bc_values[0]
                else:
                    # Neumann BC: flux specified
                    A[i, i] += 0  # Will be handled by b
            else:
                A[i, i] += T[i]  # Left face transmissibility
                A[i, i - 1] -= T[i]
            
            # Right face
            if i == nx - 1:
                if bc_type == 'dirichlet' and bc_values is not None:
                    A[i, i] += T[-1]
                    b[i] += T[-1] * bc_values[1]
            else:
                A[i, i] += T[i + 1]
                A[i, i + 1] -= T[i + 1]
        
        return A, b
    
    def solve(
        self,
        source_terms: np.ndarray,
        bc_type: str = 'dirichlet',
        bc_values: Optional[np.ndarray] = None,
        solver: str = 'direct',
        tol: float = 1e-10,
        max_iter: int = 1000,
    ) -> np.ndarray:
        """
        Solve the pressure equation.
        
        Parameters
        ----------
        source_terms : ndarray
            Source/sink terms [kg/s]
        bc_type : str
            Boundary condition type
        bc_values : ndarray, optional
            Boundary values (Dirichlet pressures or Neumann fluxes)
        solver : str
            Linear solver: 'direct' (spsolve) or 'iterative' (CG)
        tol : float
            Convergence tolerance (iterative solver)
        max_iter : int
            Maximum iterations (iterative solver)
        
        Returns
        -------
        pressure : ndarray
            Cell pressures [Pa]
        """
        A, b = self.build_matrix(source_terms, bc_type, bc_values)
        
        if solver == 'direct':
            pressure = spsolve(A, b)
        elif solver == 'iterative':
            pressure, info = cg(A, b, atol=tol, maxiter=max_iter)
            if info != 0:
                warnings.warn(f"CG solver did not converge (info={info})")
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        return pressure
    
    def compute_residual(
        self,
        pressure: np.ndarray,
        source_terms: np.ndarray,
        bc_type: str = 'dirichlet',
        bc_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute mass balance residual using system matrix A·p - b.

        Parameters
        ----------
        pressure : ndarray
            Current pressure field
        source_terms : ndarray
            Source/sink terms
        bc_type : str, optional
            Boundary condition type (needed for boundary flux accounting)
        bc_values : ndarray, optional
            Boundary condition values

        Returns
        -------
        residual : ndarray
            Mass balance residual for each cell [kg/s]
        """
        A, b = self.build_matrix(source_terms, bc_type, bc_values)
        residual = A @ pressure - b
        return residual
