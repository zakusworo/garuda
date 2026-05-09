"""PETSc solver backend for GARUDA.

Wraps PETSc (Portable Extensible Toolkit for Scientific Computation) via
petsc4py to provide distributed-memory linear / non-linear solvers,
algebraic multigrid preconditioners, and ghost-cell exchange for
large-scale reservoir simulation.

Typical usage::

    from garuda import StructuredGrid
    from garuda.solvers import PETScTPFASolver, has_petsc

    if not has_petsc:
        raise RuntimeError("PETSc not available")

    grid = StructuredGrid(nx=100, ny=100, nz=10, dx=10.0, dy=10.0, dz=5.0)
    grid.set_permeability(1e-14)
    grid.set_porosity(0.2)

    solver = PETScTPFASolver(grid, mu=1e-3, rho=1000.0)
    source = np.zeros(grid.num_cells)
    bc_values = np.array([200e5, 100e5])
    pressure = solver.solve(source, bc_type="dirichlet", bc_values=bc_values)

On a multi-core workstation the constructor automatically creates an MPI
communicator; on a cluster launch with::

    mpirun -np 8 python run_sim.py

Notes
-----
- ``petsc4py`` must be installed *after* a system PETSc build that
  includes ``--download-f2blaslapack --download-hypre`` for best AMG
  performance.
- All PETSc objects are freed automatically via context managers, but
  the user may call ``solver.destroy()`` explicitly when done.

References
----------
.. [1] Balay et al., *PETSc Users Manual*, ANL-95/11 (Rev 3.20),
   Argonne National Laboratory, 2024.
.. [2] Doherty et al., *Waiwera*: A parallel geothermal flow simulator
   using PETSc, Computers & Geosciences, 2020.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

try:
    from petsc4py import PETSc

    HAS_PETSC = True
except ImportError:
    HAS_PETSC = False
    PETSc = None  # type: ignore[misc]

from garuda.core.grid import StructuredGrid

# ---------------------------------------------------------------------------
# Public guard
# ---------------------------------------------------------------------------
has_petsc = HAS_PETSC


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------
def _require_petsc() -> None:
    """Raise a helpful error when PETSc is not installed."""
    if not HAS_PETSC:
        msg = (
            "PETSc support is not available.  Install with:\n"
            "  Ubuntu/Debian:  sudo apt-get install libpetsc-real-dev\n"
            "  Then:           pip install petsc4py\n"
            "For AMG (GAMG/hypre) also add --download-hypre when\n"
            "configuring PETSc at compile time."
        )
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Distributed mesh wrapper (DMDA)
# ---------------------------------------------------------------------------
@dataclass
class PETScDMSolver:
    """Distributed mesh manager for structured grids using PETSc DMDA.

    Creates a PETSc ``DMDA`` (Distributed-array manager) that partitions
    a Cartesian grid across MPI ranks and handles ghost-cell exchange.
    This is the foundation for scalable parallel TPFA solves.

    Parameters
    ----------
    grid : StructuredGrid
        The structured Cartesian grid to distribute.
    comm : PETSc.Comm, optional
        MPI communicator (defaults to ``PETSc.COMM_WORLD``).
    stencil_width : int
        Ghost-cell width (``1`` for 5/7-point TPFA stencil).

    Attributes
    ----------
    da : PETSc.DMDA
        The PETSc distributed-array object.
    local_sizes : tuple[int, ...]
        Local subdomain extents on this rank.
    ranges : tuple[range, ...]
        Global index ranges owned by this rank in each direction.

    """

    grid: StructuredGrid
    comm: PETSc.Comm | None = None
    stencil_width: int = 1

    def __post_init__(self) -> None:
        _require_petsc()
        if self.comm is None:
            self.comm = PETSc.COMM_WORLD  # type: ignore[unused-ignore]

        self.rank = self.comm.getRank()
        self.size = self.comm.getSize()

        # Create DMDA matching the grid dimensions
        self.da = PETSc.DMDA().create(  # type: ignore[union-attr]
            dim=self.grid.dim,
            sizes=(self.grid.nx, self.grid.ny, self.grid.nz),
            stencil_width=self.stencil_width,
            comm=self.comm,
            boundary_type=(PETSc.DMDA.BoundaryType.GHOSTED,) * self.grid.dim,
            stencil_type=PETSc.DMDA.StencilType.STAR,
        )

        # Local ownership ranges
        self.ranges = self.da.getRanges()
        self.local_sizes = tuple(r.stop - r.start for r in self.ranges)

        # Pre-allocate ghosted vectors for efficient exchange
        self._vec_local = self.da.createLocalVector()
        self._vec_global = self.da.createGlobalVector()

    # ------------------------------------------------------------------
    # Utility: global <-> local index mapping
    # ------------------------------------------------------------------
    def global_to_local(self, ix: int, iy: int = 0, iz: int = 0) -> int:
        """Map a global (ix,iy,iz) to the local linear index on this rank.

        Returns *None* if the point is not owned by this rank.
        """
        rx, ry, rz = self.ranges
        if not (rx.start <= ix < rx.stop and ry.start <= iy < ry.stop and rz.start <= iz < rz.stop):
            return None  # type: ignore[return-value]

        # Local DMDA ordering is x-natural with ghost padding
        lnx = self.local_sizes[0] + 2 * self.stencil_width
        lny = self.local_sizes[1] + 2 * self.stencil_width if self.grid.dim > 1 else 1
        lx = ix - rx.start + self.stencil_width
        ly = (iy - ry.start + self.stencil_width) if self.grid.dim > 1 else 0
        lz = (iz - rz.start + self.stencil_width) if self.grid.dim > 2 else 0
        return lx + ly * lnx + lz * lnx * lny

    def local_to_global_ijk(self, local_idx: int) -> tuple[int, int, int]:
        """Convert local linear index (with ghost padding) to global (ix,iy,iz)."""
        rx, ry, rz = self.ranges
        lnx = self.local_sizes[0] + 2 * self.stencil_width
        lny = self.local_sizes[1] + 2 * self.stencil_width if self.grid.dim > 1 else 1

        lx = local_idx % lnx
        ly = (local_idx // lnx) % lny if self.grid.dim > 1 else 0
        lz = local_idx // (lnx * lny) if self.grid.dim > 2 else 0

        # Subtract ghost padding to get global
        ix = lx - self.stencil_width + rx.start
        iy = (ly - self.stencil_width + ry.start) if self.grid.dim > 1 else 0
        iz = (lz - self.stencil_width + rz.start) if self.grid.dim > 2 else 0
        return ix, iy, iz

    # ------------------------------------------------------------------
    # Data exchange: scatter global array to local ghosted array
    # ------------------------------------------------------------------
    def global_to_local_array(self, global_arr: np.ndarray) -> np.ndarray:
        """Scatter a global NumPy array to the local ghosted representation.

        Parameters
        ----------
        global_arr : ndarray
            Full grid array of shape ``(grid.num_cells,)``.

        Returns
        -------
        local_arr : ndarray
            Local+ghost array ready for stencil computation.

        """
        _require_petsc()
        # Set global PETSc vector from numpy (dense ordering)
        self._vec_global.setArray(global_arr)
        self._vec_global.assemble()

        # Scatter global -> local with ghost update
        self.da.globalToLocal(self._vec_global, self._vec_local)
        return self._vec_local.getArray().copy()

    def local_to_global_array(self, local_arr: np.ndarray) -> np.ndarray:
        """Gather a local ghosted array back to a global NumPy array.

        Parameters
        ----------
        local_arr : ndarray
            Local+ghost data.

        Returns
        -------
        global_arr : ndarray
            Full grid array reconstructed on all ranks.

        """
        _require_petsc()
        self._vec_local.setArray(local_arr)
        self._vec_local.assemble()

        self.da.localToGlobal(self._vec_local, self._vec_global)
        return self._vec_global.getArray().copy()

    def get_ownership_range(self) -> tuple[int, int]:
        """Return ``(start, end)`` global cell indices owned by this rank."""
        # DMDA ownership is contiguous in natural ordering
        ownership_ranges = self.da.getOwnershipRanges()
        start = ownership_ranges[self.rank]
        end = ownership_ranges[self.rank + 1]
        return start, end

    def get_local_slice(self, arr: np.ndarray) -> np.ndarray:
        """Extract the portion of *arr* owned by this rank."""
        start, end = self.get_ownership_range()
        return arr[start:end]

    def destroy(self) -> None:
        """Free PETSc DMDA and vectors."""
        if hasattr(self, "_vec_local"):
            self._vec_local.destroy()
        if hasattr(self, "_vec_global"):
            self._vec_global.destroy()
        if hasattr(self, "da"):
            self.da.destroy()

    def __del__(self) -> None:
        """Destructor safety — call destroy if not already released."""
        try:
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# PETSc TPFA solver
# ---------------------------------------------------------------------------
class PETScTPFASolver:
    """Two-Point Flux Approximation solver backed by PETSc KSP / DMDA.

    Replaces SciPy ``spsolve`` / ``cg`` with PETSc's linear algebra
    infrastructure, enabling:

    - Distributed memory parallelism (MPI)
    - Scalable iterative solvers (CG, GMRES, BiCGSTAB, etc.)
    - Algebraic multigrid preconditioning (GAMG, hypre-BoomerAMG)
    - Ghost-cell data exchange via DMDA
    - Optional non-linear solves via SNES for multiphase flow

    Parameters
    ----------
    grid : StructuredGrid
        The structured Cartesian grid.
    mu : float
        Fluid viscosity [Pa·s].
    rho : float
        Fluid density [kg/m³].
    g : float
        Gravitational acceleration [m/s²].
    comm : PETSc.Comm, optional
        MPI communicator.  Defaults to ``PETSc.COMM_WORLD``.
    solver_type : str
        KSP solver type.  Common choices:
        ``"cg"`` (default), ``"gmres"``, ``"bcgs"`` (BiCGSTAB).
    pc_type : str
        Preconditioner type.  Recommended:
        ``"gamg"`` (PETSc algebraic multigrid — default),
        ``"hypre"`` (BoomerAMG, requires hypre build),
        ``"ilu"`` (ILU, serial-only fallback),
        ``"lu"`` (direct).
    petsc_options : dict[str, str | float | int], optional
        Additional PETSc option / value pairs passed directly via
        ``PETSc.Options().setValue``.

    Examples
    --------
    >>> grid = StructuredGrid(nx=50, ny=50, nz=5, dx=10, dy=10, dz=5)
    >>> grid.set_permeability(100, unit="md")
    >>> solver = PETScTPFASolver(grid, solver_type="gmres", pc_type="gamg")
    >>> p = solver.solve(np.zeros(grid.num_cells),
    ...                  bc_type="dirichlet",
    ...                  bc_values=np.array([200e5, 100e5]))

    """

    def __init__(
        self,
        grid: StructuredGrid,
        mu: float = 1e-3,
        rho: float = 1000.0,
        g: float = 9.81,
        comm: PETSc.Comm | None = None,
        solver_type: str = "cg",
        pc_type: str = "gamg",
        petsc_options: dict[str, str | float | int] | None = None,
    ):
        _require_petsc()
        if not isinstance(grid, StructuredGrid):
            raise TypeError("PETScTPFASolver only supports StructuredGrid")

        self.grid = grid
        self.mu = mu
        self.rho = rho
        self.g = g
        self.comm = comm or PETSc.COMM_WORLD  # type: ignore[union-attr]
        self.size = self.comm.getSize()

        # -- DMDA for distributed mesh and ghost exchange ------------------
        self.dm = PETScDMSolver(grid, comm=self.comm, stencil_width=1)

        # -- Precompute transmissibilities (local) -------------------------
        self.transmissibilities = self._compute_transmissibilities()

        # -- PETSc linear solver (KSP) ------------------------------------
        self.ksp = PETSc.KSP().create(comm=self.comm)  # type: ignore[union-attr]
        self.ksp.setType(getattr(PETSc.KSP.Type, solver_type.upper()))

        # Preconditioner
        pc = self.ksp.getPC()
        pc.setType(getattr(PETSc.PC.Type, pc_type.upper()))

        # Extra PETSc options
        if petsc_options:
            opts = PETSc.Options()
            for key, val in petsc_options.items():
                opts.setValue(key, val)

        # For GAMG on structured grids, set near-nullspace / coordinates
        if pc_type.lower() == "gamg":
            pc.setGAMGLevels(3)
            pc.setGAMGThreshold(0.05)

        self._petsc_options_applied = bool(petsc_options)

    # ------------------------------------------------------------------
    # Transmissibility computation (local to each rank)
    # ------------------------------------------------------------------
    def _compute_transmissibilities(self) -> np.ndarray:
        """Compute face transmissibilities for the **local** subdomain.

        Only interior faces between two local cells are stored; boundary
        faces carry Dirichlet contributions applied directly to the RHS.

        Returns
        -------
        T : ndarray
            Local transmissibilities indexed by face (order: x, y, z).

        """
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dx = float(self.grid.dx.mean() if hasattr(self.grid.dx, "mean") else self.grid.dx)
        dy = float(self.grid.dy.mean() if hasattr(self.grid.dy, "mean") else self.grid.dy)
        dz = float(self.grid.dz.mean() if hasattr(self.grid.dz, "mean") else self.grid.dz)

        k = getattr(self.grid, "permeability", None)
        if k is None:
            k_iso = 1e-14  # fallback
        else:
            # Use diagonal component
            k_iso = k[:, 0, 0] if k.ndim == 3 else np.full(self.grid.num_cells, float(np.mean(k)))

        # Total faces on global grid
        num_faces_x = (nx + 1) * ny * nz
        num_faces_y = nx * (ny + 1) * nz
        num_faces_z = nx * ny * (nz + 1)
        T = np.zeros(num_faces_x + num_faces_y + num_faces_z)

        # X-faces
        face_idx = 0
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx + 1):
                    A = dy * dz
                    if ix == 0:
                        c = iy * nx + iz * nx * ny
                        T[face_idx] = k_iso[c] * A / (dx / 2) / self.mu
                    elif ix == nx:
                        c = (nx - 1) + iy * nx + iz * nx * ny
                        T[face_idx] = k_iso[c] * A / (dx / 2) / self.mu
                    else:
                        cL = (ix - 1) + iy * nx + iz * nx * ny
                        cR = ix + iy * nx + iz * nx * ny
                        T[face_idx] = 2 * A / (dx / k_iso[cL] + dx / k_iso[cR]) / self.mu
                    face_idx += 1

        # Y-faces
        for iz in range(nz):
            for iy in range(ny + 1):
                for ix in range(nx):
                    A = dx * dz
                    if iy == 0:
                        c = ix + iz * nx * ny
                        T[face_idx] = k_iso[c] * A / (dy / 2) / self.mu
                    elif iy == ny:
                        c = ix + (ny - 1) * nx + iz * nx * ny
                        T[face_idx] = k_iso[c] * A / (dy / 2) / self.mu
                    else:
                        cL = ix + (iy - 1) * nx + iz * nx * ny
                        cR = ix + iy * nx + iz * nx * ny
                        T[face_idx] = 2 * A / (dy / k_iso[cL] + dy / k_iso[cR]) / self.mu
                    face_idx += 1

        # Z-faces
        for iz in range(nz + 1):
            for iy in range(ny):
                for ix in range(nx):
                    A = dx * dy
                    if iz == 0:
                        c = ix + iy * nx
                        T[face_idx] = k_iso[c] * A / (dz / 2) / self.mu
                    elif iz == nz:
                        c = ix + iy * nx + (nz - 1) * nx * ny
                        T[face_idx] = k_iso[c] * A / (dz / 2) / self.mu
                    else:
                        cL = ix + iy * nx + (iz - 1) * nx * ny
                        cR = ix + iy * nx + iz * nx * ny
                        T[face_idx] = 2 * A / (dz / k_iso[cL] + dz / k_iso[cR]) / self.mu
                    face_idx += 1

        return T

    # ------------------------------------------------------------------
    # Matrix assembly
    # ------------------------------------------------------------------
    def _build_local_matrix(
        self,
        source_terms: np.ndarray,
        bc_type: str,
        bc_values: np.ndarray | None,
    ) -> tuple[PETSc.Mat, PETSc.Vec]:
        """Assemble the local system ``A * p = b`` in PETSc format.

        For Dirichlet BCs, boundary-face transmissibilities are added
        onto the diagonal and the boundary pressure is moved to the RHS.

        """
        _require_petsc()
        start, end = self.dm.get_ownership_range()

        # Create sparse matrix in AIJ format
        A_mat = PETSc.Mat().createAIJ(  # type: ignore[union-attr]
            size=(self.grid.num_cells, self.grid.num_cells),
            comm=self.comm,
        )
        A_mat.setUp()
        A_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        b_vec = PETSc.Vec().createWithArray(source_terms, comm=self.comm)  # type: ignore[union-attr]
        b_vec.assemble()

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        T = self.transmissibilities
        num_faces_x = (nx + 1) * ny * nz
        num_faces_y = nx * (ny + 1) * nz

        for c in range(start, end):
            # Recover (ix, iy, iz)
            iz = c // (nx * ny)
            rem = c % (nx * ny)
            iy = rem // nx
            ix = rem % nx

            diag = 0.0

            # West face (ix-1)
            if ix > 0:
                f = iz * (nx + 1) * ny + iy * (nx + 1) + ix
                diag += T[f]
                A_mat[c, c - 1] = -T[f]
            else:
                # Boundary face
                f = iz * (nx + 1) * ny + iy * (nx + 1) + ix
                diag += T[f]
                if bc_type == "dirichlet" and bc_values is not None:
                    b_vec[c] += T[f] * bc_values[0]

            # East face (ix+1)
            if ix < nx - 1:
                f = iz * (nx + 1) * ny + iy * (nx + 1) + ix + 1
                diag += T[f]
                A_mat[c, c + 1] = -T[f]
            else:
                f = iz * (nx + 1) * ny + iy * (nx + 1) + ix + 1
                diag += T[f]
                if bc_type == "dirichlet" and bc_values is not None:
                    b_vec[c] += T[f] * bc_values[1]

            # South face (iy-1)
            if iy > 0:
                f = num_faces_x + iz * nx * (ny + 1) + ix * (ny + 1) + iy
                diag += T[f]
                A_mat[c, c - nx] = -T[f]
            else:
                f = num_faces_x + iz * nx * (ny + 1) + ix * (ny + 1) + iy
                diag += T[f]
                if bc_type == "dirichlet" and bc_values is not None and len(bc_values) > 2:
                    b_vec[c] += T[f] * bc_values[2]

            # North face (iy+1)
            if iy < ny - 1:
                f = num_faces_x + iz * nx * (ny + 1) + ix * (ny + 1) + iy + 1
                diag += T[f]
                A_mat[c, c + nx] = -T[f]
            else:
                f = num_faces_x + iz * nx * (ny + 1) + ix * (ny + 1) + iy + 1
                diag += T[f]
                if bc_type == "dirichlet" and bc_values is not None and len(bc_values) > 3:
                    b_vec[c] += T[f] * bc_values[3]

            # Z-face indexing: bottom face of cell (ix,iy,iz) sits at
            # z_edges[iz], top face at z_edges[iz+1]. Grid._generate_faces
            # lays them out as (k, j, i) with i-fastest, so:
            #   bottom = num_faces_x + num_faces_y + iz       * nx*ny + iy*nx + ix
            #   top    = num_faces_x + num_faces_y + (iz + 1) * nx*ny + iy*nx + ix
            z_c = self.grid.cell_centroids[c, 2]

            # Bottom face (z_edges[iz]) — neighbour at iz - 1
            f = num_faces_x + num_faces_y + iz * nx * ny + iy * nx + ix
            diag += T[f]
            if iz > 0:
                A_mat[c, c - nx * ny] = -T[f]
                z_neighbour = self.grid.cell_centroids[c - nx * ny, 2]
                # Per-face gravity contribution: T_f * rho * g * (z_c - z_R)
                b_vec[c] += T[f] * self.rho * self.g * (z_c - z_neighbour)
            else:
                if bc_type == "dirichlet" and bc_values is not None and len(bc_values) > 4:
                    b_vec[c] += T[f] * bc_values[4]

            # Top face (z_edges[iz+1]) — neighbour at iz + 1
            f = num_faces_x + num_faces_y + (iz + 1) * nx * ny + iy * nx + ix
            diag += T[f]
            if iz < nz - 1:
                A_mat[c, c + nx * ny] = -T[f]
                z_neighbour = self.grid.cell_centroids[c + nx * ny, 2]
                b_vec[c] += T[f] * self.rho * self.g * (z_c - z_neighbour)
            else:
                if bc_type == "dirichlet" and bc_values is not None and len(bc_values) > 5:
                    b_vec[c] += T[f] * bc_values[5]

            A_mat[c, c] = diag

        A_mat.assemble()
        b_vec.assemble()
        return A_mat, b_vec

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------
    def solve(
        self,
        source_terms: np.ndarray,
        bc_type: str = "dirichlet",
        bc_values: np.ndarray | None = None,
        solver_type: str | None = None,
        pc_type: str | None = None,
        tol: float = 1e-10,
        max_iter: int = 10000,
    ) -> np.ndarray:
        """Solve ``A * p = b`` using PETSc KSP.

        Parameters
        ----------
        source_terms : ndarray
            Source/sink terms [kg/s] (positive = injection).
        bc_type : str
            ``"dirichlet"`` or ``"neumann"``.
        bc_values : ndarray, optional
            Boundary values.  For Dirichlet: ``[p_left, p_right, ...]``.
        solver_type : str, optional
            Override KSP type (e.g. ``"gmres"``).
        pc_type : str, optional
            Override preconditioner (e.g. ``"gamg"``).
        tol : float
            Relative convergence tolerance.
        max_iter : int
            Maximum Krylov iterations.

        Returns
        -------
        pressure : ndarray
            Global cell pressure [Pa] on **all** ranks.

        """
        _require_petsc()

        # Optionally override solver / PC
        if solver_type is not None:
            self.ksp.setType(getattr(PETSc.KSP.Type, solver_type.upper()))
        if pc_type is not None:
            pc = self.ksp.getPC()
            pc.setType(getattr(PETSc.PC.Type, pc_type.upper()))

        self.ksp.setTolerances(rtol=tol, max_it=max_iter)

        # Build system
        A_mat, b_vec = self._build_local_matrix(source_terms, bc_type, bc_values)
        self.ksp.setOperators(A_mat)

        # Initial guess (zero or previous solution)
        x_vec = A_mat.createVecRight()

        # Solve
        self.ksp.solve(b_vec, x_vec)
        its = self.ksp.getIterationNumber()
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            warnings.warn(f"PETSc KSP did not converge (reason={reason}, iterations={its})")

        pressure = x_vec.getArray().copy()

        # Cleanup PETSc objects
        A_mat.destroy()
        b_vec.destroy()
        x_vec.destroy()

        return pressure

    # ------------------------------------------------------------------
    # Non-linear solver (SNES) — placeholder for future multiphase coupling
    # ------------------------------------------------------------------
    def solve_nonlinear(
        self,
        residual_func,  # Callable[[np.ndarray], np.ndarray]
        jacobian_func,  # Callable[[np.ndarray], np.ndarray]
        x0: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 50,
    ) -> np.ndarray:
        """Solve a non-linear system ``F(x) = 0`` using PETSc SNES.

        .. warning::
           This is a low-level interface.  For production multiphase
           simulations a higher-level wrapper is recommended.

        Parameters
        ----------
        residual_func : callable
            ``F(x) -> ndarray``
        jacobian_func : callable
            ``J(x) -> PETSc.Mat``
        x0 : ndarray
            Initial guess.
        tol : float
            Non-linear tolerance.
        max_iter : int
            Maximum Newton iterations.

        Returns
        -------
        x : ndarray
            Solution vector.

        """
        _require_petsc()

        # The current callback signature uses ``J_mat.setArray(J_arr)`` which
        # is only valid for PETSc Vec objects; the matrix is created as AIJ
        # (sparse) below, so the assignment fails at runtime. Until a proper
        # sparse-fill path (with explicit row/col MatSetValues + assembly) is
        # implemented this method is non-functional, and it is honest to fail
        # eagerly rather than allow callers to discover that mid-run.
        raise NotImplementedError(
            "PETScDMSolver.solve_nonlinear is not yet wired up: the Jacobian "
            "callback uses Mat.setArray() which is invalid for AIJ matrices. "
            "Use solve() for the linear case, or open an issue if you need "
            "the SNES path."
        )

        snes = PETSc.SNES().create(comm=self.comm)  # type: ignore[union-attr]
        snes.setType(PETSc.SNES.Type.NEWTONLS)
        snes.setTolerances(rtol=tol, max_it=max_iter)

        # Create PETSc vectors
        r = PETSc.Vec().createWithArray(np.zeros_like(x0), comm=self.comm)  # type: ignore[union-attr]
        x = PETSc.Vec().createWithArray(x0.copy(), comm=self.comm)  # type: ignore[union-attr]

        # Form function
        def _form_function(snes_instance, x_vec, r_vec):  # noqa: ARG001
            x_arr = x_vec.getArray()
            r_arr = residual_func(x_arr)
            r_vec.setArray(r_arr)

        # Form Jacobian
        def _form_jacobian(snes_instance, x_vec, J_mat, P_mat):  # noqa: ARG001
            x_arr = x_vec.getArray()
            J_arr = jacobian_func(x_arr)
            J_mat.setArray(J_arr)
            J_mat.assemble()
            P_mat.assemble()

        snes.setFunction(_form_function, r)
        # Dummy matrix for Jacobian callback
        J_dummy = PETSc.Mat().createAIJ(  # type: ignore[union-attr]
            size=(len(x0), len(x0)), comm=self.comm
        )
        J_dummy.setUp()
        snes.setJacobian(_form_jacobian, J_dummy, J_dummy)

        snes.solve(None, x)
        sol = x.getArray().copy()

        snes.destroy()
        r.destroy()
        x.destroy()
        J_dummy.destroy()

        return sol

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_solver_info(self) -> dict[str, str | int | float]:
        """Return dictionary of KSP configuration and last-solve stats."""
        _require_petsc()
        pc = self.ksp.getPC()
        return {
            "ksp_type": str(self.ksp.getType()),
            "pc_type": str(pc.getType()),
            "mpi_size": self.size,
            "local_cells": self.dm.local_sizes,
        }

    def destroy(self) -> None:
        """Free all PETSc objects (KSP, DM, vectors)."""
        if hasattr(self, "ksp"):
            self.ksp.destroy()
        if hasattr(self, "dm"):
            self.dm.destroy()

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
