"""Tests for the optional PETSc solver backend.

These tests are automatically skipped when ``petsc4py`` is not installed.
To run them locally::

    pip install petsc4py
    pytest tests/test_petsc_solver.py -v

For distributed-memory tests (multi-process), use::

    mpirun -np 2 pytest tests/test_petsc_solver.py -v

"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module if PETSc is unavailable
# ---------------------------------------------------------------------------
petsc = pytest.importorskip("petsc4py.PETSc")
from garuda.core.grid import StructuredGrid  # noqa: E402
from garuda.solvers.petsc_solver import PETScDMSolver, PETScTPFASolver, has_petsc  # noqa: E402


class TestAvailability:
    """Sanity checks that PETSc was imported correctly."""

    def test_has_petsc_is_true(self):
        assert has_petsc is True

    def test_petsc_comm_world(self):
        comm = petsc.COMM_WORLD  # noqa: B009
        assert comm.getSize() >= 1
        assert comm.getRank() >= 0


class TestPETScDMSolver:
    """Unit tests for the distributed mesh manager (DMDA)."""

    def test_dm_creation_1d(self):
        grid = StructuredGrid(nx=10, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
        dm = PETScDMSolver(grid)
        assert dm.grid is grid
        assert dm.rank == petsc.COMM_WORLD.getRank()
        dm.destroy()

    def test_dm_creation_3d(self):
        grid = StructuredGrid(nx=8, ny=8, nz=4, dx=10.0, dy=10.0, dz=5.0)
        dm = PETScDMSolver(grid)
        assert dm.grid is grid
        assert dm.local_sizes is not None
        dm.destroy()

    def test_ownership_range(self):
        grid = StructuredGrid(nx=20, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
        dm = PETScDMSolver(grid)
        start, end = dm.get_ownership_range()
        assert 0 <= start < end <= grid.num_cells
        dm.destroy()

    def test_global_to_local_roundtrip(self):
        grid = StructuredGrid(nx=10, ny=10, nz=1, dx=1.0, dy=1.0, dz=1.0)
        dm = PETScDMSolver(grid)

        global_arr = np.arange(grid.num_cells, dtype=float)
        local_arr = dm.global_to_local_array(global_arr)
        # local_arr contains owned cells + ghost padding
        size = dm.local_sizes[0] + 2 * dm.stencil_width
        size *= dm.local_sizes[1] + 2 * dm.stencil_width if grid.dim > 1 else 1
        size *= dm.local_sizes[2] + 2 * dm.stencil_width if grid.dim > 2 else 1
        assert len(local_arr) == size
        dm.destroy()


class TestPETScTPFASolverBasic:
    """Basic functional tests for the PETSc-backed TPFA solver."""

    @pytest.fixture
    def grid_1d(self):
        grid = StructuredGrid(nx=10, ny=1, nz=1, dx=100.0, dy=10.0, dz=10.0)
        grid.set_permeability(1e-14)
        grid.set_porosity(0.2)
        return grid

    @pytest.fixture
    def grid_2d(self):
        grid = StructuredGrid(nx=10, ny=10, nz=1, dx=10.0, dy=10.0, dz=10.0)
        grid.set_permeability(1e-14)
        grid.set_porosity(0.2)
        return grid

    @pytest.fixture
    def grid_3d(self):
        grid = StructuredGrid(nx=8, ny=8, nz=4, dx=10.0, dy=10.0, dz=5.0)
        grid.set_permeability(1e-14)
        grid.set_porosity(0.2)
        return grid

    # -----------------------------------------------------------------
    # 1D tests
    # -----------------------------------------------------------------
    def test_1d_constructor(self, grid_1d):
        solver = PETScTPFASolver(grid_1d, mu=1e-3, rho=1000.0)
        assert solver.grid is grid_1d
        assert solver.transmissibilities is not None
        assert len(solver.transmissibilities) == grid_1d.num_faces
        solver.destroy()

    def test_1d_dirichlet_solve_linear_pressure_drop(self, grid_1d):
        """A 1D domain with Dirichlet BCs should give a linear pressure profile."""
        solver = PETScTPFASolver(grid_1d, mu=1e-3, rho=1000.0)
        source = np.zeros(grid_1d.num_cells)
        p_left, p_right = 200e5, 100e5
        pressure = solver.solve(source, bc_type="dirichlet", bc_values=np.array([p_left, p_right]))

        assert len(pressure) == grid_1d.num_cells
        assert np.isfinite(pressure).all()
        assert pressure.min() >= p_right - 1e-3
        assert pressure.max() <= p_left + 1e-3
        # Monotonic decrease
        assert np.all(np.diff(pressure) <= 0)
        solver.destroy()

    def test_1d_zero_source_symmetry(self, grid_1d):
        """Zero source with equal Dirichlet pressures → flat profile."""
        solver = PETScTPFASolver(grid_1d, mu=1e-3, rho=1000.0)
        source = np.zeros(grid_1d.num_cells)
        p_target = 150e5
        pressure = solver.solve(source, bc_type="dirichlet", bc_values=np.array([p_target, p_target]))

        assert np.allclose(pressure, p_target, atol=1e-3)
        solver.destroy()

    def test_1d_solver_type_override(self, grid_1d):
        """Override solver type at solve() time."""
        solver = PETScTPFASolver(grid_1d, solver_type="gmres")
        source = np.zeros(grid_1d.num_cells)
        pressure = solver.solve(
            source,
            bc_type="dirichlet",
            bc_values=np.array([200e5, 100e5]),
            solver_type="cg",
        )
        assert len(pressure) == grid_1d.num_cells
        solver.destroy()

    def test_1d_pc_type_override(self, grid_1d):
        """Override preconditioner type at solve() time."""
        solver = PETScTPFASolver(grid_1d, pc_type="ilu")
        source = np.zeros(grid_1d.num_cells)
        pressure = solver.solve(
            source,
            bc_type="dirichlet",
            bc_values=np.array([200e5, 100e5]),
            pc_type="lu",
        )
        assert len(pressure) == grid_1d.num_cells
        solver.destroy()

    def test_1d_get_solver_info(self, grid_1d):
        solver = PETScTPFASolver(grid_1d, solver_type="cg", pc_type="gamg")
        info = solver.get_solver_info()
        assert "ksp_type" in info
        assert "pc_type" in info
        assert "mpi_size" in info
        assert "local_cells" in info
        solver.destroy()

    # -----------------------------------------------------------------
    # 2D tests
    # -----------------------------------------------------------------
    def test_2d_constructor(self, grid_2d):
        solver = PETScTPFASolver(grid_2d, mu=1e-3, rho=1000.0)
        assert solver.grid is grid_2d
        assert len(solver.transmissibilities) == grid_2d.num_faces
        solver.destroy()

    def test_2d_dirichlet_solve(self, grid_2d):
        solver = PETScTPFASolver(grid_2d, mu=1e-3, rho=1000.0)
        source = np.zeros(grid_2d.num_cells)
        pressure = solver.solve(source, bc_type="dirichlet", bc_values=np.array([200e5, 100e5]))

        assert len(pressure) == grid_2d.num_cells
        assert np.isfinite(pressure).all()
        assert pressure.min() >= 100e5 - 1e-3
        assert pressure.max() <= 200e5 + 1e-3
        solver.destroy()

    # -----------------------------------------------------------------
    # 3D tests
    # -----------------------------------------------------------------
    def test_3d_constructor(self, grid_3d):
        solver = PETScTPFASolver(grid_3d, mu=1e-3, rho=1000.0)
        assert solver.grid is grid_3d
        solver.destroy()

    def test_3d_dirichlet_solve(self, grid_3d):
        solver = PETScTPFASolver(grid_3d, mu=1e-3, rho=1000.0)
        source = np.zeros(grid_3d.num_cells)
        pressure = solver.solve(
            source,
            bc_type="dirichlet",
            bc_values=np.array([250e5, 100e5]),
        )
        assert len(pressure) == grid_3d.num_cells
        assert np.isfinite(pressure).all()
        assert pressure.min() >= 100e5 - 1e-3
        assert pressure.max() <= 250e5 + 1e-3
        solver.destroy()

    # -----------------------------------------------------------------
    # Numerical sanity
    # -----------------------------------------------------------------
    def test_mass_conservation_1d(self, grid_1d):
        """In steady state with no sources, div q = 0 -> flux in = flux out."""
        solver = PETScTPFASolver(grid_1d, mu=1e-3, rho=1000.0)
        source = np.zeros(grid_1d.num_cells)
        pressure = solver.solve(source, bc_type="dirichlet", bc_values=np.array([200e5, 100e5]))

        # Net flux into interior cells should be ~0 (|residual| < tol)
        residuals = np.zeros(grid_1d.num_cells)
        trans_x = solver.transmissibilities[: grid_1d.nx + 1]  # noqa: N806
        for i in range(grid_1d.num_cells):
            if i == 0:
                residuals[i] = (
                    trans_x[i] * (pressure[0] - pressure[1])
                    - trans_x[i + 1] * (pressure[1] - pressure[0])
                )
            elif i == grid_1d.num_cells - 1:
                residuals[i] = (
                    trans_x[i] * (pressure[i - 1] - pressure[i])
                    - trans_x[i + 1] * (pressure[i] - pressure[i - 1])
                )
            else:
                residuals[i] = (
                    trans_x[i] * (pressure[i - 1] - pressure[i])
                    - trans_x[i + 1] * (pressure[i] - pressure[i + 1])
                )
        assert np.all(np.abs(residuals) < 1e-3)
        solver.destroy()

    def test_higher_resolution_match(self):
        """Finer grid should give smoother but physically consistent pressure."""
        grid_coarse = StructuredGrid(nx=10, ny=1, nz=1, dx=100.0, dy=10.0, dz=10.0)
        grid_coarse.set_permeability(1e-14)
        grid_coarse.set_porosity(0.2)

        grid_fine = StructuredGrid(nx=100, ny=1, nz=1, dx=10.0, dy=10.0, dz=10.0)
        grid_fine.set_permeability(1e-14)
        grid_fine.set_porosity(0.2)

        solver_c = PETScTPFASolver(grid_coarse, mu=1e-3, rho=1000.0)
        solver_f = PETScTPFASolver(grid_fine, mu=1e-3, rho=1000.0)

        source_c = np.zeros(grid_coarse.num_cells)
        source_f = np.zeros(grid_fine.num_cells)

        p_c = solver_c.solve(source_c, bc_type="dirichlet", bc_values=np.array([200e5, 100e5]))
        p_f = solver_f.solve(source_f, bc_type="dirichlet", bc_values=np.array([200e5, 100e5]))

        # Coarse and fine should bracket the same physical range
        assert np.isclose(p_c.min(), p_f.min(), atol=1e-1)
        assert np.isclose(p_c.max(), p_f.max(), atol=1e-1)

        solver_c.destroy()
        solver_f.destroy()


class TestPETScTPFASolverNonlinear:
    """Tests for the SNES non-linear solver placeholder."""

    def test_nonlinear_identity(self):
        """Solve F(x)=x-5=0 → x=5."""
        grid = StructuredGrid(nx=5, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
        grid.set_permeability(1e-14)
        solver = PETScTPFASolver(grid, mu=1e-3)

        def residual(x):
            return x - 5.0

        def jacobian(_x):
            n = len(_x)
            return np.eye(n)

        x0 = np.zeros(grid.num_cells)
        sol = solver.solve_nonlinear(residual, jacobian, x0, tol=1e-12)
        assert np.allclose(sol, 5.0, atol=1e-6)
        solver.destroy()

    def test_nonlinear_quadratic(self):
        """Solve F(x)=x^2 - 4 = 0 → x=2 (positive root)."""
        grid = StructuredGrid(nx=1, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
        grid.set_permeability(1e-14)
        solver = PETScTPFASolver(grid, mu=1e-3)

        def residual(x):
            return x**2 - 4.0

        def jacobian(x):
            return np.diag(2 * x)

        x0 = np.array([1.5])
        sol = solver.solve_nonlinear(residual, jacobian, x0, tol=1e-12)
        assert np.allclose(sol, 2.0, atol=1e-4)
        solver.destroy()
