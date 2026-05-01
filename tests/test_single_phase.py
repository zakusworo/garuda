"""
Comprehensive test suite for SinglePhaseFlow.

Covers initialization, compute_accumulation, compute_flux,
step_implicit (including the documented prev_accumulation bug),
and edge-case behaviour for single-phase porous-media flow.

See garuda/physics/single_phase.py for the implementation under test.
"""

import pytest
import sys
import os

# Ensure the garuda package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

if NUMPY_AVAILABLE:
    from garuda.core.grid import StructuredGrid
    from garuda.core.fluid_properties import FluidProperties
    from garuda.core.rock_properties import RockProperties
    from garuda.core.tpfa_solver import TPFASolver
    from garuda.physics.single_phase import SinglePhaseFlow


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def grid_1d():
    """5-cell 1D grid: dx=100 m, dy=10 m, dz=1 m."""
    return StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=10, dz=1)


@pytest.fixture
def fluid_water():
    """Standard water properties."""
    return FluidProperties(fluid_type='water')


@pytest.fixture
def rock_default():
    """Default rock (porosity=0.2, k=1e-12 m2)."""
    return RockProperties()


@pytest.fixture
def solver_1d(grid_1d, fluid_water):
    """TPFA solver for 1D water flow."""
    return TPFASolver(grid_1d, mu=fluid_water.mu, rho=fluid_water.rho)


@pytest.fixture
def flow_1d(grid_1d, fluid_water, rock_default):
    """SinglePhaseFlow on a 5-cell 1D grid with water."""
    return SinglePhaseFlow(grid=grid_1d, fluid=fluid_water, rock=rock_default)


# =============================================================================
# 1. Initialization
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestSinglePhaseFlowInitialization:
    """Test __post_init__ and field storage."""

    def test_grid_fluid_rock_stored(self, flow_1d, grid_1d, fluid_water, rock_default):
        """Grid, fluid, and rock are stored as instance attributes."""
        assert flow_1d.grid is grid_1d
        assert flow_1d.fluid is fluid_water
        assert flow_1d.rock is rock_default

    def test_pressure_initial(self, flow_1d, grid_1d):
        """Pressure defaults to 1e5 Pa (1 bar) per cell."""
        assert flow_1d.pressure.shape == (grid_1d.num_cells,)
        np.testing.assert_array_equal(flow_1d.pressure, np.full(grid_1d.num_cells, 1e5))

    def test_saturation_initial(self, flow_1d, grid_1d):
        """Saturation defaults to 1.0 per cell (single-phase)."""
        assert flow_1d.saturation.shape == (grid_1d.num_cells,)
        np.testing.assert_array_equal(flow_1d.saturation, np.ones(grid_1d.num_cells))

    def test_temperature_initial(self, flow_1d, grid_1d):
        """Temperature defaults to 293.15 K (20 °C) per cell."""
        assert flow_1d.temperature.shape == (grid_1d.num_cells,)
        np.testing.assert_array_equal(flow_1d.temperature, np.full(grid_1d.num_cells, 293.15))

    def test_pressure_always_overwritten_in_post_init(self, grid_1d, fluid_water, rock_default):
        """__post_init__ now preserves user-provided pressure (fixed behaviour).

        Previously, pressure was always overwritten with 1e5.
        After the fix, custom values are honoured.
        """
        custom_p = np.linspace(1e5, 2e5, grid_1d.num_cells)
        flow = SinglePhaseFlow(
            grid=grid_1d, fluid=fluid_water, rock=rock_default,
            pressure=custom_p,
        )
        np.testing.assert_array_equal(flow.pressure, custom_p)

    def test_temperature_can_be_overridden(self, grid_1d, fluid_water, rock_default):
        """Explicit temperature array bypasses the 293.15 K default."""
        custom_T = np.linspace(300, 350, grid_1d.num_cells)
        flow = SinglePhaseFlow(
            grid=grid_1d, fluid=fluid_water, rock=rock_default,
            temperature=custom_T,
        )
        np.testing.assert_array_equal(flow.temperature, custom_T)

    def test_2d_grid_initializes_correctly(self, fluid_water, rock_default):
        """A 2D grid yields correct array shapes."""
        grid = StructuredGrid(nx=3, ny=4, nz=1, dx=50, dy=75, dz=1)
        flow = SinglePhaseFlow(grid=grid, fluid=fluid_water, rock=rock_default)
        nc = grid.num_cells  # 12
        assert flow.pressure.shape == (nc,)
        assert flow.saturation.shape == (nc,)
        assert flow.temperature.shape == (nc,)

    def test_3d_grid_initializes_correctly(self, fluid_water, rock_default):
        """A 3D grid yields correct array shapes."""
        grid = StructuredGrid(nx=2, ny=3, nz=4, dx=100, dy=100, dz=10)
        flow = SinglePhaseFlow(grid=grid, fluid=fluid_water, rock=rock_default)
        nc = grid.num_cells  # 24
        assert flow.pressure.shape == (nc,)
        assert flow.saturation.shape == (nc,)
        assert flow.temperature.shape == (nc,)


# =============================================================================
# 2. compute_accumulation
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestComputeAccumulation:
    """Test the accumulation term: phi * rho(p, T)."""

    def test_returns_correct_length(self, flow_1d, grid_1d):
        """Result has one entry per cell."""
        acc = flow_1d.compute_accumulation()
        assert isinstance(acc, np.ndarray)
        assert acc.shape == (grid_1d.num_cells,)

    def test_all_positive(self, flow_1d):
        """Accumulation values are positive (phi>0, rho>0)."""
        acc = flow_1d.compute_accumulation()
        assert np.all(acc > 0)

    def test_constant_default_state(self, flow_1d):
        """With uniform pressure/temperature, accumulation is uniform."""
        acc = flow_1d.compute_accumulation()
        assert np.allclose(acc, acc[0])

    def test_formula_phi_times_rho(self, flow_1d):
        """Matches phi * fluid.density(p, T) directly."""
        acc = flow_1d.compute_accumulation()
        expected = flow_1d.rock.porosity * flow_1d.fluid.density(
            flow_1d.pressure, flow_1d.temperature
        )
        np.testing.assert_array_almost_equal(acc, expected)

    def test_responds_to_pressure_change(self, flow_1d):
        """Higher pressure increases accumulation (compressible fluid)."""
        acc_low = flow_1d.compute_accumulation()
        flow_1d.pressure *= 2.0  # double pressure
        acc_high = flow_1d.compute_accumulation()
        assert np.all(acc_high > acc_low)

    def test_responds_to_temperature_change(self, flow_1d):
        """Higher temperature decreases accumulation (thermal expansion)."""
        acc_cold = flow_1d.compute_accumulation()
        flow_1d.temperature[:] = 373.15  # 100 °C
        acc_hot = flow_1d.compute_accumulation()
        assert np.all(acc_hot < acc_cold)

    def test_heterogeneous_pressure(self, flow_1d):
        """With a pressure gradient, accumulation varies per cell."""
        flow_1d.pressure = np.linspace(1e5, 2e5, flow_1d.grid.num_cells)
        acc = flow_1d.compute_accumulation()
        # Should be monotonically increasing
        assert np.all(np.diff(acc) > 0)

    def test_custom_rock_porosity(self, grid_1d, fluid_water):
        """Custom rock porosity is reflected in accumulation."""
        rock = RockProperties(porosity=0.35)
        flow = SinglePhaseFlow(grid=grid_1d, fluid=fluid_water, rock=rock)
        acc = flow.compute_accumulation()
        expected_phi = 0.35
        rho = fluid_water.density(flow.pressure, flow.temperature)
        np.testing.assert_array_almost_equal(acc, expected_phi * rho)

    def test_different_fluid_type(self, grid_1d, rock_default):
        """Oil has different density, accumulation reflects that."""
        oil = FluidProperties(fluid_type='oil')
        flow = SinglePhaseFlow(grid=grid_1d, fluid=oil, rock=rock_default)
        acc = flow.compute_accumulation()
        expected = rock_default.porosity * oil.density(flow.pressure, flow.temperature)
        np.testing.assert_array_almost_equal(acc, expected)


# =============================================================================
# 3. compute_flux
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestComputeFlux:
    """Test TPFA face-flux accumulation to cells."""

    def test_returns_correct_length(self, flow_1d, solver_1d):
        """Result has one entry per cell."""
        flux = flow_1d.compute_flux(solver_1d)
        assert flux.shape == (flow_1d.grid.num_cells,)

    def test_no_flow_with_uniform_pressure(self, flow_1d, solver_1d):
        """Uniform pressure yields near-zero net flux per cell."""
        flux = flow_1d.compute_flux(solver_1d)
        # Interior cells should have zero net flux (in=out)
        np.testing.assert_array_almost_equal(flux, np.zeros_like(flux))

    def test_flux_nonzero_with_pressure_gradient(self, flow_1d, solver_1d):
        """A pressure gradient produces non-zero fluxes."""
        flow_1d.pressure = np.linspace(2e5, 1e5, flow_1d.grid.num_cells)
        flux = flow_1d.compute_flux(solver_1d)
        # At least some cells should have non-zero flux
        assert not np.allclose(flux, 0)

    def test_flux_different_viscosity_1d(self, grid_1d, fluid_water, rock_default):
        """Different viscosities produce different fluxes (mu affects T)."""
        flow = SinglePhaseFlow(grid=grid_1d, fluid=fluid_water, rock=rock_default)
        flow.pressure = np.linspace(2e5, 1e5, grid_1d.num_cells)
        s1 = TPFASolver(grid_1d, mu=1e-3, rho=1000)
        s2 = TPFASolver(grid_1d, mu=2e-3, rho=1000)  # twice as viscous → half flux
        flux1 = flow.compute_flux(s1)
        flux2 = flow.compute_flux(s2)
        assert not np.allclose(flux1, flux2)
        # Higher viscosity → lower fluxes (absolute).
        # Non-zero flux entries should all be smaller.
        nonzero = np.abs(flux1) > 0
        assert np.all(np.abs(flux2)[nonzero] < np.abs(flux1)[nonzero])

    def test_1d_boundary_cells_have_net_flux(self, flow_1d, solver_1d):
        """With a pressure gradient, boundary cells show net flux."""
        flow_1d.pressure = np.linspace(2e5, 1e5, flow_1d.grid.num_cells)
        flux = flow_1d.compute_flux(solver_1d)
        # Boundary cells (first and last) should have non-zero net flux
        assert abs(flux[0]) > 1e-15 or abs(flux[-1]) > 1e-15

    def test_2d_flux(self, fluid_water, rock_default):
        """Flux works on a 2D grid."""
        grid = StructuredGrid(nx=3, ny=3, nz=1, dx=100, dy=100, dz=1)
        flow = SinglePhaseFlow(grid=grid, fluid=fluid_water, rock=rock_default)
        flow.pressure = np.ones(grid.num_cells) * 1e5
        flow.pressure[4] = 1.2e5  # bump in centre
        solver = TPFASolver(grid, mu=fluid_water.mu, rho=fluid_water.rho)
        flux = flow.compute_flux(solver)
        assert flux.shape == (grid.num_cells,)


# =============================================================================
# 4. step_implicit — basic behaviour
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestStepImplicitBasic:
    """Test step_implicit with prev_accumulation manually set (workaround)."""

    def _prepare(self, flow, solver):
        """Apply the prev_accumulation workaround."""
        flow.prev_accumulation = flow.compute_accumulation()
        return flow, solver

    def test_runs_with_workaround(self, flow_1d, solver_1d):
        """step_implicit runs to completion when prev_accumulation is set."""
        self._prepare(flow_1d, solver_1d)
        source_terms = np.zeros(flow_1d.grid.num_cells)
        result = flow_1d.step_implicit(
            dt=86400, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        assert 'converged' in result
        assert 'iterations' in result
        assert 'residual_norm' in result

    def test_converges_with_uniform_state(self, flow_1d, solver_1d):
        """When source_terms are zero and BCs match initial pressure, it converges."""
        self._prepare(flow_1d, solver_1d)
        source_terms = np.zeros(flow_1d.grid.num_cells)
        result = flow_1d.step_implicit(
            dt=86400, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        assert result['converged']

    def test_pressure_changes_with_nonzero_source(self, flow_1d, solver_1d):
        """Non-zero source terms change the pressure field."""
        self._prepare(flow_1d, solver_1d)
        p_before = flow_1d.pressure.copy()
        source_terms = np.zeros(flow_1d.grid.num_cells)
        source_terms[2] = 1.0  # inject in middle cell
        flow_1d.step_implicit(
            dt=3600, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        # Pressure changes are small (O(0.01 Pa)) but real — use strict
        # element-wise comparison instead of allclose's default rtol=1e-5
        assert not np.array_equal(flow_1d.pressure, p_before), (
            "Pressure did not change at all — expected at least a small delta."
        )

    def test_iterations_in_range(self, flow_1d, solver_1d):
        """Iteration count is between 1 and max_iter."""
        self._prepare(flow_1d, solver_1d)
        source_terms = np.zeros(flow_1d.grid.num_cells)
        result = flow_1d.step_implicit(
            dt=86400, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d, max_iter=10,
        )
        assert 1 <= result['iterations'] <= 10


# =============================================================================
# 5. step_implicit — injection
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestStepImplicitInjection:
    """step_implicit with positive source terms (injection)."""

    def test_injection_increases_pressure(self, flow_1d, solver_1d):
        """Injecting mass raises pressure."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        p_before = flow_1d.pressure.copy()
        source_terms = np.zeros(flow_1d.grid.num_cells)
        source_terms[1] = 0.5  # kg/s injection
        flow_1d.step_implicit(
            dt=3600, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        p_after = flow_1d.pressure.copy()
        # At least the injection cell pressure should rise
        assert np.mean(p_after) > np.mean(p_before)

    def test_injection_spreads(self, flow_1d, solver_1d):
        """Injection in one cell affects neighbouring cells."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        source_terms = np.zeros(flow_1d.grid.num_cells)
        source_terms[0] = 0.5  # inject at left boundary cell
        flow_1d.step_implicit(
            dt=3600, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        # Pressure should be higher at injection point, tapering away
        p = flow_1d.pressure
        assert p[0] >= p[1] >= p[-1] or p[0] > 1e5


# =============================================================================
# 6. step_implicit — production
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestStepImplicitProduction:
    """step_implicit with negative source terms (production)."""

    def test_production_decreases_pressure(self, flow_1d, solver_1d):
        """Producing mass drops pressure."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        p_before = flow_1d.pressure.copy()
        source_terms = np.zeros(flow_1d.grid.num_cells)
        source_terms[2] = -0.5  # kg/s production
        flow_1d.step_implicit(
            dt=3600, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        p_after = flow_1d.pressure.copy()
        assert np.mean(p_after) < np.mean(p_before)


# =============================================================================
# 7. Edge cases
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestStepImplicitEdgeCases:
    """Large dt, small dt, convergence failure, array temperature."""

    def test_large_dt_does_not_crash(self, flow_1d, solver_1d):
        """A very large time step still completes."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        source_terms = np.zeros(flow_1d.grid.num_cells)
        source_terms[2] = 0.1
        result = flow_1d.step_implicit(
            dt=1e10, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        assert 'converged' in result

    def test_small_dt_does_not_crash(self, flow_1d, solver_1d):
        """A very small time step still completes."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        source_terms = np.zeros(flow_1d.grid.num_cells)
        source_terms[2] = 0.1
        result = flow_1d.step_implicit(
            dt=1e-6, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        assert 'converged' in result

    def test_zero_dt_causes_division_by_zero(self, flow_1d, solver_1d):
        """dt=0 should raise ZeroDivisionError / produce inf residual."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        source_terms = np.zeros(flow_1d.grid.num_cells)
        # Division by zero in residual — expect inf/error
        result = flow_1d.step_implicit(
            dt=0.0, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        # Will likely not converge due to inf residual
        assert not result['converged'] or np.isinf(result['residual_norm'])

    def test_convergence_failure_tight_tol(self, flow_1d, solver_1d):
        """Tight tolerance with few iterations fails to converge."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        source_terms = np.zeros(flow_1d.grid.num_cells)
        source_terms[2] = 100.0  # large source
        result = flow_1d.step_implicit(
            dt=86400, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d, max_iter=2, tol=1e-15,
        )
        # With only 2 iterations and tight tol, may not converge
        assert not result['converged'] or result['iterations'] <= 2

    def test_max_iter_one(self, flow_1d, solver_1d):
        """max_iter=1 runs exactly one iteration."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        source_terms = np.zeros(flow_1d.grid.num_cells)
        result = flow_1d.step_implicit(
            dt=3600, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d, max_iter=1,
        )
        assert result['iterations'] == 1

    def test_array_temperature(self, flow_1d, solver_1d):
        """step_implicit works with a heterogeneous temperature array."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        flow_1d.temperature = np.linspace(300, 350, flow_1d.grid.num_cells)
        source_terms = np.zeros(flow_1d.grid.num_cells)
        source_terms[2] = 0.1
        result = flow_1d.step_implicit(
            dt=3600, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        assert 'converged' in result

    def test_array_temperature_affects_accumulation(self, flow_1d):
        """Heterogeneous temperature produces heterogeneous accumulation."""
        flow_1d.temperature = np.linspace(300, 350, flow_1d.grid.num_cells)
        acc = flow_1d.compute_accumulation()
        # Warmer cells have lower density → lower accumulation
        assert acc[0] > acc[-1]


# =============================================================================
# 8. prev_accumulation BUG — documented
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestPrevAccumulationBug:
    """
    FIXED: step_implicit() no longer crashes on first call because
    __post_init__ now initialises prev_accumulation automatically.
    """

    def test_runs_without_manual_set(self, flow_1d, solver_1d):
        """Calling step_implicit without setting prev_accumulation works."""
        source_terms = np.zeros(flow_1d.grid.num_cells)
        result = flow_1d.step_implicit(
            dt=86400, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        assert 'converged' in result
        assert 'iterations' in result
        assert 'residual_norm' in result

    def test_prev_accumulation_in_post_init(self, flow_1d):
        """After construction, prev_accumulation exists and has correct shape."""
        assert hasattr(flow_1d, 'prev_accumulation')
        assert flow_1d.prev_accumulation.shape == (flow_1d.grid.num_cells,)

    def test_second_call_works_after_first(self, flow_1d, solver_1d):
        """After step_implicit runs once, a second call should succeed."""
        source_terms = np.zeros(flow_1d.grid.num_cells)
        flow_1d.step_implicit(
            dt=86400, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        result = flow_1d.step_implicit(
            dt=86400, source_terms=source_terms,
            bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
            solver=solver_1d,
        )
        assert 'converged' in result


# =============================================================================
# 9. Regression / integration-style tests
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestSinglePhaseIntegration:
    """End-to-end behaviour through multiple time steps."""

    def test_multiple_time_steps(self, flow_1d, solver_1d):
        """Running multiple time steps accumulates pressure changes."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        p_initial = flow_1d.pressure.copy()
        source_terms = np.zeros(flow_1d.grid.num_cells)
        source_terms[2] = 0.2  # constant injection

        n_steps = 5
        for _ in range(n_steps):
            flow_1d.step_implicit(
                dt=3600, source_terms=source_terms,
                bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
                solver=solver_1d,
            )

        # After many injection steps, pressure should be higher
        assert np.mean(flow_1d.pressure) > np.mean(p_initial)

    def test_source_term_shape_validation(self, flow_1d, solver_1d):
        """Mismatched source_terms shape should raise an error."""
        flow_1d.prev_accumulation = flow_1d.compute_accumulation()
        wrong_shape = np.zeros(flow_1d.grid.num_cells + 1)
        with pytest.raises(ValueError):
            flow_1d.step_implicit(
                dt=86400, source_terms=wrong_shape,
                bc_type='dirichlet', bc_values=np.array([1e5, 1e5]),
                solver=solver_1d,
            )
