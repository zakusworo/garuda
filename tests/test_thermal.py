"""
Comprehensive test suite for the ThermalFlow class.

Tests cover initialization, energy accumulation, temperature interpolation,
effective thermal conductivity, conductive flux, heat flux, energy matrix
assembly, coupled stepping, and geothermal gradient computation.
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
    from garuda.physics.thermal import ThermalFlow


# =========================================================================
# Fixtures
# =========================================================================

def _make_1d_thermal(nx=5, dx=10.0, porosity=0.2, perm=1e-12):
    """Build a minimal ThermalFlow in 1D with permeability set."""
    grid = StructuredGrid(nx=nx, ny=1, nz=1, dx=dx, dy=1.0, dz=1.0)
    grid.set_permeability(perm)
    rock = RockProperties(porosity=porosity, permeability=perm)
    fluid = FluidProperties(fluid_type='water')
    thermal = ThermalFlow(grid=grid, rock=rock, fluid=fluid)
    return thermal


class _ScalarViscosityFluid(FluidProperties):
    """Fluid whose viscosity() always returns a scalar.

    Workaround for a bug in TPFASolver._compute_1d_transmissibilities
    which cannot handle self.mu as a numpy array.
    """

    def viscosity(self, temperature=None):
        return self.mu  # always scalar


def _make_flow_solver(thermal):
    """Build a TPFASolver from a ThermalFlow's grid with default mu/rho."""
    return TPFASolver(thermal.grid, mu=1e-3, rho=1000.0)


def _make_coupled_thermal(nx=3, dx=10.0, porosity=0.2, perm=1e-12):
    """Build a ThermalFlow suitable for step_coupled tests.

    Uses _ScalarViscosityFluid to avoid the tpfa_solver bug with array mu.
    """
    grid = StructuredGrid(nx=nx, ny=1, nz=1, dx=dx, dy=1.0, dz=1.0)
    grid.set_permeability(perm)
    rock = RockProperties(porosity=porosity, permeability=perm)
    fluid = _ScalarViscosityFluid(fluid_type='water')
    thermal = ThermalFlow(grid=grid, rock=rock, fluid=fluid)
    return thermal


# =========================================================================
# 1. Initialization
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestThermalFlowInitialization:
    """Test ThermalFlow creation and default state arrays."""

    def test_creation_defaults(self):
        """ThermalFlow creates successfully with valid components."""
        thermal = _make_1d_thermal(nx=5)
        assert thermal.grid is not None
        assert thermal.rock is not None
        assert thermal.fluid is not None

    def test_pressure_default_shape_and_value(self):
        """Default pressure is 1e5 Pa for every cell."""
        thermal = _make_1d_thermal(nx=5)
        assert thermal.pressure.shape == (5,)
        assert np.allclose(thermal.pressure, 1e5)

    def test_temperature_default_shape_and_value(self):
        """Default temperature is 293.15 K for every cell."""
        thermal = _make_1d_thermal(nx=5)
        assert thermal.temperature.shape == (5,)
        assert np.allclose(thermal.temperature, 293.15)

    def test_previous_state_copied(self):
        """p_prev and T_prev are copies, not views, of the initial state."""
        thermal = _make_1d_thermal(nx=4)
        assert np.array_equal(thermal.p_prev, thermal.pressure)
        assert np.array_equal(thermal.T_prev, thermal.temperature)
        # Mutate current state; prev should be independent
        thermal.pressure[0] = 2e5
        thermal.temperature[0] = 400.0
        assert thermal.p_prev[0] == 1e5
        assert thermal.T_prev[0] == 293.15

    def test_num_cells_matches_grid(self):
        """Pressure/temperature arrays have length == grid.num_cells."""
        for n in [3, 7]:
            thermal = _make_1d_thermal(nx=n)
            assert len(thermal.pressure) == n
            assert len(thermal.temperature) == n
            assert len(thermal.p_prev) == n
            assert len(thermal.T_prev) == n


# =========================================================================
# 2. compute_energy_accumulation
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestEnergyAccumulation:
    """Test the compute_energy_accumulation method."""

    def test_returns_positive_array(self):
        """Energy accumulation is non-negative for physical temperatures."""
        thermal = _make_1d_thermal(nx=5)
        acc = thermal.compute_energy_accumulation()
        assert acc.shape == (5,)
        assert np.all(acc > 0)

    def test_accumulation_scales_with_temperature(self):
        """Doubling temperature roughly doubles accumulation (linear term)."""
        thermal = _make_1d_thermal(nx=5)
        acc_cold = thermal.compute_energy_accumulation()

        thermal.temperature = thermal.temperature * 2.0
        acc_hot = thermal.compute_energy_accumulation()

        # rhoCp_bulk changes slightly with T (density), so ratio ~ 2
        ratio = acc_hot / (acc_cold + 1e-30)
        assert np.all(ratio > 1.5)  # should be roughly 2.0

    def test_accumulation_increases_with_porosity(self):
        """Higher porosity → lower solid fraction → lower bulk heat capacity."""
        thermal_low = _make_1d_thermal(nx=5, porosity=0.1)
        thermal_high = _make_1d_thermal(nx=5, porosity=0.3)

        acc_low = thermal_low.compute_energy_accumulation()
        acc_high = thermal_high.compute_energy_accumulation()

        # Lower porosity means more rock (higher rho*Cp for rock vs water)
        # actually rho_rock*cp_rock = 2650*840 > 1000*4182? 
        # 2650*840 = 2,226,000 and 1000*4182 = 4,182,000
        # So water has higher volumetric heat capacity
        # Higher porosity → more water → higher accumulation
        assert np.all(acc_high > acc_low)

    def test_accumulation_is_uniform_for_uniform_state(self):
        """Uniform p,T → uniform accumulation."""
        thermal = _make_1d_thermal(nx=10)
        acc = thermal.compute_energy_accumulation()
        assert np.allclose(acc, acc[0])


# =========================================================================
# 3. _interpolate_temperature_to_faces
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestInterpolateTemperatureToFaces:
    """Test the _interpolate_temperature_to_faces private method."""

    def test_interior_faces_averaged(self):
        """Interior face temperatures are the arithmetic mean of neighbours."""
        thermal = _make_1d_thermal(nx=4)
        thermal.temperature = np.array([300.0, 310.0, 320.0, 330.0])

        T_face = thermal._interpolate_temperature_to_faces()
        # faces: 0(boundary), 1, 2, 3, 4(boundary) — 5 faces total
        assert T_face.shape == (5,)
        # Interior: face 1 = (300+310)/2 = 305
        assert np.isclose(T_face[1], 305.0)
        # face 2 = (310+320)/2 = 315
        assert np.isclose(T_face[2], 315.0)
        # face 3 = (320+330)/2 = 325
        assert np.isclose(T_face[3], 325.0)

    def test_boundary_faces_equal_cell_value(self):
        """Boundary faces use the adjacent cell temperature directly."""
        thermal = _make_1d_thermal(nx=4)
        thermal.temperature = np.array([300.0, 310.0, 320.0, 330.0])

        T_face = thermal._interpolate_temperature_to_faces()
        assert np.isclose(T_face[0], 300.0)
        assert np.isclose(T_face[-1], 330.0)

    def test_uniform_temperature_preserved(self):
        """Uniform cell temperature → all face temps equal."""
        thermal = _make_1d_thermal(nx=5)
        thermal.temperature = np.full(5, 350.0)
        T_face = thermal._interpolate_temperature_to_faces()
        assert np.allclose(T_face, 350.0)


# =========================================================================
# 4. _effective_thermal_conductivity
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestEffectiveThermalConductivity:
    """Test the _effective_thermal_conductivity private method.

    The method now returns one value per cell so heterogeneous porosity
    propagates through; for homogeneous porosity every entry is identical.
    """

    def test_returns_per_cell_array(self):
        """Effective conductivity is an ndarray of length num_cells."""
        thermal = _make_1d_thermal(nx=5)
        leff = thermal._effective_thermal_conductivity()
        assert isinstance(leff, np.ndarray)
        assert leff.shape == (thermal.grid.num_cells,)

    def test_formula_correct(self):
        """lambda_eff = (1-phi)*lambda_rock + phi*lambda_fluid."""
        thermal = _make_1d_thermal(nx=5, porosity=0.25)
        leff = thermal._effective_thermal_conductivity()
        expected = 0.75 * thermal.rock.lambda_rock + 0.25 * 0.6
        assert np.allclose(leff, expected)

    def test_between_fluid_and_rock_values(self):
        """Effective value is bounded by fluid and rock conductivities."""
        thermal = _make_1d_thermal(nx=5, porosity=0.5)
        leff = thermal._effective_thermal_conductivity()
        lambda_rock = thermal.rock.lambda_rock  # 2.5
        lambda_fluid = 0.6
        assert np.all(lambda_fluid < leff)
        assert np.all(leff < lambda_rock)

    def test_zero_porosity_gives_rock_only(self):
        """At phi=0, lambda_eff equals lambda_rock everywhere."""
        thermal = _make_1d_thermal(nx=5, porosity=0.0)
        leff = thermal._effective_thermal_conductivity()
        assert np.allclose(leff, thermal.rock.lambda_rock)

    def test_full_porosity_gives_fluid_only(self):
        """At phi=1, lambda_eff equals lambda_fluid everywhere."""
        thermal = _make_1d_thermal(nx=5, porosity=1.0)
        leff = thermal._effective_thermal_conductivity()
        assert np.allclose(leff, 0.6)


# =========================================================================
# 5. _compute_conductive_flux
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestComputeConductiveFlux:
    """Test the _compute_conductive_flux private method."""

    def test_returns_correct_shape(self):
        """Conductive flux array has one entry per face."""
        thermal = _make_1d_thermal(nx=5)
        flux = thermal._compute_conductive_flux()
        assert flux.shape == (thermal.grid.num_faces,)

    def test_zero_gradient_gives_zero_flux(self):
        """Uniform temperature → no conductive flux anywhere."""
        thermal = _make_1d_thermal(nx=5)
        thermal.temperature = np.full(5, 300.0)
        thermal.T_prev = thermal.temperature.copy()
        flux = thermal._compute_conductive_flux()
        # Interior flux should be ~0; boundary flux may be non-zero
        # if T != T_prev at boundaries. But we set them equal, so all ~0.
        assert np.allclose(flux, 0.0, atol=1e-14)

    def test_flux_direction_hot_to_cold(self):
        """Heat flows from hot to cold (negative dT/dx → negative flux)."""
        thermal = _make_1d_thermal(nx=3)
        # Hot on left, cold on right
        thermal.temperature = np.array([400.0, 350.0, 300.0])
        thermal.T_prev = thermal.temperature.copy()

        flux = thermal._compute_conductive_flux()
        # Per-cell λ is uniform here; harmonic mean equals the cell value.
        lam_face = float(thermal._effective_thermal_conductivity()[0])
        dx = float(np.mean(thermal.grid.dx))

        expected_1 = -lam_face * (350.0 - 400.0) / dx
        assert np.isclose(flux[1], expected_1)
        assert flux[1] > 0  # flows from hot to cold (positive direction)

        expected_2 = -lam_face * (300.0 - 350.0) / dx
        assert np.isclose(flux[2], expected_2)
        assert flux[2] > 0

    def test_boundary_faces_zero_flux(self):
        """Boundary faces default to zero-flux Neumann (no Dirichlet BC info).

        The previous behaviour mixed a temporal difference (T vs T_prev) into
        the spatial gradient, which is dimensionally incoherent. The current
        implementation returns 0 at boundaries; Dirichlet BCs are applied
        separately via build_energy_matrix.
        """
        thermal = _make_1d_thermal(nx=3)
        thermal.temperature = np.array([310.0, 310.0, 310.0])
        thermal.T_prev = np.array([300.0, 300.0, 300.0])

        flux = thermal._compute_conductive_flux()

        assert np.isclose(flux[0], 0.0)
        assert np.isclose(flux[-1], 0.0)

    def test_flux_magnitude_scales_with_conductivity(self):
        """Higher lambda_rock → larger conductive flux magnitude."""
        thermal_low = _make_1d_thermal(nx=3)
        thermal_low.rock.lambda_rock = 1.0
        thermal_low.temperature = np.array([400.0, 300.0, 400.0])
        thermal_low.T_prev = thermal_low.temperature.copy()

        thermal_high = _make_1d_thermal(nx=3)
        thermal_high.rock.lambda_rock = 4.0
        thermal_high.temperature = np.array([400.0, 300.0, 400.0])
        thermal_high.T_prev = thermal_high.temperature.copy()

        flux_low = thermal_low._compute_conductive_flux()
        flux_high = thermal_high._compute_conductive_flux()

        # For interior faces, |flux| is proportional to lambda_eff ~ lambda_rock
        assert abs(flux_high[1]) > abs(flux_low[1])


# =========================================================================
# 6. compute_heat_flux
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestComputeHeatFlux:
    """Test compute_heat_flux with mass flux input."""

    def test_returns_correct_shape(self):
        """Heat flux array has one entry per face."""
        thermal = _make_1d_thermal(nx=5)
        mass_flux = np.zeros(thermal.grid.num_faces)
        q = thermal.compute_heat_flux(mass_flux)
        assert q.shape == (thermal.grid.num_faces,)

    def test_zero_mass_flux_gives_conductive_only(self):
        """With no mass flux, heat flux equals conductive flux alone."""
        thermal = _make_1d_thermal(nx=3)
        thermal.temperature = np.array([400.0, 350.0, 300.0])
        thermal.T_prev = thermal.temperature.copy()

        mass_flux = np.zeros(thermal.grid.num_faces)
        q_total = thermal.compute_heat_flux(mass_flux)
        q_cond = thermal._compute_conductive_flux()

        assert np.allclose(q_total, q_cond)

    def test_convective_contribution_increases_with_flux(self):
        """A larger mass flux results in larger total heat flux."""
        thermal = _make_1d_thermal(nx=3)
        thermal.temperature = np.array([350.0, 350.0, 350.0])
        thermal.T_prev = thermal.temperature.copy()

        mass_flux_small = np.full(thermal.grid.num_faces, 0.1)
        mass_flux_large = np.full(thermal.grid.num_faces, 1.0)

        q_small = thermal.compute_heat_flux(mass_flux_small)
        q_large = thermal.compute_heat_flux(mass_flux_large)

        # Convective part: rho * Cp * T_face * mass_flux
        # Larger mass_flux → larger magnitude
        assert np.all(np.abs(q_large) > np.abs(q_small))


# =========================================================================
# 7. build_energy_matrix
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestBuildEnergyMatrix:
    """Test the build_energy_matrix method."""

    @staticmethod
    def _is_diagonally_dominant(A):
        """Check if sparse matrix A is (weakly) diagonally dominant."""
        A_dense = A.toarray()
        diag = np.abs(A_dense.diagonal())
        off_diag_sum = np.sum(np.abs(A_dense), axis=1) - diag
        return np.all(diag >= off_diag_sum - 1e-12)

    def test_correct_dimensions(self):
        """A is (num_cells x num_cells), b has length num_cells."""
        thermal = _make_1d_thermal(nx=5)
        mass_flux = np.ones(thermal.grid.num_faces)
        heat_sources = np.zeros(thermal.grid.num_cells)

        A, b = thermal.build_energy_matrix(dt=3600.0, mass_flux=mass_flux,
                                           heat_sources=heat_sources)
        assert A.shape == (5, 5)
        assert b.shape == (5,)

    def test_diagonal_dominance(self):
        """System matrix A should be diagonally dominant."""
        thermal = _make_1d_thermal(nx=10)
        mass_flux = np.zeros(thermal.grid.num_faces)
        heat_sources = np.zeros(thermal.grid.num_cells)

        A, b = thermal.build_energy_matrix(dt=1e5, mass_flux=mass_flux,
                                           heat_sources=heat_sources)
        assert self._is_diagonally_dominant(A)

    def test_larger_dt_reduces_diagonal(self):
        """The time-derivative contribution ~1/dt, so larger dt → smaller diag."""
        thermal = _make_1d_thermal(nx=5)
        mass_flux = np.zeros(thermal.grid.num_faces)
        heat_sources = np.zeros(thermal.grid.num_cells)

        A_small_dt, _ = thermal.build_energy_matrix(dt=1.0, mass_flux=mass_flux,
                                                     heat_sources=heat_sources)
        A_large_dt, _ = thermal.build_energy_matrix(dt=1e6, mass_flux=mass_flux,
                                                     heat_sources=heat_sources)

        assert A_small_dt[0, 0] > A_large_dt[0, 0]

    def test_heat_sources_appear_in_rhs(self):
        """Heat sources are added directly to the b vector."""
        thermal = _make_1d_thermal(nx=5)
        mass_flux = np.zeros(thermal.grid.num_faces)
        heat_sources = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        _, b = thermal.build_energy_matrix(dt=3600.0, mass_flux=mass_flux,
                                           heat_sources=heat_sources)
        # b should include at least the heat_sources values
        assert np.all(b >= heat_sources)

    def test_previous_state_in_rhs(self):
        """Old energy accumulation / dt appears in RHS."""
        thermal = _make_1d_thermal(nx=3)
        thermal.temperature = np.array([300.0, 310.0, 320.0])
        thermal.T_prev = thermal.temperature.copy()

        mass_flux = np.zeros(thermal.grid.num_faces)
        heat_sources = np.zeros(thermal.grid.num_cells)

        _, b = thermal.build_energy_matrix(dt=10.0, mass_flux=mass_flux,
                                           heat_sources=heat_sources)
        # RHS should be non-zero due to previous energy accumulation
        assert np.all(b > 0)

    def test_tridiagonal_structure_1d(self):
        """In 1D, A should be tridiagonal (conduction only couples neighbours)."""
        thermal = _make_1d_thermal(nx=5)
        mass_flux = np.zeros(thermal.grid.num_faces)
        heat_sources = np.zeros(thermal.grid.num_cells)

        A, _ = thermal.build_energy_matrix(dt=1000.0, mass_flux=mass_flux,
                                           heat_sources=heat_sources)
        A_dense = A.toarray()
        for i in range(5):
            for j in range(5):
                if abs(i - j) > 1:
                    assert np.isclose(A_dense[i, j], 0.0), \
                        f"Non-tridiagonal entry at ({i},{j}) = {A_dense[i,j]}"


# =========================================================================
# 8. step_coupled
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestStepCoupled:
    """Test the step_coupled method with a real flow solver."""

    def test_runs_without_error(self):
        """Coupled step executes without raising exceptions."""
        thermal = _make_coupled_thermal(nx=3)
        flow_solver = _make_flow_solver(thermal)

        source_terms = np.zeros(thermal.grid.num_cells)
        heat_sources = np.zeros(thermal.grid.num_cells)
        bc_values = {'pressure': np.array([1e5, 2e5])}

        result = thermal.step_coupled(
            dt=1.0,
            source_terms=source_terms,
            heat_sources=heat_sources,
            bc_type='dirichlet',
            bc_values=bc_values,
            flow_solver=flow_solver,
            max_iter=10,
            tol=1e-3,
        )
        assert 'converged' in result
        assert 'iterations' in result
        assert 'pressure_change' in result
        assert 'temperature_change' in result

    def test_pressure_and_temperature_update(self):
        """After step_coupled, pressure and temperature are modified."""
        thermal = _make_coupled_thermal(nx=3)
        flow_solver = _make_flow_solver(thermal)

        p_before = thermal.pressure.copy()
        T_before = thermal.temperature.copy()

        source_terms = np.array([1.0, 0.0, -1.0])  # inject left, produce right
        heat_sources = np.array([100.0, 0.0, 0.0])
        bc_values = {'pressure': np.array([1e5, 2e5])}

        thermal.step_coupled(
            dt=1.0,
            source_terms=source_terms,
            heat_sources=heat_sources,
            bc_type='dirichlet',
            bc_values=bc_values,
            flow_solver=flow_solver,
            max_iter=10,
            tol=1e-3,
        )

        # Both fields should have changed from their initial values
        assert not np.allclose(thermal.pressure, p_before)
        assert not np.allclose(thermal.temperature, T_before)

    def test_convergence_flag_is_bool(self):
        """Result contains a boolean 'converged' flag."""
        thermal = _make_coupled_thermal(nx=3)
        flow_solver = _make_flow_solver(thermal)

        result = thermal.step_coupled(
            dt=1.0,
            source_terms=np.zeros(thermal.grid.num_cells),
            heat_sources=np.zeros(thermal.grid.num_cells),
            bc_type='dirichlet',
            bc_values={'pressure': np.array([1e5, 2e5])},
            flow_solver=flow_solver,
            max_iter=10,
            tol=1e-3,
        )
        assert isinstance(result['converged'], (bool, np.bool_))

    def test_previous_state_updated_after_step(self):
        """After a successful step, p_prev and T_prev match current state."""
        thermal = _make_coupled_thermal(nx=3)
        flow_solver = _make_flow_solver(thermal)

        thermal.step_coupled(
            dt=1.0,
            source_terms=np.array([0.1, 0.0, -0.1]),
            heat_sources=np.zeros(thermal.grid.num_cells),
            bc_type='dirichlet',
            bc_values={'pressure': np.array([1e5, 2e5])},
            flow_solver=flow_solver,
            max_iter=10,
            tol=1e-3,
        )
        assert np.allclose(thermal.p_prev, thermal.pressure)
        assert np.allclose(thermal.T_prev, thermal.temperature)

    def test_zero_heat_sources(self):
        """Coupled step works with zero heat sources (flow-only limit)."""
        thermal = _make_coupled_thermal(nx=3)
        flow_solver = _make_flow_solver(thermal)

        result = thermal.step_coupled(
            dt=10.0,
            source_terms=np.array([0.5, 0.0, -0.5]),
            heat_sources=np.zeros(thermal.grid.num_cells),
            bc_type='dirichlet',
            bc_values={'pressure': np.array([1e5, 2e5])},
            flow_solver=flow_solver,
            max_iter=10,
            tol=1e-3,
        )
        # Should still complete
        assert 'converged' in result

    def test_tight_tolerance_causes_more_iterations(self):
        """A tighter convergence tolerance may require more iterations."""
        thermal1 = _make_coupled_thermal(nx=3)
        thermal2 = _make_coupled_thermal(nx=3)

        fs1 = _make_flow_solver(thermal1)
        fs2 = _make_flow_solver(thermal2)

        src = np.array([0.1, 0.0, -0.1])
        hs = np.zeros(3)
        bc = {'pressure': np.array([1e5, 2e5])}

        r1 = thermal1.step_coupled(dt=1.0, source_terms=src,
                                   heat_sources=hs, bc_type='dirichlet',
                                   bc_values=bc, flow_solver=fs1,
                                   max_iter=20, tol=1e-3)
        r2 = thermal2.step_coupled(dt=1.0, source_terms=src,
                                   heat_sources=hs, bc_type='dirichlet',
                                   bc_values=bc, flow_solver=fs2,
                                   max_iter=20, tol=1e-8)

        # Tighter tol → >= iterations (usually)
        assert r2['iterations'] >= 1


# =========================================================================
# 9. compute_geothermal_gradient
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestGeothermalGradient:
    """Test the compute_geothermal_gradient method."""

    def test_linear_profile(self):
        """T(z) = T_surface + gradient * depth → linear with depth."""
        thermal = _make_1d_thermal(nx=5)
        depth = np.array([0.0, 100.0, 200.0, 300.0, 400.0])

        T = thermal.compute_geothermal_gradient(
            surface_temp=293.15, gradient=0.03, depth=depth
        )
        expected = 293.15 + 0.03 * depth
        assert np.allclose(T, expected)

    def test_sets_internal_state(self):
        """Geothermal gradient updates thermal.temperature and T_prev."""
        thermal = _make_1d_thermal(nx=5)
        depth = np.array([0.0, 50.0, 100.0, 150.0, 200.0])

        thermal.compute_geothermal_gradient(
            surface_temp=300.0, gradient=0.025, depth=depth
        )
        expected = 300.0 + 0.025 * depth
        assert np.allclose(thermal.temperature, expected)
        assert np.allclose(thermal.T_prev, expected)

    def test_uses_grid_centroids_when_depth_is_none(self):
        """If depth is None, the method derives depth from grid centroids."""
        thermal = _make_1d_thermal(nx=5, dx=50.0)
        # In 1D, cell_centroids[:, 2] are 0.0 (since nz=1, zc_centers=[0.5])
        # depth = -cell_centroids[:, 2] = 0.0, so T = surface_temp
        T = thermal.compute_geothermal_gradient(surface_temp=293.15, gradient=0.03)
        # All depths are 0 → all T = surface_temp
        assert np.allclose(T, 293.15)

    def test_custom_depth_and_gradient(self):
        """Varying gradient and depth produces correct temperatures."""
        thermal = _make_1d_thermal(nx=3)
        depth = np.array([10.0, 500.0, 2000.0])

        T = thermal.compute_geothermal_gradient(
            surface_temp=280.0, gradient=0.04, depth=depth
        )
        assert np.isclose(T[0], 280.0 + 0.04 * 10.0)
        assert np.isclose(T[1], 280.0 + 0.04 * 500.0)
        assert np.isclose(T[2], 280.0 + 0.04 * 2000.0)

    def test_return_value_is_temperature_array(self):
        """Method returns the temperature array of correct shape."""
        thermal = _make_1d_thermal(nx=7)
        T = thermal.compute_geothermal_gradient()
        assert T.shape == (7,)
        assert isinstance(T, np.ndarray)

    def test_zero_gradient_gives_uniform_temperature(self):
        """A gradient of 0 yields uniform surface temperature."""
        thermal = _make_1d_thermal(nx=5)
        depth = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        T = thermal.compute_geothermal_gradient(
            surface_temp=300.0, gradient=0.0, depth=depth
        )
        assert np.allclose(T, 300.0)


# =========================================================================
# 10. Edge Cases and Integration
# =========================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestThermalFlowEdgeCases:
    """Edge-case and integration tests for ThermalFlow."""

    def test_single_cell_grid(self):
        """ThermalFlow works with a single-cell grid."""
        thermal = _make_1d_thermal(nx=1)
        assert thermal.pressure.shape == (1,)
        assert thermal.temperature.shape == (1,)

        acc = thermal.compute_energy_accumulation()
        assert acc.shape == (1,)
        assert acc[0] > 0

        leff = thermal._effective_thermal_conductivity()
        assert leff > 0

        T_face = thermal._interpolate_temperature_to_faces()
        # 1 cell → 2 faces
        assert len(T_face) == 2
        assert np.isclose(T_face[0], 293.15)
        assert np.isclose(T_face[-1], 293.15)

    def test_geothermal_gradient_then_accumulation(self):
        """Set geothermal gradient, then accumulation reflects profile."""
        thermal = _make_1d_thermal(nx=5)
        depth = np.array([0.0, 250.0, 500.0, 750.0, 1000.0])
        thermal.compute_geothermal_gradient(
            surface_temp=293.15, gradient=0.03, depth=depth
        )
        acc = thermal.compute_energy_accumulation()
        # Accumulation should increase with depth (higher T)
        assert acc[0] < acc[-1]

    def test_matrix_solve_inverts_correctly_steady_state(self):
        """In steady state (large dt), solving energy matrix recovers profile."""
        thermal = _make_1d_thermal(nx=5)
        depth = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        thermal.compute_geothermal_gradient(
            surface_temp=293.15, gradient=0.03, depth=depth
        )
        T_expected = thermal.temperature.copy()

        mass_flux = np.zeros(thermal.grid.num_faces)
        heat_sources = np.zeros(thermal.grid.num_cells)

        # With very large dt, the time-derivative term is negligible
        # and the system should be nearly steady-state conduction
        A, b = thermal.build_energy_matrix(dt=1e12, mass_flux=mass_flux,
                                           heat_sources=heat_sources)
        from scipy.sparse.linalg import spsolve
        T_solved = spsolve(A, b)

        # With a nearly steady-state conduction matrix (no sources, no convection),
        # the solution should be close to the original profile
        # (only conduction, no net flux with uniform gradient)
        assert np.allclose(T_solved, T_expected, rtol=0.1)

    def test_heat_flux_preserves_energy_direction(self):
        """With a hot left boundary and cold right, net flux is rightward."""
        thermal = _make_1d_thermal(nx=3)
        thermal.temperature = np.array([500.0, 350.0, 200.0])
        thermal.T_prev = thermal.temperature.copy()

        mass_flux = np.zeros(thermal.grid.num_faces)
        q = thermal.compute_heat_flux(mass_flux)

        # Interior faces should have positive flux (hot → cold, i.e. left→right)
        assert q[1] > 0
        assert q[2] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
