"""Tests for MultiphaseFlow — Corey perm, phase equilibrium, SIM step, gradient."""

import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

if NUMPY_AVAILABLE:
    from garuda.core.grid import StructuredGrid
    from garuda.core.rock_properties import RockProperties
    from garuda.core.fluid_properties import FluidProperties
    from garuda.core.iapws_properties import WaterSteamProperties
    from garuda.physics.multiphase import MultiphaseFlow, MultiphaseState


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestRelativePermeability:
    def setup_method(self):
        self.rock = RockProperties(porosity=0.12, permeability=100, permeability_unit='md')
        self.fluid = FluidProperties(fluid_type='geothermal')
        self.grid = StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=10, dz=1)
        self.flow = MultiphaseFlow(self.grid, self.rock, self.fluid)

    def test_endpoints_liquid(self):
        krw, krs = self.flow.relative_permeability(np.ones(5))
        assert np.allclose(krw, 1.0), f"krw={krw}"
        assert np.allclose(krs, 0.0), f"krs={krs}"

    def test_endpoints_steam(self):
        krw, krs = self.flow.relative_permeability(np.zeros(5))
        assert np.allclose(krw, 0.0)
        assert np.allclose(krs, 1.0)

    def test_monotonic_krw(self):
        Sw = np.linspace(0, 1, 20)
        krw, _ = self.flow.relative_permeability(Sw)
        assert np.all(np.diff(krw) >= 0), "krw should increase monotonically"

    def test_monotonic_krs(self):
        Sw = np.linspace(0, 1, 20)
        _, krs = self.flow.relative_permeability(Sw)
        assert np.all(np.diff(krs) <= 0), "krs should decrease monotonically"

    def test_sum_le_one(self):
        Sw = np.linspace(0, 1, 10)
        krw, krs = self.flow.relative_permeability(Sw)
        assert np.all(krw + krs <= 1.0 + 1e-10)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestPhaseEquilibrium:
    def setup_method(self):
        self.grid = StructuredGrid(nx=3, ny=1, nz=1, dx=100, dy=10, dz=1)
        self.rock = RockProperties(porosity=0.12, permeability=100, permeability_unit='md')
        self.fluid = FluidProperties(fluid_type='geothermal')
        self.flow = MultiphaseFlow(self.grid, self.rock, self.fluid)

    def test_liquid_subcooled(self):
        p = np.full(3, 10e5)   # 10 bar
        T = np.full(3, 350.0)  # ~77°C, well below saturation at 10 bar (~180°C)
        self.flow.set_initial_state(p, T, np.zeros(3))
        self.flow.apply_phase_equilibrium()
        assert np.allclose(self.flow.state.saturation, 0.0)

    def test_two_phase_near_saturation(self):
        # 10 bar → T_sat ≈ 453K (180°C). Set T slightly above.
        p = np.full(3, 10e5)
        self.flow.set_initial_state(p, np.full(3, 460.0), np.full(3, 0.3))
        self.flow.apply_phase_equilibrium()
        assert np.all(self.flow.state.saturation >= 0.0)
        assert np.all(self.flow.state.saturation <= 1.0)

    def test_saturation_clamped_zero_one(self):
        p = np.full(3, 10e5)
        self.flow.set_initial_state(p, np.full(3, 460.0), np.array([-0.5, 0.5, 1.5]))
        self.flow.apply_phase_equilibrium()
        assert np.all(self.flow.state.saturation >= 0.0)
        assert np.all(self.flow.state.saturation <= 1.0)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestStep:
    def setup_method(self):
        self.grid = StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=10, dz=1)
        self.rock = RockProperties(porosity=0.2, permeability=100, permeability_unit='md')
        self.fluid = FluidProperties(fluid_type='water')
        self.flow = MultiphaseFlow(self.grid, self.rock, self.fluid)

    def test_runs(self):
        p = np.full(5, 200e5)
        T = np.full(5, 350.0)
        self.flow.set_initial_state(p, T, np.zeros(5))
        source = np.zeros(5)
        bc = np.array([200e5, 100e5])
        result = self.flow.step(86400, source, bc_values=bc)
        assert result['converged'] in (True, False)
        assert result['iterations'] >= 1

    def test_production_lowers_pressure(self):
        p = np.full(5, 200e5)
        T = np.full(5, 350.0)
        self.flow.set_initial_state(p, T, np.zeros(5))
        source = np.zeros(5)
        source[2] = -50.0
        bc = np.array([200e5, 100e5])
        self.flow.step(86400, source, bc_values=bc)
        assert self.flow.state.pressure[2] < 200e5

    def test_converges_small_dt(self):
        p = np.full(5, 200e5)
        T = np.full(5, 350.0)
        self.flow.set_initial_state(p, T, np.zeros(5))
        result = self.flow.step(100, np.zeros(5), max_iter=20, tol=1e-4,
                                 bc_values=np.array([200e5, 100e5]))
        assert result['converged']


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestInitialState:
    def setup_method(self):
        self.grid = StructuredGrid(nx=3, ny=1, nz=1, dx=100, dy=10, dz=1)
        self.rock = RockProperties(porosity=0.15, permeability=100, permeability_unit='md')
        self.fluid = FluidProperties(fluid_type='water')
        self.flow = MultiphaseFlow(self.grid, self.rock, self.fluid)

    def test_sets_all_arrays(self):
        p = np.full(3, 250e5)
        T = np.full(3, 500.0)
        self.flow.set_initial_state(p, T, np.full(3, 0.05))
        assert len(self.flow.state.pressure) == 3
        assert len(self.flow.state.temperature) == 3
        assert len(self.flow.state.saturation) == 3
        assert len(self.flow.state.enthalpy) == 3

    def test_prev_state_initialized(self):
        p = np.full(3, 250e5)
        T = np.full(3, 500.0)
        self.flow.set_initial_state(p, T)
        assert np.allclose(self.flow.state.prev_pressure, p)
        assert np.allclose(self.flow.state.prev_temperature, T)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestGeothermalGradient:
    def setup_method(self):
        self.grid = StructuredGrid(nx=2, ny=2, nz=2, dx=100, dy=100, dz=50)
        self.rock = RockProperties(porosity=0.12, permeability=100, permeability_unit='md')
        self.fluid = FluidProperties(fluid_type='geothermal')
        self.flow = MultiphaseFlow(self.grid, self.rock, self.fluid)

    def test_linear_increase_with_depth(self):
        T = self.flow.compute_geothermal_gradient(298.15, 0.06)
        # Grid z grows from 0 upward; treat z as depth below the top of the
        # reservoir so temperature must increase with z.
        depths = self.grid.cell_centroids[:, 2]
        assert np.allclose(T, 298.15 + 0.06 * depths)
        # Strict monotonicity along z
        top = self.grid.get_cell_index(0, 0, 0)
        bottom = self.grid.get_cell_index(0, 0, self.grid.nz - 1)
        assert T[bottom] > T[top]

    def test_tropical_defaults(self):
        T = self.flow.compute_geothermal_gradient()
        # 25°C surface + 60°C/km — avg across domain should be warm
        assert np.mean(T) > 293.0, f"avg T={np.mean(T):.1f}K expected >293K"

    def test_custom_surface_temp(self):
        T = self.flow.compute_geothermal_gradient(280.0, 0.05)
        depths = self.grid.cell_centroids[:, 2]
        assert np.allclose(T, 280.0 + 0.05 * depths)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestGetSummary:
    def setup_method(self):
        self.grid = StructuredGrid(nx=3, ny=1, nz=1, dx=100, dy=10, dz=1)
        self.rock = RockProperties(porosity=0.15, permeability=100, permeability_unit='md')
        self.fluid = FluidProperties(fluid_type='water')
        self.flow = MultiphaseFlow(self.grid, self.rock, self.fluid)

    def test_all_keys(self):
        self.flow.set_initial_state(np.full(3, 200e5), np.full(3, 350.0))
        s = self.flow.get_summary()
        for k in ('p_min', 'p_max', 'p_avg', 'T_min', 'T_max', 'T_avg',
                   'S_min', 'S_max', 'S_avg', 'phase'):
            assert k in s

    def test_phase_liquid_dominated(self):
        self.flow.set_initial_state(np.full(3, 200e5), np.full(3, 350.0), np.zeros(3))
        assert self.flow.get_summary()['phase'] == 'liquid-dominated'

    def test_values_reasonable(self):
        self.flow.set_initial_state(np.full(3, 200e5), np.full(3, 350.0))
        s = self.flow.get_summary()
        assert 100 < s['p_avg'] < 500
        assert 0 < s['T_avg'] < 200
