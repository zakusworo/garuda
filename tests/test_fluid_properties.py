"""
Comprehensive tests for FluidProperties dataclass.

Covers: defaults per fluid type, custom overrides, density, viscosity,
formation volume factor, total compressibility, scalar and array inputs.
"""

import pytest
import numpy as np
from garuda.core.fluid_properties import FluidProperties


# ---------------------------------------------------------------------------
# 1. Default property values per fluid type
# ---------------------------------------------------------------------------

class TestDefaults:
    """Verify default mu, rho, rho_ref, cp, beta, c_fluid after __post_init__."""

    def test_water_defaults(self):
        fp = FluidProperties(fluid_type='water')
        assert fp.mu == 1e-3
        assert fp.rho == 998
        assert fp.rho_ref == 998
        assert fp.cp == 4182
        assert fp.beta == 2.1e-4
        assert fp.c_fluid == 4.4e-10

    def test_oil_defaults(self):
        fp = FluidProperties(fluid_type='oil')
        assert fp.mu == 5e-3
        assert fp.rho == 850
        assert fp.rho_ref == 850

    def test_gas_defaults(self):
        fp = FluidProperties(fluid_type='gas')
        assert fp.mu == 1.8e-5
        assert fp.rho == 1.2
        assert fp.rho_ref == 1.2

    def test_geothermal_defaults(self):
        fp = FluidProperties(fluid_type='geothermal')
        assert fp.mu == 1.5e-4
        assert fp.rho == 800
        assert fp.rho_ref == 1000   # differs from rho — intentional
        assert fp.cp == 4500
        assert fp.beta == 5e-4

    def test_default_fluid_type_is_water(self):
        fp = FluidProperties()
        assert fp.fluid_type == 'water'
        assert fp.mu == 1e-3
        assert fp.rho == 998


# ---------------------------------------------------------------------------
# 2. Custom values override defaults (where applicable)
# ---------------------------------------------------------------------------

class TestCustomValues:
    """Pass explicit values and ensure they are not clobbered by post_init."""

    def test_custom_mu_rho_water(self):
        fp = FluidProperties(fluid_type='water', mu=2e-3, rho=1050)
        assert fp.mu == 2e-3
        assert fp.rho == 1050
        assert fp.rho_ref == 1050   # rho_ref mirrors rho when not given

    def test_explicit_rho_ref(self):
        fp = FluidProperties(fluid_type='water', rho_ref=1000)
        assert fp.rho_ref == 1000
        assert fp.rho == 998        # rho stays at default

    def test_geothermal_always_overwrites_cp_beta(self):
        # __post_init__ unconditionally sets cp=4500, beta=5e-4 for geothermal
        fp = FluidProperties(fluid_type='geothermal', cp=9999, beta=9e-4)
        assert fp.cp == 4500
        assert fp.beta == 5e-4

    def test_geothermal_custom_rho_ref_respected(self):
        fp = FluidProperties(fluid_type='geothermal', rho_ref=950)
        assert fp.rho_ref == 950
        assert fp.rho == 800

    def test_custom_c_fluid(self):
        fp = FluidProperties(fluid_type='water', c_fluid=1e-9)
        assert fp.c_fluid == 1e-9


# ---------------------------------------------------------------------------
# 3. Density
# ---------------------------------------------------------------------------

class TestDensity:
    """Test density(pressure, temperature=None)."""

    def test_at_reference_conditions_returns_rho_ref(self):
        fp = FluidProperties(fluid_type='water')
        rho = fp.density(pressure=fp.p_ref, temperature=fp.T_ref)
        assert rho == pytest.approx(fp.rho_ref)

    def test_increases_with_pressure(self):
        fp = FluidProperties(fluid_type='water')
        rho_low = fp.density(pressure=fp.p_ref, temperature=fp.T_ref)
        rho_high = fp.density(pressure=1e7, temperature=fp.T_ref)
        assert rho_high > rho_low

    def test_decreases_with_temperature(self):
        fp = FluidProperties(fluid_type='water')
        rho_cold = fp.density(pressure=fp.p_ref, temperature=fp.T_ref)
        rho_hot = fp.density(pressure=fp.p_ref, temperature=350.0)
        assert rho_hot < rho_cold

    def test_temperature_none_defaults_to_T_ref(self):
        fp = FluidProperties(fluid_type='water')
        rho1 = fp.density(pressure=1e5, temperature=None)
        rho2 = fp.density(pressure=1e5, temperature=fp.T_ref)
        assert rho1 == pytest.approx(rho2)

    def test_scalar_input_returns_float(self):
        fp = FluidProperties(fluid_type='water')
        rho = fp.density(pressure=1e5, temperature=300.0)
        assert isinstance(rho, float)

    def test_array_input_returns_ndarray(self):
        fp = FluidProperties(fluid_type='water')
        p = np.array([1e5, 2e5, 3e5])
        T = np.array([293.15, 300.0, 310.0])
        rho = fp.density(pressure=p, temperature=T)
        assert isinstance(rho, np.ndarray)
        assert rho.shape == (3,)

    def test_array_pressure_monotonically_increasing(self):
        fp = FluidProperties(fluid_type='water')
        p = np.linspace(1e5, 1e7, 50)
        rho = fp.density(pressure=p, temperature=fp.T_ref)
        assert np.all(np.diff(rho) > 0)

    def test_array_temperature_monotonically_decreasing(self):
        fp = FluidProperties(fluid_type='water')
        T = np.linspace(293.15, 500, 50)
        rho = fp.density(pressure=fp.p_ref, temperature=T)
        assert np.all(np.diff(rho) < 0)

    def test_geothermal_density_at_reference(self):
        fp = FluidProperties(fluid_type='geothermal')
        # rho_ref=1000, different from self.rho=800
        rho = fp.density(pressure=fp.p_ref, temperature=fp.T_ref)
        assert rho == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# 4. Viscosity
# ---------------------------------------------------------------------------

class TestViscosity:
    """Test viscosity(temperature=None)."""

    def test_at_reference_returns_mu(self):
        fp = FluidProperties(fluid_type='water')
        mu = fp.viscosity(temperature=fp.T_ref)
        assert mu == pytest.approx(1e-3)

    def test_no_temperature_returns_mu_directly(self):
        # Short-circuit: when mu is not None and temperature is None,
        # self.mu is returned directly.
        fp = FluidProperties(fluid_type='oil')
        mu = fp.viscosity()
        assert mu == pytest.approx(5e-3)

    def test_decreases_with_increasing_temperature(self):
        fp = FluidProperties(fluid_type='water')
        mu_cold = fp.viscosity(temperature=293.15)
        mu_hot = fp.viscosity(temperature=350.0)
        assert mu_hot < mu_cold

    def test_scalar_input_returns_float(self):
        fp = FluidProperties(fluid_type='water')
        mu = fp.viscosity(temperature=300.0)
        assert isinstance(mu, float)

    def test_array_input_returns_ndarray(self):
        fp = FluidProperties(fluid_type='water')
        T = np.array([293.15, 300.0, 310.0, 350.0])
        mu = fp.viscosity(temperature=T)
        assert isinstance(mu, np.ndarray)
        assert mu.shape == (4,)

    def test_array_monotonically_decreasing(self):
        fp = FluidProperties(fluid_type='water')
        T = np.linspace(293.15, 500, 50)
        mu = fp.viscosity(temperature=T)
        assert np.all(np.diff(mu) < 0)

    def test_higher_temperature_produces_lower_viscosity_for_all_fluids(self):
        for ftype in ('water', 'oil', 'gas', 'geothermal'):
            fp = FluidProperties(fluid_type=ftype)
            mu_cold = fp.viscosity(temperature=293.15)
            mu_hot = fp.viscosity(temperature=373.15)
            assert mu_hot < mu_cold, f"{ftype}: {mu_hot} >= {mu_cold}"


# ---------------------------------------------------------------------------
# 5. Formation Volume Factor
# ---------------------------------------------------------------------------

class TestFormationVolumeFactor:
    """Test formation_volume_factor(pressure, temperature=None)."""

    def test_at_reference_conditions_is_one(self):
        fp = FluidProperties(fluid_type='water')
        B = fp.formation_volume_factor(pressure=fp.p_ref, temperature=fp.T_ref)
        assert B == pytest.approx(1.0)

    def test_decreases_with_pressure(self):
        fp = FluidProperties(fluid_type='water')
        B_low = fp.formation_volume_factor(pressure=fp.p_ref, temperature=fp.T_ref)
        B_high = fp.formation_volume_factor(pressure=1e7, temperature=fp.T_ref)
        assert B_high < B_low

    def test_increases_with_temperature(self):
        # Hotter fluid is less dense → same mass occupies more volume → B > 1
        fp = FluidProperties(fluid_type='water')
        B_cold = fp.formation_volume_factor(pressure=fp.p_ref, temperature=fp.T_ref)
        B_hot = fp.formation_volume_factor(pressure=fp.p_ref, temperature=350.0)
        assert B_hot > B_cold

    def test_scalar_input_returns_float(self):
        fp = FluidProperties(fluid_type='water')
        B = fp.formation_volume_factor(pressure=1e5, temperature=300.0)
        assert isinstance(B, float)

    def test_array_input_returns_ndarray(self):
        fp = FluidProperties(fluid_type='water')
        p = np.linspace(1e5, 1e7, 20)
        B = fp.formation_volume_factor(pressure=p, temperature=fp.T_ref)
        assert isinstance(B, np.ndarray)
        assert np.all(np.diff(B) < 0)  # decreasing with pressure

    def test_geothermal_fvf_at_ref_is_one(self):
        fp = FluidProperties(fluid_type='geothermal')
        B = fp.formation_volume_factor(pressure=fp.p_ref, temperature=fp.T_ref)
        assert B == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. Total Compressibility
# ---------------------------------------------------------------------------

class TestTotalCompressibility:
    """Test total_compressibility(pressure=None, temperature=None)."""

    def test_returns_c_fluid(self):
        fp = FluidProperties(fluid_type='water')
        ct = fp.total_compressibility()
        assert ct == 4.4e-10

    def test_ignores_pressure_and_temperature_arguments(self):
        fp = FluidProperties(fluid_type='water')
        ct = fp.total_compressibility(pressure=1e7, temperature=500.0)
        assert ct == 4.4e-10

    def test_respects_custom_c_fluid(self):
        fp = FluidProperties(fluid_type='water', c_fluid=1e-9)
        ct = fp.total_compressibility()
        assert ct == 1e-9

    def test_scalar_return_type(self):
        fp = FluidProperties(fluid_type='water')
        ct = fp.total_compressibility()
        assert isinstance(ct, float)


# ---------------------------------------------------------------------------
# 7. Edge cases / regression guards
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Miscellaneous edge cases and cross-checks."""

    def test_density_formula_correctness(self):
        """Spot-check density against the exact formula."""
        fp = FluidProperties(fluid_type='water', c_fluid=1e-9, beta=1e-4,
                             rho_ref=1000.0)
        p = 2e5
        T = 300.0
        expected = 1000.0 * np.exp(1e-9 * (2e5 - 1e5) - 1e-4 * (300.0 - 293.15))
        actual = fp.density(pressure=p, temperature=T)
        assert actual == pytest.approx(expected)

    def test_viscosity_formula_correctness(self):
        """Spot-check viscosity against the exact formula."""
        fp = FluidProperties(fluid_type='water', mu=1e-3)
        T = 350.0
        expected = 1e-3 * np.exp(0.02 * (293.15 - 350.0))
        actual = fp.viscosity(temperature=T)
        assert actual == pytest.approx(expected)

    def test_unknown_fluid_type_uses_water_defaults(self):
        """Fluid type not in (water, oil, gas, geothermal) skips post_init branches."""
        fp = FluidProperties(fluid_type='brine', mu=None, rho=None)
        # mu and rho stay None, rho_ref stays None
        assert fp.mu is None
        assert fp.rho is None
        assert fp.rho_ref is None
        # cp, beta, c_fluid keep class defaults (same as water)
        assert fp.cp == 4182
        assert fp.beta == 2.1e-4
        assert fp.c_fluid == 4.4e-10
