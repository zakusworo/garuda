"""
Test suite for GARUDA physics modules.

Tests IAPWS-97 properties and Peaceman well models.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

if NUMPY_AVAILABLE:
    from garuda.core.iapws_properties import WaterSteamProperties, IAPWSFluidProperties
    from garuda.physics.well_models import (
        PeacemanWell, WellManager, WellParameters, WellOperatingConditions
    )


# =============================================================================
# IAPWS-97 PROPERTY TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestIAPWSProperties:
    """Test IAPWS-97 water/steam properties."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.props = WaterSteamProperties()
    
    def test_saturation_pressure(self):
        """Test saturation pressure calculation."""
        # At 100°C (373.15 K), p_sat should be ~0.101 MPa (1 atm)
        p_sat = self.props.saturation_pressure(373.15)
        
        # Allow for approximation error
        assert 0.09 < p_sat < 0.12, f"Expected ~0.1 MPa, got {p_sat}"
    
    def test_saturation_pressure_geothermal(self):
        """Test saturation pressure in geothermal range."""
        # At 250°C (523.15 K), p_sat should be ~4 MPa
        p_sat = self.props.saturation_pressure(523.15)
        
        # Geothermal range: should be between 1-10 MPa
        assert 1.0 < p_sat < 10.0, f"Expected 1-10 MPa, got {p_sat}"
    
    def test_density_liquid(self):
        """Test liquid water density."""
        # At 20°C, 0.1 MPa: rho ≈ 998 kg/m³
        rho = self.props.density(0.1, 293.15)
        
        assert 990 < rho < 1010, f"Expected ~998 kg/m³, got {rho}"
    
    def test_density_geothermal(self):
        """Test density at geothermal conditions."""
        # At 250°C, 10 MPa: rho ≈ 800 kg/m³
        rho = self.props.density(10.0, 523.15)
        
        # Should be less than room temperature water
        assert 700 < rho < 900, f"Expected 700-900 kg/m³, got {rho}"
    
    def test_viscosity_liquid(self):
        """Test liquid water viscosity."""
        # At 20°C: mu ≈ 1.0 cP = 0.001 Pa·s
        mu = self.props.viscosity(0.1, 293.15)
        
        assert 0.0008 < mu < 0.0012, f"Expected ~0.001 Pa·s, got {mu}"
    
    def test_viscosity_geothermal(self):
        """Test viscosity at geothermal conditions."""
        # At 250°C: mu ≈ 0.1 cP = 0.0001 Pa·s
        mu = self.props.viscosity(10.0, 523.15)
        
        # Should be much lower than room temperature
        assert 0.00005 < mu < 0.0003, f"Expected ~0.0001 Pa·s, got {mu}"
    
    def test_phase_identification(self):
        """Test phase identification."""
        # Liquid: high pressure, moderate temperature
        phase = self.props.phase(20.0, 500.0)
        assert phase == 'liquid', f"Expected liquid, got {phase}"
        
        # Vapor: low pressure, high temperature
        phase = self.props.phase(0.5, 500.0)
        assert phase == 'vapor', f"Expected vapor, got {phase}"
    
    def test_get_all_properties(self):
        """Test getting all properties at once."""
        props = self.props.get_all_properties(15.0, 550.0)
        
        assert 'density' in props
        assert 'viscosity' in props
        assert 'enthalpy' in props
        assert 'specific_heat_cp' in props
        assert 'thermal_conductivity' in props
        assert 'phase' in props
        
        # Check reasonable values
        assert 600 < props['density'] < 900
        assert 5e-05 < props['viscosity'] < 0.001
        assert props['enthalpy'] > 0


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestIAPWSFluidProperties:
    """Test IAPWS fluid properties wrapper for GARUDA."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fluid = IAPWSFluidProperties()
    
    def test_get_properties(self):
        """Test property retrieval for solver."""
        p = 20e6  # 20 MPa
        T = 550.0  # K
        
        mu, rho = self.fluid.get_properties(p, T)
        
        assert mu > 0
        assert rho > 0
        assert mu < 0.01  # Should be < 10 cP
        assert 500 < rho < 1000  # kg/m³
    
    def test_get_enthalpy(self):
        """Test enthalpy calculation."""
        p = 15e6
        T = 500.0
        
        h = self.fluid.get_enthalpy(p, T)
        
        # Should be in J/kg, reasonable range for liquid water
        assert 500e3 < h < 2000e3


# =============================================================================
# PEACEMAN WELL MODEL TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestPeacemanWell:
    """Test Peaceman well model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        params = WellParameters(
            name='TEST-1',
            cell_index=10,
            well_radius=0.12,
            skin_factor=0.0,
        )
        
        operating = WellOperatingConditions(
            constraint_type='rate',
            target_value=-10.0,  # 10 kg/s production
            min_bhp=100e5,
            max_bhp=300e5,
            max_rate=20.0,
        )
        
        self.well = PeacemanWell(params, operating)
    
    def test_effective_radius(self):
        """Test effective wellblock radius calculation."""
        dx = dy = 100.0
        kx = ky = 1e-13  # ~100 md
        
        r_0 = self.well.compute_effective_radius(dx, dy, kx, ky)
        
        # Should be ~10-20% of grid block size
        assert 10 < r_0 < 30, f"Expected 10-30 m, got {r_0}"
    
    def test_productivity_index(self):
        """Test productivity index calculation."""
        k = 100e-15  # 100 md
        mu = 1e-3  # 1 cP
        dx = dy = 100.0
        dz = 10.0
        
        PI = self.well.compute_productivity_index(k, mu, dx=dx, dy=dy, dz=dz)
        
        # PI should be positive
        assert PI > 0, f"Expected positive PI, got {PI}"
        
        # Typical range: 1e-10 to 1e-8 kg/(s·Pa)
        assert 1e-12 < PI < 1e-6, f"PI {PI} out of expected range"
    
    def test_compute_rate(self):
        """Test rate calculation from pressure difference.

        Sign convention: negative rate = production (mass leaves cell),
        positive rate = injection (mass enters cell). p_wf < p_cell creates
        drawdown that produces, so q must be negative.
        """
        # First compute PI
        k = 100e-15
        mu = 1e-3
        self.well.compute_productivity_index(k, mu, dx=100, dy=100, dz=10)

        # Drawdown: p_wf below p_cell -> production
        p_cell = 200e5
        p_wf = 190e5
        rho = 1000.0

        q = self.well.compute_rate(p_cell, p_wf, rho)
        assert q < 0, f"Expected negative (production) rate, got {q}"

        # Overpressure: p_wf above p_cell -> injection
        q_inj = self.well.compute_rate(p_cell, p_cell + 10e5, rho)
        assert q_inj > 0, f"Expected positive (injection) rate, got {q_inj}"
    
    def test_skin_factor_effect(self):
        """Test skin factor effect on productivity."""
        k = 100e-15
        mu = 1e-3
        
        # Zero skin
        params_no_skin = WellParameters('TEST', 10, skin_factor=0.0)
        well_no_skin = PeacemanWell(params_no_skin, WellOperatingConditions())
        PI_no_skin = well_no_skin.compute_productivity_index(k, mu, dx=100, dy=100, dz=10)
        
        # Positive skin (damage)
        params_damage = WellParameters('TEST', 10, skin_factor=5.0)
        well_damage = PeacemanWell(params_damage, WellOperatingConditions())
        PI_damage = well_damage.compute_productivity_index(k, mu, dx=100, dy=100, dz=10)
        
        # Negative skin (stimulation)
        params_stim = WellParameters('TEST', 10, skin_factor=-3.0)
        well_stim = PeacemanWell(params_stim, WellOperatingConditions())
        PI_stim = well_stim.compute_productivity_index(k, mu, dx=100, dy=100, dz=10)
        
        # Damaged well should have lower PI
        assert PI_damage < PI_no_skin, "Damaged well should have lower PI"
        
        # Stimulated well should have higher PI
        assert PI_stim > PI_no_skin, "Stimulated well should have higher PI"
    
    def test_apply_constraints_rate(self):
        """Test rate constraint application."""
        k = 100e-15
        mu = 1e-3
        self.well.compute_productivity_index(k, mu, dx=100, dy=100, dz=10)
        
        p_cell = 200e5
        rho = 1000.0
        
        rate, bhp = self.well.apply_constraints(p_cell, rho)
        
        # Rate should be close to target (10 kg/s)
        assert 8 < abs(rate) < 12, f"Expected ~10 kg/s, got {rate}"
        
        # BHP should be within limits
        assert self.well.operating.min_bhp <= bhp <= self.well.operating.max_bhp
    
    def test_apply_constraints_pressure(self):
        """Test pressure constraint application."""
        # Switch to pressure constraint
        self.well.operating.constraint_type = 'pressure'
        self.well.operating.target_value = 150e5  # 150 bar
        
        k = 100e-15
        mu = 1e-3
        self.well.compute_productivity_index(k, mu, dx=100, dy=100, dz=10)
        
        p_cell = 200e5
        rho = 1000.0
        
        rate, bhp = self.well.apply_constraints(p_cell, rho)
        
        # BHP should be at target
        assert np.isclose(bhp, 150e5, rtol=0.1), f"Expected 150 bar, got {bhp/1e5} bar"


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestWellManager:
    """Test well manager for multiple wells."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = WellManager()
    
    def test_add_well(self):
        """Test adding wells."""
        self.manager.add_well(
            name='PROD-1',
            cell_index=50,
            well_type='producer',
            target_rate=45.0,
        )
        
        assert 'PROD-1' in self.manager.wells
        assert len(self.manager.wells) == 1
    
    def test_add_multiple_wells(self):
        """Test adding multiple wells."""
        self.manager.add_well('PROD-1', 50, 'producer', 45.0)
        self.manager.add_well('PROD-2', 75, 'producer', 45.0)
        self.manager.add_well('INJ-1', 100, 'injector', 40.0)
        
        assert len(self.manager.wells) == 3
        
        summary = self.manager.get_well_summary()
        assert summary['total_wells'] == 3
        assert summary['producers'] == 2
        assert summary['injectors'] == 1
    
    def test_remove_well(self):
        """Test removing wells."""
        self.manager.add_well('TEST-1', 10, 'producer', 50.0)
        self.manager.remove_well('TEST-1')
        
        assert 'TEST-1' not in self.manager.wells
    
    def test_compute_well_rates(self):
        """Test computing well rates from pressure field."""
        # Add wells
        self.manager.add_well('PROD-1', 5, 'producer', 45.0)
        self.manager.add_well('INJ-1', 15, 'injector', 40.0)
        
        # Create simple grid mock
        class MockGrid:
            num_cells = 20
            dx = dy = 100.0
            dz = 10.0
            permiability = np.ones((num_cells, 3, 3)) * 100e-15  # 100 md
        
        grid = MockGrid()
        pressure = np.ones(grid.num_cells) * 200e5  # 200 bar
        density = 1000.0
        viscosity = 1e-3
        
        source_terms = self.manager.compute_well_rates(
            grid, pressure, density, viscosity
        )
        
        # Should have non-zero source terms
        assert np.any(source_terms != 0)
        
        # Producer should be negative, injector positive
        assert source_terms[5] < 0, "Producer should have negative rate"
        assert source_terms[15] > 0, "Injector should have positive rate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
