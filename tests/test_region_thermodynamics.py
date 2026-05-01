"""Tests for region-based thermodynamics."""

import numpy as np
import pytest

from garuda.core.region_thermodynamics import (
    CRITICAL_PRESSURE_PA,
    CRITICAL_TEMPERATURE_K,
    FluidThermoState,
    RegionThermodynamics,
    SaturationCurve,
    SteamRegion,
    SupercriticalRegion,
    WaterRegion,
)


class TestWaterRegion:
    def test_density(self):
        water = WaterRegion()
        rho = water.density(1e5, 300.0)  # near atmospheric
        assert rho > 900.0  # kg/m3 for liquid water

    def test_enthalpy(self):
        water = WaterRegion()
        h = water.enthalpy(1e5, 300.0)
        assert h > 100_000.0  # J/kg for liquid water at ~27C

    def test_viscosity(self):
        water = WaterRegion()
        mu = water.viscosity(300.0)
        assert 1e-4 < mu < 2e-3  # Pa.s for liquid water

    def test_dh_dt(self):
        water = WaterRegion()
        cp = water.dh_dt(1e5, 300.0)
        assert pytest.approx(cp, rel=0.1) == 4_200.0

    def test_vectorized(self):
        water = WaterRegion()
        T = np.array([300.0, 350.0, 400.0])
        p = np.full_like(T, 1e5)
        rho = water.density(p, T)
        assert rho.shape == T.shape
        assert np.all(rho > 900.0)


class TestSteamRegion:
    def test_density(self):
        steam = SteamRegion()
        rho = steam.density(1e5, 400.0)
        assert rho > 0.0
        assert rho < 100.0  # steam is much less dense than water

    def test_enthalpy(self):
        steam = SteamRegion()
        h = steam.enthalpy(1e5, 400.0)
        assert h > 2_000_000.0  # J/kg for steam

    def test_viscosity(self):
        steam = SteamRegion()
        mu = steam.viscosity(400.0)
        assert 1e-6 < mu < 5e-5  # Pa.s for steam


class TestSupercriticalRegion:
    def test_density(self):
        sc = SupercriticalRegion()
        rho = sc.density(25e6, 700.0)
        assert rho > 0.0

    def test_enthalpy(self):
        sc = SupercriticalRegion()
        h = sc.enthalpy(25e6, 700.0)
        assert h > 1_000_000.0

    def test_viscosity(self):
        sc = SupercriticalRegion()
        mu = sc.viscosity(700.0)
        assert mu > 1e-6


class TestSaturationCurve:
    def test_saturation_temperature(self):
        sat = SaturationCurve()
        Ts = sat.saturation_temperature(0.101325e6)  # 1 atm
        assert Ts > 370.0  # ~373 K at 1 atm
        assert Ts < 380.0

    def test_saturation_pressure(self):
        sat = SaturationCurve()
        ps = sat.saturation_pressure(373.15)
        assert ps > 0.0
        # Should be near 1 atm


class TestRegionThermodynamics:
    def test_get_region_water(self):
        thermo = RegionThermodynamics()
        region = thermo.get_region(1e5, 300.0)  # below saturation at 1 atm
        assert region == "water"

    def test_get_region_steam(self):
        thermo = RegionThermodynamics()
        region = thermo.get_region(0.5e6, 500.0)  # above saturation temp at 0.5 MPa
        assert region == "steam"

    def test_get_region_supercritical(self):
        thermo = RegionThermodynamics()
        region = thermo.get_region(25e6, 700.0)
        assert region == "supercritical"

    def test_get_properties_scalar(self):
        thermo = RegionThermodynamics()
        state = thermo.get_properties(1e5, 300.0)
        assert state.region == "water"
        assert state.density > 900.0
        assert state.enthalpy > 100_000.0
        assert state.viscosity > 0.0

    def test_get_properties_vector(self):
        thermo = RegionThermodynamics()
        p = np.array([1e5, 25e6, 0.5e6])
        T = np.array([300.0, 700.0, 500.0])
        state = thermo.get_properties(p, T)
        assert state.region.shape == p.shape
        assert state.density.shape == p.shape
        # Check regions are correctly identified
        assert state.region[0] == "water"
        assert state.region[1] == "supercritical"
        assert state.region[2] == "steam"  # 500K above Ts at 0.5 MPa
        # Supercritical enthalpy should be higher than liquid water enthalpy
        assert state.enthalpy[1] > state.enthalpy[0]
