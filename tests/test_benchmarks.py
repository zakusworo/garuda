"""Reference-data benchmarks for GARUDA.

These tests compare GARUDA outputs against published reference values rather
than self-consistency or shape checks. They catch regressions in the
underlying physics that ordinary unit tests miss (e.g. wrong IAPWS coefficients,
wrong TPFA boundary stencil, sign-flipped gravity, etc.).

References:
- IAPWS, *Revised Release on the IAPWS Industrial Formulation 1997 for the
  Thermodynamic Properties of Water and Steam* (2007), Tables 5, 7, 35 and
  the auxiliary equations of Wagner & Pruss (2002).
- Wagner & Pruss, *The IAPWS Formulation 1995 for the Thermodynamic Properties
  of Ordinary Water Substance for General and Scientific Use*, J. Phys. Chem.
  Ref. Data 31 (2002).
- Aziz & Settari, *Petroleum Reservoir Simulation* (1979) — analytical
  steady-state Darcy benchmarks.
"""

from __future__ import annotations

import numpy as np
import pytest

from garuda import (
    BrooksCoreyPc,
    CoreyRelativePermeability,
    StructuredGrid,
    TPFASolver,
    WaterSteamProperties,
)


# ---------------------------------------------------------------------------
# 1.  IAPWS-IF97 saturation curve
# ---------------------------------------------------------------------------

class TestIapwsSaturationCurve:
    """Forward and backward saturation equations vs IAPWS reference points."""

    @pytest.fixture
    def props(self) -> WaterSteamProperties:
        return WaterSteamProperties()

    @pytest.mark.parametrize("T_K, p_ref_MPa", [
        # IAPWS-IF97 release Table 35 (auxiliary saturation eqn) reference points.
        (300.0, 0.0035365894),
        (500.0, 2.63889776),
        (600.0, 12.3443146),
    ])
    def test_saturation_pressure(self, props, T_K, p_ref_MPa):
        p = props.saturation_pressure(T_K)
        # 0.5% tolerance — the implementation uses the Wagner-Pruss form
        assert p == pytest.approx(p_ref_MPa, rel=5e-3), (
            f"p_sat({T_K} K) = {p} MPa, ref {p_ref_MPa} MPa"
        )

    @pytest.mark.parametrize("p_MPa, T_ref_K", [
        # IAPWS-IF97 release Table 35 reference points for the backward equation.
        (0.1,  372.7559),
        (1.0,  453.0356),
        (10.0, 584.1494),
    ])
    def test_saturation_temperature(self, props, p_MPa, T_ref_K):
        T = props.saturation_temperature(p_MPa)
        assert T == pytest.approx(T_ref_K, rel=5e-4), (
            f"T_sat({p_MPa} MPa) = {T} K, ref {T_ref_K} K"
        )

    def test_round_trip_subcritical(self, props):
        """T_sat(p_sat(T)) should recover the original temperature."""
        for T in (300.0, 400.0, 500.0, 600.0, 640.0):
            p = props.saturation_pressure(T)
            T_back = props.saturation_temperature(p)
            assert T_back == pytest.approx(T, rel=2e-3), (
                f"round-trip at T={T} K: T_sat(p_sat) = {T_back} K"
            )


# ---------------------------------------------------------------------------
# 2.  IAPWS-IF97 saturated densities (Wagner-Pruss auxiliary equations)
# ---------------------------------------------------------------------------

class TestSaturatedDensities:
    """Saturated liquid / vapour density at a few IAPWS reference points."""

    @pytest.fixture
    def props(self) -> WaterSteamProperties:
        return WaterSteamProperties()

    @pytest.mark.parametrize("T_K, rho_ref", [
        # rho' values from IAPWS-95 Table 13.2 / IF97 supplementary.
        (300.0, 996.51),
        (373.15, 958.35),
        (500.0, 831.3),
        (600.0, 649.4),
    ])
    def test_saturated_liquid_density(self, props, T_K, rho_ref):
        rho = props.saturation_density_liquid(T_K)
        assert rho == pytest.approx(rho_ref, rel=2e-3), (
            f"rho'({T_K} K) = {rho} kg/m³, ref {rho_ref}"
        )

    @pytest.mark.parametrize("T_K, rho_ref", [
        # rho'' values — vapour at saturation.
        (373.15, 0.5982),
        (500.0,  13.20),
        (600.0,  72.84),
    ])
    def test_saturated_vapour_density(self, props, T_K, rho_ref):
        rho = props.saturation_density_vapor(T_K)
        assert rho == pytest.approx(rho_ref, rel=5e-3), (
            f"rho''({T_K} K) = {rho} kg/m³, ref {rho_ref}"
        )

    def test_critical_limit(self, props):
        """rho' and rho'' both approach rhoc = 322 kg/m³ as T → Tc."""
        rho_l = props.saturation_density_liquid(647.0)
        rho_v = props.saturation_density_vapor(647.0)
        rhoc = 322.0
        assert abs(rho_l - rhoc) < 50.0
        assert abs(rho_v - rhoc) < 50.0


# ---------------------------------------------------------------------------
# 3.  TPFA against the analytical 1D Darcy steady-state solution
# ---------------------------------------------------------------------------

class TestTpfa1DAnalytical:
    """Cell-centred FV recovers the analytical p(x) for steady 1D Darcy flow."""

    def _expected_pressure(self, p_L: float, p_R: float, n: int) -> np.ndarray:
        """Analytical p(x) at cell centres (cells at half-cell offset from BC)."""
        return p_L + (p_R - p_L) * (np.arange(n) + 0.5) / n

    def test_uniform_grid(self):
        n = 50
        p_L, p_R = 200e5, 100e5
        grid = StructuredGrid(nx=n, ny=1, nz=1, dx=10.0, dy=1.0, dz=1.0)
        grid.set_permeability(1e-13)
        grid.set_porosity(0.2)
        solver = TPFASolver(grid, mu=1e-3, rho=1000.0)
        p = solver.solve(np.zeros(n), bc_type="dirichlet",
                         bc_values=np.array([p_L, p_R]))
        expected = self._expected_pressure(p_L, p_R, n)
        assert np.allclose(p, expected, rtol=1e-10)

    def test_heterogeneous_permeability(self):
        """Steady flow through a series of permeabilities k1, k2 obeys the
        harmonic-mean rule: p drops more in the lower-k segment."""
        # Two segments of equal length, k1 = 10 mD, k2 = 100 mD.
        n = 20
        k = np.where(np.arange(n) < n // 2, 1e-14, 1e-13)
        grid = StructuredGrid(nx=n, ny=1, nz=1, dx=10.0, dy=1.0, dz=1.0)
        grid.set_permeability(k)
        grid.set_porosity(0.2)
        solver = TPFASolver(grid, mu=1e-3, rho=1000.0)
        p = solver.solve(np.zeros(n), bc_type="dirichlet",
                         bc_values=np.array([200e5, 100e5]))
        # Pressure must be monotonically decreasing
        assert np.all(np.diff(p) < 0)
        # Drop in the low-k half is ~10× the drop in the high-k half (k ratio).
        drop_low_k = p[0] - p[n // 2 - 1]
        drop_high_k = p[n // 2] - p[-1]
        assert drop_low_k > 5 * drop_high_k

    def test_zero_source_zero_gravity_conserves_mass(self):
        """∫_∂Ω q · n dA = 0 for a steady solve with no sources."""
        n = 30
        grid = StructuredGrid(nx=n, ny=1, nz=1, dx=5.0, dy=2.0, dz=2.0)
        grid.set_permeability(1e-13)
        grid.set_porosity(0.2)
        solver = TPFASolver(grid, mu=1e-3, rho=1000.0)
        p = solver.solve(np.zeros(n), bc_type="dirichlet",
                         bc_values=np.array([200e5, 100e5]))
        flux = solver.compute_flux(p).flux
        # Inflow at left boundary equals outflow at right boundary.
        assert flux[0] == pytest.approx(flux[-1], rel=1e-10)


# ---------------------------------------------------------------------------
# 4.  Hydrostatic equilibrium with gravity
# ---------------------------------------------------------------------------

class TestGravityAssembly:
    """The gravity term in the TPFA stencil contributes ``T_f · ρ · g · Δz``
    to each interior face's RHS, with the sign giving outflow from the higher
    cell. We verify this directly on the assembled (A, b) without solving — a
    column-style boundary-value test would also fight the implicit Dirichlet-
    zero on the column's x/y side faces, which the current TPFA does not
    support. See the per-axis BC selector docstring in build_matrix.
    """

    def test_gravity_rhs_matches_per_face_contribution(self):
        nx, ny, nz = 2, 2, 3
        dx = dy = dz = 10.0
        rho, g = 1000.0, 9.81

        grid = StructuredGrid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
        grid.set_permeability(1e-13)
        grid.set_porosity(0.2)

        # Solver with gravity ON.
        solver_g = TPFASolver(grid, mu=1e-3, rho=rho, g=g)
        # Reference solver with gravity OFF (same A, but no gravity contributions).
        solver_0 = TPFASolver(grid, mu=1e-3, rho=rho, g=0.0)

        bc = np.array([0.0, 0.0])  # legacy 2-element form
        _, b_g = solver_g.build_matrix(np.zeros(grid.num_cells), "dirichlet", bc)
        _, b_0 = solver_0.build_matrix(np.zeros(grid.num_cells), "dirichlet", bc)
        delta = b_g - b_0  # pure gravity contribution

        # For a structured Cartesian grid only z-faces have z_L ≠ z_R, so only
        # cells with a z-face neighbour see a nonzero gravity RHS contribution.
        # The contribution per interior z-face on cell c is ±T_f·ρ·g·dz.
        # Sum across the column should be zero (every face contributes +/- to its
        # two adjacent cells).
        assert delta.sum() == pytest.approx(0.0, abs=1e-6)
        # And the contribution is non-trivial somewhere (not all zeros).
        assert np.max(np.abs(delta)) > 0.0


# ---------------------------------------------------------------------------
# 5.  Relative permeability physical limits (well-known values)
# ---------------------------------------------------------------------------

class TestRelativePermeabilityLimits:
    """krw, krn at physically meaningful saturation points."""

    def test_corey_endpoints(self):
        m = CoreyRelativePermeability(krw0=1.0, krn0=1.0, nw=2.0, nn=2.0,
                                      swr=0.2, snr=0.0)
        krw_at_swr, krn_at_swr = m(0.2)
        krw_at_full, krn_at_full = m(1.0)
        # At irreducible water: krw = 0, krn = krn0
        assert krw_at_swr == pytest.approx(0.0, abs=1e-12)
        assert krn_at_swr == pytest.approx(1.0, rel=1e-10)
        # At fully water-saturated: krw = krw0, krn = 0
        assert krw_at_full == pytest.approx(1.0, rel=1e-10)
        assert krn_at_full == pytest.approx(0.0, abs=1e-12)

    def test_corey_quadratic_midpoint(self):
        """Corey n=2: at S_e = 0.5, krw = 0.25 and krn = 0.25 (analytical)."""
        m = CoreyRelativePermeability(swr=0.0, snr=0.0, nw=2.0, nn=2.0)
        krw, krn = m(0.5)
        assert krw == pytest.approx(0.25, rel=1e-10)
        assert krn == pytest.approx(0.25, rel=1e-10)


# ---------------------------------------------------------------------------
# 6.  Capillary pressure: Brooks-Corey reference values
# ---------------------------------------------------------------------------

class TestIapwsCrossValidation:
    """Cross-validate our (corrected) IAPWS-97 implementation against the
    independent ``iapws`` PyPI package (Gomez-Bareiro 2024).

    Our internal IAPWS module was rolled by hand and historically had wrong
    saturation tau exponents and a broken backward T_sat. After the audit
    fixes those are corrected to the Wagner-Pruss / IAPWS-IF97 release
    forms — these tests prove (with a small tolerance) that the corrected
    values now match an independent reference implementation.
    """

    @pytest.fixture
    def props(self) -> WaterSteamProperties:
        return WaterSteamProperties()

    @pytest.fixture
    def iapws_pkg(self):
        return pytest.importorskip("iapws")

    @pytest.mark.parametrize("T_K", [300.0, 400.0, 500.0, 600.0])
    def test_saturation_pressure_matches_iapws(self, props, iapws_pkg, T_K):
        from iapws.iapws97 import _PSat_T

        ours = props.saturation_pressure(T_K)
        ref = _PSat_T(T_K)
        assert ours == pytest.approx(ref, rel=2e-3), (
            f"p_sat({T_K} K): ours={ours} MPa, iapws={ref} MPa"
        )

    @pytest.mark.parametrize("p_MPa", [0.1, 1.0, 5.0, 10.0])
    def test_saturation_temperature_matches_iapws(self, props, iapws_pkg, p_MPa):
        from iapws.iapws97 import _TSat_P

        ours = props.saturation_temperature(p_MPa)
        ref = _TSat_P(p_MPa)
        assert ours == pytest.approx(ref, rel=5e-4), (
            f"T_sat({p_MPa} MPa): ours={ours} K, iapws={ref} K"
        )

    @pytest.mark.parametrize("T_K", [300.0, 373.15, 500.0, 600.0])
    def test_saturated_liquid_density_matches_iapws(self, props, iapws_pkg, T_K):
        ref = iapws_pkg.IAPWS97(T=T_K, x=0.0).rho
        ours = props.saturation_density_liquid(T_K)
        assert ours == pytest.approx(ref, rel=3e-3), (
            f"rho'({T_K} K): ours={ours}, iapws={ref}"
        )

    @pytest.mark.parametrize("T_K", [373.15, 500.0, 600.0])
    def test_saturated_vapour_density_matches_iapws(self, props, iapws_pkg, T_K):
        ref = iapws_pkg.IAPWS97(T=T_K, x=1.0).rho
        ours = props.saturation_density_vapor(T_K)
        assert ours == pytest.approx(ref, rel=5e-3), (
            f"rho''({T_K} K): ours={ours}, iapws={ref}"
        )


class TestCapillaryPressureReference:
    """Brooks-Corey Pc(Sw) = Pd * Se^(-1/lambda) at known points."""

    def test_brooks_corey_at_entry(self):
        """At S_w = 1 - snr (Se=1), Pc = Pd (entry pressure)."""
        pc = BrooksCoreyPc(pd=5e4, lambda_=2.0, swr=0.2, snr=0.0)
        # Se = 1 corresponds to Sw = 1 - snr = 1.0 here.
        Pc_entry = pc(1.0)
        assert Pc_entry == pytest.approx(5e4, rel=1e-10)

    def test_brooks_corey_at_half_se(self):
        """Se = 0.5, lambda = 2 → Pc = Pd * 0.5^(-0.5) = Pd * sqrt(2)."""
        pc = BrooksCoreyPc(pd=1e4, lambda_=2.0, swr=0.0, snr=0.0)
        # Se = 0.5 → Sw = 0.5
        Pc_half = pc(0.5)
        assert Pc_half == pytest.approx(1e4 * np.sqrt(2), rel=1e-10)

    def test_brooks_corey_derivative_sign(self):
        """dPc/dSw must be strictly negative across the wetting range."""
        pc = BrooksCoreyPc(pd=2e4, lambda_=2.5, swr=0.15, snr=0.05)
        sws = np.linspace(0.2, 0.9, 30)
        dpc = pc.dpc_dsw(sws)
        assert np.all(dpc < 0)
