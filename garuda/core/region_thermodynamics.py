"""Region-based fluid thermodynamics for geothermal simulation.

Inspired by Waiwera's thermodynamics module. Organises fluid properties
by thermodynamic region (liquid water, steam, supercritical) and provides
a top-level manager that dispatches to the correct region based on
pressure/temperature state.

Typical usage::

    from garuda.core.region_thermodynamics import RegionThermodynamics

    thermo = RegionThermodynamics()
    props = thermo.get_properties(pressure=10e6, temperature=573.15)
    # props["region"] == "water"
    # props["density"], props["enthalpy"], props["viscosity"]

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from garuda.core.iapws_properties import WaterSteamProperties


__all__ = [
    "ThermodynamicsRegion",
    "WaterRegion",
    "SteamRegion",
    "SupercriticalRegion",
    "SaturationCurve",
    "RegionThermodynamics",
    "FluidThermoState",
]


# IAPWS-97 critical point
CRITICAL_PRESSURE_PA = 22.064e6
CRITICAL_TEMPERATURE_K = 647.096
CRITICAL_ENTHALPY_J_KG = 2_087_000.0  # approximate

# Robust saturation temperature approximation via log–space interpolation
# on IAPWS-97 table values.  Avoids the NaN-producing backward equation.
_SAT_PTS_PA = np.array(
    [611.0, 1_000.0, 10_000.0, 50_000.0, 101_325.0, 500_000.0, 1e6, 5e6, 10e6, 15e6, 20e6, CRITICAL_PRESSURE_PA],
    dtype=float,
)
_SAT_T_PTS_K = np.array(
    [273.15, 280.12, 318.96, 354.48, 373.124, 424.98, 453.03, 537.11, 584.11, 613.15, 635.82, CRITICAL_TEMPERATURE_K],
    dtype=float,
)


def _saturation_temperature_approx(pressure_pa: float | np.ndarray) -> float | np.ndarray:
    r"""Saturation temperature [K] from pressure [Pa] via robust interpolation."""
    scalar = np.isscalar(pressure_pa)
    p_arr = np.asarray(pressure_pa, dtype=float)
    # Clamp to valid IAPWS-97 range
    p_clamped = np.clip(p_arr, _SAT_PTS_PA[0], _SAT_PTS_PA[-1])
    t_interp = np.interp(np.log(p_clamped), np.log(_SAT_PTS_PA), _SAT_T_PTS_K)
    if scalar:
        return float(t_interp)
    return t_interp


def _saturation_temperature_scalar(pressure_pa: float) -> float:
    return float(_saturation_temperature_approx(pressure_pa))


@dataclass
class FluidThermoState:
    """Container for fluid thermodynamic properties.

    Attributes
    ----------
    region : str
        Thermodynamic region name.
    pressure : float | np.ndarray
        Pressure [Pa].
    temperature : float | np.ndarray
        Temperature [K].
    density : float | np.ndarray
        Density [kg/m³].
    enthalpy : float | np.ndarray
        Specific enthalpy [J/kg].
    viscosity : float | np.ndarray
        Dynamic viscosity [Pa·s].

    """

    region: str
    pressure: float | np.ndarray
    temperature: float | np.ndarray
    density: float | np.ndarray
    enthalpy: float | np.ndarray
    viscosity: float | np.ndarray


class ThermodynamicsRegion(ABC):
    """Abstract base for a fluid thermodynamic region."""

    name: str = ""

    @abstractmethod
    def density(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Return density [kg/m³]."""

    @abstractmethod
    def enthalpy(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Return specific enthalpy [J/kg]."""

    @abstractmethod
    def viscosity(
        self,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Return dynamic viscosity [Pa·s]."""

    @abstractmethod
    def dh_dt(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Return enthalpy derivative w.r.t. temperature [J/(kg·K)]."""

    @abstractmethod
    def dh_dp(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Return enthalpy derivative w.r.t. pressure [J/(kg·Pa)]."""


class WaterRegion(ThermodynamicsRegion):
    """Sub-critical liquid water region (IAPWS-97 Region 1)."""

    name = "water"

    def __init__(self) -> None:
        self.iapws = WaterSteamProperties()

    def _to_scalar_or_array(self, pm: float | np.ndarray) -> tuple[float | np.ndarray, bool]:
        """Check whether input is scalar or array."""
        return pm, np.isscalar(pm)

    def density(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Liquid water density via IAPWS backward equation."""
        pm, scalar = self._to_scalar_or_array(pressure)
        if scalar:
            pm_val = pm / 1e6
            T_val = float(temperature)
            return float(self.iapws.density_region1(pm_val, T_val))
        pm_arr = np.asarray(pm) / 1e6
        T_arr = np.asarray(temperature)
        return np.array([self.iapws.density_region1(p, t) for p, t in zip(pm_arr, T_arr)])

    def enthalpy(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Liquid water specific enthalpy [J/kg]."""
        pm, scalar = self._to_scalar_or_array(pressure)
        if scalar:
            return float(self.iapws.enthalpy_liquid(float(temperature))) * 1_000.0
        T_arr = np.asarray(temperature)
        return np.array([self.iapws.enthalpy_liquid(t) for t in T_arr]) * 1_000.0

    def viscosity(
        self,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Liquid water dynamic viscosity [Pa·s]."""
        T, scalar = self._to_scalar_or_array(temperature)
        if scalar:
            return float(self.iapws.viscosity_liquid(float(T)))
        T_arr = np.asarray(T)
        return np.array([self.iapws.viscosity_liquid(t) for t in T_arr])

    def dh_dt(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Approximate specific heat :math:`c_p` [J/(kg·K)]."""
        # For liquid water cp ≈ 4.2 kJ/(kg·K)
        return 4_200.0

    def dh_dp(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Approximate liquid enthalpy pressure derivative [J/(kg·Pa)]."""
        # For incompressible liquid dh/dp ≈ v = 1/rho
        rho = self.density(pressure, temperature)
        return 1.0 / rho


class SteamRegion(ThermodynamicsRegion):
    """Sub-critical steam (vapor) region (IAPWS-97 Region 2)."""

    name = "steam"

    def __init__(self) -> None:
        self.iapws = WaterSteamProperties()

    def _to_scalar_or_array(self, pm: float | np.ndarray) -> tuple[float | np.ndarray, bool]:
        return pm, np.isscalar(pm)

    def density(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Steam density via ideal-gas + IAPWS Region 2 approximations."""
        pm, scalar = self._to_scalar_or_array(pressure)
        if scalar:
            pm_val = pm / 1e6
            T_val = float(temperature)
            # IAPWS-IF97 region 4 backward eq is now well-defined for the full
            # subcritical range, but guard against NaN/inf (e.g. p ≤ 0).
            T_sat = self.iapws.saturation_temperature(pm_val)
            if np.isfinite(T_sat) and T_val < T_sat * 0.95:
                # Below saturation: shouldn't happen in the steam branch,
                # clamp to saturation to keep the ideal-gas density sane.
                T_val = T_sat
            R_specific = 461.5  # J/(kg·K) for steam
            return float(pressure / (R_specific * T_val))
        pm_arr = np.asarray(pressure) / 1e6
        T_arr = np.asarray(temperature)
        T_sat_arr = np.array(
            [self.iapws.saturation_temperature(p) if p > 0.1 and p < 22.064 else 647.096 for p in pm_arr]
        )
        T_safe = np.where(T_arr < T_sat_arr * 0.95, T_sat_arr, T_arr)
        R_specific = 461.5
        return np.asarray(pressure) / (R_specific * T_safe)

    def enthalpy(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Steam specific enthalpy [J/kg]."""
        pm, scalar = self._to_scalar_or_array(pressure)
        if scalar:
            return float(self.iapws.enthalpy_vapor(float(temperature))) * 1_000.0
        T_arr = np.asarray(temperature)
        return np.array([self.iapws.enthalpy_vapor(t) for t in T_arr]) * 1_000.0

    def viscosity(
        self,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Steam dynamic viscosity [Pa·s].

        Approximate with temperature-dependent power law.
        """
        T_ref = 373.15
        mu_ref = 1.25e-5
        # mu ~ T^0.6 for steam
        T, scalar = self._to_scalar_or_array(temperature)
        if scalar:
            return mu_ref * (float(T) / T_ref) ** 0.6
        T_arr = np.asarray(T)
        return mu_ref * (T_arr / T_ref) ** 0.6

    def dh_dt(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Approximate steam isobaric specific heat [J/(kg·K)]."""
        # For steam cp ≈ 2.0 kJ/(kg·K)
        return 2_000.0

    def dh_dp(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Approximate steam enthalpy pressure derivative [J/(kg·Pa)]."""
        # For ideal gas dh/dp ≈ 0 (enthalpy of ideal gas depends only on T)
        return 0.0


class SupercriticalRegion(ThermodynamicsRegion):
    """Supercritical water / steam region (p > Pc or T > Tc)."""

    name = "supercritical"

    def __init__(self) -> None:
        self.iapws = WaterSteamProperties()

    def _to_scalar_or_array(self, pm: float | np.ndarray) -> tuple[float | np.ndarray, bool]:
        return pm, np.isscalar(pm)

    def density(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Supercritical density via ideal-gas with compressibility factor Z."""
        R_specific = 461.5
        return pressure / (R_specific * temperature)

    def enthalpy(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Supercritical specific enthalpy [J/kg]."""
        pm, scalar = self._to_scalar_or_array(pressure)
        if scalar:
            # Approximate: supercritical enthalpy ≈ liquid enthalpy at T
            # plus a pressure correction term
            h_liq = float(self.iapws.enthalpy_liquid(float(temperature))) * 1_000.0
            # Small pressure correction for water (v dh/dp ≈ 1e-3 m³/kg * pressure)
            return h_liq + 1e-6 * float(pressure)
        T_arr = np.asarray(temperature)
        h_liq_arr = np.array([self.iapws.enthalpy_liquid(t) for t in T_arr]) * 1_000.0
        p_arr = np.asarray(pressure)
        return h_liq_arr + 1e-6 * p_arr

    def viscosity(
        self,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Supercritical dynamic viscosity [Pa·s].

        Interpolate between liquid and steam correlations.
        """
        T, scalar = self._to_scalar_or_array(temperature)
        mu_water = (
            self.iapws.viscosity_liquid(float(T))
            if scalar
            else np.array([self.iapws.viscosity_liquid(t) for t in np.asarray(T)])
        )
        if scalar:
            mu_steam = 1.25e-5 * (float(T) / 373.15) ** 0.6
        else:
            T_arr = np.asarray(T)
            mu_steam = 1.25e-5 * (T_arr / 373.15) ** 0.6
        # Weight toward liquid below Tc, toward steam above Tc
        T_arr = np.asarray(T)
        w = np.clip((T_arr - CRITICAL_TEMPERATURE_K) / 100.0 + 0.5, 0.0, 1.0)
        if scalar:
            return float(w * mu_steam + (1.0 - w) * mu_water)
        return w * mu_steam + (1.0 - w) * mu_water

    def dh_dt(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Approximate supercritical isobaric specific heat [J/(kg·K)]."""
        # Near critical point cp diverges; use a safe average
        T_arr = np.asarray(temperature)
        # Smooth transition from liquid cp to steam cp around Tc
        T_arr = np.asarray(temperature)
        w = np.clip((T_arr - CRITICAL_TEMPERATURE_K) / 50.0 + 0.5, 0.0, 1.0)
        return 4_200.0 * (1.0 - w) + 2_000.0 * w

    def dh_dp(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Approximate supercritical enthalpy pressure derivative [J/(kg·Pa)]."""
        rho = self.density(pressure, temperature)
        return 1.0 / rho


class SaturationCurve:
    """Saturation temperature/pressure curve for water."""

    def saturation_temperature(
        self,
        pressure_pa: float | np.ndarray,
    ) -> float | np.ndarray:
        """Saturation temperature [K] from pressure [Pa]."""
        return _saturation_temperature_approx(pressure_pa)

    def saturation_pressure(
        self,
        temperature_k: float | np.ndarray,
    ) -> float | np.ndarray:
        """Saturation pressure [Pa] from temperature [K].

        Approximate inverse of saturation_temperature using
        the Clausius-Clapeyron relation for water.
        """
        # Antoine-like approximation for water (Pa)
        T = temperature_k
        if np.isscalar(T):
            if float(T) >= CRITICAL_TEMPERATURE_K:
                return float(CRITICAL_PRESSURE_PA)
            # Valid range 273.15 - 647 K
            return 610.78 * np.exp(17.269 * (T - 273.15) / (T - 35.85))
        T_arr = np.asarray(T)
        psat = np.where(
            T_arr >= CRITICAL_TEMPERATURE_K,
            CRITICAL_PRESSURE_PA,
            610.78 * np.exp(17.269 * (T_arr - 273.15) / (T_arr - 35.85)),
        )
        return psat

    @staticmethod
    def region_from_pt(
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> str | np.ndarray:
        """Determine thermodynamic region from P-T state.

        Returns
        -------
        str or ndarray of str
            One of ``'water'``, ``'steam'``, ``'supercritical'``.

        """
        p = np.asarray(pressure, dtype=float)
        T = np.asarray(temperature, dtype=float)

        result = np.empty(p.shape, dtype=object)
        result.fill("")

        # Critical dome boundary
        sc_mask = (p >= CRITICAL_PRESSURE_PA) | (T >= CRITICAL_TEMPERATURE_K)
        result[sc_mask] = "supercritical"

        # Below critical: compare T with saturation temperature at given P
        below_critical = ~sc_mask
        if np.any(below_critical):
            Ts = _saturation_temperature_approx(p[below_critical])
            result[below_critical] = np.where(
                T[below_critical] < Ts,
                "water",
                "steam",
            )

        if np.isscalar(pressure) and np.isscalar(temperature):
            return str(result.flat[0])
        return result


class RegionThermodynamics:
    """Top-level thermodynamics manager dispatching to region handlers."""

    def __init__(self) -> None:
        self.saturation_curve = SaturationCurve()
        self.regions: dict[str, ThermodynamicsRegion] = {
            "water": WaterRegion(),
            "steam": SteamRegion(),
            "supercritical": SupercriticalRegion(),
        }

    def get_region(self, pressure: float | np.ndarray, temperature: float | np.ndarray) -> str | np.ndarray:
        """Determine region name for a P-T state."""
        if np.isscalar(pressure) and np.isscalar(temperature):
            if pressure >= CRITICAL_PRESSURE_PA or temperature >= CRITICAL_TEMPERATURE_K:
                return "supercritical"
            Ts = _saturation_temperature_approx(pressure)
            return "water" if temperature < Ts else "steam"
        return SaturationCurve.region_from_pt(pressure, temperature)

    def get_properties(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> FluidThermoState:
        """Return all thermodynamic properties for a P-T state.

        Parameters
        ----------
        pressure : float | np.ndarray
            Pressure [Pa].
        temperature : float | np.ndarray
            Temperature [K].

        Returns
        -------
        FluidThermoState
            Complete thermodynamic state with region label.

        """
        region = self.get_region(pressure, temperature)

        if isinstance(region, str):
            handler = self.regions[region]
            return FluidThermoState(
                region=region,
                pressure=pressure,
                temperature=temperature,
                density=handler.density(pressure, temperature),
                enthalpy=handler.enthalpy(pressure, temperature),
                viscosity=handler.viscosity(temperature),
            )

        # Vector case
        region_arr = np.asarray(region)
        p_arr = np.asarray(pressure)
        T_arr = np.asarray(temperature)

        density = np.empty_like(p_arr, dtype=float)
        enthalpy = np.empty_like(p_arr, dtype=float)
        viscosity = np.empty_like(p_arr, dtype=float)

        for r_name in self.regions:
            mask = region_arr == r_name
            if not np.any(mask):
                continue
            handler = self.regions[r_name]
            density[mask] = handler.density(p_arr[mask], T_arr[mask])
            enthalpy[mask] = handler.enthalpy(p_arr[mask], T_arr[mask])
            viscosity[mask] = handler.viscosity(T_arr[mask])

        return FluidThermoState(
            region=region_arr,
            pressure=p_arr,
            temperature=T_arr,
            density=density,
            enthalpy=enthalpy,
            viscosity=viscosity,
        )
