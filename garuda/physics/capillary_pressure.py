"""Capillary pressure models for multiphase flow in porous media.

Implements Brooks-Corey and van Genuchten parametric models commonly
used in reservoir and geothermal simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class CapillaryPressureModel(ABC):
    """Abstract base class for capillary pressure models.

    Capillary pressure is defined as the pressure difference across
    the interface between two immiscible fluids in a porous medium:
        Pc = P_nonwetting - P_wetting

    In water/steam (or water/oil) systems this is:
        Pc = P_vapor - P_liquid   (or P_oil - P_water)

    All subclasses must implement ``__call__(S_w)`` returning Pc.
    """

    @abstractmethod
    def __call__(self, S_w: float | np.ndarray) -> float | np.ndarray:
        """Evaluate capillary pressure [Pa] at water saturation(s)."""
        ...

    def dpc_dsw(self, S_w: float | np.ndarray) -> float | np.ndarray:
        """Derivative dPc/dSw [Pa] at water saturation(s).

        Default implementation uses finite differences.
        Subclasses may override with analytical expressions.
        """
        S_w = np.asarray(S_w, dtype=float)
        h = 1e-8
        return (self(S_w + h) - self(S_w - h)) / (2.0 * h)

    def effective_saturation(self, S_w: float | np.ndarray) -> np.ndarray:
        """Compute effective water saturation Se.

        Se = (S_w - Swr) / (1 - Swr - Snr)
        """
        S_w = np.asarray(S_w, dtype=float)
        d = 1.0 - self.swr - self.snr
        return np.clip((S_w - self.swr) / d, 0.0, 1.0) if d > 0 else np.clip(S_w, 0.0, 1.0)


@dataclass
class BrooksCoreyPc(CapillaryPressureModel):
    """Brooks-Corey capillary pressure model.

    Pc(Sw) = Pd * Se^(-1/λ)

    where
        Se = (Sw - Swr) / (1 - Swr - Snr)
        Pd = entry pressure [Pa]
        λ  = pore-size distribution index (dimensionless, > 0)

    Parameters
    ----------
    pd : float
        Entry (threshold) pressure [Pa].
    lambda_ : float
        Pore-size distribution index λ (default 2.0).
    swr : float
        Irreducible water saturation (default 0.2).
    snr : float
        Residual non-wetting saturation (default 0.0).

    """

    pd: float = 1e4
    lambda_: float = 2.0
    swr: float = 0.2
    snr: float = 0.0

    def __post_init__(self):
        if self.lambda_ <= 0:
            raise ValueError("lambda_ must be > 0")
        if self.pd < 0:
            raise ValueError("pd (entry pressure) must be non-negative")
        if not (0 <= self.swr < 1):
            raise ValueError("swr must be in [0, 1)")
        if not (0 <= self.snr < 1 - self.swr):
            raise ValueError("snr must be in [0, 1 - swr)")

    def __call__(self, S_w: float | np.ndarray) -> float | np.ndarray:
        """Return capillary pressure [Pa] for water saturation(s)."""
        Se = self.effective_saturation(S_w)
        # Avoid division by zero at Se=0 by flooring Se to a small value
        Se_safe = np.where(Se < 1e-12, 1e-12, Se)
        return self.pd * Se_safe ** (-1.0 / self.lambda_)

    def dpc_dsw(self, S_w: float | np.ndarray) -> float | np.ndarray:
        """Analytical derivative dPc/dSw [Pa]."""
        Se = self.effective_saturation(S_w)
        d = 1.0 - self.swr - self.snr
        if d == 0:
            return np.zeros_like(S_w, dtype=float)
        Se_safe = np.where(Se < 1e-12, 1e-12, Se)
        return -(self.pd / (self.lambda_ * d)) * Se_safe ** (-1.0 / self.lambda_ - 1.0)


@dataclass
class VanGenuchtenPc(CapillaryPressureModel):
    """van Genuchten capillary pressure model.

    Pc(Sw) = P0 * (Se^(-1/m) - 1)^(1 - m)

    where
        Se = (Sw - Swr) / (1 - Swr - Snr)
        m  = 1 - 1/n
        P0 = scaling pressure [Pa]
        n  = pore-size distribution parameter (dimensionless, n > 1)

    Parameters
    ----------
    p0 : float
        Scaling pressure P0 [Pa] (default 1e4).
    n : float
        Pore-size distribution parameter n (default 2.0, must be > 1).
    swr : float
        Irreducible water saturation (default 0.2).
    snr : float
        Residual non-wetting saturation (default 0.0).

    """

    p0: float = 1e4
    n: float = 2.0
    swr: float = 0.2
    snr: float = 0.0

    def __post_init__(self):
        if self.n <= 1.0:
            raise ValueError("n must be > 1")
        if self.p0 < 0:
            raise ValueError("p0 (scaling pressure) must be non-negative")
        if not (0 <= self.swr < 1):
            raise ValueError("swr must be in [0, 1)")
        if not (0 <= self.snr < 1 - self.swr):
            raise ValueError("snr must be in [0, 1 - swr)")
        self._m = 1.0 - 1.0 / self.n

    def __call__(self, S_w: float | np.ndarray) -> float | np.ndarray:
        """Return capillary pressure [Pa] for water saturation(s)."""
        Se = self.effective_saturation(S_w)
        # Clamp Se away from 0 to avoid singularities
        Se_safe = np.where(Se < 1e-12, 1e-12, Se)
        term = Se_safe ** (-1.0 / self._m) - 1.0
        term_safe = np.where(term < 0, 0.0, term)
        return self.p0 * term_safe ** (1.0 - self._m)

    def dpc_dsw(self, S_w: float | np.ndarray) -> float | np.ndarray:
        """Analytical derivative dPc/dSw [Pa]."""
        Se = self.effective_saturation(S_w)
        d = 1.0 - self.swr - self.snr
        if d == 0:
            return np.zeros_like(S_w, dtype=float)
        Se_safe = np.where(Se < 1e-12, 1e-12, Se)
        term = Se_safe ** (-1.0 / self._m) - 1.0
        # Cap term away from zero to avoid 0**(-m) singularity near Se=1
        term_safe = np.where(term < 1e-12, 1e-12, term)
        dpc_dse = (self.p0 * (1.0 - self._m) / self._m) * Se_safe ** (-1.0 / self._m - 1.0) * term_safe ** (-self._m)
        return dpc_dse / d


__all__ = [
    "CapillaryPressureModel",
    "BrooksCoreyPc",
    "VanGenuchtenPc",
]
