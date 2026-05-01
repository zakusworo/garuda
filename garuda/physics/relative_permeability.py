"""Relative permeability models for multiphase flow in porous media.

Implements standard parametric models (Corey/Brooks-Corey, van Genuchten-Mualem,
linear) commonly used in petroleum reservoir and geothermal simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class RelativePermeabilityModel(ABC):
    """Abstract base class for relative permeability models.

    Relative permeability describes how the presence of one fluid phase
    affects the mobility of another phase in a porous medium.

    All subclasses must implement ``__call__(S_w)`` returning ``(krw, krn)``,
    where *krw* is the wetting-phase (water) relative permeability and
    *krn* is the non-wetting-phase (oil/gas/steam) relative permeability.
    """

    @abstractmethod
    def __call__(self, S_w: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Return (krw, krn) for water saturation(s)."""
        ...

    def effective_saturation(self, S_w: float | np.ndarray) -> np.ndarray:
        """Compute effective water saturation Se.

        Se = (S_w - Swr) / (1 - Swr - Snr)

        Parameters
        ----------
        S_w : float or np.ndarray
            Water saturation.

        Returns
        -------
        np.ndarray
            Effective water saturation clipped to [0, 1].
        """
        S_w = np.asarray(S_w, dtype=float)
        d = 1.0 - self.swr - self.snr
        return np.clip((S_w - self.swr) / d, 0.0, 1.0) if d > 0 else np.clip(S_w, 0.0, 1.0)


@dataclass
class CoreyRelativePermeability(RelativePermeabilityModel):
    r"""Brooks-Corey (power-law) relative permeability model.

    .. math::
        k_{rw} = k_{rw0} \cdot S_e^{N_w}
        k_{rn} = k_{rn0} \cdot (1 - S_e)^{N_n}

    where :math:`S_e` is the effective water saturation.

    Parameters
    ----------
    krw0 : float
        End-point wetting-phase relative permeability (default 1.0).
    krn0 : float
        End-point non-wetting-phase relative permeability (default 1.0).
    nw : float
        Wetting-phase Corey exponent :math:`N_w` (default 2.0, > 0).
    nn : float
        Non-wetting-phase Corey exponent :math:`N_n` (default 2.0, > 0).
    swr : float
        Irreducible water saturation :math:`S_{wr}` (default 0.2).
    snr : float
        Residual non-wetting saturation :math:`S_{nr}` (default 0.0).

    """

    krw0: float = 1.0
    krn0: float = 1.0
    nw: float = 2.0
    nn: float = 2.0
    swr: float = 0.2
    snr: float = 0.0

    def __post_init__(self):
        if self.krw0 < 0:
            raise ValueError("krw0 must be non-negative")
        if self.krn0 < 0:
            raise ValueError("krn0 must be non-negative")
        if self.nw <= 0:
            raise ValueError("nw (wetting exponent) must be > 0")
        if self.nn <= 0:
            raise ValueError("nn (non-wetting exponent) must be > 0")
        if not (0 <= self.swr < 1):
            raise ValueError("swr must be in [0, 1)")
        if not (0 <= self.snr < 1 - self.swr):
            raise ValueError("snr must be in [0, 1 - swr)")

    def __call__(self, S_w: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Return (krw, krn) for water saturation(s)."""
        Se = self.effective_saturation(S_w)
        krw = self.krw0 * Se**self.nw
        krn = self.krn0 * (1.0 - Se) ** self.nn
        return krw, krn


@dataclass
class VanGenuchtenMualem(RelativePermeabilityModel):
    r"""van Genuchten-Mualem relative permeability model.

    Commonly used for unsaturated flow and geothermal systems where
    the retention curve is described by the van Genuchten model.

    .. math::
        m = 1 - 1/n

        k_{rw} = S_e^{0.5} \cdot \bigl[1 - (1 - S_e^{1/m})^m\bigr]^2

        k_{rn} = (1 - S_e)^{0.5} \cdot \bigl[1 - S_e^{1/m}\bigr]^{2m}

    Parameters
    ----------
    n : float
        Pore-size distribution parameter :math:`n` (default 2.0, must be > 1).
    swr : float
        Irreducible water saturation :math:`S_{wr}` (default 0.2).
    snr : float
        Residual non-wetting saturation :math:`S_{nr}` (default 0.0).

    """

    n: float = 2.0
    swr: float = 0.2
    snr: float = 0.0

    def __post_init__(self):
        if self.n <= 1.0:
            raise ValueError("n must be > 1")
        if not (0 <= self.swr < 1):
            raise ValueError("swr must be in [0, 1)")
        if not (0 <= self.snr < 1 - self.swr):
            raise ValueError("snr must be in [0, 1 - swr)")
        self._m = 1.0 - 1.0 / self.n

    def __call__(self, S_w: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Return (krw, krn) for water saturation(s)."""
        Se = self.effective_saturation(S_w)
        Se_safe = np.where(Se < 1e-12, 1e-12, Se)
        one_minus_Se = np.where(Se > 1.0 - 1e-12, 1e-12, 1.0 - Se)

        term = 1.0 - Se_safe ** (1.0 / self._m)
        term_safe = np.where(term < 0, 0.0, term)

        krw = Se_safe**0.5 * (1.0 - term_safe**self._m) ** 2
        krn = one_minus_Se**0.5 * term_safe ** (2.0 * self._m)
        return krw, krn


@dataclass
class LinearRelativePermeability(RelativePermeabilityModel):
    r"""Linear relative permeability model.

    Simple straight-line interpolation between end points, useful for
    testing, validation, and as a conservative (pessimistic) estimate.

    .. math::
        k_{rw} = k_{rw0} \cdot S_e
        k_{rn} = k_{rn0} \cdot (1 - S_e)

    Parameters
    ----------
    krw0 : float
        End-point wetting-phase relative permeability (default 1.0).
    krn0 : float
        End-point non-wetting-phase relative permeability (default 1.0).
    swr : float
        Irreducible water saturation (default 0.2).
    snr : float
        Residual non-wetting saturation (default 0.0).

    """

    krw0: float = 1.0
    krn0: float = 1.0
    swr: float = 0.2
    snr: float = 0.0

    def __post_init__(self):
        if self.krw0 < 0:
            raise ValueError("krw0 must be non-negative")
        if self.krn0 < 0:
            raise ValueError("krn0 must be non-negative")
        if not (0 <= self.swr < 1):
            raise ValueError("swr must be in [0, 1)")
        if not (0 <= self.snr < 1 - self.swr):
            raise ValueError("snr must be in [0, 1 - swr)")

    def __call__(self, S_w: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Return (krw, krn) for water saturation(s)."""
        Se = self.effective_saturation(S_w)
        krw = self.krw0 * Se
        krn = self.krn0 * (1.0 - Se)
        return krw, krn


@dataclass
class StoneIRelativePermeability:
    r"""Stone I three-phase relative permeability model.

    Estimates oil relative permeability in a three-phase system
    (water/oil/gas) from two-phase data.

    .. math::
        k_{ro} = k_{row} \cdot k_{rog} \cdot
                  \frac{S_{o}}{(1 - S_{wc})(1 - S_{orw})}

    where :math:`k_{row}` is the oil/water two-phase kr for oil at
    water saturation :math:`S_w + S_o`, and :math:`k_{rog}` is the
    oil/gas two-phase kr for oil at gas saturation :math:`S_g + S_o`.

    Parameters
    ----------
    krow_model : RelativePermeabilityModel
        Two-phase oil/water relative permeability model.
    krog_model : RelativePermeabilityModel
        Two-phase oil/gas relative permeability model.
    swc : float
        Connate water saturation (default 0.2).
    sorw : float
        Residual oil saturation in oil/water system (default 0.15).

    """

    krow_model: RelativePermeabilityModel
    krog_model: RelativePermeabilityModel
    swc: float = 0.2
    sorw: float = 0.15

    def __post_init__(self):
        if not (0 <= self.swc < 1):
            raise ValueError("swc must be in [0, 1)")
        if not (0 <= self.sorw < 1 - self.swc):
            raise ValueError("sorw must be in [0, 1 - swc)")

    def __call__(self, S_w: float | np.ndarray, S_o: float | np.ndarray) -> tuple:
        """Return (krw, kro, krg) three-phase relative permeabilities.

        Parameters
        ----------
        S_w : float or np.ndarray
            Water saturation.
        S_o : float or np.ndarray
            Oil saturation.

        Returns
        -------
        tuple
            (krw, kro, krg)

        """
        S_w = np.asarray(S_w, dtype=float)
        S_o = np.asarray(S_o, dtype=float)
        # Evaluate two-phase models at effective saturations
        krw, _ = self.krow_model(S_w)
        _, kro_w = self.krow_model(S_w + S_o)  # oil in oil/water system
        _, kro_g = self.krog_model(S_o)  # oil in oil/gas system
        krg, _ = self.krog_model(S_o + S_w)  # gas rel-perm

        denom = (1.0 - self.swc) * (1.0 - self.sorw)
        if denom == 0:
            kro = np.zeros_like(S_o, dtype=float)
        else:
            kro = kro_w * kro_g * (S_o / denom)
        return krw, kro, krg


__all__ = [
    "RelativePermeabilityModel",
    "CoreyRelativePermeability",
    "VanGenuchtenMualem",
    "LinearRelativePermeability",
    "StoneIRelativePermeability",
]
