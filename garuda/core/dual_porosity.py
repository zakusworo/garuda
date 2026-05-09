"""
dual_porosity.py

Dual-porosity models for fractured reservoir simulation.

Implements the classical Warren-Root pseudo-steady-state approach as well as
transient (fine-grid) matrix-fracture transfer formulations commonly used in
groundwater, petroleum, and geothermal reservoir engineering.

The module supports:
  - Warren-Root shape-factor based interporosity flow
  - Kazemi-Gilman-Elsharkawy extension for anisotropic fractures
  - Lim-Aguilera transient transfer function
  - Conversion between single-porosity and dual-porosity parameters
  - Temperature-dependent fracture/matrix property scaling

Typical usage::

    from garuda.core.dual_porosity import DualPorosityModel

    model = DualPorosityModel(
        matrix_porosity=0.15,
        matrix_permeability=1e-15,   # m^2 (~1 mD)
        fracture_porosity=0.01,
        fracture_permeability=1e-12, # m^2 (~1000 mD)
        fracture_spacing=(10.0, 10.0, 10.0),  # mx, my, mz in metres
    )

    # Shape factor for cubic matrix blocks
    sigma = model.warren_root_shape_factor()

    # Interporosity flow coefficient
    alpha = model.interporosity_flow_coefficient(
        compressibility=1e-9,  # 1/Pa
        viscosity=1e-3,        # Pa.s
    )
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "TransferModel",
    "BlockGeometry",
    "DualPorosityParams",
    "DualPorosityModel",
    "convert_single_to_dual",
    "convert_dual_to_single",
]


class TransferModel(enum.Enum):
    """Supported matrix–fracture transfer formulations."""

    WARREN_ROOT_PSS = "warren_root_pss"
    """Pseudo-steady-state ( Warren & Root, 1963 )"""

    KAZEMI_GILMAN = "kazemi_gilman"
    """Anisotropic extension ( Kazemi, Gilman & Elsharkawy, 1992 )"""

    LIM_AGUILERA = "lim_aguilera"
    """Transient transfer with series solution ( Lim & Aguilera, 1996 )"""


class BlockGeometry(enum.Enum):
    """Idealised matrix-block geometries for shape-factor selection."""

    SLAB_X = "slab_x"
    SLAB_Y = "slab_y"
    SLAB_Z = "slab_z"
    SLAB_RADIAL = "slab_radial"
    CUBE = "cube"
    SPHERE = "sphere"
    PRISM = "prism"


@dataclass(frozen=True)
class DualPorosityParams:
    """
    Immutable container for dual-porosity petrophysical properties.

    Parameters
    ----------
    phi_m : float
        Matrix porosity (fraction, 0–1).
    phi_f : float
        Fracture porosity (fraction, 0–1).
    k_m : float
        Matrix intrinsic permeability (m²).
    k_f : float
        Fracture intrinsic permeability (m²).
    Lx, Ly, Lz : float
        Average matrix-block dimensions (m).
    tau : float, optional
        Tortuosity factor (>1). Default 1.0.
    alpha_wr : float, optional
        Warren-Root shape-factor constant (geometry-dependent).
    """

    phi_m: float
    phi_f: float
    k_m: float
    k_f: float
    Lx: float
    Ly: float
    Lz: float
    tau: float = 1.0
    alpha_wr: float | None = None

    @property
    def total_porosity(self) -> float:
        """Return total storativity-weighted porosity."""
        return self.phi_m + self.phi_f

    @property
    def storativity_ratio(self) -> float:
        """
        omega = phi_f*c_f / (phi_m*c_m + phi_f*c_f).

        When compressibilities are equal this reduces to phi_f / (phi_m + phi_f).
        """
        return self.phi_f / (self.phi_m + self.phi_f + 1e-30)


# ---------------------------------------------------------------------------
# Shape-factor tables (constant part of σ·L²)
# ---------------------------------------------------------------------------

_SHAPE_FACTOR_CONSTANT: dict[BlockGeometry, float] = {
    # Geometry          constant C in  σ = C / L²
    BlockGeometry.SLAB_X: math.pi**2 / (4.0**2),
    BlockGeometry.SLAB_Y: math.pi**2 / (4.0**2),
    BlockGeometry.SLAB_Z: math.pi**2 / (4.0**2),
    BlockGeometry.SLAB_RADIAL: math.pi**2 / (4.0**2),
    BlockGeometry.CUBE: math.pi**2,
    BlockGeometry.SPHERE: math.pi**2,
    BlockGeometry.PRISM: 2.0 * math.pi**2,
}


def _pick_geometry(Lx: float, Ly: float, Lz: float) -> BlockGeometry:
    """Heuristic to select a default geometry from block dimensions."""
    dims = sorted([Lx, Ly, Lz])
    ratio = dims[2] / (dims[0] + 1e-30)
    if ratio > 5.0:
        if dims[0] == Lx:
            return BlockGeometry.SLAB_X
        if dims[0] == Ly:
            return BlockGeometry.SLAB_Y
        return BlockGeometry.SLAB_Z
    if abs(Lx - Ly) < 1e-6 and abs(Ly - Lz) < 1e-6:
        return BlockGeometry.CUBE
    return BlockGeometry.PRISM


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class DualPorosityModel:
    """
    Warren-Root type dual-porosity engine.

    Provides shape factors, interporosity transfer coefficients, and
    dimensionless groups used in analytical solutions and numerical
    reservoir simulators (e.g. TOUGH2, ECLIPSE-DUALS, CMG).
    """

    def __init__(
        self,
        matrix_porosity: float,
        matrix_permeability: float,
        fracture_porosity: float,
        fracture_permeability: float,
        fracture_spacing: tuple[float, float, float],
        geometry: BlockGeometry | None = None,
        tortuosity: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        matrix_porosity : float
            dimensionless matrix porosity.
        matrix_permeability : float
            matrix permeability in m².
        fracture_porosity : float
            dimensionless fracture porosity.
        fracture_permeability : float
            fracture permeability in m².
        fracture_spacing : tuple(float, float, float)
            Average distance between fractures in x, y, z (m).
        geometry : BlockGeometry | None
            If None, inferred from ``fracture_spacing`` ratios.
        tortuosity : float
            Tortuosity factor ≥ 1.0.
        """
        self.Lx, self.Ly, self.Lz = fracture_spacing
        self.phi_m = matrix_porosity
        self.phi_f = fracture_porosity
        self.k_m = matrix_permeability
        self.k_f = fracture_permeability
        self.tau = tortuosity
        self.geometry = geometry or _pick_geometry(self.Lx, self.Ly, self.Lz)
        self._params = DualPorosityParams(
            phi_m=self.phi_m,
            phi_f=self.phi_f,
            k_m=self.k_m,
            k_f=self.k_f,
            Lx=self.Lx,
            Ly=self.Ly,
            Lz=self.Lz,
            tau=self.tau,
        )

    # ------------------------------------------------------------------
    # Shape factors
    # ------------------------------------------------------------------

    def warren_root_shape_factor(self, geometry: BlockGeometry | None = None) -> float:
        r"""
        Return the Warren-Root shape factor σ (1/m²).

        For isotropic cubic blocks of side ``L``:

        .. math::
            \sigma = \pi^2 \left( \frac{1}{L_x^2} + \frac{1}{L_y^2}
                     + \frac{1}{L_z^2} \right)

        For slabs use :math:`\pi^2 / 4L^2`.
        """
        geom = geometry or self.geometry
        if geom in (
            BlockGeometry.SLAB_X,
            BlockGeometry.SLAB_Y,
            BlockGeometry.SLAB_Z,
            BlockGeometry.SLAB_RADIAL,
        ):
            # Identify the dominant fracture spacing
            dims = {"x": self.Lx, "y": self.Ly, "z": self.Lz}
            if geom == BlockGeometry.SLAB_X:
                L = self.Lx
            elif geom == BlockGeometry.SLAB_Y:
                L = self.Ly
            elif geom == BlockGeometry.SLAB_Z:
                L = self.Lz
            else:
                L = min(dims.values())
            return math.pi**2 / (4.0 * L**2)

        # General orthogonal (Kazemi / prism) form
        return math.pi**2 * (1.0 / (self.Lx**2) + 1.0 / (self.Ly**2) + 1.0 / (self.Lz**2))

    def kazemi_shape_factor(self) -> float:
        r"""
        Kazemi-Gilman shape factor for anisotropic fracture networks.

        .. math::
            \sigma = 4 \left( \frac{1}{L_x^2} + \frac{1}{L_y^2}
                     + \frac{1}{L_z^2} \right)
        """
        return 4.0 * (1.0 / (self.Lx**2) + 1.0 / (self.Ly**2) + 1.0 / (self.Lz**2))

    def lim_aguilera_shape_factor(self) -> float:
        r"""
        Lim–Aguilera pseudo-steady-state shape factor derived from
        matrix-block transient pressure analysis.

        For a cubic block::

            σ = 25.1 / L²

        For slab::

            σ = π² / (4L²)
        """
        if self.geometry == BlockGeometry.CUBE:
            L = self.Lx  # assumed equal sides
            return 25.13 / (L**2)
        return self.warren_root_shape_factor()

    # ------------------------------------------------------------------
    # Transfer coefficient
    # ------------------------------------------------------------------

    def interporosity_flow_coefficient(
        self,
        compressibility: float | None = None,
        viscosity: float | None = None,
        model: TransferModel = TransferModel.WARREN_ROOT_PSS,
    ) -> float:
        r"""
        Interporosity flow coefficient λ (dimensionless).

        .. math::
            \lambda = \alpha \frac{k_m}{k_f} L_f^2

        where :math:`\alpha = \sigma / \tau` and :math:`L_f` is a
        characteristic fracture length (taken as the harmonic mean of the
        fracture spacings).

        Parameters
        ----------
        compressibility, viscosity : float, optional
            Accepted for API stability with earlier releases but **not used**
            in this geometric definition of λ. Retained as keyword arguments
            so existing call sites (and ``lambda_group``) continue to work.
        model : TransferModel
            Which shape factor to employ.

        Returns
        -------
        float
            λ (dimensionless).
        """
        # compressibility / viscosity intentionally unused — see docstring.
        del compressibility, viscosity

        if model == TransferModel.KAZEMI_GILMAN:
            sigma = self.kazemi_shape_factor()
        elif model == TransferModel.LIM_AGUILERA:
            sigma = self.lim_aguilera_shape_factor()
        else:
            sigma = self.warren_root_shape_factor()

        alpha = sigma / self.tau
        # Characteristic length = harmonic mean of spacings
        Lf = 3.0 / (1.0 / self.Lx + 1.0 / self.Ly + 1.0 / self.Lz)

        return alpha * self.k_m / self.k_f * Lf**2

    # ------------------------------------------------------------------
    # Dimensionless groups
    # ------------------------------------------------------------------

    def omega(self, c_m: float = 1.0, c_f: float = 1.0) -> float:
        r"""
        Storativity ratio ω (dimensionless).

        .. math::
            \omega = \frac{\phi_f c_f}{\phi_m c_m + \phi_f c_f}
        """
        return (self.phi_f * c_f) / (self.phi_m * c_m + self.phi_f * c_f + 1e-30)

    def lambda_group(
        self,
        compressibility: float,
        viscosity: float,
        model: TransferModel = TransferModel.WARREN_ROOT_PSS,
    ) -> float:
        r"""
        Alias for :meth:`interporosity_flow_coefficient` with
        :math:`c_m = c_f = c_t`.
        """
        return self.interporosity_flow_coefficient(
            compressibility=compressibility,
            viscosity=viscosity,
            model=model,
        )

    # ------------------------------------------------------------------
    # Analytical helpers — transient pressure in matrix block
    # ------------------------------------------------------------------

    @staticmethod
    def lim_aguilera_transfer_function(
        t_dim: NDArray[np.float64],
        n_terms: int = 20,
    ) -> NDArray[np.float64]:
        r"""
        Dimensionless transient interporosity transfer function for
        a slab matrix block.

        .. math::
            f(t_D) = \sum_{n=1}^{\infty}
                     \frac{2}{n^2 \pi^2}
                     \left[1 - \exp(-n^2 \pi^2 t_D)\right]

        Parameters
        ----------
        t_dim : NDArray[np.float64]
            Dimensionless time :math:`t_D = k_m t / (\phi_m c_m \mu L^2)`.
        n_terms : int
            Number of series terms.

        Returns
        -------
        NDArray[np.float64]
            Transfer-function values, same shape as ``t_dim``.
        """
        t = np.asarray(t_dim, dtype=np.float64)
        result = np.zeros_like(t)
        for n in range(1, n_terms + 1):
            coeff = 2.0 / ((n * math.pi) ** 2)
            result += coeff * (1.0 - np.exp(-((n * math.pi) ** 2) * t))
        return result

    @staticmethod
    def pseudo_steady_state_time(
        matrix_diffusivity: float,
        characteristic_length: float,
    ) -> float:
        r"""
        Estimate the time to reach pseudo-steady-state (PSS) within a
        matrix block.

        .. math::
            t_{\text{PSS}} \approx 0.1 \frac{L^2}{\eta_m}

        where :math:`\eta_m = k_m / (\phi_m c_m \mu)` is the matrix
        hydraulic diffusivity.

        Parameters
        ----------
        matrix_diffusivity : float
            η_m (m²/s).
        characteristic_length : float
            L (m), typically half the fracture spacing.

        Returns
        -------
        float
            t_PSS (s).
        """
        return 0.1 * characteristic_length**2 / (matrix_diffusivity + 1e-30)

    # ------------------------------------------------------------------
    # Property scaling (temperature-dependent)
    # ------------------------------------------------------------------

    @staticmethod
    def temperature_scale_permeability(
        k_ref: float,
        T_ref: float,
        T_new: float,
        activation_energy: float = 0.0,
    ) -> float:
        r"""
        Scale permeability with temperature using an Arrhenius-like law.

        .. math::
            k(T) = k_{\text{ref}} \exp\left[ -\frac{E_a}{R}
                   \left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right) \right]

        By default :math:`E_a = 0` so permeability is independent of T.

        Parameters
        ----------
        k_ref : float
            Reference permeability (m²).
        T_ref, T_new : float
            Temperatures in **K**.
        activation_energy : float
            E_a (J/mol).  Default 0.

        Returns
        -------
        float
            Scaled permeability.
        """
        if activation_energy == 0.0:
            return k_ref
        R = 8.314462618  # J/(mol·K)
        return k_ref * math.exp(-activation_energy / R * (1.0 / T_new - 1.0 / T_ref))

    @staticmethod
    def temperature_scale_porosity(
        phi_ref: float,
        T_ref: float,
        T_new: float,
        thermal_expansion_coeff: float = 0.0,
    ) -> float:
        r"""
        Linear thermal expansion of porosity.

        .. math::
            \phi(T) = \phi_{\text{ref}} \left[1 + \alpha_T (T - T_{\text{ref}})\right]
        """
        if thermal_expansion_coeff == 0.0:
            return phi_ref
        return phi_ref * (1.0 + thermal_expansion_coeff * (T_new - T_ref))

    # ------------------------------------------------------------------
    # String / repr helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<{self.__class__.__name__} "
            f"phi_m={self.phi_m:.3f} phi_f={self.phi_f:.4f} "
            f"k_m={self.k_m:.3e} k_f={self.k_f:.3e} "
            f"geometry={self.geometry.value}>"
        )


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------


def convert_single_to_dual(
    single_porosity: float,
    single_permeability: float,
    fracture_intensity: float,
    aperture: float,
    matrix_block_size: tuple[float, float, float],
) -> DualPorosityParams:
    """
    Convert measured single-porosity properties into an equivalent
    dual-porosity parameter set.

    Parameters
    ----------
    single_porosity : float
        Measured (bulk) porosity.
    single_permeability : float
        Measured (bulk) permeability (m²).
    fracture_intensity : float
        Number of fractures per unit length (1/m).
    aperture : float
        Average fracture aperture (m).
    matrix_block_size : tuple(float, float, float)
        Dimensions of the equivalent matrix block (m).

    Returns
    -------
    DualPorosityParams
    """
    # Fracture porosity from parallel-plate approximation
    phi_f = fracture_intensity * aperture
    if phi_f >= single_porosity:
        phi_f = single_porosity * 0.5
    phi_m = single_porosity - phi_f

    # Fracture permeability — cubic law (with 1/12 factor)
    k_f = (aperture**3) * fracture_intensity / 12.0
    # Matrix permeability — residual
    # Using weighted harmonic mean relation: 1/k_bulk = (1-phi_f)/k_m + phi_f/k_f
    # Solve for k_m
    inv_km = (1.0 / single_permeability - phi_f / k_f) / (1.0 - phi_f)
    k_m = 1.0 / inv_km if inv_km > 0 else 1e-18  # fallback

    return DualPorosityParams(
        phi_m=phi_m,
        phi_f=phi_f,
        k_m=k_m,
        k_f=k_f,
        Lx=matrix_block_size[0],
        Ly=matrix_block_size[1],
        Lz=matrix_block_size[2],
    )


def convert_dual_to_single(
    phi_m: float,
    phi_f: float,
    k_m: float,
    k_f: float,
) -> tuple[float, float]:
    """
    Upscaling dual-porosity properties to an equivalent single-porosity
    continuum using arithmetic averaging for porosity and a harmonic average
    weighted by ``(1 − phi_f, phi_f)`` for permeability.

    The permeability weights are chosen to make this routine the inverse of
    :func:`convert_single_to_dual`, which uses
    ``1 / k_bulk = (1 − phi_f) / k_m + phi_f / k_f``.

    Returns
    -------
    (single_porosity, single_permeability) : tuple[float, float]
    """
    phi_bulk = phi_m + phi_f
    # Inverse of convert_single_to_dual's harmonic-mean rearrangement:
    #   1/k_bulk = (1 - phi_f)/k_m + phi_f/k_f
    w_f = phi_f
    w_m = 1.0 - phi_f
    k_bulk = 1.0 / (w_m / (k_m + 1e-30) + w_f / (k_f + 1e-30))
    return phi_bulk, k_bulk
