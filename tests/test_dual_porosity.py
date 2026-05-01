"""Tests for dual-porosity module."""

import numpy as np
import pytest

from garuda.core.dual_porosity import (
    BlockGeometry,
    DualPorosityModel,
    DualPorosityParams,
    TransferModel,
    convert_dual_to_single,
    convert_single_to_dual,
)


class TestDualPorosityParams:
    def test_creation(self):
        params = DualPorosityParams(phi_m=0.15, phi_f=0.01, k_m=1e-15, k_f=1e-12, Lx=10, Ly=10, Lz=10)
        assert params.total_porosity == pytest.approx(0.16, abs=1e-6)
        assert params.storativity_ratio > 0.0


class TestDualPorosityModel:
    def test_init(self):
        model = DualPorosityModel(
            matrix_porosity=0.15,
            matrix_permeability=1e-15,
            fracture_porosity=0.01,
            fracture_permeability=1e-12,
            fracture_spacing=(10.0, 10.0, 10.0),
        )
        assert model.geometry == BlockGeometry.CUBE

    def test_slab_geometry(self):
        model = DualPorosityModel(
            matrix_porosity=0.15,
            matrix_permeability=1e-15,
            fracture_porosity=0.01,
            fracture_permeability=1e-12,
            fracture_spacing=(10.0, 10.0, 100.0),
        )
        # aspect ratio = 100/10 = 10 > 5, auto-detects slab
        assert model.geometry in (BlockGeometry.SLAB_X, BlockGeometry.SLAB_Y, BlockGeometry.SLAB_Z)

    def test_warren_root_shape_factor(self):
        model = DualPorosityModel(
            matrix_porosity=0.15,
            matrix_permeability=1e-15,
            fracture_porosity=0.01,
            fracture_permeability=1e-12,
            fracture_spacing=(10.0, 10.0, 10.0),
        )
        sigma = model.warren_root_shape_factor()
        assert sigma > 0.0
        # For cube: sigma = pi^2 * (1/Lx^2 + 1/Ly^2 + 1/Lz^2)
        expected = np.pi**2 * 3 / 100.0
        assert pytest.approx(sigma, rel=0.01) == expected

    def test_kazemi_shape_factor(self):
        model = DualPorosityModel(
            matrix_porosity=0.15,
            matrix_permeability=1e-15,
            fracture_porosity=0.01,
            fracture_permeability=1e-12,
            fracture_spacing=(10.0, 10.0, 10.0),
        )
        sigma = model.kazemi_shape_factor()
        assert sigma > 0.0
        expected = 4.0 * 3 / 100.0
        assert pytest.approx(sigma, rel=0.01) == expected

    def test_interporosity_flow_coefficient(self):
        model = DualPorosityModel(
            matrix_porosity=0.15,
            matrix_permeability=1e-15,
            fracture_porosity=0.01,
            fracture_permeability=1e-12,
            fracture_spacing=(10.0, 10.0, 10.0),
        )
        lam = model.interporosity_flow_coefficient(compressibility=1e-9, viscosity=1e-3)
        assert lam > 0.0

    def test_omega(self):
        model = DualPorosityModel(
            matrix_porosity=0.15,
            matrix_permeability=1e-15,
            fracture_porosity=0.01,
            fracture_permeability=1e-12,
            fracture_spacing=(10.0, 10.0, 10.0),
        )
        omega_val = model.omega(c_m=1e-9, c_f=1e-9)
        assert 0.0 < omega_val < 1.0

    def test_lim_aguilera_transfer_function(self):
        t_dim = np.linspace(0.0, 1.0, 11)
        f = DualPorosityModel.lim_aguilera_transfer_function(t_dim, n_terms=10)
        assert f.shape == t_dim.shape
        assert np.all(np.isfinite(f))
        assert np.all(f >= 0.0)

    def test_pseudo_steady_state_time(self):
        t_pss = DualPorosityModel.pseudo_steady_state_time(
            matrix_diffusivity=1e-7,
            characteristic_length=5.0,
        )
        assert t_pss > 0.0

    def test_temperature_scale_permeability_no_change(self):
        k = DualPorosityModel.temperature_scale_permeability(k_ref=1e-15, T_ref=300, T_new=300)
        assert pytest.approx(k, rel=1e-6) == 1e-15

    def test_temperature_scale_porosity_no_change(self):
        phi = DualPorosityModel.temperature_scale_porosity(phi_ref=0.15, T_ref=300, T_new=300)
        assert pytest.approx(phi, rel=1e-6) == 0.15

    def test_conversion_functions(self):
        dual = convert_single_to_dual(
            single_porosity=0.16,
            single_permeability=1e-13,
            fracture_intensity=1.0,
            aperture=1e-3,
            matrix_block_size=(10, 10, 10),
        )
        # dual is a DualPorosityParams dataclass
        assert dual.phi_m > 0.0
        assert dual.phi_f > 0.0
        phi_total, k_total = convert_dual_to_single(phi_m=0.15, phi_f=0.01, k_m=1e-15, k_f=1e-12)
        assert pytest.approx(phi_total, abs=1e-6) == 0.16
        assert k_total > 0.0
