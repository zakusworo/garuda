"""Tests for relative permeability models."""

import numpy as np
import pytest

from garuda.physics.relative_permeability import (
    CoreyRelativePermeability,
    LinearRelativePermeability,
    RelativePermeabilityModel,
    StoneIRelativePermeability,
    VanGenuchtenMualem,
)


class TestCoreyRelativePermeability:
    def test_scalar_saturation(self):
        model = CoreyRelativePermeability(nw=2.0, nn=2.0, swr=0.2, snr=0.1, krw0=0.8, krn0=0.9)
        krw, krn = model(0.5)
        assert isinstance(krw, (float, np.floating))
        assert isinstance(krn, (float, np.floating))
        assert 0.0 <= krw <= 1.0
        assert 0.0 <= krn <= 1.0

    def test_endpoint_behavior(self):
        model = CoreyRelativePermeability(nw=2.0, nn=2.0, swr=0.2, snr=0.1)
        # At residual water saturation: krw = 0
        krw, krn = model(0.2)
        assert pytest.approx(krw, abs=1e-6) == 0.0
        assert krn > 0.0
        # At max water saturation (1 - snr): krn = 0
        krw, krn = model(0.9)
        assert pytest.approx(krn, abs=1e-6) == 0.0
        assert krw > 0.0

    def test_vectorized(self):
        model = CoreyRelativePermeability(nw=2.0, nn=2.0, swr=0.2, snr=0.1)
        sw = np.linspace(0.2, 0.9, 10)
        krw, krn = model(sw)
        assert krw.shape == sw.shape
        assert krn.shape == sw.shape
        assert np.all(krw >= 0.0)
        assert np.all(krn >= 0.0)

    def test_sum_less_than_one(self):
        model = CoreyRelativePermeability(nw=2.0, nn=2.0, swr=0.2, snr=0.1)
        sw = np.linspace(0.21, 0.89, 20)
        krw, krn = model(sw)
        assert np.all(krw + krn <= 1.0 + 1e-6)

    def test_invalid_exponents_raise(self):
        with pytest.raises(ValueError):
            CoreyRelativePermeability(nw=0.0)
        with pytest.raises(ValueError):
            CoreyRelativePermeability(nw=-1.0)


class TestVanGenuchtenMualem:
    def test_scalar(self):
        model = VanGenuchtenMualem(n=2.0, swr=0.2, snr=0.1)
        krw, krn = model(0.5)
        assert 0.0 <= krw <= 1.0
        assert 0.0 <= krn <= 1.0

    def test_vectorized(self):
        model = VanGenuchtenMualem(n=2.0, swr=0.2, snr=0.1)
        sw = np.linspace(0.2, 0.9, 10)
        krw, krn = model(sw)
        assert krw.shape == sw.shape


class TestLinearRelativePermeability:
    def test_linearity(self):
        model = LinearRelativePermeability(swr=0.2, snr=0.1)
        sw = np.linspace(0.2, 0.9, 8)
        krw, krn = model(sw)
        expected_krw = (sw - 0.2) / (0.9 - 0.2)
        np.testing.assert_allclose(krw, expected_krw, atol=1e-6)

    def test_endpoints(self):
        model = LinearRelativePermeability(swr=0.2, snr=0.1)
        krw, krn = model(0.2)
        assert pytest.approx(krw, abs=1e-6) == 0.0
        assert pytest.approx(krn, abs=1e-6) == 1.0


class TestStoneIRelativePermeability:
    def test_three_phase(self):
        krow_model = CoreyRelativePermeability(nw=2.0, swr=0.2, snr=0.1)
        krog_model = CoreyRelativePermeability(nn=2.0, swr=0.2, snr=0.1)
        stone = StoneIRelativePermeability(krow_model=krow_model, krog_model=krog_model)
        krw, kro, krg = stone(S_w=0.3, S_o=0.2)
        assert isinstance(kro, (float, np.floating))
        assert kro >= 0.0
        assert kro <= 1.0
