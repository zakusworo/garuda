"""Tests for capillary pressure models."""

import numpy as np
import pytest

from garuda.physics.capillary_pressure import BrooksCoreyPc, VanGenuchtenPc


class TestBrooksCoreyPc:
    def test_scalar(self):
        model = BrooksCoreyPc(pd=1e5, lambda_=2.0, swr=0.2, snr=0.1)
        pc = model(0.5)
        assert pc > 0.0

    def test_endpoints(self):
        model = BrooksCoreyPc(pd=1e5, lambda_=2.0, swr=0.2, snr=0.1)
        # At residual water saturation: Pc -> max (clamped in implementation)
        pc_swr = model(0.2)
        # At max water: Pc -> entry pressure
        pc_max = model(0.9)
        assert pc_max < pc_swr

    def test_vectorized(self):
        model = BrooksCoreyPc(pd=1e5, lambda_=2.0, swr=0.2, snr=0.1)
        sw = np.linspace(0.21, 0.89, 10)
        pc = model(sw)
        assert pc.shape == sw.shape
        assert np.all(np.isfinite(pc))

    def test_derivative(self):
        model = BrooksCoreyPc(pd=1e5, lambda_=2.0, swr=0.2, snr=0.1)
        sw = np.linspace(0.21, 0.89, 10)
        dpc = model.dpc_dsw(sw)
        assert dpc.shape == sw.shape
        assert np.all(np.isfinite(dpc))


class TestVanGenuchtenPc:
    def test_scalar(self):
        model = VanGenuchtenPc(p0=1e5, n=2.0, swr=0.2, snr=0.1)
        pc = model(0.5)
        assert pc > 0.0

    def test_vectorized(self):
        model = VanGenuchtenPc(p0=1e5, n=2.0, swr=0.2, snr=0.1)
        sw = np.linspace(0.21, 0.89, 10)
        pc = model(sw)
        assert pc.shape == sw.shape

    def test_derivative_finite(self):
        model = VanGenuchtenPc(p0=1e5, n=2.0, swr=0.2, snr=0.1)
        sw = np.linspace(0.21, 0.89, 10)
        dpc = model.dpc_dsw(sw)
        assert np.all(np.isfinite(dpc))
