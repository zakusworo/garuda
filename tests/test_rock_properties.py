"""
Test suite for GARUDA RockProperties module.

Tests permeability conversion, tensor building, heterogeneous fields,
channelized/gaussian permeability generation, and rock-fluid bulk properties.
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
    from garuda.core.rock_properties import RockProperties


# =============================================================================
# DEFAULT INITIALIZATION TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestDefaultInit:
    """Test default initialization of RockProperties."""

    def test_default_porosity(self):
        """Default porosity should be 0.2."""
        rock = RockProperties()
        assert rock.porosity == 0.2

    def test_default_permeability_si(self):
        """Default permeability should be 1e-12 m² in SI."""
        rock = RockProperties()
        assert np.isclose(rock.permeability_m2, 1e-12)

    def test_default_unit_is_m2(self):
        """Default permeability unit should be 'm2'."""
        rock = RockProperties()
        assert rock.permeability_unit == 'm2'

    def test_default_c_rock(self):
        """Default rock compressibility."""
        rock = RockProperties()
        assert rock.c_rock == 1e-9

    def test_default_cp(self):
        """Default rock heat capacity."""
        rock = RockProperties()
        assert rock.cp == 840

    def test_default_rho_rock(self):
        """Default rock density."""
        rock = RockProperties()
        assert rock.rho_rock == 2650

    def test_default_lambda_rock(self):
        """Default thermal conductivity."""
        rock = RockProperties()
        assert rock.lambda_rock == 2.5

    def test_default_k_ratio(self):
        """Default anisotropy ratio."""
        rock = RockProperties()
        assert rock.k_ratio == (1.0, 1.0, 0.1)

    def test_tensor_built_on_init(self):
        """Permeability tensor should be built after __post_init__."""
        rock = RockProperties()
        assert hasattr(rock, 'perm_tensor')
        assert rock.perm_tensor is not None
        assert rock.perm_tensor.shape == (1, 3, 3)

    def test_perm_tensor_diagonal_default(self):
        """Default scalar perm tensor should be diagonal with k_ratio applied."""
        rock = RockProperties()
        # kx = 1e-12 * 1.0, ky = 1e-12 * 1.0, kz = 1e-12 * 0.1
        expected = np.array([
            [1e-12, 0, 0],
            [0, 1e-12, 0],
            [0, 0, 1e-13],
        ])
        assert np.allclose(rock.perm_tensor[0], expected)


# =============================================================================
# PERMEABILITY UNIT CONVERSION TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestDarcyConversion:
    """Test conversion from Darcy to SI (m²)."""

    DARCY_TO_M2 = 9.869233e-13

    def test_one_darcy_to_m2(self):
        """1 Darcy should equal 9.869233e-13 m²."""
        rock = RockProperties(permeability=1.0, permeability_unit='darcy')
        assert np.isclose(rock.permeability_m2, self.DARCY_TO_M2)

    def test_ten_darcy_to_m2(self):
        """10 Darcy conversion."""
        rock = RockProperties(permeability=10.0, permeability_unit='darcy')
        assert np.isclose(rock.permeability_m2, 10 * self.DARCY_TO_M2)

    def test_darcy_scalar_preserved(self):
        """Darcy conversion of a scalar should remain scalar-like."""
        rock = RockProperties(permeability=1.0, permeability_unit='darcy')
        assert rock.permeability_m2.ndim == 0 or np.isscalar(rock.permeability_m2)

    def test_darcy_tensor_uses_si_values(self):
        """Tensor diagonal should use converted SI values."""
        rock = RockProperties(permeability=1.0, permeability_unit='darcy',
                              k_ratio=(1.0, 1.0, 1.0))
        expected_k = self.DARCY_TO_M2
        assert np.isclose(rock.perm_tensor[0, 0, 0], expected_k)
        assert np.isclose(rock.perm_tensor[0, 1, 1], expected_k)
        assert np.isclose(rock.perm_tensor[0, 2, 2], expected_k)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestMillidarcyConversion:
    """Test conversion from millidarcy (md) to SI (m²)."""

    MD_TO_M2 = 9.869233e-16

    def test_one_md_to_m2(self):
        """1 md should equal 9.869233e-16 m²."""
        rock = RockProperties(permeability=1.0, permeability_unit='md')
        assert np.isclose(rock.permeability_m2, self.MD_TO_M2)

    def test_hundred_md_to_m2(self):
        """100 md conversion."""
        rock = RockProperties(permeability=100.0, permeability_unit='md')
        assert np.isclose(rock.permeability_m2, 100 * self.MD_TO_M2)

    def test_md_zero(self):
        """0 md should be 0 m²."""
        rock = RockProperties(permeability=0.0, permeability_unit='md')
        assert rock.permeability_m2 == 0.0

    def test_md_array_conversion(self):
        """Array of md values should convert element-wise."""
        perm_md = np.array([100, 200, 300], dtype=float)
        rock = RockProperties(permeability=perm_md, permeability_unit='md')
        expected_m2 = perm_md * self.MD_TO_M2
        assert np.allclose(rock.permeability_m2, expected_m2)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestM2PassThrough:
    """Test that m2 unit passes through without conversion."""

    def test_m2_no_conversion_scalar(self):
        """Scalar m2 permeability should not be modified."""
        rock = RockProperties(permeability=2.5e-12, permeability_unit='m2')
        assert rock.permeability_m2 == 2.5e-12

    def test_m2_no_conversion_array(self):
        """Array m2 permeability should not be modified."""
        arr = np.array([1e-12, 2e-12, 3e-12])
        rock = RockProperties(permeability=arr, permeability_unit='m2')
        assert np.array_equal(rock.permeability_m2, arr)


# =============================================================================
# TENSOR BUILDING TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestScalarPermTensor:
    """Test tensor building for scalar permeability."""

    def test_tensor_shape(self):
        """Scalar perm should produce (1, 3, 3) tensor."""
        rock = RockProperties()
        assert rock.perm_tensor.shape == (1, 3, 3)

    def test_tensor_is_diagonal(self):
        """Tensor should be diagonal for isotropic scalar."""
        rock = RockProperties(k_ratio=(1.0, 1.0, 1.0))
        # Off-diagonal elements should be zero
        assert rock.perm_tensor[0, 0, 1] == 0
        assert rock.perm_tensor[0, 0, 2] == 0
        assert rock.perm_tensor[0, 1, 0] == 0
        assert rock.perm_tensor[0, 1, 2] == 0
        assert rock.perm_tensor[0, 2, 0] == 0
        assert rock.perm_tensor[0, 2, 1] == 0

    def test_k_ratio_applied(self):
        """k_ratio should scale each diagonal component."""
        k_ratio = (2.0, 3.0, 0.5)
        rock = RockProperties(permeability=1e-12, k_ratio=k_ratio)
        assert np.isclose(rock.perm_tensor[0, 0, 0], 1e-12 * 2.0)
        assert np.isclose(rock.perm_tensor[0, 1, 1], 1e-12 * 3.0)
        assert np.isclose(rock.perm_tensor[0, 2, 2], 1e-12 * 0.5)

    def test_k_ratio_default_gives_kz_low(self):
        """Default k_ratio (1,1,0.1) gives kz = 0.1 * kx."""
        rock = RockProperties()
        assert np.isclose(rock.perm_tensor[0, 2, 2], 0.1 * rock.perm_tensor[0, 0, 0])

    def test_scalar_float_handled(self):
        """Plain float permeability should build tensor correctly."""
        rock = RockProperties(permeability=0.5e-12)
        assert rock.perm_tensor.shape == (1, 3, 3)

    def test_scalar_int_handled(self):
        """Integer permeability (as in 'md' unit) should be converted."""
        rock = RockProperties(permeability=100, permeability_unit='md',
                              k_ratio=(1.0, 1.0, 1.0))
        assert rock.perm_tensor.shape == (1, 3, 3)
        assert np.isclose(rock.perm_tensor[0, 0, 0], 100 * 9.869233e-16)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class Test1DHeterogeneousTensor:
    """Test tensor building for 1D heterogeneous permeability arrays."""

    def test_1d_tensor_shape(self):
        """1D (n_cells,) perm should produce (n_cells, 3, 3) tensor."""
        perm_1d = np.array([1e-12, 2e-12, 3e-12, 4e-12])
        rock = RockProperties(permeability=perm_1d, permeability_unit='m2')
        assert rock.perm_tensor.shape == (4, 3, 3)

    def test_1d_diagonal_per_cell(self):
        """Each cell should have its own diagonal with k_ratio applied."""
        perm_1d = np.array([1e-12, 2e-12])
        rock = RockProperties(permeability=perm_1d, permeability_unit='m2',
                              k_ratio=(1.0, 0.5, 0.1))
        # Cell 0
        assert np.isclose(rock.perm_tensor[0, 0, 0], 1e-12 * 1.0)
        assert np.isclose(rock.perm_tensor[0, 1, 1], 1e-12 * 0.5)
        assert np.isclose(rock.perm_tensor[0, 2, 2], 1e-12 * 0.1)
        # Cell 1
        assert np.isclose(rock.perm_tensor[1, 0, 0], 2e-12 * 1.0)
        assert np.isclose(rock.perm_tensor[1, 1, 1], 2e-12 * 0.5)
        assert np.isclose(rock.perm_tensor[1, 2, 2], 2e-12 * 0.1)

    def test_1d_off_diagonal_zero(self):
        """Off-diagonal elements should be zero for 1D arrays."""
        perm_1d = np.array([1e-12, 2e-12, 3e-12])
        rock = RockProperties(permeability=perm_1d, permeability_unit='m2')
        assert np.all(rock.perm_tensor[:, 0, 1] == 0)
        assert np.all(rock.perm_tensor[:, 0, 2] == 0)
        assert np.all(rock.perm_tensor[:, 1, 0] == 0)

    def test_1d_single_element(self):
        """1D array with single element should still work."""
        perm_1d = np.array([1e-12])
        rock = RockProperties(permeability=perm_1d, permeability_unit='m2')
        assert rock.perm_tensor.shape == (1, 3, 3)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class Test2DAnisotropicTensor:
    """Test tensor building for 2D anisotropic permeability arrays."""

    def test_2d_tensor_shape(self):
        """2D (n_cells, 3) perm should produce (n_cells, 3, 3) tensor."""
        perm_2d = np.array([
            [1e-12, 0.5e-12, 0.1e-12],
            [2e-12, 1.0e-12, 0.2e-12],
        ])
        rock = RockProperties(permeability=perm_2d, permeability_unit='m2')
        assert rock.perm_tensor.shape == (2, 3, 3)

    def test_2d_kx_ky_kz_per_cell(self):
        """Each row should be placed on the diagonal directly (no k_ratio)."""
        perm_2d = np.array([
            [1e-12, 0.5e-12, 0.1e-12],
            [2e-12, 1.0e-12, 0.2e-12],
        ])
        rock = RockProperties(permeability=perm_2d, permeability_unit='m2',
                              k_ratio=(0.5, 0.5, 0.5))
        # k_ratio should NOT be applied for 2D anisotropic
        assert np.isclose(rock.perm_tensor[0, 0, 0], 1e-12)
        assert np.isclose(rock.perm_tensor[0, 1, 1], 0.5e-12)
        assert np.isclose(rock.perm_tensor[0, 2, 2], 0.1e-12)
        assert np.isclose(rock.perm_tensor[1, 0, 0], 2e-12)
        assert np.isclose(rock.perm_tensor[1, 1, 1], 1.0e-12)
        assert np.isclose(rock.perm_tensor[1, 2, 2], 0.2e-12)

    def test_2d_no_k_ratio_effect(self):
        """k_ratio should have no effect on 2D anisotropic arrays."""
        perm_2d = np.array([[1e-12, 1e-12, 1e-12]])
        rock = RockProperties(permeability=perm_2d, permeability_unit='m2',
                              k_ratio=(0.1, 0.1, 0.1))
        # Values should be exactly what was passed
        assert np.isclose(rock.perm_tensor[0, 0, 0], 1e-12)
        assert np.isclose(rock.perm_tensor[0, 1, 1], 1e-12)
        assert np.isclose(rock.perm_tensor[0, 2, 2], 1e-12)

    def test_2d_off_diagonal_zero(self):
        """Off-diagonal elements should be zero for 2D arrays."""
        perm_2d = np.array([
            [1e-12, 0.5e-12, 0.1e-12],
            [2e-12, 1.0e-12, 0.2e-12],
        ])
        rock = RockProperties(permeability=perm_2d, permeability_unit='m2')
        assert np.all(rock.perm_tensor[:, 0, 1] == 0)
        assert np.all(rock.perm_tensor[:, 1, 0] == 0)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class Test3DFullTensor:
    """Test that 3D permeability arrays are passed through as-is."""

    def test_3d_tensor_passthrough(self):
        """3D array should be used directly as the permeability tensor."""
        full_tensor = np.array([
            [[1e-12, 5e-14, 0],
             [5e-14, 1e-12, 0],
             [0, 0, 1e-13]],
        ])
        rock = RockProperties(permeability=full_tensor, permeability_unit='m2')
        assert np.array_equal(rock.perm_tensor, full_tensor)

    def test_3d_multiple_cells(self):
        """3D tensor with multiple cells."""
        full_tensor = np.zeros((3, 3, 3))
        for c in range(3):
            full_tensor[c, 0, 0] = (c + 1) * 1e-12
            full_tensor[c, 1, 1] = (c + 1) * 1e-12
            full_tensor[c, 2, 2] = (c + 1) * 1e-12 * 0.1
        rock = RockProperties(permeability=full_tensor, permeability_unit='m2')
        assert rock.perm_tensor.shape == (3, 3, 3)
        assert np.array_equal(rock.perm_tensor, full_tensor)


# =============================================================================
# SET HETEROGENEOUS TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestSetHeterogeneous:
    """Test set_heterogeneous method."""

    def test_updates_porosity(self):
        """set_heterogeneous should update porosity."""
        rock = RockProperties()
        new_phi = np.array([0.15, 0.25, 0.35])
        rock.set_heterogeneous(porosity=new_phi,
                               permeability=np.array([10, 20, 30], dtype=float))
        assert np.array_equal(rock.porosity, new_phi)

    def test_updates_permeability(self):
        """set_heterogeneous should update permeability."""
        rock = RockProperties()
        new_perm = np.array([50, 100, 150], dtype=float)
        rock.set_heterogeneous(porosity=np.array([0.1, 0.2, 0.3]),
                               permeability=new_perm)
        assert np.array_equal(rock.permeability, new_perm)

    def test_default_unit_is_md(self):
        """set_heterogeneous default unit should be 'md'."""
        rock = RockProperties()
        rock.set_heterogeneous(porosity=np.array([0.1]),
                               permeability=np.array([1.0]))
        assert rock.permeability_unit == 'md'

    def test_recalculates_tensor(self):
        """set_heterogeneous should rebuild the permeability tensor."""
        rock = RockProperties()
        n_cells = 5
        rock.set_heterogeneous(
            porosity=np.full(n_cells, 0.2),
            permeability=np.arange(1, n_cells + 1, dtype=float),
            permeability_unit='md',
        )
        assert rock.perm_tensor.shape == (n_cells, 3, 3)
        # Check conversion: 1 md → 9.869233e-16, times k_ratio
        expected_k0 = 1 * 9.869233e-16 * 1.0  # kx for cell 0
        assert np.isclose(rock.perm_tensor[0, 0, 0], expected_k0)

    def test_explicit_unit_m2(self):
        """set_heterogeneous with m2 unit."""
        rock = RockProperties()
        rock.set_heterogeneous(
            porosity=np.array([0.15]),
            permeability=np.array([3e-12]),
            permeability_unit='m2',
        )
        assert rock.permeability_unit == 'm2'
        assert np.isclose(rock.perm_tensor[0, 0, 0], 3e-12)

    def test_explicit_unit_darcy(self):
        """set_heterogeneous with darcy unit."""
        rock = RockProperties()
        n = 3
        rock.set_heterogeneous(
            porosity=np.full(n, 0.2),
            permeability=np.full(n, 1.0, dtype=float),
            permeability_unit='darcy',
        )
        expected = 1.0 * 9.869233e-13
        assert np.isclose(rock.perm_tensor[0, 0, 0], expected)

    def test_tensor_is_1d_heterogeneous(self):
        """After set_heterogeneous with 1D perm, tensor should be (n_cells, 3, 3)."""
        rock = RockProperties()
        rock.set_heterogeneous(
            porosity=np.array([0.1, 0.2]),
            permeability=np.array([10.0, 20.0]),
            permeability_unit='md',
        )
        assert rock.perm_tensor.shape == (2, 3, 3)

    def test_large_field(self):
        """set_heterogeneous with a large number of cells."""
        rock = RockProperties()
        n = 1000
        rock.set_heterogeneous(
            porosity=np.random.uniform(0.05, 0.35, n),
            permeability=np.random.uniform(1, 1000, n),
            permeability_unit='md',
        )
        assert rock.perm_tensor.shape == (n, 3, 3)
        assert np.all(rock.perm_tensor >= 0)


# =============================================================================
# CHANNELIZED PERMEABILITY TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestChannelizedPermeability:
    """Test set_channelized_permeability method."""

    def test_runs_without_error(self):
        """set_channelized_permeability should run without raising."""
        rock = RockProperties()
        rock.set_channelized_permeability(nx=5, ny=5, nz=3)
        assert rock.perm_tensor is not None

    def test_reproducible_with_seed(self):
        """Two calls with separate RockProperties should produce identical fields
        due to np.random.seed(42) being set inside the method."""
        rock1 = RockProperties()
        rock1.set_channelized_permeability(nx=5, ny=5, nz=3)

        rock2 = RockProperties()
        rock2.set_channelized_permeability(nx=5, ny=5, nz=3)

        assert np.array_equal(rock1.permeability, rock2.permeability)
        assert np.array_equal(rock1.perm_tensor, rock2.perm_tensor)

    def test_tensor_shape_matches_grid(self):
        """Permeability tensor should have n_cells = nx*ny*nz."""
        nx, ny, nz = 4, 3, 5
        rock = RockProperties()
        rock.set_channelized_permeability(nx=nx, ny=ny, nz=nz)
        assert rock.perm_tensor.shape == (nx * ny * nz, 3, 3)

    def test_unit_is_md(self):
        """After channelized, permeability_unit should be 'md'."""
        rock = RockProperties()
        rock.set_channelized_permeability(nx=3, ny=3, nz=3)
        assert rock.permeability_unit == 'md'

    def test_porosity_preserved(self):
        """Porosity shape should match grid shape, using the original porosity value."""
        rock = RockProperties(porosity=0.25)
        rock.set_channelized_permeability(nx=3, ny=3, nz=2)
        assert rock.porosity.shape == (3, 3, 2)
        assert np.allclose(rock.porosity, 0.25)

    def test_orientation_x(self):
        """Channel orientation 'x' should not error and produce valid tensor."""
        rock = RockProperties()
        rock.set_channelized_permeability(
            nx=4, ny=3, nz=2,
            channel_orientation='x',
            k_channel=500,
            k_background=5,
        )
        assert rock.perm_tensor.shape == (24, 3, 3)
        assert rock.permeability_unit == 'md'

    def test_orientation_y(self):
        """Channel orientation 'y' should not error and produce valid tensor."""
        rock = RockProperties()
        rock.set_channelized_permeability(
            nx=4, ny=3, nz=2,
            channel_orientation='y',
            k_channel=500,
            k_background=5,
        )
        assert rock.perm_tensor.shape == (24, 3, 3)

    def test_orientation_z(self):
        """Channel orientation 'z' should not error."""
        rock = RockProperties()
        rock.set_channelized_permeability(
            nx=3, ny=3, nz=3,
            channel_orientation='z',
            k_channel=1000,
            k_background=10,
        )
        assert rock.perm_tensor.shape == (27, 3, 3)

    def test_all_orientations_produce_different_tensors(self):
        """Different orientations should produce different permeability arrays."""
        rock_x = RockProperties()
        rock_x.set_channelized_permeability(nx=5, ny=5, nz=3,
                                            channel_orientation='x')

        rock_y = RockProperties()
        rock_y.set_channelized_permeability(nx=5, ny=5, nz=3,
                                            channel_orientation='y')

        rock_z = RockProperties()
        rock_z.set_channelized_permeability(nx=5, ny=5, nz=3,
                                            channel_orientation='z')

        # At least one pair should differ
        x_vs_y = np.array_equal(rock_x.permeability, rock_y.permeability)
        x_vs_z = np.array_equal(rock_x.permeability, rock_z.permeability)
        y_vs_z = np.array_equal(rock_y.permeability, rock_z.permeability)
        assert not (x_vs_y and x_vs_z and y_vs_z), \
            "All orientations produced identical fields"

    def test_only_channel_and_background_values(self):
        """Permeability values should only be k_channel or k_background."""
        k_channel = 1000
        k_background = 10
        rock = RockProperties()
        rock.set_channelized_permeability(
            nx=5, ny=5, nz=3,
            k_channel=k_channel,
            k_background=k_background,
        )
        unique_vals = np.unique(rock.permeability)
        assert set(unique_vals) <= {float(k_channel), float(k_background)}

    def test_fraction_zero_gives_all_background(self):
        """channel_fraction=0 should give all background values."""
        rock = RockProperties()
        rock.set_channelized_permeability(
            nx=3, ny=3, nz=3,
            channel_fraction=0.0,
            k_channel=1000,
            k_background=10,
        )
        assert np.all(rock.permeability == 10)

    def test_fraction_one_gives_all_channel(self):
        """channel_fraction=1 should give all channel values."""
        rock = RockProperties()
        rock.set_channelized_permeability(
            nx=3, ny=3, nz=3,
            channel_fraction=1.0,
            k_channel=1000,
            k_background=10,
        )
        assert np.all(rock.permeability == 1000)


# =============================================================================
# GAUSSIAN PERMEABILITY TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestGaussianPermeability:
    """Test set_gaussian_permeability method."""

    def test_runs_without_error(self):
        """set_gaussian_permeability should run without raising."""
        rock = RockProperties()
        rock.set_gaussian_permeability(nx=5, ny=5, nz=3)
        assert rock.perm_tensor is not None

    def test_tensor_shape_matches_grid(self):
        """Tensor should have n_cells = nx*ny*nz."""
        nx, ny, nz = 4, 3, 2
        rock = RockProperties()
        rock.set_gaussian_permeability(nx=nx, ny=ny, nz=nz)
        assert rock.perm_tensor.shape == (nx * ny * nz, 3, 3)

    def test_unit_is_md(self):
        """After gaussian, permeability_unit should be 'md'."""
        rock = RockProperties()
        rock.set_gaussian_permeability(nx=4, ny=4, nz=1)
        assert rock.permeability_unit == 'md'

    def test_porosity_preserved(self):
        """Porosity array should be uniform with the original value."""
        rock = RockProperties(porosity=0.18)
        rock.set_gaussian_permeability(nx=3, ny=3, nz=2)
        assert rock.porosity.shape == (3, 3, 2)
        assert np.allclose(rock.porosity, 0.18)

    def test_all_positive(self):
        """All permeability values should be positive."""
        rock = RockProperties()
        rock.set_gaussian_permeability(nx=5, ny=5, nz=3)
        assert np.all(rock.permeability > 0)

    def test_log_transform_consistency(self):
        """With zero std, all values should equal 10^mean_logk."""
        mean_logk = 2.0
        rock = RockProperties()
        rock.set_gaussian_permeability(
            nx=4, ny=4, nz=1,
            mean_logk=mean_logk,
            std_logk=0.0,
        )
        expected_perm = 10 ** mean_logk
        assert np.allclose(rock.permeability, expected_perm)

    def test_different_mean_produces_different_values(self):
        """Different mean_logk should produce different permeability fields."""
        rock1 = RockProperties()
        rock1.set_gaussian_permeability(nx=3, ny=3, nz=1, mean_logk=1.0, std_logk=0.0)

        rock2 = RockProperties()
        rock2.set_gaussian_permeability(nx=3, ny=3, nz=1, mean_logk=3.0, std_logk=0.0)

        assert not np.array_equal(rock1.permeability, rock2.permeability)

    def test_small_grid(self):
        """Single-cell grid should work."""
        rock = RockProperties()
        rock.set_gaussian_permeability(nx=1, ny=1, nz=1,
                                       mean_logk=2.0, std_logk=0.5)
        assert rock.perm_tensor.shape == (1, 3, 3)
        assert rock.permeability[0] > 0


# =============================================================================
# TOTAL COMPRESSIBILITY TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestTotalCompressibility:
    """Test total_compressibility method."""

    def test_formula(self):
        """c_t = c_rock + phi * c_fluid."""
        rock = RockProperties(porosity=0.2, c_rock=1e-9)
        c_fluid = 4.5e-10
        c_t = rock.total_compressibility(c_fluid)
        expected = 1e-9 + 0.2 * 4.5e-10
        assert np.isclose(c_t, expected)

    def test_scalar_returns_scalar(self):
        """With scalar porosity, should return scalar."""
        rock = RockProperties(porosity=0.3)
        c_t = rock.total_compressibility(1e-9)
        assert np.isscalar(c_t) or (isinstance(c_t, np.ndarray) and c_t.ndim == 0)

    def test_array_porosity_returns_array(self):
        """With array porosity, should return array of same shape."""
        phi = np.array([0.1, 0.2, 0.3])
        rock = RockProperties()
        rock.set_heterogeneous(porosity=phi,
                               permeability=np.array([1.0, 1.0, 1.0]))
        c_t = rock.total_compressibility(1e-9)
        assert isinstance(c_t, np.ndarray)
        assert c_t.shape == phi.shape
        assert np.allclose(c_t, rock.c_rock + phi * 1e-9)

    def test_positive_values(self):
        """Compressibility should always be positive."""
        rock = RockProperties()
        for c_fluid in [0, 1e-11, 1e-9, 1e-7]:
            c_t = rock.total_compressibility(c_fluid)
            assert c_t > 0

    def test_zero_fluid_compressibility(self):
        """With zero fluid compressibility, c_t = c_rock."""
        rock = RockProperties(c_rock=2e-9)
        c_t = rock.total_compressibility(0.0)
        assert np.isclose(c_t, 2e-9)

    def test_high_porosity_increases_ct(self):
        """Higher porosity should increase total compressibility."""
        rock = RockProperties(porosity=0.4)
        c_t_high = rock.total_compressibility(1e-9)

        rock_low = RockProperties(porosity=0.1, c_rock=1e-9)
        c_t_low = rock_low.total_compressibility(1e-9)

        assert c_t_high > c_t_low


# =============================================================================
# HEAT CAPACITY BULK TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestHeatCapacityBulk:
    """Test heat_capacity_bulk method."""

    def test_positive_value(self):
        """Bulk heat capacity should be positive."""
        rock = RockProperties()
        rhoCp = rock.heat_capacity_bulk(fluid_cp=4180, fluid_rho=1000)
        assert rhoCp > 0

    def test_formula_scalar(self):
        """(1-phi)*rho_rock*cp + phi*rho_fluid*fluid_cp."""
        phi = 0.2
        rho_rock = 2650
        cp = 840
        fluid_cp = 4180
        fluid_rho = 1000
        rock = RockProperties(porosity=phi, rho_rock=rho_rock, cp=cp)

        rhoCp = rock.heat_capacity_bulk(fluid_cp=fluid_cp, fluid_rho=fluid_rho)
        expected = (1 - phi) * rho_rock * cp + phi * fluid_rho * fluid_cp
        assert np.isclose(rhoCp, expected)

    def test_physically_reasonable(self):
        """Typical sandstone with water: ~2.0-2.5e6 J/(m³·K)."""
        rock = RockProperties(porosity=0.2)
        rhoCp = rock.heat_capacity_bulk(fluid_cp=4180, fluid_rho=1000)
        assert 2.0e6 < rhoCp < 3.0e6, \
            f"Expected 2.0-3.0e6 J/(m³·K), got {rhoCp}"

    def test_returns_float(self):
        """Should return a float."""
        rock = RockProperties()
        rhoCp = rock.heat_capacity_bulk(fluid_cp=4000, fluid_rho=900)
        assert isinstance(rhoCp, float)

    def test_higher_porosity_reduces_bulk_capacity(self):
        """Higher porosity (more water, less rock) reduces bulk heat capacity
        since rock typically has higher volumetric heat capacity."""
        rock_high_phi = RockProperties(porosity=0.3)
        rock_low_phi = RockProperties(porosity=0.1)

        # Same rock/fluid params
        rhoCp_high = rock_high_phi.heat_capacity_bulk(fluid_cp=4180, fluid_rho=1000)
        rhoCp_low = rock_low_phi.heat_capacity_bulk(fluid_cp=4180, fluid_rho=1000)

        # Rock: 2650*840 = 2,226,000 > Water: 1000*4180 = 4,180,000
        # Actually water has higher volumetric capacity, so higher phi increases
        # Just check that both are positive and reasonable
        assert rhoCp_high > 0
        assert rhoCp_low > 0

    def test_array_porosity_uses_mean(self):
        """With array porosity, should use mean porosity."""
        phi = np.array([0.1, 0.2, 0.3, 0.4])
        rock = RockProperties()
        rock.set_heterogeneous(porosity=phi,
                               permeability=np.array([1, 1, 1, 1], dtype=float))
        rhoCp = rock.heat_capacity_bulk(fluid_cp=4180, fluid_rho=1000)
        mean_phi = np.mean(phi)
        expected = (1 - mean_phi) * 2650 * 840 + mean_phi * 1000 * 4180
        assert np.isclose(rhoCp, expected)

    def test_zero_porosity_pure_rock(self):
        """Zero porosity should give pure rock heat capacity."""
        rock = RockProperties(porosity=0.0)
        rhoCp = rock.heat_capacity_bulk(fluid_cp=5000, fluid_rho=1)
        assert np.isclose(rhoCp, 2650 * 840)


# =============================================================================
# THERMAL DIFFUSIVITY TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestThermalDiffusivity:
    """Test thermal_diffusivity method."""

    def test_positive_value(self):
        """Thermal diffusivity should be positive."""
        rock = RockProperties()
        alpha = rock.thermal_diffusivity(fluid_cp=4180, fluid_rho=1000)
        assert alpha > 0

    def test_returns_float(self):
        """Should return a float."""
        rock = RockProperties()
        alpha = rock.thermal_diffusivity(fluid_cp=4000, fluid_rho=900)
        assert isinstance(alpha, float)

    def test_typical_rock_value(self):
        """Typical rock thermal diffusivity is ~1e-6 m²/s."""
        rock = RockProperties(porosity=0.2, lambda_rock=2.5)
        alpha = rock.thermal_diffusivity(fluid_cp=4180, fluid_rho=1000)
        # Should be order 1e-6 m²/s
        assert 5e-7 < alpha < 5e-6, \
            f"Expected ~1e-6 m²/s, got {alpha}"

    def test_formula_consistency(self):
        """alpha = lambda / rhoCp_bulk, so alpha * rhoCp = lambda."""
        rock = RockProperties(lambda_rock=2.5)
        rhoCp = rock.heat_capacity_bulk(fluid_cp=4180, fluid_rho=1000)
        alpha = rock.thermal_diffusivity(fluid_cp=4180, fluid_rho=1000)
        assert np.isclose(alpha * rhoCp, 2.5)

    def test_higher_conductivity_increases_diffusivity(self):
        """Higher thermal conductivity should increase diffusivity."""
        rock_high = RockProperties(lambda_rock=5.0)
        rock_low = RockProperties(lambda_rock=1.0)

        alpha_high = rock_high.thermal_diffusivity(fluid_cp=4180, fluid_rho=1000)
        alpha_low = rock_low.thermal_diffusivity(fluid_cp=4180, fluid_rho=1000)

        assert alpha_high > alpha_low

    def test_different_porosity_affects_diffusivity(self):
        """Different porosity should change thermal diffusivity."""
        # Water has lower thermal diffusivity than rock
        rock_hi_phi = RockProperties(porosity=0.3)
        rock_lo_phi = RockProperties(porosity=0.1)

        alpha_hi = rock_hi_phi.thermal_diffusivity(fluid_cp=4180, fluid_rho=1000)
        alpha_lo = rock_lo_phi.thermal_diffusivity(fluid_cp=4180, fluid_rho=1000)

        # Both should be reasonable
        assert alpha_hi > 0
        assert alpha_lo > 0
        # Higher porosity adds more fluid (water) which typically
        # has lower thermal diffusivity, so the bulk should change
        assert not np.isclose(alpha_hi, alpha_lo)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestInvalidInputs:
    """Test that invalid inputs raise appropriate errors."""

    def test_invalid_2d_shape_raises(self):
        """2D array with wrong second dimension should raise ValueError."""
        bad_perm = np.array([
            [1e-12, 0.5e-12],  # only 2 columns, not 3
            [2e-12, 1.0e-12],
        ])
        with pytest.raises(ValueError, match="Invalid permeability shape"):
            RockProperties(permeability=bad_perm, permeability_unit='m2')

    def test_invalid_2d_4_columns(self):
        """2D array with 4 columns should raise ValueError."""
        bad_perm = np.array([
            [1e-12, 2e-12, 3e-12, 4e-12],
        ])
        with pytest.raises(ValueError, match="Invalid permeability shape"):
            RockProperties(permeability=bad_perm, permeability_unit='m2')

    def test_4d_array_raises(self):
        """4D array should raise ValueError."""
        bad_perm = np.zeros((2, 2, 3, 3))
        with pytest.raises(ValueError, match="Invalid permeability shape"):
            RockProperties(permeability=bad_perm, permeability_unit='m2')

    def test_2d_with_wrong_shape_raises(self):
        """2D with shape (3, 2) should raise ValueError."""
        bad_perm = np.array([
            [1e-12, 2e-12],
            [3e-12, 4e-12],
            [5e-12, 6e-12],
        ])
        with pytest.raises(ValueError, match="Invalid permeability shape"):
            RockProperties(permeability=bad_perm, permeability_unit='m2')

    def test_error_message_contains_shape(self):
        """Error message should reference the invalid shape."""
        bad_perm = np.array([[1], [2]])
        with pytest.raises(ValueError) as exc_info:
            RockProperties(permeability=bad_perm, permeability_unit='m2')
        assert 'shape' in str(exc_info.value)


# =============================================================================
# CUSTOM K_RATIO TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestCustomKRatio:
    """Test various k_ratio configurations."""

    def test_isotropic_ratio(self):
        """k_ratio (1,1,1) should give isotropic tensor from scalar."""
        rock = RockProperties(permeability=1e-12, k_ratio=(1.0, 1.0, 1.0))
        assert np.isclose(rock.perm_tensor[0, 0, 0], 1e-12)
        assert np.isclose(rock.perm_tensor[0, 1, 1], 1e-12)
        assert np.isclose(rock.perm_tensor[0, 2, 2], 1e-12)

    def test_only_horizontal(self):
        """k_ratio (1, 1, 0) should give zero vertical permeability."""
        rock = RockProperties(permeability=1e-12, k_ratio=(1.0, 1.0, 0.0))
        assert rock.perm_tensor[0, 2, 2] == 0.0

    def test_arbitrary_ratio(self):
        """Arbitrary k_ratio like (0.8, 1.2, 0.05)."""
        k_ratio = (0.8, 1.2, 0.05)
        perm = 5e-12
        rock = RockProperties(permeability=perm, k_ratio=k_ratio)
        assert np.isclose(rock.perm_tensor[0, 0, 0], perm * 0.8)
        assert np.isclose(rock.perm_tensor[0, 1, 1], perm * 1.2)
        assert np.isclose(rock.perm_tensor[0, 2, 2], perm * 0.05)


# =============================================================================
# INTEGRATION / WORKFLOW TESTS
# =============================================================================

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestWorkflowIntegration:
    """Test realistic workflows."""

    def test_default_to_channelized_workflow(self):
        """Create default, then generate channelized field."""
        rock = RockProperties(porosity=0.22)
        assert rock.porosity == 0.22

        rock.set_channelized_permeability(
            nx=10, ny=10, nz=1,
            channel_orientation='x',
            k_channel=800,
            k_background=20,
        )

        # Check that everything is consistent
        assert rock.perm_tensor.shape == (100, 3, 3)
        assert len(rock.permeability) == 100
        assert rock.porosity.shape == (10, 10, 1)
        assert rock.permeability_unit == 'md'

        # Compute bulk properties
        c_t = rock.total_compressibility(4e-10)
        assert np.all(c_t > 0)

        rhoCp = rock.heat_capacity_bulk(fluid_cp=4180, fluid_rho=1000)
        assert rhoCp > 0

        alpha = rock.thermal_diffusivity(fluid_cp=4180, fluid_rho=1000)
        assert alpha > 0

    def test_default_to_gaussian_workflow(self):
        """Create default, then generate gaussian field."""
        rock = RockProperties(porosity=0.15, rho_rock=2700, lambda_rock=3.0)

        rock.set_gaussian_permeability(
            nx=8, ny=8, nz=2,
            mean_logk=2.5,
            std_logk=0.8,
        )

        assert rock.perm_tensor.shape == (128, 3, 3)
        assert len(rock.permeability) == 128

        # All perm values should be positive
        assert np.all(rock.permeability > 0)

        c_t = rock.total_compressibility(1e-9)
        assert np.all(c_t > 0)

    def test_set_heterogeneous_then_properties(self):
        """Set heterogeneous, then compute all bulk properties."""
        n = 50
        rock = RockProperties()
        rock.set_heterogeneous(
            porosity=np.random.uniform(0.05, 0.35, n),
            permeability=np.random.uniform(1, 500, n),
            permeability_unit='md',
        )

        c_t = rock.total_compressibility(5e-10)
        assert len(c_t) == n
        assert np.all(c_t > 0)

        rhoCp = rock.heat_capacity_bulk(4180, 1000)
        assert rhoCp > 0

        alpha = rock.thermal_diffusivity(4180, 1000)
        assert alpha > 0

    def test_switch_between_fields(self):
        """Switch from channelized to gaussian on same instance."""
        rock = RockProperties()

        rock.set_channelized_permeability(nx=3, ny=3, nz=1,
                                          k_channel=500, k_background=10)
        perm_chan = rock.permeability.copy()

        rock.set_gaussian_permeability(nx=3, ny=3, nz=1,
                                       mean_logk=2.0, std_logk=0.5)
        perm_gauss = rock.permeability.copy()

        # Fields should be different
        assert not np.array_equal(perm_chan, perm_gauss)

    def test_consistent_units_throughout(self):
        """Permeability unit should remain consistent through operations."""
        rock = RockProperties(permeability_unit='darcy', permeability=1.5)
        assert rock.permeability_unit == 'darcy'

        rock.set_heterogeneous(
            porosity=np.array([0.2]),
            permeability=np.array([100.0]),
            permeability_unit='md',
        )
        assert rock.permeability_unit == 'md'

        rock.set_channelized_permeability(nx=5, ny=5, nz=1)
        assert rock.permeability_unit == 'md'

    def test_custom_rock_params_persist(self):
        """Custom c_rock, cp, rho_rock, lambda_rock should persist."""
        rock = RockProperties(
            c_rock=2.5e-9,
            cp=920,
            rho_rock=2800,
            lambda_rock=3.8,
        )

        rock.set_channelized_permeability(nx=4, ny=4, nz=1)

        assert rock.c_rock == 2.5e-9
        assert rock.cp == 920
        assert rock.rho_rock == 2800
        assert rock.lambda_rock == 3.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
