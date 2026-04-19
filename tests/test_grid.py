"""
Test suite for GARUDA grid module.

Tests structured grid generation, connectivity, and geometry.
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
    from garuda.core.grid import StructuredGrid


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestStructuredGrid1D:
    """Test 1D structured grid."""
    
    def test_grid_creation(self):
        """Test basic 1D grid creation."""
        grid = StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=1, dz=1)
        
        assert grid.dim == 1
        assert grid.num_cells == 5
        assert grid.num_faces == 6
        assert grid.nx == 5
        assert grid.ny == 1
        assert grid.nz == 1
    
    def test_cell_volumes(self):
        """Test cell volume computation."""
        grid = StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=1, dz=1)
        
        expected_volume = 100 * 1 * 1
        assert np.allclose(grid.cell_volumes, expected_volume)
        assert len(grid.cell_volumes) == 5
    
    def test_cell_centroids(self):
        """Test cell centroid positions."""
        grid = StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=1, dz=1)
        
        # First cell centroid should be at dx/2
        assert np.isclose(grid.cell_centroids[0, 0], 50)
        # Last cell centroid should be at (nx - 0.5) * dx
        assert np.isclose(grid.cell_centroids[-1, 0], 450)
    
    def test_face_connectivity(self):
        """Test face-cell connectivity."""
        grid = StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=1, dz=1)
        
        # First face should be boundary (left)
        assert grid.face_cells[0, 0] == -1
        assert grid.face_cells[0, 1] == 0
        
        # Interior face
        assert grid.face_cells[1, 0] == 0
        assert grid.face_cells[1, 1] == 1
        
        # Last face should be boundary (right)
        assert grid.face_cells[-1, 0] == 4
        assert grid.face_cells[-1, 1] == -1
    
    def test_cell_faces(self):
        """Test cell-face connectivity."""
        grid = StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=1, dz=1)
        
        # First cell has faces 0 (left) and 1 (right)
        assert grid.cell_faces[0, 0] == 0
        assert grid.cell_faces[0, 1] == 1
        
        # Last cell has faces 4 (left) and 5 (right)
        assert grid.cell_faces[-1, 0] == 5
        assert grid.cell_faces[-1, 1] == 6
    
    def test_index_conversion(self):
        """Test linear to ijk index conversion."""
        grid = StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=1, dz=1)
        
        for i in range(grid.num_cells):
            ijk = grid.get_ijk(i)
            linear = grid.get_cell_index(*ijk)
            assert linear == i


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestStructuredGrid2D:
    """Test 2D structured grid."""
    
    def test_grid_creation(self):
        """Test basic 2D grid creation."""
        grid = StructuredGrid(nx=3, ny=2, nz=1, dx=100, dy=50, dz=1)
        
        assert grid.dim == 2
        assert grid.num_cells == 6
        assert grid.num_faces == 17  # (3+1)*2 + 3*(2+1) = 8 + 9
        assert grid.nx == 3
        assert grid.ny == 2
    
    def test_face_count(self):
        """Test face count formula."""
        grid = StructuredGrid(nx=4, ny=3, nz=1, dx=100, dy=50, dz=1)
        
        expected_faces = (grid.nx + 1) * grid.ny + grid.nx * (grid.ny + 1)
        assert grid.num_faces == expected_faces
    
    def test_cell_volumes(self):
        """Test cell volumes in 2D."""
        grid = StructuredGrid(nx=3, ny=2, nz=1, dx=100, dy=50, dz=1)
        
        expected_volume = 100 * 50 * 1
        assert np.allclose(grid.cell_volumes, expected_volume)
        assert len(grid.cell_volumes) == 6
    
    def test_face_normals(self):
        """Test face normal directions."""
        grid = StructuredGrid(nx=3, ny=2, nz=1, dx=100, dy=50, dz=1)
        
        # X-faces should have x-component of normal
        num_faces_x = (grid.nx + 1) * grid.ny
        for i in range(num_faces_x):
            # Normal should be in x-direction (±1, 0, 0)
            assert np.isclose(np.abs(grid.face_normals[i, 0]), 1)
            assert np.isclose(grid.face_normals[i, 1], 0)
        
        # Y-faces should have y-component of normal
        for i in range(num_faces_x, grid.num_faces):
            # Normal should be in y-direction (0, ±1, 0)
            assert np.isclose(grid.face_normals[i, 0], 0)
            assert np.isclose(np.abs(grid.face_normals[i, 1]), 1)
    
    def test_boundary_faces(self):
        """Test boundary face identification."""
        grid = StructuredGrid(nx=3, ny=2, nz=1, dx=100, dy=50, dz=1)
        
        boundary_count = np.sum(np.any(grid.face_cells < 0, axis=1))
        
        # For 3x2 grid: 10 boundary faces
        # Left: 2, Right: 2, Bottom: 3, Top: 3
        assert boundary_count == 10


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestStructuredGrid3D:
    """Test 3D structured grid."""
    
    def test_grid_creation(self):
        """Test basic 3D grid creation."""
        grid = StructuredGrid(nx=2, ny=2, nz=2, dx=100, dy=50, dz=20)
        
        assert grid.dim == 3
        assert grid.num_cells == 8
        assert grid.num_faces == 36  # (2+1)*2*2 + 2*(2+1)*2 + 2*2*(2+1) = 12+12+12
        assert grid.nx == 2
        assert grid.ny == 2
        assert grid.nz == 2
    
    def test_face_count(self):
        """Test face count formula for 3D."""
        grid = StructuredGrid(nx=3, ny=2, nz=2, dx=100, dy=50, dz=20)
        
        expected_faces = (
            (grid.nx + 1) * grid.ny * grid.nz +
            grid.nx * (grid.ny + 1) * grid.nz +
            grid.nx * grid.ny * (grid.nz + 1)
        )
        assert grid.num_faces == expected_faces
    
    def test_cell_volumes(self):
        """Test cell volumes in 3D."""
        grid = StructuredGrid(nx=2, ny=2, nz=2, dx=100, dy=50, dz=20)
        
        expected_volume = 100 * 50 * 20
        assert np.allclose(grid.cell_volumes, expected_volume)
        assert len(grid.cell_volumes) == 8
    
    def test_face_normals_3d(self):
        """Test face normals in 3D."""
        grid = StructuredGrid(nx=2, ny=2, nz=2, dx=100, dy=50, dz=20)
        
        num_faces_x = (grid.nx + 1) * grid.ny * grid.nz
        num_faces_y = grid.nx * (grid.ny + 1) * grid.nz
        
        # X-faces
        for i in range(num_faces_x):
            assert np.isclose(np.abs(grid.face_normals[i, 0]), 1)
            assert np.isclose(grid.face_normals[i, 1], 0)
            assert np.isclose(grid.face_normals[i, 2], 0)
        
        # Y-faces
        for i in range(num_faces_x, num_faces_x + num_faces_y):
            assert np.isclose(grid.face_normals[i, 0], 0)
            assert np.isclose(np.abs(grid.face_normals[i, 1]), 1)
            assert np.isclose(grid.face_normals[i, 2], 0)
        
        # Z-faces
        for i in range(num_faces_x + num_faces_y, grid.num_faces):
            assert np.isclose(grid.face_normals[i, 0], 0)
            assert np.isclose(grid.face_normals[i, 1], 0)
            assert np.isclose(np.abs(grid.face_normals[i, 2]), 1)
    
    def test_cell_faces_3d(self):
        """Test cell-face connectivity in 3D."""
        grid = StructuredGrid(nx=2, ny=2, nz=2, dx=100, dy=50, dz=20)
        
        # Each cell should have 6 faces
        assert grid.cell_faces.shape == (8, 6)
        
        # All faces should be valid (>= -1)
        assert np.all(grid.cell_faces >= -1)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestHeterogeneousGrid:
    """Test grids with heterogeneous cell sizes."""
    
    def test_heterogeneous_1d(self):
        """Test 1D grid with variable dx."""
        dx = np.array([50, 100, 150])
        grid = StructuredGrid(nx=3, ny=1, nz=1, dx=dx, dy=1, dz=1)
        
        # Volumes should vary
        assert not np.allclose(grid.cell_volumes, grid.cell_volumes[0])
        
        # Total length should be sum of dx
        total_length = np.sum(dx)
        assert np.isclose(grid.cell_centroids[-1, 0] + dx[-1]/2, total_length)
    
    def test_heterogeneous_2d(self):
        """Test 2D grid with variable dx and dy."""
        dx = np.array([50, 100])
        dy = np.array([30, 70])
        grid = StructuredGrid(nx=2, ny=2, nz=1, dx=dx, dy=dy, dz=1)
        
        # Volumes should vary
        expected_volumes = np.array([
            dx[0] * dy[0],
            dx[1] * dy[0],
            dx[0] * dy[1],
            dx[1] * dy[1],
        ])
        
        assert np.allclose(grid.cell_volumes, expected_volumes)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestGridProperties:
    """Test grid property assignment."""
    
    def test_set_porosity(self):
        """Test porosity assignment."""
        grid = StructuredGrid(nx=3, ny=2, nz=1, dx=100, dy=50, dz=1)
        
        # Homogeneous
        grid.set_porosity(0.2)
        assert np.allclose(grid.porosity, 0.2)
        
        # Heterogeneous
        porosity = np.linspace(0.1, 0.3, grid.num_cells)
        grid.set_porosity(porosity)
        assert np.allclose(grid.porosity, porosity)
    
    def test_set_permiability_scalar(self):
        """Test scalar permeability assignment."""
        grid = StructuredGrid(nx=2, ny=2, nz=1, dx=100, dy=50, dz=1)
        
        # Scalar in millidarcy
        grid.set_permiability(100, unit='md')
        
        # Check shape
        assert grid.permiability.shape == (grid.num_cells, 3, 3)
        
        # Check isotropic
        for i in range(grid.num_cells):
            assert np.isclose(grid.permiability[i, 0, 0], grid.permiability[i, 1, 1])
    
    def test_set_permiability_array(self):
        """Test array permeability assignment."""
        grid = StructuredGrid(nx=2, ny=2, nz=1, dx=100, dy=50, dz=1)
        
        # Heterogeneous
        perm = np.linspace(50, 200, grid.num_cells)
        grid.set_permiability(perm, unit='md')
        
        # Check first and last
        assert grid.permiability[0, 0, 0] < grid.permiability[-1, 0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
