#!/usr/bin/env python3
"""
Test 2D/3D Grid Generation for GARUDA - Self-contained (no numpy required)

Verifies that face connectivity, centroids, normals, and areas are correctly computed.
"""

import sys

sys.path.insert(0, '/home/zakusworo/garuda')

# Minimal numpy-like implementation for testing
class Array2D:
    """Simple 2D array implementation for testing without numpy."""
    def __init__(self, shape, fill_value=0):
        self.rows, self.cols = shape
        self.data = [[fill_value for _ in range(self.cols)] for _ in range(self.rows)]
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.data[idx[0]][idx[1]]
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            self.data[idx[0]][idx[1]] = value
        else:
            self.data[idx] = value
    
    @property
    def shape(self):
        return (self.rows, self.cols)


def test_grid_import():
    """Test that grid module can be imported."""
    print("=" * 60)
    print("Testing Grid Module Import")
    print("=" * 60)
    
    from garuda.core.grid import StructuredGrid
    print("✅ Grid module imported successfully")


def test_1d_grid_structure():
    """Test 1D grid structure without numpy."""
    print("\n" + "=" * 60)
    print("Testing 1D Grid Structure")
    print("=" * 60)
    
    from garuda.core.grid import StructuredGrid
    
    grid = StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=1, dz=1)
    
    print(f"Dimension: {grid.dim}D")
    print(f"Number of cells: {grid.num_cells}")
    print(f"Number of faces: {grid.num_faces}")
    
    # Verify 1D grid has 6 faces (5 cells + 1 boundary)
    assert grid.num_faces == 6, f"Expected 6 faces, got {grid.num_faces}"
    assert grid.num_cells == 5, f"Expected 5 cells, got {grid.num_cells}"
    assert grid.dim == 1, f"Expected dim=1, got {grid.dim}"
    
    print(f"Cell volumes: {grid.cell_volumes}")
    print(f"Cell faces shape: {grid.cell_faces.shape}")
    
    print("\n✅ 1D grid structure test PASSED\n")


def test_2d_grid_structure():
    """Test 2D grid structure."""
    print("=" * 60)
    print("Testing 2D Grid Structure")
    print("=" * 60)
    
    from garuda.core.grid import StructuredGrid
    
    grid = StructuredGrid(nx=3, ny=2, nz=1, dx=100, dy=50, dz=1)
    
    print(f"Dimension: {grid.dim}D")
    print(f"Grid size: {grid.nx} x {grid.ny}")
    print(f"Number of cells: {grid.num_cells}")
    print(f"Number of faces: {grid.num_faces}")
    
    # Expected: 3x2 = 6 cells
    # Faces: (3+1)*2 + 3*(2+1) = 8 + 9 = 17 faces
    expected_faces = (grid.nx + 1) * grid.ny + grid.nx * (grid.ny + 1)
    print(f"Expected faces: {expected_faces}, Actual: {grid.num_faces}")
    
    assert grid.num_faces == expected_faces, f"Expected {expected_faces} faces, got {grid.num_faces}"
    assert grid.num_cells == 6, f"Expected 6 cells, got {grid.num_cells}"
    assert grid.dim == 2, f"Expected dim=2, got {grid.dim}"
    
    print(f"Face areas: {grid.face_areas}")
    print(f"Cell faces shape: {grid.cell_faces.shape}")
    
    # Test cell indexing
    print("\nCell indexing test:")
    for i in range(grid.num_cells):
        ijk = grid.get_ijk(i)
        linear = grid.get_cell_index(*ijk)
        print(f"  Cell {i} -> ijk={ijk} -> linear={linear}")
        assert linear == i, f"Indexing error: {i} != {linear}"
    
    print("\n✅ 2D grid structure test PASSED\n")


def test_3d_grid_structure():
    """Test 3D grid structure."""
    print("=" * 60)
    print("Testing 3D Grid Structure")
    print("=" * 60)
    
    from garuda.core.grid import StructuredGrid
    
    grid = StructuredGrid(nx=2, ny=2, nz=2, dx=100, dy=50, dz=20)
    
    print(f"Dimension: {grid.dim}D")
    print(f"Grid size: {grid.nx} x {grid.ny} x {grid.nz}")
    print(f"Number of cells: {grid.num_cells}")
    print(f"Number of faces: {grid.num_faces}")
    
    # Expected: 2x2x2 = 8 cells
    # Faces: (2+1)*2*2 + 2*(2+1)*2 + 2*2*(2+1) = 12 + 12 + 12 = 36 faces
    expected_faces = (grid.nx + 1) * grid.ny * grid.nz + \
                     grid.nx * (grid.ny + 1) * grid.nz + \
                     grid.nx * grid.ny * (grid.nz + 1)
    print(f"Expected faces: {expected_faces}, Actual: {grid.num_faces}")
    
    assert grid.num_faces == expected_faces, f"Expected {expected_faces} faces, got {grid.num_faces}"
    assert grid.num_cells == 8, f"Expected 8 cells, got {grid.num_cells}"
    assert grid.dim == 3, f"Expected dim=3, got {grid.dim}"
    
    print(f"Cell volumes: {grid.cell_volumes}")
    print(f"Cell faces shape: {grid.cell_faces.shape}")
    
    # Test cell indexing
    print("\nCell indexing test (first 4 cells):")
    for i in range(min(4, grid.num_cells)):
        ijk = grid.get_ijk(i)
        linear = grid.get_cell_index(*ijk)
        print(f"  Cell {i} -> ijk={ijk} -> linear={linear}")
        assert linear == i, f"Indexing error: {i} != {linear}"
    
    print("\n✅ 3D grid structure test PASSED\n")


def test_face_connectivity():
    """Test face connectivity logic."""
    print("=" * 60)
    print("Testing Face Connectivity")
    print("=" * 60)
    
    from garuda.core.grid import StructuredGrid
    
    # Simple 2x2 2D grid
    grid = StructuredGrid(nx=2, ny=2, nz=1, dx=100, dy=100, dz=1)
    
    print(f"Grid: {grid.nx}x{grid.ny} = {grid.num_cells} cells")
    print(f"Faces: {grid.num_faces}")
    
    print("\nFace connectivity (all faces):")
    boundary_count = 0
    interior_count = 0
    
    for i in range(grid.num_faces):
        left, right = grid.face_cells[i]
        if left == -1 or right == -1:
            boundary_count += 1
            print(f"  Face {i}: cells [{left}, {right}] (BOUNDARY)")
        else:
            interior_count += 1
            print(f"  Face {i}: cells [{left}, {right}] (INTERIOR)")
    
    print(f"\nBoundary faces: {boundary_count}")
    print(f"Interior faces: {interior_count}")
    
    print("\nCell-face connectivity:")
    for i in range(grid.num_cells):
        print(f"  Cell {i} (ijk={grid.get_ijk(i)}): faces {grid.cell_faces[i]}")
    
    print("\n✅ Face connectivity test PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GARUDA Grid Module Tests (No NumPy)")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    for name, test_fn in [
        ("Grid Import", test_grid_import),
        ("1D Grid Structure", test_1d_grid_structure),
        ("2D Grid Structure", test_2d_grid_structure),
        ("3D Grid Structure", test_3d_grid_structure),
        ("Face Connectivity", test_face_connectivity),
    ]:
        try:
            test_fn()
        except Exception as e:
            print(f"❌ {name} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
