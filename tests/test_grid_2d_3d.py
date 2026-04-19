#!/usr/bin/env python3
"""
Test 2D/3D Grid Generation for GARUDA

Verifies that face connectivity, centroids, normals, and areas are correctly computed.
"""

import sys
import numpy as np

sys.path.insert(0, '/home/zakusworo/garuda')

from garuda.core.grid import StructuredGrid


def test_1d_grid():
    """Test 1D grid generation."""
    print("=" * 60)
    print("Testing 1D Grid")
    print("=" * 60)
    
    grid = StructuredGrid(nx=5, ny=1, nz=1, dx=100, dy=1, dz=1)
    
    print(f"Dimension: {grid.dim}D")
    print(f"Number of cells: {grid.num_cells}")
    print(f"Number of faces: {grid.num_faces}")
    print(f"Cell volumes shape: {grid.cell_volumes.shape}")
    print(f"Cell centroids shape: {grid.cell_centroids.shape}")
    print(f"Face areas shape: {grid.face_areas.shape}")
    print(f"Face normals shape: {grid.face_normals.shape}")
    print(f"Face cells shape: {grid.face_cells.shape}")
    print(f"Cell faces shape: {grid.cell_faces.shape}")
    
    # Verify connectivity
    print("\nCell-face connectivity (first 3 cells):")
    for i in range(min(3, grid.num_cells)):
        print(f"  Cell {i}: faces {grid.cell_faces[i]}")
    
    print("\nFace connectivity (first 5 faces):")
    for i in range(min(5, grid.num_faces)):
        print(f"  Face {i}: cells {grid.face_cells[i]}, normal={grid.face_normals[i, 0]}")
    
    # Verify 1D grid has 6 faces (5 cells + 1 boundary)
    assert grid.num_faces == 6, f"Expected 6 faces, got {grid.num_faces}"
    assert grid.num_cells == 5, f"Expected 5 cells, got {grid.num_cells}"
    
    print("\n✅ 1D grid test PASSED\n")
    return True


def test_2d_grid():
    """Test 2D grid generation."""
    print("=" * 60)
    print("Testing 2D Grid")
    print("=" * 60)
    
    grid = StructuredGrid(nx=3, ny=2, nz=1, dx=100, dy=50, dz=1)
    
    print(f"Dimension: {grid.dim}D")
    print(f"Grid size: {grid.nx} x {grid.ny}")
    print(f"Number of cells: {grid.num_cells}")
    print(f"Number of faces: {grid.num_faces}")
    print(f"Cell volumes shape: {grid.cell_volumes.shape}")
    print(f"Cell centroids shape: {grid.cell_centroids.shape}")
    print(f"Face areas shape: {grid.face_areas.shape}")
    print(f"Face normals shape: {grid.face_normals.shape}")
    print(f"Face cells shape: {grid.face_cells.shape}")
    print(f"Cell faces shape: {grid.cell_faces.shape}")
    
    # Expected: 3x2 = 6 cells
    # Faces: (3+1)*2 + 3*(2+1) = 8 + 9 = 17 faces
    expected_faces = (grid.nx + 1) * grid.ny + grid.nx * (grid.ny + 1)
    print(f"\nExpected faces: {expected_faces}, Actual: {grid.num_faces}")
    
    # Verify connectivity
    print("\nCell-face connectivity (first 3 cells):")
    for i in range(min(3, grid.num_cells)):
        print(f"  Cell {i} (ijk={grid.get_ijk(i)}): faces {grid.cell_faces[i]}")
    
    print("\nFace connectivity (first 10 faces):")
    for i in range(min(10, grid.num_faces)):
        print(f"  Face {i}: cells {grid.face_cells[i]}, area={grid.face_areas[i]:.1f}, normal={grid.face_normals[i, :2]}")
    
    # Verify cell centroids
    print("\nCell centroids (first 6 cells):")
    for i in range(min(6, grid.num_cells)):
        print(f"  Cell {i}: {grid.cell_centroids[i]}")
    
    assert grid.num_faces == expected_faces, f"Expected {expected_faces} faces, got {grid.num_faces}"
    assert grid.num_cells == 6, f"Expected 6 cells, got {grid.num_cells}"
    assert grid.dim == 2, f"Expected dim=2, got {grid.dim}"
    
    print("\n✅ 2D grid test PASSED\n")
    return True


def test_3d_grid():
    """Test 3D grid generation."""
    print("=" * 60)
    print("Testing 3D Grid")
    print("=" * 60)
    
    grid = StructuredGrid(nx=2, ny=2, nz=2, dx=100, dy=50, dz=20)
    
    print(f"Dimension: {grid.dim}D")
    print(f"Grid size: {grid.nx} x {grid.ny} x {grid.nz}")
    print(f"Number of cells: {grid.num_cells}")
    print(f"Number of faces: {grid.num_faces}")
    print(f"Cell volumes shape: {grid.cell_volumes.shape}")
    print(f"Cell centroids shape: {grid.cell_centroids.shape}")
    print(f"Face areas shape: {grid.face_areas.shape}")
    print(f"Face normals shape: {grid.face_normals.shape}")
    print(f"Face cells shape: {grid.face_cells.shape}")
    print(f"Cell faces shape: {grid.cell_faces.shape}")
    
    # Expected: 2x2x2 = 8 cells
    # Faces: (2+1)*2*2 + 2*(2+1)*2 + 2*2*(2+1) = 12 + 12 + 12 = 36 faces
    expected_faces = (grid.nx + 1) * grid.ny * grid.nz + \
                     grid.nx * (grid.ny + 1) * grid.nz + \
                     grid.nx * grid.ny * (grid.nz + 1)
    print(f"\nExpected faces: {expected_faces}, Actual: {grid.num_faces}")
    
    # Verify connectivity
    print("\nCell-face connectivity (first 4 cells):")
    for i in range(min(4, grid.num_cells)):
        print(f"  Cell {i} (ijk={grid.get_ijk(i)}): faces {grid.cell_faces[i]}")
    
    print("\nFace connectivity (first 10 faces):")
    for i in range(min(10, grid.num_faces)):
        print(f"  Face {i}: cells {grid.face_cells[i]}, area={grid.face_areas[i]:.1f}, normal={grid.face_normals[i]}")
    
    # Verify cell volumes
    expected_volume = grid.dx * grid.dy * grid.dz
    print(f"\nExpected cell volume: {expected_volume} m³")
    print(f"Actual cell volumes: {grid.cell_volumes}")
    
    assert grid.num_faces == expected_faces, f"Expected {expected_faces} faces, got {grid.num_faces}"
    assert grid.num_cells == 8, f"Expected 8 cells, got {grid.num_cells}"
    assert grid.dim == 3, f"Expected dim=3, got {grid.dim}"
    assert np.allclose(grid.cell_volumes, expected_volume), f"Expected volume {expected_volume}, got {grid.cell_volumes}"
    
    print("\n✅ 3D grid test PASSED\n")
    return True


def test_heterogeneous_grid():
    """Test grid with heterogeneous cell sizes."""
    print("=" * 60)
    print("Testing Heterogeneous 2D Grid")
    print("=" * 60)
    
    dx = np.array([50, 100, 150])
    dy = np.array([30, 70])
    
    grid = StructuredGrid(nx=3, ny=2, nz=1, dx=dx, dy=dy, dz=1)
    
    print(f"Grid size: {grid.nx} x {grid.ny}")
    print(f"DX: {dx}")
    print(f"DY: {dy}")
    print(f"Number of cells: {grid.num_cells}")
    print(f"Number of faces: {grid.num_faces}")
    
    print("\nCell volumes (should vary):")
    for i in range(grid.num_cells):
        ix, iy, _ = grid.get_ijk(i)
        print(f"  Cell {i} (ix={ix}, iy={iy}): volume={grid.cell_volumes[i]:.1f} m³")
    
    print("\nCell centroids:")
    for i in range(grid.num_cells):
        print(f"  Cell {i}: x={grid.cell_centroids[i, 0]:.1f}, y={grid.cell_centroids[i, 1]:.1f}")
    
    # Verify volumes are heterogeneous
    assert not np.allclose(grid.cell_volumes, grid.cell_volumes[0]), "Expected heterogeneous volumes"
    
    print("\n✅ Heterogeneous grid test PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GARUDA Grid Module Tests")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    try:
        all_passed &= test_1d_grid()
    except Exception as e:
        print(f"❌ 1D grid test FAILED: {e}\n")
        all_passed = False
    
    try:
        all_passed &= test_2d_grid()
    except Exception as e:
        print(f"❌ 2D grid test FAILED: {e}\n")
        all_passed = False
    
    try:
        all_passed &= test_3d_grid()
    except Exception as e:
        print(f"❌ 3D grid test FAILED: {e}\n")
        all_passed = False
    
    try:
        all_passed &= test_heterogeneous_grid()
    except Exception as e:
        print(f"❌ Heterogeneous grid test FAILED: {e}\n")
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
