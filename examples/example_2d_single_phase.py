#!/usr/bin/env python3
"""
GARUDA 2D Example - Conceptual demonstration (requires numpy)

This example shows how to use GARUDA for 2D single-phase flow simulation.
Requires: pip install numpy scipy

Problem setup:
    - 2D domain: 500m x 500m, 5x5 cells (100m x 100m each)
    - Left boundary: p = 200 bar
    - Right boundary: p = 100 bar
    - Top/Bottom: no-flow (Neumann)
    - Permeability: 100 md (homogeneous)
    - Porosity: 0.2
    - Fluid: water (mu = 1 cP, rho = 1000 kg/m³)

Expected result:
    - Pressure gradient from left to right
    - Symmetric pressure field in y-direction
"""

import sys
sys.path.insert(0, '/home/zakusworo/garuda')

try:
    import numpy as np
    from garuda import StructuredGrid, TPFASolver, FluidProperties, RockProperties
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available. Running conceptual demo only.")


def run_2d_simulation():
    """Run 2D single-phase flow simulation."""
    
    print("=" * 70)
    print("GARUDA 2D Single-Phase Flow Simulation")
    print("=" * 70)
    
    # Create 2D grid
    print("\n1. Creating 2D grid...")
    grid = StructuredGrid(nx=5, ny=5, nz=1, dx=100, dy=100, dz=10)
    
    print(f"   Grid dimensions: {grid.nx} x {grid.ny} x {grid.nz}")
    print(f"   Cell count: {grid.num_cells}")
    print(f"   Face count: {grid.num_faces}")
    print(f"   Grid type: {grid.dim}D")
    
    # Set rock properties
    print("\n2. Setting rock properties...")
    rock = RockProperties(porosity=0.2, permeability=100, permeability_unit='md')
    grid.set_permiability(rock.permiability_m2)
    grid.set_porosity(rock.porosity)
    print(f"   Permeability: 100 md = {rock.permiability_m2:.2e} m²")
    print(f"   Porosity: {rock.porosity}")
    
    # Set fluid properties
    print("\n3. Setting fluid properties...")
    fluid = FluidProperties(fluid_type='water')
    print(f"   Viscosity: {fluid.mu*1000:.1f} cP")
    print(f"   Density: {fluid.rho:.0f} kg/m³")
    
    # Create solver
    print("\n4. Creating TPFA solver...")
    solver = TPFASolver(grid, mu=fluid.mu, rho=fluid.rho)
    print(f"   Transmissibilities computed: min={solver.transmissibilities.min():.2e}, max={solver.transmissibilities.max():.2e}")
    
    # Define boundary conditions
    print("\n5. Setting boundary conditions...")
    p_left = 200e5  # 200 bar
    p_right = 100e5  # 100 bar
    bc_values = [p_left, p_right]
    print(f"   Left boundary: {p_left/1e5:.0f} bar (Dirichlet)")
    print(f"   Right boundary: {p_right/1e5:.0f} bar (Dirichlet)")
    print(f"   Top/Bottom: no-flow (Neumann)")
    
    # No source terms
    source_terms = np.zeros(grid.num_cells)
    
    # Solve
    print("\n6. Solving pressure equation...")
    pressure = solver.solve(
        source_terms=source_terms,
        bc_type='dirichlet',
        bc_values=bc_values,
        solver='direct'
    )
    
    # Post-process
    print("\n7. Results:")
    print(f"   Pressure range: {pressure.min()/1e5:.1f} - {pressure.max()/1e5:.1f} bar")
    print(f"   Pressure drop: {(pressure.max() - pressure.min())/1e5:.1f} bar")
    
    # Reshape for visualization
    p_2d = pressure.reshape((grid.nx, grid.ny))
    
    print("\n   Pressure field (bar):")
    print("   " + "-" * 50)
    for j in range(grid.ny - 1, -1, -1):
        row = "   |"
        for i in range(grid.nx):
            row += f" {p_2d[i, j]/1e5:5.1f} |"
        print(row)
    print("   " + "-" * 50)
    
    # Compute fluxes
    print("\n8. Computing fluxes...")
    flux_data = solver.compute_flux(pressure)
    total_flux = np.sum(flux_data.flux[flux_data.flux > 0])  # Sum of positive fluxes
    print(f"   Total flux through domain: {total_flux:.4f} kg/s")
    
    # Mass balance check
    print("\n9. Mass balance check...")
    residual = solver.compute_residual(pressure, source_terms)
    max_residual = np.max(np.abs(residual))
    print(f"   Max residual: {max_residual:.2e} kg/s")
    if max_residual < 1e-10:
        print("   ✅ Mass balance satisfied!")
    else:
        print("   ⚠️  Mass balance error detected")
    
    print("\n" + "=" * 70)
    print("✅ 2D simulation completed successfully!")
    print("=" * 70)
    
    return pressure, flux_data


def run_3d_simulation():
    """Run 3D single-phase flow simulation (small grid for demo)."""
    
    print("\n" + "=" * 70)
    print("GARUDA 3D Single-Phase Flow Simulation")
    print("=" * 70)
    
    # Create 3D grid
    print("\n1. Creating 3D grid...")
    grid = StructuredGrid(nx=3, ny=3, nz=3, dx=50, dy=50, dz=10)
    
    print(f"   Grid dimensions: {grid.nx} x {grid.ny} x {grid.nz}")
    print(f"   Cell count: {grid.num_cells}")
    print(f"   Face count: {grid.num_faces}")
    print(f"   Grid type: {grid.dim}D")
    
    # Set rock properties
    print("\n2. Setting rock properties...")
    rock = RockProperties(porosity=0.15, permeability=50, permeability_unit='md')
    grid.set_permiability(rock.permiability_m2)
    grid.set_porosity(rock.porosity)
    
    # Set fluid properties
    print("\n3. Setting fluid properties...")
    fluid = FluidProperties(fluid_type='water')
    
    # Create solver
    print("\n4. Creating TPFA solver...")
    solver = TPFASolver(grid, mu=fluid.mu, rho=fluid.rho)
    
    # Boundary conditions
    print("\n5. Setting boundary conditions...")
    p_top = 150e5  # 150 bar
    p_bottom = 200e5  # 200 bar
    bc_values = [p_bottom, p_top]
    
    # Gravity will create hydrostatic gradient
    source_terms = np.zeros(grid.num_cells)
    
    # Solve
    print("\n6. Solving pressure equation (with gravity)...")
    pressure = solver.solve(
        source_terms=source_terms,
        bc_type='dirichlet',
        bc_values=bc_values,
        solver='direct'
    )
    
    # Results
    print("\n7. Results:")
    print(f"   Pressure range: {pressure.min()/1e5:.1f} - {pressure.max()/1e5:.1f} bar")
    
    # Show pressure by layer
    print("\n   Pressure by layer (bar):")
    for k in range(grid.nz):
        layer_pressure = pressure[k * grid.nx * grid.ny:(k + 1) * grid.nx * grid.ny]
        print(f"   Layer {k} (z={k*grid.dz:.0f}m): avg={layer_pressure.mean()/1e5:.1f} bar")
    
    print("\n" + "=" * 70)
    print("✅ 3D simulation completed successfully!")
    print("=" * 70)
    
    return pressure


if __name__ == "__main__":
    if not NUMPY_AVAILABLE:
        print("\n" + "=" * 70)
        print("CONCEPTUAL DEMO - NumPy not installed")
        print("=" * 70)
        print("""
This example demonstrates GARUDA's 2D/3D capabilities.

To run the actual simulation:
  1. Install dependencies: pip install numpy scipy numba
  2. Run: python examples/example_2d_single_phase.py

What this example does:
  - Creates a 5x5 2D Cartesian grid (500m x 500m)
  - Sets homogeneous rock properties (100 md, 0.2 porosity)
  - Applies Dirichlet BCs: 200 bar (left), 100 bar (right)
  - Solves TPFA finite volume discretization
  - Computes pressure field and fluxes
  - Verifies mass balance

Expected output:
  - Linear pressure gradient from left to right
  - Symmetric pressure distribution in y-direction
  - Total flux consistent with Darcy's law
  - Mass balance residual < 1e-10 kg/s

The same workflow applies to 3D simulations.
""")
        print("=" * 70)
    else:
        # Run simulations
        pressure_2d, flux_2d = run_2d_simulation()
        pressure_3d = run_3d_simulation()
