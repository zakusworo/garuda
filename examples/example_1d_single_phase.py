#!/usr/bin/env python3
"""
GARUDA Example: 1D Single-Phase Flow

Demonstrates basic TPFA solver usage for 1D flow with Dirichlet boundary conditions.

Problem setup:
    - 1D domain: 1000m length, 10 cells (100m each)
    - Left boundary: p = 200 bar
    - Right boundary: p = 100 bar
    - Permeability: 100 md (homogeneous)
    - Porosity: 0.2
    - Fluid: water (mu = 1 cP, rho = 1000 kg/m³)

Expected result:
    - Linear pressure drop from 200 to 100 bar
    - Constant flux throughout domain

GARUDA: Geothermal And Reservoir Understanding with Data-driven Analytics
"""

import numpy as np
import sys
sys.path.insert(0, '/home/zakusworo/garuda')

from garuda.core.grid import StructuredGrid
from garuda.core.tpfa_solver import TPFASolver
from garuda.core.fluid_properties import FluidProperties
from garuda.core.rock_properties import RockProperties


def main():
    print("=" * 60)
    print("GARUDA 1D Single-Phase Flow Example")
    print("=" * 60)
    
    # === Grid Setup ===
    print("\n[1] Creating grid...")
    nx = 10
    dx = 100.0  # meters
    dy = 10.0   # meters (cross-section)
    dz = 10.0   # meters
    
    grid = StructuredGrid(nx=nx, ny=1, nz=1, dx=dx, dy=dy, dz=dz)
    print(f"    Grid: {nx} cells × {dx}m = {nx*dx}m domain")
    print(f"    Cross-section: {dy}m × {dz}m = {dy*dz}m²")
    print(f"    Total cells: {grid.num_cells}")
    
    # === Rock Properties ===
    print("\n[2] Setting rock properties...")
    rock = RockProperties(
        porosity=0.2,
        permeability=100,  # millidarcy
        permeability_unit='md',
    )
    print(f"    Porosity: {rock.porosity}")
    print(f"    Permeability: {100} md = {rock.permiability_m2[0]:.2e} m²")
    
    # Set properties on grid
    grid.set_permiability(rock.permiability_m2)
    grid.set_porosity(rock.porosity)
    
    # === Fluid Properties ===
    print("\n[3] Setting fluid properties...")
    fluid = FluidProperties(fluid_type='water')
    print(f"    Viscosity: {fluid.mu*1000:.1f} cP")
    print(f"    Density: {fluid.rho:.0f} kg/m³")
    
    # === TPFA Solver ===
    print("\n[4] Initializing TPFA solver...")
    solver = TPFASolver(grid, mu=fluid.mu, rho=fluid.rho)
    print(f"    Transmissibilities computed: {len(solver.transmissibilities)} faces")
    print(f"    Interior T: {solver.transmissibilities[1:-1].mean():.2e} m³·s/kg")
    
    # === Boundary Conditions ===
    print("\n[5] Setting boundary conditions...")
    p_left = 200e5  # 200 bar in Pa
    p_right = 100e5  # 100 bar in Pa
    bc_values = np.array([p_left, p_right])
    print(f"    Left boundary (x=0): {p_left/1e5:.0f} bar")
    print(f"    Right boundary (x=L): {p_right/1e5:.0f} bar")
    
    # === Solve ===
    print("\n[6] Solving pressure equation...")
    source_terms = np.zeros(grid.num_cells)  # No sources/sinks
    
    pressure = solver.solve(
        source_terms=source_terms,
        bc_type='dirichlet',
        bc_values=bc_values,
    )
    
    # === Post-processing ===
    print("\n[7] Results:")
    print("-" * 60)
    print(f"    Pressure range: {pressure.min()/1e5:.1f} - {pressure.max()/1e5:.1f} bar")
    print(f"    Pressure drop: {(pressure.max() - pressure.min())/1e5:.1f} bar")
    
    # Compute flux
    flux_data = solver.compute_flux(pressure)
    print(f"    Flux at left boundary: {flux_data.flux[0]:.4f} kg/s")
    print(f"    Flux at right boundary: {flux_data.flux[-1]:.4f} kg/s")
    
    # Mass balance check
    residual = solver.compute_residual(pressure, source_terms)
    print(f"    Mass balance residual: {np.abs(residual).max():.2e} kg/s")
    
    # === Display pressure profile ===
    print("\n[8] Pressure Profile:")
    print("-" * 60)
    print(f"    {'Cell':>4} | {'x (m)':>8} | {'p (bar)':>10} | {'p (exact)':>10} | {'Error (%)':>10}")
    print("    " + "-" * 50)
    
    for i in range(nx):
        x = (i + 0.5) * dx
        p_numerical = pressure[i] / 1e5
        # Analytical solution: linear drop
        p_exact = p_left/1e5 - (p_left/1e5 - p_right/1e5) * (x / (nx * dx))
        error = abs(p_numerical - p_exact) / p_exact * 100
        print(f"    {i:>4} | {x:>8.1f} | {p_numerical:>10.2f} | {p_exact:>10.2f} | {error:>10.4f}")
    
    # === Verify analytical solution ===
    print("\n[9] Verification:")
    print("-" * 60)
    
    # Analytical flux (Darcy's law)
    A = dy * dz  # Cross-sectional area
    k = rock.permiability_m2[0]  # m²
    dp = p_left - p_right  # Pa
    L = nx * dx  # m
    Q_analytical = k * A * dp / (fluid.mu * L)  # m³/s
    Q_mass_analytical = Q_analytical * fluid.rho  # kg/s
    
    Q_numerical = flux_data.flux[0]  # kg/s
    
    print(f"    Analytical mass flux: {Q_mass_analytical:.4f} kg/s")
    print(f"    Numerical mass flux:  {Q_numerical:.4f} kg/s")
    print(f"    Relative error: {abs(Q_numerical - Q_mass_analytical) / Q_mass_analytical * 100:.4f}%")
    
    # Check if solution is correct
    if abs(Q_numerical - Q_mass_analytical) / Q_mass_analytical < 0.01:
        print("\n    ✅ Solution verified! Results match analytical solution.")
    else:
        print("\n    ⚠️  Warning: Numerical solution differs from analytical.")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    return pressure, flux_data


if __name__ == '__main__':
    pressure, flux = main()
