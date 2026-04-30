#!/usr/bin/env python3
"""
GARUDA Indonesian Geothermal Field Simulation
==============================================
10-year production forecast for volcanic geothermal reservoir.
Uses MultiphaseFlow with IAPWS-97 water/steam properties.

Scenario: Wayang Windu-type liquid-dominated reservoir
- 280°C, 250 bar initial conditions
- Two production wells (45 kg/s each)
- One reinjection well (80% mass return at 25°C)
- 1-day time steps for 1 year (365 steps)
"""

import sys
sys.path.insert(0, '/home/zakusworo/garuda')

import numpy as np
from garuda import StructuredGrid, RockProperties, FluidProperties, MultiphaseFlow
from garuda.core.iapws_properties import WaterSteamProperties

# ==============================================================================
# Reservoir Configuration
# ==============================================================================

print("=" * 70)
print("  GARUDA — INDONESIAN GEOTHERMAL FIELD SIMULATION")
print("  Wayang Windu-type liquid-dominated reservoir")
print("=" * 70)

# Grid: 500m x 500m x 250m, 5x5x5 cells
NX, NY, NZ = 5, 5, 5
DX, DY, DZ = 100.0, 100.0, 50.0
grid = StructuredGrid(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ)

print(f"\nGrid: {NX}x{NY}x{NZ} = {grid.num_cells} cells")
print(f"Domain: {NX*DX:.0f}m x {NY*DY:.0f}m x {NZ*DZ:.0f}m")

# Rock: fractured andesite (volcanic)
rock = RockProperties(
    porosity=0.12,
    permeability=150,              # md
    permeability_unit='md',
    k_ratio=(1.0, 1.0, 0.1),      # kz << kx,ky
    rho_rock=2700,                  # andesite [kg/m³]
    cp=850,                         # heat capacity [J/(kg·K)]
    lambda_rock=2.8,                # thermal conductivity [W/(m·K)]
)

# Fluid: geothermal water/steam
fluid = FluidProperties(fluid_type='geothermal')

# IAPWS-97 for water/steam phase equilibrium
iapws = WaterSteamProperties()

# Create simulator
flow = MultiphaseFlow(grid, rock, fluid, iapws)

# ==============================================================================
# Initial Conditions
# ==============================================================================

P_INIT = 250e5           # 250 bar hydrostatic ~2500m
T_SURFACE = 298.15       # 25°C tropical surface
GRADIENT = 0.06          # 60°C/km volcanic gradient

# Compute geothermal gradient
T_init = flow.compute_geothermal_gradient(T_SURFACE, GRADIENT)
T_init_C = T_init - 273.15

# Adjust temperature to target reservoir conditions
depth_avg = np.mean(-grid.cell_centroids[:, 2])
T_target = T_SURFACE + GRADIENT * depth_avg
T_init = np.full(grid.num_cells, T_target)
S_init = np.full(grid.num_cells, 0.02)  # 2% initial steam

flow.set_initial_state(np.full(grid.num_cells, P_INIT), T_init, S_init)

print(f"Initial: p={P_INIT/1e5:.0f} bar, T={T_target-273.15:.0f}°C, S_s=2%")
print(f"Depth range: {depth_avg:.0f}m, Phase: {flow.get_summary()['phase']}")

# ==============================================================================
# Well Configuration
# ==============================================================================

# Production wells at center
prod1_idx = grid.get_cell_index(NX//2, NY//2, NZ//2)
prod2_idx = grid.get_cell_index(NX//2, NY//2+1, NZ//2)
# Injection well at corner (cold water reinjection)
inj_idx = grid.get_cell_index(0, 0, 2)

Q_PROD = 45.0             # kg/s per production well
T_INJ = 298.15             # Reinjection temperature (25°C)
REINJECTION_FRACTION = 0.80

print(f"\nWells:")
print(f"  PROD-01: cell {prod1_idx} ({NX//2},{NY//2},{NZ//2}), target {Q_PROD} kg/s")
print(f"  PROD-02: cell {prod2_idx} ({NX//2},{NY//2+1},{NZ//2}), target {Q_PROD} kg/s")
print(f"  INJ-01:  cell {inj_idx} (0,0,2), reinject {REINJECTION_FRACTION*100:.0f}% at {T_INJ-273.15:.0f}°C")

# ==============================================================================
# Simulation
# ==============================================================================

DT = 86400.0               # 1 day
N_STEPS = 365              # 1 year
REPORT_EVERY = 30          # Report every 30 days

print(f"\nSimulation: {N_STEPS} steps @ {DT/3600:.0f}h = {N_STEPS*DT/86400/365:.1f} years")
print("-" * 70)
print(f"{'Day':>6} {'T_avg':>7} {'P_avg':>7} {'S_avg':>8} {'Q_total':>8} {'Conv':>5}")
print("-" * 70)

# Track history
p_history = []
T_history = []

for step in range(N_STEPS):
    # Build source terms
    source_terms = np.zeros(grid.num_cells)
    source_terms[prod1_idx] = -Q_PROD
    source_terms[prod2_idx] = -Q_PROD

    # Reinjection
    q_inj = (Q_PROD + Q_PROD) * REINJECTION_FRACTION
    source_terms[inj_idx] = q_inj

    # Heat: reinjection cooling
    heat_sources = np.zeros(grid.num_cells)
    cell_T = flow.state.temperature[inj_idx]
    heat_sources[inj_idx] = q_inj * fluid.cp * (T_INJ - cell_T) / grid.cell_volumes[inj_idx]

    # Execute step
    result = flow.step(DT, source_terms, heat_sources=heat_sources, max_iter=10, tol=1e-4)

    # Track
    p_history.append(np.mean(flow.state.pressure) / 1e5)
    T_history.append(np.mean(flow.state.temperature - 273.15))

    # Report
    if step % REPORT_EVERY == 0 or step == N_STEPS - 1:
        s = flow.get_summary()
        q_total = Q_PROD + Q_PROD
        conv = "yes" if result['converged'] else "no"
        print(f"{step+1:>6} {s['T_avg']:>7.1f} {s['p_avg']:>7.1f} {s['S_avg']:>7.1f}% {q_total:>8.1f} {conv:>5}")

# ==============================================================================
# Final Summary
# ==============================================================================

final = flow.get_summary()
p_decline = (P_INIT/1e5 - final['p_avg']) / (P_INIT/1e5) * 100.0
t_decline = T_init_C - final['T_avg']

print("\n" + "=" * 70)
print("  SIMULATION RESULTS")
print("=" * 70)
print(f"\n  FINAL STATE:")
print(f"    Pressure:          {final['p_avg']:.1f} bar  (decline: {p_decline:.1f}%)")
print(f"    Temperature:       {final['T_avg']:.1f} °C  (decline: {t_decline:.1f}°C)")
print(f"    Steam saturation:  {final['S_avg']:.1f}%")
print(f"    Phase:             {final['phase']}")

# Mass balance
total_produced = Q_PROD * 2 * N_STEPS * DT / 1000  # tons
total_injected = total_produced * REINJECTION_FRACTION
net = total_produced - total_injected

print(f"\n  MASS BALANCE:")
print(f"    Total produced:    {total_produced:>12,.0f} tons")
print(f"    Total injected:    {total_injected:>12,.0f} tons")
print(f"    Net extraction:    {net:>12,.0f} tons")
print(f"    Reinjection ratio: {REINJECTION_FRACTION*100:>11.0f}%")

# Thermal power estimate
h_prod = flow.iapws.enthalpy_liquid(T_init[0]) * 1000  # J/kg
h_inj = flow.iapws.enthalpy_liquid(T_INJ) * 1000
power_mw = (Q_PROD * 2) * (h_prod - h_inj) / 1e6

print(f"\n  THERMAL POWER:")
print(f"    h_prod @ {T_init[0]-273.15:.0f}°C:   {h_prod/1e3:>12.0f} kJ/kg")
print(f"    h_inj  @ {T_INJ-273.15:.0f}°C:     {h_inj/1e3:>12.0f} kJ/kg")
print(f"    Power (thermal):   {power_mw:>12.1f} MW")
print(f"    Annual energy:     {power_mw*8760/1000:>12,.0f} GWh")

print("\n" + "=" * 70)
print("  Simulation completed successfully!")
print("=" * 70)
