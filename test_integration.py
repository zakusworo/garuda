#!/usr/bin/env python3
"""End-to-end integration test — exercises the real GARUDA API.

This script verifies that all core modules integrate correctly:
  1. Grid creation + property assignment
  2. TPFA single-phase solve (1D/2D/3D)
  3. Well model coupling
  4. IAPWS-IF97 thermophysical properties
  5. Relative permeability + capillary pressure (multiphase prep)
  6. Dual-porosity transfer
  7. Source network
  8. Region thermodynamics
"""

from __future__ import annotations

import numpy as np

from garuda import (
    BlockGeometry,
    BrooksCoreyPc,
    CoreyRelativePermeability,
    DualPorosityModel,
    IAPWSFluidProperties,
    PeacemanWell,
    RegionThermodynamics,
    RockProperties,
    SourceNetwork,
    SourceNode,
    StructuredGrid,
    TPFASolver,
    WellOperatingConditions,
    WellParameters,
)


def test_1d_single_phase() -> None:
    """1D single-phase flow with Dirichlet BCs."""
    print("\n[1] 1D Single-Phase Flow")
    print("-" * 50)

    grid = StructuredGrid(nx=20, ny=1, nz=1, dx=100.0, dy=10.0, dz=10.0)
    grid.set_permeability(100, unit="md")
    grid.set_porosity(0.2)

    solver = TPFASolver(grid, mu=1e-3, rho=1000.0)
    source = np.zeros(grid.num_cells)
    bc = np.array([200e5, 100e5])

    p = solver.solve(source, bc_type="dirichlet", bc_values=bc, solver="direct")

    # Cell-centred FV: cells are at half-cell offset from the boundaries,
    # so the linear pressure profile is sampled at x=(i+0.5)/nx.
    expected = 200e5 + (100e5 - 200e5) * (np.arange(grid.num_cells) + 0.5) / grid.num_cells
    err = np.max(np.abs(p - expected))

    print(f"  Pressure range: {p.min()/1e5:.1f} - {p.max()/1e5:.1f} bar")
    print(f"  Max deviation from linear: {err:.3e} Pa")
    assert err < 1e-3, f"1D solve inaccurate: {err}"
    print("  PASS")


def test_2d_heterogeneous() -> None:
    """2D flow with heterogeneous permeability (channelized)."""
    print("\n[2] 2D Heterogeneous Flow")
    print("-" * 50)

    grid = StructuredGrid(nx=30, ny=30, nz=1, dx=50.0, dy=50.0, dz=10.0)
    rock = RockProperties()
    rock.set_channelized_permeability(
        nx=30, ny=30, nz=1,
        channel_orientation="x",
        channel_fraction=0.2,
        k_channel=1000.0,
        k_background=10.0,
    )
    grid.set_permeability(rock.permeability_m2)
    grid.set_porosity(0.2)

    solver = TPFASolver(grid, mu=1e-3, rho=1000.0)
    source = np.zeros(grid.num_cells)
    p = solver.solve(source, bc_type="dirichlet", bc_values=np.array([150e5, 100e5]))

    print(f"  Grid: {grid.nx}x{grid.ny} = {grid.num_cells} cells")
    print(f"  Perm range: {rock.permeability_m2.min():.2e} - {rock.permeability_m2.max():.2e} m²")
    print(f"  Pressure range: {p.min()/1e5:.1f} - {p.max()/1e5:.1f} bar")
    assert np.isfinite(p).all()
    print("  PASS")


def test_well_model() -> None:
    """Peaceman well with BHP constraint."""
    print("\n[3] Well Model (Peaceman)")
    print("-" * 50)

    params = WellParameters(
        name="PROD-1",
        cell_index=5,
        well_radius=0.1,
        skin_factor=0.0,
        well_depth=1000.0,
    )
    ops = WellOperatingConditions(
        constraint_type="pressure",
        target_value=150e5,
        max_rate=50.0,
        min_bhp=80e5,
    )
    well = PeacemanWell(params, ops)

    # Must compute PI before calling compute_rate
    well.compute_productivity_index(
        permeability=1e-14,
        viscosity=1e-3,
        dx=100.0,
        dy=100.0,
        dz=10.0,
    )

    rate = well.compute_rate(
        cell_pressure=200e5,
        wellbore_pressure=150e5,
        density=780.0,
    )

    print(f"  Well rate: {rate:.3f} kg/s")
    print(f"  BHP: {well.current_bhp/1e5:.1f} bar")
    # Sign convention: negative = production (drawdown). p_wf < p_cell here.
    assert rate < 0, "Production well should have negative rate"
    print("  PASS")


def test_iapws_properties() -> None:
    """IAPWS-IF97 water/steam properties."""
    print("\n[4] IAPWS-IF97 Thermophysical Properties")
    print("-" * 50)

    fluid = IAPWSFluidProperties()
    rho = fluid.get_density(p=15e6, T=550.0)
    mu = fluid.get_viscosity(p=15e6, T=550.0)
    h = fluid.get_enthalpy(p=15e6, T=550.0)
    all_props = fluid.get_all(p=15e6, T=550.0)

    print(f"  @ 15 MPa, 550 K:")
    print(f"    Density:   {rho:.2f} kg/m³")
    print(f"    Viscosity: {mu:.6f} Pa·s")
    print(f"    Enthalpy:  {h:.2f} kJ/kg")
    print(f"    Cp:        {all_props.get('specific_heat_cp', 'N/A')} kJ/(kg·K)")
    print(f"    Thermal k: {all_props.get('thermal_conductivity', 'N/A')} W/(m·K)")

    assert rho > 0 and mu > 0 and h > 0
    assert isinstance(all_props, dict)
    print("  PASS")


def test_multiphase_models() -> None:
    """Relative permeability + capillary pressure models."""
    print("\n[5] Multiphase Models (Rel-Perm + Capillary Pressure)")
    print("-" * 50)

    # Two-phase oil-water
    relperm = CoreyRelativePermeability(krw0=0.3, krn0=0.8, nw=2.0, nn=2.0, swr=0.15, snr=0.2)
    pc = BrooksCoreyPc(pd=1e5, lambda_=2.0, swr=0.15, snr=0.2)

    sw = np.linspace(0.2, 0.8, 20)
    krw, krn = relperm(sw)
    pc_val = pc(sw)

    print(f"  Sw range: {sw.min():.2f} - {sw.max():.2f}")
    print(f"  krw range: {krw.min():.3f} - {krw.max():.3f}")
    print(f"  krn range: {krn.min():.3f} - {krn.max():.3f}")
    print(f"  Pc range: {pc_val.min()/1e5:.2f} - {pc_val.max()/1e5:.2f} bar")

    assert (krw >= 0).all() and (krw <= 1).all()
    assert (krn >= 0).all() and (krn <= 1).all()
    assert (pc_val >= 0).all()
    print("  PASS")


def test_dual_porosity() -> None:
    """Dual-porosity MINC model."""
    print("\n[6] Dual-Porosity / MINC")
    print("-" * 50)

    dp = DualPorosityModel(
        matrix_porosity=0.15,
        matrix_permeability=1e-16,
        fracture_porosity=0.05,
        fracture_permeability=1e-13,
        fracture_spacing=(0.5, 0.5, 0.5),
        geometry=BlockGeometry.SLAB_Z,
    )

    sigma = dp.warren_root_shape_factor()
    print(f"  Shape factor: {sigma:.4f} 1/m²")

    # Interporosity flow coefficient (real attribute name on DualPorosityModel)
    k_m = dp.k_m
    lambda_coeff = sigma * k_m / dp.Lx**2
    print(f"  Interporosity coeff: {lambda_coeff:.2e}")

    assert sigma > 0
    assert lambda_coeff > 0
    print("  PASS")

def test_source_network() -> None:
    """Source-network wiring: groups, separator, reinjector, and connect()."""
    print("\n[7] Source Network")
    print("-" * 50)

    from garuda import Reinjector, Separator, SourceGroup

    network = SourceNetwork()

    # Group with one producer (rate +50) and one injector (rate -40).
    group = SourceGroup(name="WELLS")
    group.add_node(SourceNode(name="PROD-01", cell_index=0, rate=50.0))
    group.add_node(SourceNode(name="INJ-01", cell_index=99, rate=-40.0))
    network.add_group(group)

    # Surface separator and reinjector wired together.
    sep = Separator(name="SEP-01", separation_curve={"water": 1.0})
    reinj = Reinjector(name="REINJ-01", cell_index=99, target_rate=10.0,
                       inlet_stream="water")
    network.add_separator(sep)
    network.add_reinjector(reinj)
    network.connect("group", "WELLS", "separator", "SEP-01", stream_phase="water")
    network.connect("separator", "SEP-01", "reinjector", "REINJ-01",
                    stream_phase="water")

    total_rate = group.compute_group_rate()
    print(f"  Group total rate: {total_rate:.1f} kg/s")
    print(f"  Number of nodes:  {len(group.nodes)}")
    print(f"  Connections:      {len(network.connections)}")
    assert abs(total_rate - 10.0) < 1e-6
    assert len(network.connections) == 2
    print("  PASS")


def test_region_thermodynamics() -> None:
    """Region-based EOS with saturation curve."""
    print("\n[8] Region Thermodynamics")
    print("-" * 50)

    rt = RegionThermodynamics()

    # Water region (subcritical liquid)
    state1 = rt.get_properties(pressure=10e6, temperature=573.15)
    print(f"  @ 10 MPa, 300°C: region={state1.region}, rho={state1.density:.2f} kg/m³")

    # Steam region
    state2 = rt.get_properties(pressure=1e6, temperature=623.15)
    print(f"  @ 1 MPa, 350°C: region={state2.region}, rho={state2.density:.2f} kg/m³")

    # Supercritical
    state3 = rt.get_properties(pressure=25e6, temperature=673.15)
    print(f"  @ 25 MPa, 400°C: region={state3.region}, rho={state3.density:.2f} kg/m³")

    # Saturation curve
    t_sat = rt.saturation_curve.saturation_temperature(5e6)
    print(f"  Saturation T @ 5 MPa: {t_sat - 273.15:.1f}°C")
    p_sat = rt.saturation_curve.saturation_pressure(573.15)
    print(f"  Saturation P @ 300°C: {p_sat/1e6:.2f} MPa")

    assert state1.region == "water"
    assert state2.region == "steam"
    assert state3.region == "supercritical"
    print("  PASS")


def test_3d_with_gravity() -> None:
    """3D flow with gravity term."""
    print("\n[9] 3D Flow with Gravity")
    print("-" * 50)

    grid = StructuredGrid(nx=10, ny=10, nz=5, dx=50.0, dy=50.0, dz=20.0)
    grid.set_permeability(1e-14)
    grid.set_porosity(0.2)

    solver = TPFASolver(grid, mu=1e-3, rho=1000.0, g=9.81)
    source = np.zeros(grid.num_cells)

    # Bottom = high pressure, top = low pressure
    p = solver.solve(source, bc_type="dirichlet", bc_values=np.array([300e5, 100e5]))

    print(f"  Grid: {grid.nx}x{grid.ny}x{grid.nz} = {grid.num_cells} cells")
    print(f"  Pressure range: {p.min()/1e5:.1f} - {p.max()/1e5:.1f} bar")
    assert np.isfinite(p).all()
    print("  PASS")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  GARUDA Integration Test Suite")
    print("=" * 60)

    try:
        test_1d_single_phase()
        test_2d_heterogeneous()
        test_well_model()
        test_iapws_properties()
        test_multiphase_models()
        test_dual_porosity()
        test_source_network()
        test_region_thermodynamics()
        test_3d_with_gravity()

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        raise
