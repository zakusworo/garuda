#!/usr/bin/env python3
"""GARUDA Reservoir Simulator — Interactive Streamlit GUI.

Run with:
    cd /home/zakusworo/garuda
    source .venv/bin/activate
    streamlit run garuda_gui.py

Modules exposed:
    - Grid Builder (1D/2D/3D structured)
    - Single-Phase Flow (TPFA pressure solve)
    - Well Model (Peaceman productivity)
    - IAPWS-IF97 Fluid Properties
    - Multiphase Models (rel-perm + capillary pressure)
    - Dual Porosity / MINC
    - Source Network
    - Region Thermodynamics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── GARUDA imports ───────────────────────────────────────────────────────────
from garuda import (
    BlockGeometry,
    BrooksCoreyPc,
    CoreyRelativePermeability,
    DualPorosityModel,
    IAPWSFluidProperties,
    LinearRelativePermeability,
    PeacemanWell,
    RegionThermodynamics,
    RockProperties,
    SaturationCurve,
    SourceNetwork,
    SourceNode,
    StoneIRelativePermeability,
    StructuredGrid,
    TPFASolver,
    VanGenuchtenMualem,
    VanGenuchtenPc,
    WellOperatingConditions,
    WellParameters,
)

st.set_page_config(
    page_title="GARUDA Reservoir Simulator",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.title("🔥 GARUDA Simulator")
st.sidebar.markdown("*Open-Source Reservoir & Geothermal Simulation*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Home",
        "📐 Grid Builder",
        "💧 Single-Phase Flow",
        "🎯 Well Model",
        "🌡️ IAPWS-IF97 Properties",
        "⚗️ Multiphase Models",
        "🪨 Dual Porosity / MINC",
        "🌐 Source Network",
        "🔬 Region Thermodynamics",
    ],
)

# ── Helper: unit conversion ────────────────────────────────────────────────
def md_to_m2(k_md: float) -> float:
    """Millidarcy -> m²."""
    return k_md * 9.869233e-16


def m2_to_md(k_m2: float) -> float:
    """m² -> millidarcy."""
    return k_m2 / 9.869233e-16


# ═══════════════════════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("GARUDA Reservoir Simulator v0.2.0")
    st.markdown(
        """
        **GARUDA** is an open-source Python reservoir simulator for petroleum
        and geothermal systems, featuring:

        - ✅ **Structured grid generation** (1D/2D/3D)
        - ✅ **TPFA finite-volume pressure solver** with Numba acceleration
        - ✅ **Peaceman well model** (BHP / rate constraints)
        - ✅ **IAPWS-IF97** water/steam thermophysical properties
        - ✅ **Relative permeability** (Corey, van Genuchten-Mualem, Stone I)
        - ✅ **Capillary pressure** (Brooks-Corey, van Genuchten)
        - ✅ **Dual-porosity / MINC** (Warren-Root, Kazemi, Lim-Aguilera)
        - ✅ **Source networks** (producers, injectors, separators, reinjectors)
        - ✅ **Region-based thermodynamics** (water / steam / supercritical)
        - ✅ **Optional PETSc backend** for distributed-memory HPC

        Choose a module from the sidebar to start simulating.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Suite", "557 passed")
    c2.metric("Code Coverage", "~89 %")
    c3.metric("Version", "0.2.0")

    st.divider()
    st.info(
        "💡 **Tip:** All inputs use SI units (Pa, m, kg, K) unless otherwise noted."
    )

# ═══════════════════════════════════════════════════════════════════════════
#  GRID BUILDER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📐 Grid Builder":
    st.title("📐 Grid Builder")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Dimensions")
        nx = st.number_input("NX", 1, 200, 20)
        ny = st.number_input("NY", 1, 200, 1)
        nz = st.number_input("NZ", 1, 100, 1)

        st.subheader("Cell Sizes (m)")
        dx = st.number_input("ΔX", 0.1, 10000.0, 100.0)
        dy = st.number_input("ΔY", 0.1, 10000.0, 100.0)
        dz = st.number_input("ΔZ", 0.1, 10000.0, 10.0)

        st.subheader("Properties")
        k_md = st.number_input("Permeability (md)", 0.001, 1e6, 100.0)
        poro = st.slider("Porosity", 0.01, 0.5, 0.2)

        build_btn = st.button("Build Grid", type="primary")

    with col2:
        if build_btn:
            grid = StructuredGrid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
            grid.set_permeability(k_md, unit="md")
            grid.set_porosity(poro)

            st.success(
                f"Grid built: {nx}×{ny}×{nz} = **{grid.num_cells:,} cells**  |  "
                f"Total volume = **{grid.cell_volumes.sum():.3e} m³**"
            )

            # Visualise cross-section for 2D/3D
            if ny > 1 or nz > 1:
                kx = grid.permeability[:, 0, 0]
                perm_2d = kx.reshape((nz, ny, nx)).mean(axis=0)
                fig = px.imshow(
                    perm_2d,
                    color_continuous_scale="Viridis",
                    labels={"color": "Permeability (m²)"},
                    title="Mean Kx Permeability (Z-averaged)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                x = np.arange(nx) * dx + dx / 2
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=grid.permeability[:, 0, 0],
                        mode="lines+markers",
                        name="Kx Permeability",
                    )
                )
                fig.update_layout(
                    title="1D Kx Permeability Profile",
                    xaxis_title="Distance (m)",
                    yaxis_title="Permeability (m²)",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Export dataframe
            df = pd.DataFrame(
                {
                    "Cell": np.arange(grid.num_cells),
                    "Kx (m²)": grid.permeability[:, 0, 0],
                    "Ky (m²)": grid.permeability[:, 1, 1],
                    "Kz (m²)": grid.permeability[:, 2, 2],
                    "Porosity": grid.porosity,
                    "Volume (m³)": grid.cell_volumes,
                }
            )
            st.dataframe(df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  SINGLE-PHASE FLOW
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💧 Single-Phase Flow":
    st.title("💧 Single-Phase Flow Solver")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Grid")
        nx = st.number_input("NX", 2, 200, 50)
        ny = st.number_input("NY", 1, 200, 1)
        nz = st.number_input("NZ", 1, 100, 1)
        dx = st.number_input("ΔX (m)", 1.0, 10000.0, 50.0)
        dy = st.number_input("ΔY (m)", 1.0, 10000.0, 50.0)
        dz = st.number_input("ΔZ (m)", 0.1, 10000.0, 10.0)

        st.subheader("Fluid & Rock")
        k_md = st.number_input("Permeability (md)", 0.001, 1e6, 100.0)
        poro = st.slider("Porosity", 0.01, 0.5, 0.2)
        mu = st.number_input("Viscosity (Pa·s)", 1e-6, 1.0, 1e-3, format="%.1e")
        rho = st.number_input("Density (kg/m³)", 10.0, 2000.0, 1000.0)

        st.subheader("Boundary Conditions")
        p_left = st.number_input("Left / Bottom P (bar)", 1.0, 1000.0, 200.0)
        p_right = st.number_input("Right / Top P (bar)", 1.0, 1000.0, 100.0)

        st.subheader("Source / Sink")
        source_type = st.selectbox("Source pattern", ["None", "Uniform", "Gaussian", "Point"])
        source_strength = st.number_input("Source strength (kg/s)", -1000.0, 1000.0, 0.0)

        run_btn = st.button("Run Simulation", type="primary")

    with col2:
        if run_btn:
            grid = StructuredGrid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
            grid.set_permeability(k_md, unit="md")
            grid.set_porosity(poro)

            solver = TPFASolver(grid, mu=mu, rho=rho)

            # Build source term
            source = np.zeros(grid.num_cells)
            if source_type == "Uniform":
                source[:] = source_strength / grid.num_cells
            elif source_type == "Gaussian":
                cx, cy = nx // 2, ny // 2
                for j in range(ny):
                    for i in range(nx):
                        idx = j * nx + i
                        r2 = ((i - cx) / (nx / 4)) ** 2 + ((j - cy) / (ny / 4)) ** 2
                        source[idx] = source_strength * np.exp(-r2)
            elif source_type == "Point":
                source[grid.num_cells // 2] = source_strength

            bc = np.array([p_left * 1e5, p_right * 1e5])
            p = solver.solve(source, bc_type="dirichlet", bc_values=bc, solver="direct")

            st.success(
                f"Converged  |  Pressure range: **{p.min() / 1e5:.2f} – {p.max() / 1e5:.2f} bar**"
            )

            # 1D plot
            if ny == 1 and nz == 1:
                x = np.arange(nx) * dx + dx / 2
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=p / 1e5,
                        mode="lines+markers",
                        name="Pressure",
                        line=dict(color="#1f77b4"),
                    )
                )
                fig.update_layout(
                    title="1D Pressure Profile",
                    xaxis_title="Distance (m)",
                    yaxis_title="Pressure (bar)",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 2D heatmap
                p2d = p.reshape((nz, ny, nx)).mean(axis=0)
                fig = px.imshow(
                    p2d / 1e5,
                    color_continuous_scale="RdBu_r",
                    labels={"color": "Pressure (bar)"},
                    title="Pressure Field (Z-averaged)",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Flow-rate summary
            dx_m = dx
            area = dy * dz
            grad_p = (p[0] - p[-1]) / (nx * dx_m)
            k_m2 = md_to_m2(k_md)
            q_darcy = -k_m2 * area / mu * grad_p  # m³/s
            st.metric("Darcy flow rate (approx)", f"{q_darcy:.3e} m³/s")

# ═══════════════════════════════════════════════════════════════════════════
#  WELL MODEL
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Well Model":
    st.title("🎯 Peaceman Well Model")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Well Parameters")
        well_name = st.text_input("Well name", "PROD-01")
        rw = st.number_input("Well radius (m)", 0.01, 1.0, 0.1)
        skin = st.number_input("Skin factor", -5.0, 50.0, 0.0)
        depth = st.number_input("Well depth (m)", 0.0, 10000.0, 1000.0)

        st.subheader("Operating Constraint")
        constraint = st.selectbox("Constraint", ["pressure", "rate"])
        target = st.number_input("Target BHP (bar) or Rate (kg/s)", 1.0, 1000.0, 150.0)
        max_rate = st.number_input("Max rate (kg/s)", 0.0, 1000.0, 50.0)
        min_bhp = st.number_input("Min BHP (bar)", 1.0, 1000.0, 80.0)

        st.subheader("Reservoir Conditions")
        k_md = st.number_input("Permeability (md)", 0.001, 1e6, 100.0)
        mu = st.number_input("Viscosity (Pa·s)", 1e-6, 1.0, 1e-3, format="%.1e")
        dx = st.number_input("ΔX (m)", 1.0, 1000.0, 100.0)
        dy = st.number_input("ΔY (m)", 1.0, 1000.0, 100.0)
        dz = st.number_input("ΔZ (m)", 0.1, 1000.0, 10.0)
        p_res = st.number_input("Reservoir pressure (bar)", 1.0, 1000.0, 200.0)
        p_wf = st.number_input("Wellbore pressure (bar)", 1.0, 1000.0, 150.0)
        rho = st.number_input("Fluid density (kg/m³)", 10.0, 2000.0, 780.0)

        run_btn = st.button("Compute Rate", type="primary")

    with col2:
        if run_btn:
            params = WellParameters(
                name=well_name,
                cell_index=0,
                well_radius=rw,
                skin_factor=skin,
                well_depth=depth,
            )
            ops = WellOperatingConditions(
                constraint_type=constraint,
                target_value=target * 1e5 if constraint == "pressure" else target,
                max_rate=max_rate,
                min_bhp=min_bhp * 1e5,
            )
            well = PeacemanWell(params, ops)

            well.compute_productivity_index(
                permeability=md_to_m2(k_md),
                viscosity=mu,
                dx=dx,
                dy=dy,
                dz=dz,
            )

            rate = well.compute_rate(
                cell_pressure=p_res * 1e5,
                wellbore_pressure=p_wf * 1e5,
                density=rho,
            )

            st.success(
                f"**{well_name}**  |  Rate = **{rate:.3f} kg/s**  |  "
                f"BHP = **{well.current_bhp / 1e5:.2f} bar**"
            )

            # IPR / VLP style plot
            pwf_range = np.linspace(p_res * 0.3, p_res * 1.05, 100)
            rates = []
            for pwf in pwf_range:
                r = well.compute_rate(
                    cell_pressure=p_res * 1e5,
                    wellbore_pressure=pwf * 1e5,
                    density=rho,
                )
                rates.append(r)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=rates,
                    y=pwf_range,
                    mode="lines",
                    name="IPR",
                    line=dict(color="#ff7f0e", width=2),
                )
            )
            fig.add_hline(
                y=p_wf,
                line_dash="dash",
                annotation_text=f"Operating Pwf = {p_wf:.1f} bar",
            )
            fig.add_vline(
                x=rate,
                line_dash="dash",
                annotation_text=f"Rate = {rate:.2f} kg/s",
            )
            fig.update_layout(
                title="Inflow Performance Relationship (IPR)",
                xaxis_title="Flow Rate (kg/s)",
                yaxis_title="Bottom-Hole Pressure (bar)",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # PI display
            if well.productivity_index is not None:
                st.metric(
                    "Productivity Index (PI)",
                    f"{well.productivity_index:.3e} m³/(s·Pa)",
                )

# ═══════════════════════════════════════════════════════════════════════════
#  IAPWS-IF97
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌡️ IAPWS-IF97 Properties":
    st.title("🌡️ IAPWS-IF97 Water / Steam Properties")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Conditions")
        p_mpa = st.slider("Pressure (MPa)", 0.1, 100.0, 15.0)
        t_c = st.slider("Temperature (°C)", 0.0, 800.0, 280.0)
        t_k = t_c + 273.15

        calc_btn = st.button("Calculate", type="primary")

    with col2:
        if calc_btn:
            fluid = IAPWSFluidProperties()
            props = fluid.get_all(p=p_mpa * 1e6, T=t_k)

            st.success(f"Phase: **{props.get('phase', 'unknown')}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Density", f"{props.get('density', 0):.2f}", "kg/m³")
            c2.metric("Viscosity", f"{props.get('viscosity', 0) * 1e6:.1f}", "µPa·s")
            c3.metric("Enthalpy", f"{props.get('enthalpy', 0):.1f}", "kJ/kg")
            c4.metric("Cp", f"{props.get('specific_heat_cp', 0):.2f}", "kJ/(kg·K)")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Thermal k", f"{props.get('thermal_conductivity', 0):.3f}", "W/(m·K)")
            c6.metric("Prandtl", f"{props.get('prandtl', 0):.3f}")
            c7.metric("Compressibility", f"{props.get('compressibility', 0):.3e}", "1/Pa")
            c8.metric("Expansivity", f"{props.get('thermal_expansivity', 0):.3e}", "1/K")

            st.divider()
            st.subheader("Full Property Table")
            df = pd.DataFrame(
                [{k: (f"{v:.4e}" if isinstance(v, float) else v) for k, v in props.items()}]
            ).T
            df.columns = ["Value"]
            st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Saturation Curve Explorer")
    p_sat_mpa = st.slider("Pressure for T_sat (MPa)", 0.1, 22.064, 5.0)
    t_sat_k = fluid.saturation_temperature(p_sat_mpa * 1e6)
    st.info(f"Saturation temperature @ {p_sat_mpa:.2f} MPa = **{t_sat_k - 273.15:.2f} °C**")

# ═══════════════════════════════════════════════════════════════════════════
#  MULTIPHASE MODELS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚗️ Multiphase Models":
    st.title("⚗️ Multiphase Flow Models")

    tab1, tab2 = st.tabs(["Relative Permeability", "Capillary Pressure"])

    with tab1:
        st.subheader("Relative Permeability Model")
        model_type = st.selectbox(
            "Model",
            ["Corey", "van Genuchten-Mualem", "Linear", "Stone I (3-phase)"],
        )

        sw = np.linspace(0.0, 1.0, 200)

        if model_type == "Corey":
            c1, c2 = st.columns(2)
            krw0 = c1.number_input("krw0", 0.0, 1.0, 0.3)
            krn0 = c2.number_input("krn0", 0.0, 1.0, 0.8)
            nw = c1.number_input("nw", 0.1, 10.0, 2.0)
            nn = c2.number_input("nn", 0.1, 10.0, 2.0)
            swr = c1.number_input("Swr", 0.0, 0.99, 0.15)
            snr = c2.number_input("Snr", 0.0, 0.99, 0.2)
            relperm = CoreyRelativePermeability(krw0, krn0, nw, nn, swr, snr)

        elif model_type == "van Genuchten-Mualem":
            c1, c2 = st.columns(2)
            n = c1.number_input("n", 1.01, 10.0, 2.0)
            l_ = c2.number_input("l (Mualem)", -5.0, 5.0, 0.5)
            swr = c1.number_input("Swr", 0.0, 0.99, 0.15)
            snr = c2.number_input("Snr", 0.0, 0.99, 0.0)
            relperm = VanGenuchtenMualem(n, l_, swr, snr)

        elif model_type == "Linear":
            swr = st.number_input("Swr", 0.0, 0.99, 0.15)
            snr = st.number_input("Snr", 0.0, 0.99, 0.0)
            relperm = LinearRelativePermeability(swr, snr)

        else:  # Stone I
            st.info("Stone I: oil relative permeability from water/gas curves")
            c1, c2 = st.columns(2)
            swc = c1.number_input("Swc (connate)", 0.0, 0.99, 0.15)
            sorw = c2.number_input("Sorw (residual oil)", 0.0, 0.99, 0.2)
            # Simple two-phase models for demo
            krow = CoreyRelativePermeability(krw0=0.3, krn0=0.8, nw=2.0, nn=2.0, swr=swc, snr=sorw)
            krog = CoreyRelativePermeability(krw0=0.3, krn0=0.8, nw=2.0, nn=2.0, swr=swc, snr=sorw)
            relperm = StoneIRelativePermeability(krow_model=krow, krog_model=krog, swc=swc, sorw=sorw)
            # Evaluate at fixed oil saturation for demo
            sw_eval = np.linspace(swc + 0.01, 1 - sorw - 0.01, 100)
            so_eval = np.full_like(sw_eval, 0.3)
            krw, kro, krg = relperm(sw_eval, so_eval)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sw_eval, y=kro, name="kro", line=dict(color="#2ca02c")))
            fig.update_layout(title="Stone I — Oil Rel-Perm", xaxis_title="Sw", yaxis_title="kro")
            st.plotly_chart(fig, use_container_width=True)
            st.stop()

        krw, krn = relperm(sw)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sw, y=krw, name="krw", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=sw, y=krn, name="krn", line=dict(color="#ff7f0e")))
        fig.update_layout(
            title=f"{model_type} Relative Permeability",
            xaxis_title="Water Saturation (Sw)",
            yaxis_title="Relative Permeability",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Capillary Pressure Model")
        pc_type = st.selectbox("Model", ["Brooks-Corey", "van Genuchten"])

        sw = np.linspace(0.01, 1.0, 200)

        if pc_type == "Brooks-Corey":
            pd = st.number_input("Entry pressure (Pa)", 1.0, 1e7, 1e5)
            lam = st.number_input("λ", 0.1, 10.0, 2.0)
            swr = st.number_input("Swr", 0.0, 0.99, 0.2)
            snr = st.number_input("Snr", 0.0, 0.99, 0.0)
            pc_model = BrooksCoreyPc(pd, lam, swr, snr)
        else:
            alpha = st.number_input("α (1/Pa)", 1e-8, 1.0, 1e-4, format="%.1e")
            n = st.number_input("n", 1.01, 10.0, 2.0)
            swr = st.number_input("Swr", 0.0, 0.99, 0.2)
            snr = st.number_input("Snr", 0.0, 0.99, 0.0)
            pc_model = VanGenuchtenPc(alpha, n, swr, snr)

        pc_val = pc_model(sw)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sw, y=pc_val / 1e5, name="Pc", line=dict(color="#d62728")))
        fig.update_layout(
            title=f"{pc_type} Capillary Pressure",
            xaxis_title="Water Saturation (Sw)",
            yaxis_title="Capillary Pressure (bar)",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  DUAL POROSITY / MINC
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🪨 Dual Porosity / MINC":
    st.title("🪨 Dual-Porosity & MINC")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Matrix Properties")
        phi_m = st.slider("Matrix porosity", 0.01, 0.5, 0.10)
        k_m_md = st.number_input("Matrix perm (md)", 1e-6, 1.0, 0.001)

        st.subheader("Fracture Properties")
        phi_f = st.slider("Fracture porosity", 0.001, 0.5, 0.02)
        k_f_md = st.number_input("Fracture perm (md)", 0.1, 1e6, 1000.0)

        st.subheader("Geometry")
        lx = st.number_input("Lx (m)", 0.01, 1000.0, 50.0)
        ly = st.number_input("Ly (m)", 0.01, 1000.0, 50.0)
        lz = st.number_input("Lz (m)", 0.01, 1000.0, 50.0)
        geom = st.selectbox(
            "Block geometry",
            ["SLAB_X", "SLAB_Y", "SLAB_Z", "CUBE", "SPHERE", "PRISM"],
        )
        tau = st.number_input("Tortuosity", 1.0, 10.0, 1.0)

        run_btn = st.button("Calculate", type="primary")

    with col2:
        if run_btn:
            geom_enum = BlockGeometry[geom]
            dp = DualPorosityModel(
                matrix_porosity=phi_m,
                matrix_permeability=md_to_m2(k_m_md),
                fracture_porosity=phi_f,
                fracture_permeability=md_to_m2(k_f_md),
                fracture_spacing=(lx, ly, lz),
                geometry=geom_enum,
                tortuosity=tau,
            )

            sigma_wr = dp.warren_root_shape_factor()
            sigma_kaz = dp.kazemi_shape_factor()
            sigma_la = dp.lim_aguilera_shape_factor()
            alpha = dp.interporosity_transfer(sigma_wr)

            st.success(f"σ (Warren-Root) = **{sigma_wr:.4e} 1/m²**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Warren-Root σ", f"{sigma_wr:.4e}", "1/m²")
            c2.metric("Kazemi σ", f"{sigma_kaz:.4e}", "1/m²")
            c3.metric("Lim-Aguilera σ", f"{sigma_la:.4e}", "1/m²")

            st.metric("Interporosity transfer α", f"{alpha:.4e}", "1/s")

            # Shape factor comparison chart
            models = ["Warren-Root", "Kazemi", "Lim-Aguilera"]
            sigmas = [sigma_wr, sigma_kaz, sigma_la]
            fig = go.Figure(
                data=[go.Bar(x=models, y=sigmas, marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"])]
            )
            fig.update_layout(
                title="Shape Factor Comparison",
                yaxis_title="σ (1/m²)",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  SOURCE NETWORK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌐 Source Network":
    st.title("🌐 Source Network (Waiwera-style)")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Network Builder")
        num_sources = st.number_input("Number of sources", 1, 10, 2)

        network = SourceNetwork()
        sources: list[SourceNode] = []
        for i in range(int(num_sources)):
            with st.expander(f"Source {i + 1}"):
                name = st.text_input(f"Name", f"SRC-{i + 1:02d}", key=f"name_{i}")
                cell = st.number_input(f"Cell index", 0, 10000, i * 10, key=f"cell_{i}")
                rate = st.number_input(f"Rate (kg/s)", -1000.0, 1000.0, 50.0, key=f"rate_{i}")
                bhp = st.number_input(f"BHP (bar)", 1.0, 1000.0, 150.0, key=f"bhp_{i}")
                src = SourceNode(name=name, cell_index=cell, rate=rate, bhp=bhp * 1e5)
                sources.append(src)
                network.add_source(src)

        add_sep = st.checkbox("Add Separator")
        add_reinj = st.checkbox("Add Reinjector")

        if add_sep:
            sep = network.add_separator(name="SEP-01", separator_type="flash")
            st.info("Separator added")
        if add_reinj:
            reinj = network.add_reinjector(name="REINJ-01", fraction=0.85)
            if add_sep:
                network.connect(sep, reinj)
            st.info("Reinjector added (85 % fraction)")

        run_btn = st.button("Analyse Network", type="primary")

    with col2:
        if run_btn:
            st.subheader("Network Summary")
            st.metric("Total rate", f"{network.total_rate:.2f} kg/s")
            st.metric("Net rate", f"{network.net_rate:.2f} kg/s")
            st.metric("Number of sources", len(network.sources))

            # Source table
            data = []
            for src in network.sources:
                data.append(
                    {
                        "Name": src.name,
                        "Cell": src.cell_index,
                        "Rate (kg/s)": src.rate,
                        "BHP (bar)": src.bhp / 1e5 if src.bhp else None,
                        "Type": src.node_type,
                    }
                )
            st.dataframe(pd.DataFrame(data), use_container_width=True)

            # Rate balance chart
            names = [s.name for s in network.sources]
            rates = [s.rate for s in network.sources]
            colors = ["#2ca02c" if r >= 0 else "#d62728" for r in rates]
            fig = go.Figure(
                data=[go.Bar(x=names, y=rates, marker_color=colors)]
            )
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(
                title="Source Rate Balance",
                yaxis_title="Rate (kg/s)",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  REGION THERMODYNAMICS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Region Thermodynamics":
    st.title("🔬 Region-Based Thermodynamics")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("State Point")
        p_mpa = st.slider("Pressure (MPa)", 0.1, 100.0, 10.0)
        t_c = st.slider("Temperature (°C)", 0.0, 800.0, 300.0)
        t_k = t_c + 273.15

        eval_btn = st.button("Evaluate State", type="primary")

    with col2:
        if eval_btn:
            rt = RegionThermodynamics()
            state = rt.evaluate(p=p_mpa * 1e6, T=t_k)

            st.success(f"Region: **{state.region}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Density", f"{state.density:.2f}", "kg/m³")
            c2.metric("Viscosity", f"{state.viscosity * 1e6:.1f}", "µPa·s")
            c3.metric("Enthalpy", f"{state.enthalpy:.1f}", "kJ/kg")

            c4, c5, c6 = st.columns(3)
            c4.metric("Cp", f"{state.cp:.2f}", "kJ/(kg·K)")
            c5.metric("Thermal k", f"{state.thermal_conductivity:.3f}", "W/(m·K)")
            c6.metric("Compressibility", f"{state.compressibility:.3e}", "1/Pa")

            # Phase diagram: P-T with saturation curve
            p_range = np.linspace(0.1, 22.0, 100)
            t_sat = [rt.saturation_curve.saturation_temperature(p * 1e6) - 273.15 for p in p_range]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=t_sat,
                    y=p_range,
                    mode="lines",
                    name="Saturation Curve",
                    line=dict(color="#d62728", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[t_c],
                    y=[p_mpa],
                    mode="markers",
                    marker=dict(size=14, color="#1f77b4"),
                    name="Current State",
                )
            )
            fig.update_layout(
                title="P-T Phase Diagram",
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (MPa)",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Saturation Curve Look-up")
    p_lookup = st.number_input("Pressure for T_sat (MPa)", 0.1, 22.064, 5.0)
    t_sat = rt.saturation_curve.saturation_temperature(p_lookup * 1e6) - 273.15
    st.info(f"Saturation temperature @ {p_lookup:.2f} MPa = **{t_sat:.2f} °C**")

    t_lookup = st.number_input("Temperature for P_sat (°C)", 0.0, 374.0, 250.0)
    p_sat = rt.saturation_curve.saturation_pressure(t_lookup + 273.15) / 1e6
    st.info(f"Saturation pressure @ {t_lookup:.1f} °C = **{p_sat:.3f} MPa**")

# ── Footer ─────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption(
    "GARUDA v0.2.0  ·  [GitHub](https://github.com/zakusworo/garuda)"
)
