#!/usr/bin/env python3
"""GARUDA Reservoir Simulator — Premium Streamlit GUI v1.0.

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

import io
import json
import base64
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM THEME
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GARUDA Reservoir Simulator",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium dark/reservoir-engineering look
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar */
        .css-1cyp8cj.e16nr0p33 {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }

        /* Cards */
        .card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #38bdf8;
        }

        .metric-label {
            font-size: 0.875rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }

        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);
        }

        /* Sliders & inputs */
        .stSlider>div>div>div {
            background: #38bdf8 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background: #1e293b !important;
            border-bottom: 2px solid #38bdf8;
        }

        /* Tables */
        .dataframe {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Success / info messages */
        .stSuccess {
            border-radius: 12px;
            border-left: 4px solid #22c55e;
        }

        .stInfo {
            border-radius: 12px;
            border-left: 4px solid #38bdf8;
        }

        /* Section headers */
        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.025em;
        }

        h2, h3 {
            font-weight: 600 !important;
            color: #e2e8f0 !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            font-weight: 500;
            border-radius: 8px;
            background: rgba(30, 41, 59, 0.5);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 1.5rem; margin: 0; color: #38bdf8;">🔥 GARUDA</h1>
            <p style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
                Open-Source Reservoir & Geothermal Simulation
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📐 Grid Builder",
            "💧 Single-Phase Flow",
            "🎯 Well Model",
            "🌡️ IAPWS-IF97",
            "⚗️ Multiphase",
            "🪨 Dual Porosity",
            "🌐 Source Network",
            "🔬 Thermodynamics",
            "🧊 3D Visualizer",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Quick stats in sidebar
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span>Tests</span><span style="color: #22c55e; font-weight: 600;">557 passed</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span>Coverage</span><span style="color: #38bdf8; font-weight: 600;">~89%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Version</span><span style="color: #f59e0b; font-weight: 600;">v0.2.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("[GitHub](https://github.com/zakusworo/garuda) · Politeknik Energi dan Pertambangan Bandung")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def md_to_m2(k_md: float) -> float:
    return k_md * 9.869233e-16


def m2_to_md(k_m2: float) -> float:
    return k_m2 / 9.869233e-16


def _metric_card(label: str, value: str, unit: str = "") -> str:
    unit_html = f'<div style="font-size: 0.75rem; color: #64748b;">{unit}</div>' if unit else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {unit_html}
    </div>
    """


def _export_csv(df: pd.DataFrame, filename: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def _export_json(data: dict, filename: str) -> None:
    j = json.dumps(data, indent=2, default=str).encode("utf-8")
    st.download_button(
        label="📥 Download JSON",
        data=j,
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 2.5rem; color: #38bdf8;">🔥 GARUDA</h1>
            <p style="font-size: 1.125rem; color: #94a3b8; max-width: 600px; margin: 1rem auto;">
                Open-source Python reservoir simulator for petroleum & geothermal systems
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards in 3 columns
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="card">
                <h3 style="color: #38bdf8; margin-top: 0;">📐 Grid & Flow</h3>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Structured 1D/2D/3D grids with TPFA finite-volume solver and Numba acceleration.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
                <h3 style="color: #f59e0b; margin-top: 0;">🎯 Wells & Sources</h3>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Peaceman well model, source networks, separators & reinjectors.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="card">
                <h3 style="color: #22c55e; margin-top: 0;">⚗️ Physics</h3>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    IAPWS-IF97 thermodynamics, relative permeability, capillary pressure, dual-porosity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Quick-start cards linking to each module
    st.subheader("🚀 Quick Start")
    qc1, qc2, qc3, qc4 = st.columns(4)
    modules = [
        ("📐 Grid Builder", "Build structured grids", "#38bdf8"),
        ("💧 Single-Phase Flow", "TPFA pressure solver", "#22c55e"),
        ("🎯 Well Model", "Peaceman productivity", "#f59e0b"),
        ("🌡️ IAPWS-IF97", "Water/steam properties", "#ef4444"),
    ]
    for col, (name, desc, color) in zip([qc1, qc2, qc3, qc4], modules):
        with col:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                            border-left: 3px solid {color};
                            border-radius: 12px; padding: 1rem;
                            margin-bottom: 0.5rem;">
                    <div style="font-weight: 600; color: #e2e8f0;">{name}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info("💡 All inputs use SI units (Pa, m, kg, K) unless otherwise noted.")


# ═══════════════════════════════════════════════════════════════════════════
#  GRID BUILDER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📐 Grid Builder":
    st.markdown("<h1>📐 Grid Builder</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.subheader("Dimensions")
            nx = st.slider("NX", 1, 200, 20, help="Number of cells in X direction")
            ny = st.slider("NY", 1, 200, 1, help="Number of cells in Y direction")
            nz = st.slider("NZ", 1, 100, 1, help="Number of cells in Z direction")

        with st.container(border=True):
            st.subheader("Cell Sizes (m)")
            dx = st.number_input("ΔX", 0.1, 10000.0, 100.0)
            dy = st.number_input("ΔY", 0.1, 10000.0, 100.0)
            dz = st.number_input("ΔZ", 0.1, 10000.0, 10.0)

        with st.container(border=True):
            st.subheader("Properties")
            k_md = st.number_input("Permeability (md)", 0.001, 1e6, 100.0)
            poro = st.slider("Porosity", 0.01, 0.5, 0.2)

        build_btn = st.button("🔨 Build Grid", type="primary", use_container_width=True)

    with col2:
        if build_btn:
            with st.spinner("Building grid..."):
                grid = StructuredGrid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
                grid.set_permeability(k_md, unit="md")
                grid.set_porosity(poro)

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cells", f"{grid.num_cells:,}")
            m2.metric("Total Volume", f"{grid.cell_volumes.sum():.3e}", "m³")
            m3.metric("Avg Porosity", f"{grid.porosity.mean():.3f}")
            m4.metric("Avg Kx", f"{grid.permeability[:,0,0].mean():.3e}", "m²")

            st.success(f"Grid built: {nx}×{ny}×{nz} = **{grid.num_cells:,} cells**")

            # Visualization
            st.subheader("Visualisation")
            if ny > 1 or nz > 1:
                kx = grid.permeability[:, 0, 0]
                perm_2d = kx.reshape((nz, ny, nx)).mean(axis=0)
                fig = px.imshow(
                    perm_2d,
                    color_continuous_scale="Viridis",
                    labels={"color": "Kx (m²)"},
                    title="Mean Kx Permeability (Z-averaged)",
                    aspect="auto",
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                x = np.arange(nx) * dx + dx / 2
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x, y=grid.permeability[:, 0, 0],
                        mode="lines+markers", name="Kx",
                        line=dict(color="#38bdf8"),
                        fill="tozeroy", fillcolor="rgba(56,189,248,0.1)",
                    )
                )
                fig.update_layout(
                    title="1D Kx Permeability Profile",
                    xaxis_title="Distance (m)",
                    yaxis_title="Permeability (m²)",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Data table with export
            st.subheader("Cell Data")
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
            st.dataframe(df, use_container_width=True, height=300)
            _export_csv(df, "grid_cells.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  SINGLE-PHASE FLOW
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💧 Single-Phase Flow":
    st.markdown("<h1>💧 Single-Phase Flow Solver</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.subheader("Grid")
            nx = st.slider("NX", 2, 200, 50)
            ny = st.slider("NY", 1, 200, 1)
            nz = st.slider("NZ", 1, 100, 1)
            dx = st.number_input("ΔX (m)", 1.0, 10000.0, 50.0)
            dy = st.number_input("ΔY (m)", 1.0, 10000.0, 50.0)
            dz = st.number_input("ΔZ (m)", 0.1, 10000.0, 10.0)

        with st.container(border=True):
            st.subheader("Fluid & Rock")
            k_md = st.number_input("Permeability (md)", 0.001, 1e6, 100.0)
            poro = st.slider("Porosity", 0.01, 0.5, 0.2, key="sp_poro")
            mu = st.number_input("Viscosity (Pa·s)", 1e-6, 1.0, 1e-3, format="%.1e")
            rho = st.number_input("Density (kg/m³)", 10.0, 2000.0, 1000.0)

        with st.container(border=True):
            st.subheader("Boundary Conditions")
            p_left = st.number_input("Left / Bottom P (bar)", 1.0, 1000.0, 200.0)
            p_right = st.number_input("Right / Top P (bar)", 1.0, 1000.0, 100.0)

        with st.container(border=True):
            st.subheader("Source / Sink")
            source_type = st.selectbox(
                "Source pattern", ["None", "Uniform", "Gaussian", "Point"],
                help="Distribution of source/sink term across grid"
            )
            source_strength = st.number_input("Source strength (kg/s)", -1000.0, 1000.0, 0.0)

        run_btn = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("Solving pressure equation..."):
                grid = StructuredGrid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
                grid.set_permeability(k_md, unit="md")
                grid.set_porosity(poro)

                solver = TPFASolver(grid, mu=mu, rho=rho)

                source = np.zeros(grid.num_cells)
                if source_type == "Uniform":
                    source[:] = source_strength / grid.num_cells
                elif source_type == "Gaussian":
                    cx, cy = nx // 2, ny // 2
                    for j in range(ny):
                        for i in range(nx):
                            idx = j * nx + i
                            r2 = ((i - cx) / max(nx / 4, 1)) ** 2 + ((j - cy) / max(ny / 4, 1)) ** 2
                            source[idx] = source_strength * np.exp(-r2)
                elif source_type == "Point":
                    source[grid.num_cells // 2] = source_strength

                bc = np.array([p_left * 1e5, p_right * 1e5])
                p = solver.solve(source, bc_type="dirichlet", bc_values=bc, solver="direct")

            st.success(
                f"Converged  |  Pressure range: **{p.min() / 1e5:.2f} – {p.max() / 1e5:.2f} bar**"
            )

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Min Pressure", f"{p.min() / 1e5:.2f}", "bar")
            m2.metric("Max Pressure", f"{p.max() / 1e5:.2f}", "bar")
            dx_m = dx
            area = dy * dz
            grad_p = (p[0] - p[-1]) / (nx * dx_m)
            k_m2 = md_to_m2(k_md)
            q_darcy = -k_m2 * area / mu * grad_p
            m3.metric("Darcy Flow", f"{q_darcy:.3e}", "m³/s")

            # Plots
            st.subheader("Pressure Field")
            if ny == 1 and nz == 1:
                x = np.arange(nx) * dx + dx / 2
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x, y=p / 1e5,
                        mode="lines+markers", name="Pressure",
                        line=dict(color="#38bdf8", width=3),
                        fill="tozeroy", fillcolor="rgba(56,189,248,0.15)",
                    )
                )
                fig.add_hline(y=p_left, line_dash="dash", line_color="#22c55e", annotation_text="Left BC")
                fig.add_hline(y=p_right, line_dash="dash", line_color="#ef4444", annotation_text="Right BC")
                fig.update_layout(
                    title="1D Pressure Profile",
                    xaxis_title="Distance (m)",
                    yaxis_title="Pressure (bar)",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                p2d = p.reshape((nz, ny, nx)).mean(axis=0)
                fig = px.imshow(
                    p2d / 1e5,
                    color_continuous_scale="RdBu_r",
                    labels={"color": "Pressure (bar)"},
                    title="Pressure Field (Z-averaged)",
                    aspect="auto",
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            # Export
            df = pd.DataFrame({
                "Cell": np.arange(grid.num_cells),
                "Pressure (Pa)": p,
                "Pressure (bar)": p / 1e5,
                "Source (kg/s)": source,
            })
            st.dataframe(df, use_container_width=True, height=250)
            _export_csv(df, "pressure_solution.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  WELL MODEL
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Well Model":
    st.markdown("<h1>🎯 Peaceman Well Model</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.subheader("Well Parameters")
            well_name = st.text_input("Well name", "PROD-01")
            rw = st.number_input("Well radius (m)", 0.01, 1.0, 0.1)
            skin = st.number_input("Skin factor", -5.0, 50.0, 0.0)
            depth = st.number_input("Well depth (m)", 0.0, 10000.0, 1000.0)

        with st.container(border=True):
            st.subheader("Operating Constraint")
            constraint = st.selectbox("Constraint", ["pressure", "rate"])
            target = st.number_input("Target BHP (bar) or Rate (kg/s)", 1.0, 1000.0, 150.0)
            max_rate = st.number_input("Max rate (kg/s)", 0.0, 1000.0, 50.0)
            min_bhp = st.number_input("Min BHP (bar)", 1.0, 1000.0, 80.0)

        with st.container(border=True):
            st.subheader("Reservoir Conditions")
            k_md = st.number_input("Permeability (md)", 0.001, 1e6, 100.0, key="well_k")
            mu = st.number_input("Viscosity (Pa·s)", 1e-6, 1.0, 1e-3, format="%.1e", key="well_mu")
            dx = st.number_input("ΔX (m)", 1.0, 1000.0, 100.0, key="well_dx")
            dy = st.number_input("ΔY (m)", 1.0, 1000.0, 100.0, key="well_dy")
            dz = st.number_input("ΔZ (m)", 0.1, 1000.0, 10.0, key="well_dz")
            p_res = st.number_input("Reservoir pressure (bar)", 1.0, 1000.0, 200.0)
            p_wf = st.number_input("Wellbore pressure (bar)", 1.0, 1000.0, 150.0)
            rho = st.number_input("Fluid density (kg/m³)", 10.0, 2000.0, 780.0)

        run_btn = st.button("▶️ Compute Rate", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("Computing well performance..."):
                params = WellParameters(
                    name=well_name, cell_index=0,
                    well_radius=rw, skin_factor=skin, well_depth=depth,
                )
                ops = WellOperatingConditions(
                    constraint_type=constraint,
                    target_value=target * 1e5 if constraint == "pressure" else target,
                    max_rate=max_rate, min_bhp=min_bhp * 1e5,
                )
                well = PeacemanWell(params, ops)
                well.compute_productivity_index(
                    permeability=md_to_m2(k_md), viscosity=mu,
                    dx=dx, dy=dy, dz=dz,
                )
                rate = well.compute_rate(
                    cell_pressure=p_res * 1e5,
                    wellbore_pressure=p_wf * 1e5,
                    density=rho,
                )

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Flow Rate", f"{rate:.3f}", "kg/s")
            m2.metric("Current BHP", f"{well.current_bhp / 1e5:.2f}", "bar")
            m3.metric("Productivity Index", f"{well.productivity_index:.3e}", "m³/(s·Pa)")

            st.success(f"**{well_name}** — Rate = **{rate:.3f} kg/s**  |  BHP = **{well.current_bhp / 1e5:.2f} bar**")

            # IPR curve
            st.subheader("Inflow Performance Relationship (IPR)")
            pwf_range = np.linspace(p_res * 0.3, p_res * 1.05, 100)
            rates = [
                well.compute_rate(
                    cell_pressure=p_res * 1e5,
                    wellbore_pressure=pwf * 1e5,
                    density=rho,
                )
                for pwf in pwf_range
            ]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=rates, y=pwf_range,
                    mode="lines", name="IPR",
                    line=dict(color="#38bdf8", width=3),
                    fill="tozerox", fillcolor="rgba(56,189,248,0.1)",
                )
            )
            fig.add_hline(
                y=p_wf, line_dash="dash", line_color="#f59e0b",
                annotation_text=f"Operating Pwf = {p_wf:.1f} bar",
            )
            fig.add_vline(
                x=rate, line_dash="dash", line_color="#22c55e",
                annotation_text=f"Rate = {rate:.2f} kg/s",
            )
            fig.update_layout(
                title="IPR Curve",
                xaxis_title="Flow Rate (kg/s)",
                yaxis_title="Bottom-Hole Pressure (bar)",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Operating envelope data
            df = pd.DataFrame({
                "Pwf (bar)": pwf_range,
                "Rate (kg/s)": rates,
            })
            st.dataframe(df, use_container_width=True, height=250)
            _export_csv(df, "ipr_curve.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  IAPWS-IF97
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌡️ IAPWS-IF97":
    st.markdown("<h1>🌡️ IAPWS-IF97 Water / Steam</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.subheader("State Point")
            p_mpa = st.slider("Pressure (MPa)", 0.1, 100.0, 15.0)
            t_c = st.slider("Temperature (°C)", 0.0, 800.0, 280.0)
            t_k = t_c + 273.15

        calc_btn = st.button("▶️ Calculate Properties", type="primary", use_container_width=True)

    with col2:
        if calc_btn:
            with st.spinner("Computing IAPWS-IF97 properties..."):
                fluid = IAPWSFluidProperties()
                props = fluid.get_all(p=p_mpa * 1e6, T=t_k)

            phase = props.get("phase", "unknown")
            phase_color = {"liquid": "#38bdf8", "vapor": "#f59e0b", "supercritical": "#ef4444"}.get(phase, "#94a3b8")
            st.markdown(
                f"""
                <div style="background: {phase_color}20; border: 1px solid {phase_color};
                            border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
                    <span style="color: {phase_color}; font-weight: 700; font-size: 1.25rem;">
                        ● {phase.upper()}
                    </span>
                    <span style="color: #94a3b8; margin-left: 0.5rem;">
                        @ {p_mpa:.1f} MPa / {t_c:.1f} °C
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metrics in 4x2 grid
            rows = [
                [("Density", f"{props.get('density', 0):.2f}", "kg/m³"),
                 ("Viscosity", f"{props.get('viscosity', 0) * 1e6:.1f}", "µPa·s"),
                 ("Enthalpy", f"{props.get('enthalpy', 0):.1f}", "kJ/kg"),
                 ("Cp", f"{props.get('specific_heat_cp', 0):.2f}", "kJ/(kg·K)"),],
                [("Thermal k", f"{props.get('thermal_conductivity', 0):.3f}", "W/(m·K)"),
                 ("Prandtl", f"{props.get('prandtl', 0):.3f}" if props.get('prandtl') is not None else "N/A", ""),
                 ("Compressibility", f"{props.get('compressibility', 0):.3e}" if props.get('compressibility') is not None else "N/A", "1/Pa"),
                 ("Expansivity", f"{props.get('thermal_expansivity', 0):.3e}" if props.get('thermal_expansivity') is not None else "N/A", "1/K"),],
            ]
            for row in rows:
                cols = st.columns(4)
                for col, (label, value, unit) in zip(cols, row):
                    col.markdown(_metric_card(label, value, unit), unsafe_allow_html=True)

            # Full property table
            st.subheader("Full Property Table")
            df = pd.DataFrame(
                [{k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in props.items()}]
            ).T
            df.columns = ["Value"]
            st.dataframe(df, use_container_width=True)
            _export_csv(df.reset_index().rename(columns={"index": "Property"}), "iapws_properties.csv")

    st.divider()
    st.subheader("Saturation Curve Explorer")
    p_sat_mpa = st.slider("Pressure for T_sat (MPa)", 0.1, 22.064, 5.0, key="sat_p")
    from garuda.core.iapws_properties import WaterSteamProperties
    _iapws = WaterSteamProperties()
    t_sat_k = _iapws.saturation_temperature(p_sat_mpa)
    st.info(f"Saturation temperature @ **{p_sat_mpa:.2f} MPa** = **{t_sat_k - 273.15:.2f} °C**")


# ═══════════════════════════════════════════════════════════════════════════
#  MULTIPHASE MODELS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚗️ Multiphase":
    st.markdown("<h1>⚗️ Multiphase Flow Models</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Relative Permeability", "📉 Capillary Pressure"])

    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            with st.container(border=True):
                st.subheader("Model Settings")
                model_type = st.selectbox(
                    "Model",
                    ["Corey", "van Genuchten-Mualem", "Linear", "Stone I (3-phase)"],
                )

            sw = np.linspace(0.0, 1.0, 200)

            with st.container(border=True):
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
                    swr = c1.number_input("Swr", 0.0, 0.99, 0.15, key="vg_swr")
                    snr = c2.number_input("Snr", 0.0, 0.99, 0.0, key="vg_snr")
                    relperm = VanGenuchtenMualem(n=n, swr=swr, snr=snr)

                elif model_type == "Linear":
                    swr = st.number_input("Swr", 0.0, 0.99, 0.15, key="lin_swr")
                    snr = st.number_input("Snr", 0.0, 0.99, 0.0, key="lin_snr")
                    relperm = LinearRelativePermeability(swr, snr)

                else:  # Stone I
                    st.info("Stone I: oil relative permeability from water/gas curves")
                    c1, c2 = st.columns(2)
                    swc = c1.number_input("Swc (connate)", 0.0, 0.99, 0.15)
                    sorw = c2.number_input("Sorw (residual oil)", 0.0, 0.99, 0.2)
                    krow = CoreyRelativePermeability(0.3, 0.8, 2.0, 2.0, swc, sorw)
                    krog = CoreyRelativePermeability(0.3, 0.8, 2.0, 2.0, swc, sorw)
                    relperm = StoneIRelativePermeability(krow, krog, swc, sorw)
                    sw_eval = np.linspace(swc + 0.01, 1 - sorw - 0.01, 100)
                    so_eval = np.full_like(sw_eval, 0.3)
                    krw, kro, krg = relperm(sw_eval, so_eval)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sw_eval, y=kro, name="kro", line=dict(color="#22c55e", width=3)))
                    fig.update_layout(
                        title="Stone I — Oil Relative Permeability",
                        xaxis_title="Sw", yaxis_title="kro",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.stop()

            krw, krn = relperm(sw)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sw, y=krw, name="krw", line=dict(color="#38bdf8", width=3), fill="tozeroy", fillcolor="rgba(56,189,248,0.1)"))
            fig.add_trace(go.Scatter(x=sw, y=krn, name="krn", line=dict(color="#f59e0b", width=3), fill="tozeroy", fillcolor="rgba(245,158,11,0.1)"))
            fig.update_layout(
                title=f"{model_type} Relative Permeability",
                xaxis_title="Water Saturation (Sw)",
                yaxis_title="Relative Permeability",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Data export
            df = pd.DataFrame({"Sw": sw, "krw": krw, "krn": krn})
            st.dataframe(df, use_container_width=True, height=250)
            _export_csv(df, "relperm.csv")

    with tab2:
        col1, col2 = st.columns([1, 2])

        with col1:
            with st.container(border=True):
                st.subheader("Capillary Pressure Model")
                pc_type = st.selectbox("Model", ["Brooks-Corey", "van Genuchten"], key="pc_type")

            sw = np.linspace(0.01, 1.0, 200)

            with st.container(border=True):
                if pc_type == "Brooks-Corey":
                    p_entry = st.number_input("Entry pressure (Pa)", 1.0, 1e7, 1e5)
                    lam = st.number_input("λ", 0.1, 10.0, 2.0, key="bc_lam")
                    swr = st.number_input("Swr", 0.0, 0.99, 0.2, key="bc_swr")
                    snr = st.number_input("Snr", 0.0, 0.99, 0.0, key="bc_snr")
                    pc_model = BrooksCoreyPc(p_entry, lam, swr, snr)
                else:
                    p0 = st.number_input("P₀ (Pa)", 1.0, 1e7, 1e4)
                    n = st.number_input("n", 1.01, 10.0, 2.0, key="vg_pc_n")
                    swr = st.number_input("Swr", 0.0, 0.99, 0.2, key="vg_pc_swr")
                    snr = st.number_input("Snr", 0.0, 0.99, 0.0, key="vg_pc_snr")
                    pc_model = VanGenuchtenPc(p0=p0, n=n, swr=swr, snr=snr)

            pc_val = pc_model(sw)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sw, y=pc_val / 1e5, name="Pc", line=dict(color="#ef4444", width=3), fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"))
            fig.update_layout(
                title=f"{pc_type} Capillary Pressure",
                xaxis_title="Water Saturation (Sw)",
                yaxis_title="Capillary Pressure (bar)",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

            df = pd.DataFrame({"Sw": sw, "Pc (Pa)": pc_val, "Pc (bar)": pc_val / 1e5})
            st.dataframe(df, use_container_width=True, height=250)
            _export_csv(df, "capillary_pressure.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  DUAL POROSITY
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🪨 Dual Porosity":
    st.markdown("<h1>🪨 Dual-Porosity & MINC</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.subheader("Matrix Properties")
            phi_m = st.slider("Matrix porosity", 0.01, 0.5, 0.10)
            k_m_md = st.number_input("Matrix perm (md)", 1e-6, 1.0, 0.001)

        with st.container(border=True):
            st.subheader("Fracture Properties")
            phi_f = st.slider("Fracture porosity", 0.001, 0.5, 0.02)
            k_f_md = st.number_input("Fracture perm (md)", 0.1, 1e6, 1000.0)

        with st.container(border=True):
            st.subheader("Geometry")
            lx = st.number_input("Lx (m)", 0.01, 1000.0, 50.0)
            ly = st.number_input("Ly (m)", 0.01, 1000.0, 50.0)
            lz = st.number_input("Lz (m)", 0.01, 1000.0, 50.0)
            geom = st.selectbox(
                "Block geometry",
                ["SLAB_X", "SLAB_Y", "SLAB_Z", "CUBE", "SPHERE", "PRISM"],
            )
            tau = st.number_input("Tortuosity", 1.0, 10.0, 1.0)

        run_btn = st.button("▶️ Calculate", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("Computing dual-porosity parameters..."):
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
                lam = dp.interporosity_flow_coefficient(compressibility=1e-9, viscosity=1e-3)
                omega = dp.omega()

            # Results cards
            st.markdown(
                f"""
                <div style="background: rgba(34,197,94,0.1); border: 1px solid #22c55e;
                            border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
                    <span style="color: #22c55e; font-weight: 700;">
                        ● Warren-Root σ = {sigma_wr:.4e} 1/m²
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Warren-Root σ", f"{sigma_wr:.4e}", "1/m²")
            m2.metric("Kazemi σ", f"{sigma_kaz:.4e}", "1/m²")
            m3.metric("Lim-Aguilera σ", f"{sigma_la:.4e}", "1/m²")
            m4.metric("λ (flow coeff.)", f"{lam:.4e}", "dimensionless")

            st.metric("Storativity ratio ω", f"{omega:.4f}", "dimensionless")

            # Shape factor comparison
            st.subheader("Shape Factor Comparison")
            fig = go.Figure()
            colors = ["#38bdf8", "#f59e0b", "#22c55e"]
            for model, sigma, color in zip(["Warren-Root", "Kazemi", "Lim-Aguilera"], [sigma_wr, sigma_kaz, sigma_la], colors):
                fig.add_trace(go.Bar(x=[model], y=[sigma], name=model, marker_color=color))
            fig.update_layout(
                title="Shape Factor Models",
                yaxis_title="σ (1/m²)",
                template="plotly_dark",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Export
            data = {
                "model": "dual_porosity",
                "geometry": geom,
                "matrix_porosity": phi_m,
                "matrix_perm_m2": md_to_m2(k_m_md),
                "fracture_porosity": phi_f,
                "fracture_perm_m2": md_to_m2(k_f_md),
                "fracture_spacing": [lx, ly, lz],
                "tortuosity": tau,
                "sigma_warren_root": sigma_wr,
                "sigma_kazemi": sigma_kaz,
                "sigma_lim_aguilera": sigma_la,
                "interporosity_flow_coefficient": lam,
                "storativity_ratio": omega,
            }
            _export_json(data, "dual_porosity_results.json")


# ═══════════════════════════════════════════════════════════════════════════
#  SOURCE NETWORK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌐 Source Network":
    st.markdown("<h1>🌐 Source Network</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.subheader("Network Builder")
            num_sources = st.slider("Number of sources", 1, 10, 2)

        from garuda.core.source_network import SourceGroup, Separator, Reinjector
        group = SourceGroup(name="FIELD", group_rate_target=None)
        sources: list[SourceNode] = []

        with st.container(border=True):
            for i in range(int(num_sources)):
                with st.expander(f"Source {i + 1}"):
                    name = st.text_input("Name", f"SRC-{i + 1:02d}", key=f"name_{i}")
                    cell = st.number_input("Cell index", 0, 10000, i * 10, key=f"cell_{i}")
                    rate = st.number_input("Rate (kg/s)", -1000.0, 1000.0, 50.0, key=f"rate_{i}")
                    src = SourceNode(name=name, cell_index=cell, rate=rate)
                    sources.append(src)
                    group.add_node(src)

        with st.container(border=True):
            add_sep = st.checkbox("Add Separator")
            add_reinj = st.checkbox("Add Reinjector")
            sep = None
            reinj = None
            if add_sep:
                sep = Separator(name="SEP-01")
                st.info("Separator added")
            if add_reinj:
                reinj = Reinjector(name="REINJ-01", cell_index=0, target_rate=50.0)
                st.info("Reinjector added")

        run_btn = st.button("▶️ Analyse Network", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("Analysing network..."):
                total_rate = group.compute_group_rate()

            st.success(f"Network **{group.name}** — Total rate = **{total_rate:.2f} kg/s**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Rate", f"{total_rate:.2f}", "kg/s")
            m2.metric("Number of Nodes", len(group.nodes))
            m3.metric("Net Flow", "Producer" if total_rate >= 0 else "Injector", "")

            # Source table
            st.subheader("Source Details")
            data = []
            for src in group.nodes.values():
                data.append({
                    "Name": src.name,
                    "Cell": src.cell_index,
                    "Rate (kg/s)": src.rate,
                    "Phase": src.phase,
                    "Active": "✅" if src.active else "❌",
                })
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, height=300)
            _export_csv(df, "source_network.csv")

            # Rate balance chart
            st.subheader("Rate Balance")
            names = [s.name for s in group.nodes.values()]
            rates = [s.rate for s in group.nodes.values()]
            colors = ["#22c55e" if r >= 0 else "#ef4444" for r in rates]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=names, y=rates, marker_color=colors, text=[f"{r:.1f}" for r in rates], textposition="outside"))
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            fig.update_layout(
                title="Source Rate Balance",
                yaxis_title="Rate (kg/s)",
                template="plotly_dark",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  REGION THERMODYNAMICS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Thermodynamics":
    st.markdown("<h1>🔬 Region-Based Thermodynamics</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.subheader("State Point")
            p_mpa = st.slider("Pressure (MPa)", 0.1, 100.0, 10.0)
            t_c = st.slider("Temperature (°C)", 0.0, 800.0, 300.0)
            t_k = t_c + 273.15

        eval_btn = st.button("▶️ Evaluate State", type="primary", use_container_width=True)

    with col2:
        if eval_btn:
            with st.spinner("Evaluating thermodynamic state..."):
                rt = RegionThermodynamics()
                state = rt.get_properties(pressure=p_mpa * 1e6, temperature=t_k)

            region_colors = {
                "liquid": "#38bdf8",
                "vapor": "#f59e0b",
                "supercritical": "#ef4444",
                "two-phase": "#22c55e",
            }
            color = region_colors.get(state.region, "#94a3b8")
            st.markdown(
                f"""
                <div style="background: {color}20; border: 1px solid {color};
                            border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
                    <span style="color: {color}; font-weight: 700; font-size: 1.25rem;">
                        ● {state.region.upper()}
                    </span>
                    <span style="color: #94a3b8; margin-left: 0.5rem;">
                        @ {p_mpa:.1f} MPa / {t_c:.1f} °C
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Density", f"{state.density:.2f}", "kg/m³")
            m2.metric("Viscosity", f"{state.viscosity * 1e6:.1f}", "µPa·s")
            m3.metric("Enthalpy", f"{state.enthalpy / 1000:.1f}", "kJ/kg")

            # P-T Phase Diagram
            st.subheader("P-T Phase Diagram")
            p_range = np.linspace(0.1, 22.0, 100)
            t_sat = [rt.saturation_curve.saturation_temperature(p * 1e6) - 273.15 for p in p_range]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_sat, y=p_range,
                mode="lines", name="Saturation Curve",
                line=dict(color="#ef4444", width=3),
                fill="tozerox", fillcolor="rgba(239,68,68,0.1)",
            ))
            fig.add_trace(go.Scatter(
                x=[t_c], y=[p_mpa],
                mode="markers", name="Current State",
                marker=dict(size=16, color="#38bdf8", symbol="star", line=dict(width=2, color="white")),
            ))
            fig.update_layout(
                title="Phase Diagram",
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (MPa)",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Saturation Curve Look-up")
    rt2 = RegionThermodynamics()
    p_lookup = st.number_input("Pressure for T_sat (MPa)", 0.1, 22.064, 5.0)
    t_sat = rt2.saturation_curve.saturation_temperature(p_lookup * 1e6) - 273.15
    st.info(f"Saturation temperature @ **{p_lookup:.2f} MPa** = **{t_sat:.2f} °C**")

    t_lookup = st.number_input("Temperature for P_sat (°C)", 0.0, 374.0, 250.0)
    p_sat = rt2.saturation_curve.saturation_pressure(t_lookup + 273.15) / 1e6
    st.info(f"Saturation pressure @ **{t_lookup:.1f} °C** = **{p_sat:.3f} MPa**")


# ═══════════════════════════════════════════════════════════════════════════
#  3D VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧊 3D Visualizer":
    st.markdown("<h1>🧊 3D Reservoir Visualizer</h1>", unsafe_allow_html=True)
    st.markdown("""<p style='color:#94a3b8;'>Publication-quality 3D rendering with isothermal surfaces, cross-sections, streamlines, and well trajectories.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.subheader("Grid Settings")
            nx = st.slider("NX", 10, 80, 40, help="Grid cells in X")
            ny = st.slider("NY", 10, 80, 40, help="Grid cells in Y")
            nz = st.slider("NZ", 5, 40, 20, help="Grid cells in Z (depth)")
            dx = st.number_input("ΔX (m)", 10.0, 500.0, 50.0)
            dy = st.number_input("ΔY (m)", 10.0, 500.0, 50.0)
            dz = st.number_input("ΔZ (m)", 5.0, 200.0, 25.0)

        with st.container(border=True):
            st.subheader("Temperature Field")
            t_surface = st.number_input("Surface temperature (°C)", 10.0, 150.0, 80.0)
            t_bottom = st.number_input("Bottom temperature (°C)", 100.0, 400.0, 280.0)
            heat_source_radius = st.number_input("Heat source radius (m)", 100.0, 2000.0, 500.0)

        with st.container(border=True):
            st.subheader("Production Well")
            well_x = st.number_input("Well X (m)", 0.0, 5000.0, 800.0)
            well_y = st.number_input("Well Y (m)", 0.0, 5000.0, 1200.0)
            well_depth = st.number_input("Well depth (m)", 50.0, 2000.0, 450.0)
            drawdown = st.number_input("Drawdown (MPa)", 0.0, 10.0, 5.0)

        with st.container(border=True):
            st.subheader("Visualization Options")
            show_isotherms = st.multiselect("Isothermal surfaces (°C)", [120, 150, 180, 200, 220], default=[150, 200])
            show_slices = st.checkbox("Show cross-section slices", value=True)
            show_streamlines = st.checkbox("Show flow streamlines", value=True)
            show_well = st.checkbox("Show well trajectory", value=True)

        render_btn = st.button("▶️ Render 3D Scene", type="primary", use_container_width=True)

    with col2:
        if render_btn:
            import pyvista as pv

            with st.spinner("Building 3D grid and computing fields..."):
                grid = pv.ImageData()
                grid.dimensions = [nx, ny, nz]
                grid.spacing = [dx, dy, dz]
                grid.origin = [0, 0, -nz * dz]

                x = np.linspace(0, (nx - 1) * dx, nx)
                y = np.linspace(0, (ny - 1) * dy, ny)
                z = np.linspace(grid.origin[2], 0, nz)
                XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')

                cx, cy = (nx - 1) * dx / 2, (ny - 1) * dy / 2
                r_hot = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
                depth = (-ZZ) / (nz * dz)

                T = t_surface + (t_bottom - t_surface) * depth * np.exp(-r_hot / heat_source_radius)

                # Well cooling
                r_well = np.sqrt((XX - well_x) ** 2 + (YY - well_y) ** 2)
                cooling = 60.0 * np.exp(-r_well / 80.0) * np.exp(-(-ZZ - well_depth) ** 2 / 5000.0)
                T -= cooling

                grid.point_data['temperature'] = T.ravel(order='F')

                # Pressure field
                P = 20e6 + 850.0 * 9.81 * (-ZZ)
                P -= drawdown * 1e6 * np.exp(-r_well / 100.0) * np.exp(-(-ZZ - well_depth) ** 2 / 8000.0)
                grid.point_data['pressure'] = P.ravel(order='F')

                # Velocity (for streamlines)
                k, mu = 1e-13, 2.5e-4
                P_3d = P.reshape((nx, ny, nz), order='F')
                dPx = np.gradient(P_3d, dx, axis=0)
                dPy = np.gradient(P_3d, dy, axis=1)
                dPz = np.gradient(P_3d, dz, axis=2)
                vx = -k / mu * dPx
                vy = -k / mu * dPy
                vz = -k / mu * dPz
                vectors = np.stack([vx.ravel(order='F'), vy.ravel(order='F'), vz.ravel(order='F')], axis=1)
                grid.point_data['velocity'] = vectors

            with st.spinner("Rendering 3D scene..."):
                pv.OFF_SCREEN = True
                pl = pv.Plotter(off_screen=True, window_size=(1600, 1100))

                # Isothermal surfaces
                for temp in show_isotherms:
                    contour = grid.contour([float(temp)], scalars='temperature')
                    if contour.n_points > 0:
                        pl.add_mesh(contour, cmap='coolwarm', opacity=0.4,
                                    clim=[T.min(), T.max()], show_edges=False, smooth_shading=True)

                if show_slices:
                    slice_x = grid.slice(normal='x', origin=(cx, cy, -nz * dz / 2))
                    slice_y = grid.slice(normal='y', origin=(cx, cy, -nz * dz / 2))
                    slice_z = grid.slice(normal='z', origin=(cx, cy, -nz * dz / 4))
                    pl.add_mesh(slice_x, cmap='coolwarm', opacity=0.6, clim=[T.min(), T.max()],
                                scalar_bar_args={'title': 'Temperature (°C)', 'vertical': True, 'position_x': 0.88})
                    pl.add_mesh(slice_y, cmap='coolwarm', opacity=0.35, clim=[T.min(), T.max()])
                    pl.add_mesh(slice_z, cmap='coolwarm', opacity=0.25, clim=[T.min(), T.max()])

                if show_streamlines:
                    seed_pts = np.array([
                        [well_x - 300, well_y, -30], [well_x + 300, well_y, -30],
                        [well_x, well_y - 300, -30], [well_x, well_y + 300, -30],
                        [well_x - 150, well_y - 150, -30], [well_x + 150, well_y + 150, -30],
                    ])
                    seed = pv.PolyData(seed_pts)
                    try:
                        streamlines = grid.streamlines_from_source(
                            seed, vectors_name='velocity',
                            integration_direction='both', max_time=5e7,
                            initial_step_length=0.1, n_points=200)
                        if streamlines.n_points > 0:
                            pl.add_mesh(streamlines.tube(radius=5.0), color='#00e5ff', opacity=0.9, smooth_shading=True)
                    except Exception:
                        pass

                if show_well:
                    well_pts = np.linspace([well_x, well_y, 50], [well_x, well_y, -well_depth], 60)
                    well_line = pv.lines_from_points(well_pts)
                    pl.add_mesh(well_line.tube(radius=8.0), color='#ff1744', smooth_shading=True)
                    pl.add_mesh(pv.Sphere(radius=50, center=[well_x, well_y, 50]), color='#ff1744')

                pl.add_text("Geothermal Reservoir 3D Model", font_size=16, color='white', position='upper_edge')
                pl.add_text(f"T: {T.min():.0f}-{T.max():.0f} °C | P: {P.min()/1e6:.0f}-{P.max()/1e6:.0f} MPa",
                            font_size=11, color='#94a3b8', position='lower_edge')
                pl.view_isometric()
                pl.camera.zoom(1.1)

                img_path = '/tmp/garuda_3d_render.png'
                pl.screenshot(img_path, transparent_background=False)

            # Display the rendered image
            st.image(img_path, use_container_width=True)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Grid Size", f"{nx}×{ny}×{nz}")
            m2.metric("Cells", f"{grid.n_cells:,}")
            m3.metric("T Range", f"{T.min():.0f}–{T.max():.0f}", "°C")
            m4.metric("P Range", f"{P.min()/1e6:.0f}–{P.max()/1e6:.0f}", "MPa")

            # Export
            with open(img_path, 'rb') as fimg:
                img_bytes = fimg.read()
            st.download_button(
                label="📥 Download PNG",
                data=img_bytes,
                file_name="garuda_3d_reservoir.png",
                mime="image/png",
                use_container_width=True,
            )

            # VTK export for ParaView
            vtk_path = '/tmp/garuda_3d_grid.vtk'
            grid.save(vtk_path)
            with open(vtk_path, 'rb') as fvtk:
                vtk_bytes = fvtk.read()
            st.download_button(
                label="📥 Download VTK (ParaView)",
                data=vtk_bytes,
                file_name="garuda_3d_reservoir.vtk",
                mime="application/vnd.vtk",
                use_container_width=True,
            )
