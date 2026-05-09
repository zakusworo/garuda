"""GUI page: single_phase.

Auto-extracted from monolithic garuda_gui.py during the GUI split refactor.
"""

from __future__ import annotations

import base64
import io
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
from garuda.gui.utils import _export_csv, _export_json, _metric_card, m2_to_md, md_to_m2


def render() -> None:
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
