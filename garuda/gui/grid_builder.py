"""GUI page: grid_builder.

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
