"""GUI page: dual_porosity.

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
