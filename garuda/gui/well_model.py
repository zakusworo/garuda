"""GUI page: well_model.

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
