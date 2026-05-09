"""GUI page: thermodynamics.

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
