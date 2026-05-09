"""GUI page: home.

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
