"""GUI page: multiphase.

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
