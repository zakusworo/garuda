"""GUI page: source_network.

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
    st.markdown("<h1>🌐 Source Network</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.subheader("Network Builder")
            num_sources = st.slider("Number of sources", 1, 10, 2)

        from garuda.core.source_network import Reinjector, Separator, SourceGroup
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
