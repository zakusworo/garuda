"""GUI page: iapws.

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
