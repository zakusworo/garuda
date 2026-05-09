"""Shared helpers used by every GUI page."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st


def md_to_m2(k_md: float) -> float:
    """Convert millidarcy to m²."""
    return k_md * 9.869233e-16


def m2_to_md(k_m2: float) -> float:
    """Convert m² to millidarcy."""
    return k_m2 / 9.869233e-16


def metric_card(label: str, value: str, unit: str = "") -> str:
    """HTML for a small metric card; render via ``st.markdown(..., unsafe_allow_html=True)``."""
    unit_html = f'<div style="font-size: 0.75rem; color: #64748b;">{unit}</div>' if unit else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {unit_html}
    </div>
    """


def export_csv(df: pd.DataFrame, filename: str) -> None:
    """Render a Streamlit download button for a DataFrame as CSV."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def export_json(data: dict, filename: str) -> None:
    """Render a Streamlit download button for a dict as JSON."""
    j = json.dumps(data, indent=2, default=str).encode("utf-8")
    st.download_button(
        label="📥 Download JSON",
        data=j,
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


# Backwards-compat aliases (the original GUI used a leading underscore).
_metric_card = metric_card
_export_csv = export_csv
_export_json = export_json
