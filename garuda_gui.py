#!/usr/bin/env python3
"""GARUDA Reservoir Simulator — Streamlit GUI entry-point.

Run with::

    streamlit run garuda_gui.py

This file owns the page configuration, theme CSS, and sidebar; the actual
panels are implemented in ``garuda/gui/*.py`` and registered in
``garuda.gui.PAGES``.
"""

from __future__ import annotations

import streamlit as st

from garuda.gui import PAGES

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM THEME
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GARUDA Reservoir Simulator",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium dark/reservoir-engineering look
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .css-1cyp8cj.e16nr0p33 {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }

        .card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #38bdf8;
        }

        .metric-label {
            font-size: 0.875rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }

        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }

        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);
        }

        .stSlider>div>div>div {
            background: #38bdf8 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background: #1e293b !important;
            border-bottom: 2px solid #38bdf8;
        }

        .dataframe {
            border-radius: 12px;
            overflow: hidden;
        }

        .stSuccess {
            border-radius: 12px;
            border-left: 4px solid #22c55e;
        }

        .stInfo {
            border-radius: 12px;
            border-left: 4px solid #38bdf8;
        }

        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.025em;
        }

        h2, h3 {
            font-weight: 600 !important;
            color: #e2e8f0 !important;
        }

        .streamlit-expanderHeader {
            font-weight: 500;
            border-radius: 8px;
            background: rgba(30, 41, 59, 0.5);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 1.5rem; margin: 0; color: #38bdf8;">🔥 GARUDA</h1>
            <p style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
                Open-Source Reservoir & Geothermal Simulation
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span>Tests</span><span style="color: #22c55e; font-weight: 600;">560 passed</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Version</span><span style="color: #f59e0b; font-weight: 600;">v0.2.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("[GitHub](https://github.com/zakusworo/garuda)")


# ═══════════════════════════════════════════════════════════════════════════
#  DISPATCH
# ═══════════════════════════════════════════════════════════════════════════
PAGES[page].render()
