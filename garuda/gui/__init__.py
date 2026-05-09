"""GARUDA Streamlit GUI — modular page registry.

Each page lives in its own module and exposes a ``render()`` function that
draws the page's UI. The dispatcher in ``garuda_gui.py`` selects the page
to call based on the sidebar radio.
"""

from __future__ import annotations

from garuda.gui import (
    dual_porosity,
    grid_builder,
    home,
    iapws,
    multiphase,
    single_phase,
    source_network,
    thermodynamics,
    visualizer_3d,
    well_model,
)

# Mapping displayed in the sidebar -> module exposing render().
PAGES = {
    "🏠 Home": home,
    "📐 Grid Builder": grid_builder,
    "💧 Single-Phase Flow": single_phase,
    "🎯 Well Model": well_model,
    "🌡️ IAPWS-IF97": iapws,
    "⚗️ Multiphase": multiphase,
    "🪨 Dual Porosity": dual_porosity,
    "🌐 Source Network": source_network,
    "🔬 Thermodynamics": thermodynamics,
    "🧊 3D Visualizer": visualizer_3d,
}

__all__ = ["PAGES"]
