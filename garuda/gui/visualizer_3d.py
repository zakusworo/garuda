"""GUI page: visualizer_3d.

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
    st.markdown("<h1>🧊 3D Reservoir Visualizer</h1>", unsafe_allow_html=True)
    st.markdown("""<p style='color:#94a3b8;'>Publication-quality 3D rendering with isothermal surfaces, cross-sections, streamlines, and well trajectories.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.subheader("Grid Settings")
            nx = st.slider("NX", 10, 80, 40, help="Grid cells in X")
            ny = st.slider("NY", 10, 80, 40, help="Grid cells in Y")
            nz = st.slider("NZ", 5, 40, 20, help="Grid cells in Z (depth)")
            dx = st.number_input("ΔX (m)", 10.0, 500.0, 50.0)
            dy = st.number_input("ΔY (m)", 10.0, 500.0, 50.0)
            dz = st.number_input("ΔZ (m)", 5.0, 200.0, 25.0)

        with st.container(border=True):
            st.subheader("Temperature Field")
            t_surface = st.number_input("Surface temperature (°C)", 10.0, 150.0, 80.0)
            t_bottom = st.number_input("Bottom temperature (°C)", 100.0, 400.0, 280.0)
            heat_source_radius = st.number_input("Heat source radius (m)", 100.0, 2000.0, 500.0)

        with st.container(border=True):
            st.subheader("Production Well")
            well_x = st.number_input("Well X (m)", 0.0, 5000.0, 800.0)
            well_y = st.number_input("Well Y (m)", 0.0, 5000.0, 1200.0)
            well_depth = st.number_input("Well depth (m)", 50.0, 2000.0, 450.0)
            drawdown = st.number_input("Drawdown (MPa)", 0.0, 10.0, 5.0)

        with st.container(border=True):
            st.subheader("Visualization Options")
            show_isotherms = st.multiselect("Isothermal surfaces (°C)", [120, 150, 180, 200, 220], default=[150, 200])
            show_slices = st.checkbox("Show cross-section slices", value=True)
            show_streamlines = st.checkbox("Show flow streamlines", value=True)
            show_well = st.checkbox("Show well trajectory", value=True)

        render_btn = st.button("▶️ Render 3D Scene", type="primary", use_container_width=True)

    with col2:
        if render_btn:
            import pyvista as pv

            with st.spinner("Building 3D grid and computing fields..."):
                grid = pv.ImageData()
                grid.dimensions = [nx, ny, nz]
                grid.spacing = [dx, dy, dz]
                grid.origin = [0, 0, -nz * dz]

                x = np.linspace(0, (nx - 1) * dx, nx)
                y = np.linspace(0, (ny - 1) * dy, ny)
                z = np.linspace(grid.origin[2], 0, nz)
                XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')

                cx, cy = (nx - 1) * dx / 2, (ny - 1) * dy / 2
                r_hot = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
                depth = (-ZZ) / (nz * dz)

                T = t_surface + (t_bottom - t_surface) * depth * np.exp(-r_hot / heat_source_radius)

                # Well cooling
                r_well = np.sqrt((XX - well_x) ** 2 + (YY - well_y) ** 2)
                cooling = 60.0 * np.exp(-r_well / 80.0) * np.exp(-(-ZZ - well_depth) ** 2 / 5000.0)
                T -= cooling

                grid.point_data['temperature'] = T.ravel(order='F')

                # Pressure field
                P = 20e6 + 850.0 * 9.81 * (-ZZ)
                P -= drawdown * 1e6 * np.exp(-r_well / 100.0) * np.exp(-(-ZZ - well_depth) ** 2 / 8000.0)
                grid.point_data['pressure'] = P.ravel(order='F')

                # Velocity (for streamlines)
                k, mu = 1e-13, 2.5e-4
                P_3d = P.reshape((nx, ny, nz), order='F')
                dPx = np.gradient(P_3d, dx, axis=0)
                dPy = np.gradient(P_3d, dy, axis=1)
                dPz = np.gradient(P_3d, dz, axis=2)
                vx = -k / mu * dPx
                vy = -k / mu * dPy
                vz = -k / mu * dPz
                vectors = np.stack([vx.ravel(order='F'), vy.ravel(order='F'), vz.ravel(order='F')], axis=1)
                grid.point_data['velocity'] = vectors

            with st.spinner("Rendering 3D scene..."):
                pv.OFF_SCREEN = True
                pl = pv.Plotter(off_screen=True, window_size=(1600, 1100))

                # Isothermal surfaces
                for temp in show_isotherms:
                    contour = grid.contour([float(temp)], scalars='temperature')
                    if contour.n_points > 0:
                        pl.add_mesh(contour, cmap='coolwarm', opacity=0.4,
                                    clim=[T.min(), T.max()], show_edges=False, smooth_shading=True)

                if show_slices:
                    slice_x = grid.slice(normal='x', origin=(cx, cy, -nz * dz / 2))
                    slice_y = grid.slice(normal='y', origin=(cx, cy, -nz * dz / 2))
                    slice_z = grid.slice(normal='z', origin=(cx, cy, -nz * dz / 4))
                    pl.add_mesh(slice_x, cmap='coolwarm', opacity=0.6, clim=[T.min(), T.max()],
                                scalar_bar_args={'title': 'Temperature (°C)', 'vertical': True, 'position_x': 0.88})
                    pl.add_mesh(slice_y, cmap='coolwarm', opacity=0.35, clim=[T.min(), T.max()])
                    pl.add_mesh(slice_z, cmap='coolwarm', opacity=0.25, clim=[T.min(), T.max()])

                if show_streamlines:
                    seed_pts = np.array([
                        [well_x - 300, well_y, -30], [well_x + 300, well_y, -30],
                        [well_x, well_y - 300, -30], [well_x, well_y + 300, -30],
                        [well_x - 150, well_y - 150, -30], [well_x + 150, well_y + 150, -30],
                    ])
                    seed = pv.PolyData(seed_pts)
                    try:
                        streamlines = grid.streamlines_from_source(
                            seed, vectors_name='velocity',
                            integration_direction='both', max_time=5e7,
                            initial_step_length=0.1, n_points=200)
                        if streamlines.n_points > 0:
                            pl.add_mesh(streamlines.tube(radius=5.0), color='#00e5ff', opacity=0.9, smooth_shading=True)
                    except Exception:
                        pass

                if show_well:
                    well_pts = np.linspace([well_x, well_y, 50], [well_x, well_y, -well_depth], 60)
                    well_line = pv.lines_from_points(well_pts)
                    pl.add_mesh(well_line.tube(radius=8.0), color='#ff1744', smooth_shading=True)
                    pl.add_mesh(pv.Sphere(radius=50, center=[well_x, well_y, 50]), color='#ff1744')

                pl.add_text("Geothermal Reservoir 3D Model", font_size=16, color='white', position='upper_edge')
                pl.add_text(f"T: {T.min():.0f}-{T.max():.0f} °C | P: {P.min()/1e6:.0f}-{P.max()/1e6:.0f} MPa",
                            font_size=11, color='#94a3b8', position='lower_edge')
                pl.view_isometric()
                pl.camera.zoom(1.1)

                img_path = '/tmp/garuda_3d_render.png'
                pl.screenshot(img_path, transparent_background=False)

            # Display the rendered image
            st.image(img_path, use_container_width=True)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Grid Size", f"{nx}×{ny}×{nz}")
            m2.metric("Cells", f"{grid.n_cells:,}")
            m3.metric("T Range", f"{T.min():.0f}–{T.max():.0f}", "°C")
            m4.metric("P Range", f"{P.min()/1e6:.0f}–{P.max()/1e6:.0f}", "MPa")

            # Export
            with open(img_path, 'rb') as fimg:
                img_bytes = fimg.read()
            st.download_button(
                label="📥 Download PNG",
                data=img_bytes,
                file_name="garuda_3d_reservoir.png",
                mime="image/png",
                use_container_width=True,
            )

            # VTK export for ParaView
            vtk_path = '/tmp/garuda_3d_grid.vtk'
            grid.save(vtk_path)
            with open(vtk_path, 'rb') as fvtk:
                vtk_bytes = fvtk.read()
            st.download_button(
                label="📥 Download VTK (ParaView)",
                data=vtk_bytes,
                file_name="garuda_3d_reservoir.vtk",
                mime="application/vnd.vtk",
                use_container_width=True,
            )
