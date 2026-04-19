"""
GARUDA - Geothermal And Reservoir Understanding with Data-driven Analytics

Modern reservoir simulation for petroleum and geothermal energy.

Acronym: G-A-R-U-D-A
  G = Geothermal
  A = And
  R = Reservoir
  U = Understanding
  D = Data-driven
  A = Analytics
  (GARU + DA = GARUDA)

╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗  ██████╗  ██████╗ ██╗  ██╗███████╗███████╗         ║
║  ██╔════╝ ██╔═══██╗██╔═══██╗██║ ██╔╝██╔════╝██╔════╝         ║
║  ██║  ███╗██████╔╝██║   ██║█████╔╝ █████╗  ███████╗         ║
║  ██║   ██║██╔══██╗██║   ██║██╔═██╗ ██╔══╝  ╚════██║         ║
║  ╚██████╔╝██║  ██║╚██████╔╝██║  ██╗███████╗███████║         ║
║   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝         ║
║                                                               ║
║         Geothermal And Reservoir Understanding               ║
║              with Data-driven Analytics                      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Simulator for:
  • Petroleum reservoirs (oil & gas)
  • Geothermal reservoirs (volcanic systems)
  • CO₂ sequestration
  • Groundwater flow

Features:
  • TPFA finite volume solver
  • Non-isothermal multiphase flow
  • Heterogeneous permeability
  • AI/ML integration ready
  • Indonesian geothermal optimized

Author: Zulfikar Aji Kusworo
License: MIT
"""

from garuda.core.grid import Grid, StructuredGrid
from garuda.core.tpfa_solver import TPFASolver
from garuda.core.fluid_properties import FluidProperties
from garuda.core.rock_properties import RockProperties
from garuda.core.iapws_properties import IAPWSFluidProperties, WaterSteamProperties
from garuda.physics.single_phase import SinglePhaseFlow
from garuda.physics.thermal import ThermalFlow
from garuda.physics.well_models import PeacemanWell, WellManager, WellParameters, WellOperatingConditions

__version__ = "0.1.0-dev"
__author__ = "Zulfikar Aji Kusworo"
__email__ = "greataji13@gmail.com"

# Garuda ASCII logo
GARUDA_LOGO = """
╔═══════════════════════════════════════════════════════════════╗
║   GARUDA - Geothermal And Reservoir Understanding            ║
║          with Data-driven Analytics                          ║
║                                                              ║
║          🇮🇩  Powered by Indonesia  🇮🇩                       ║
╚═══════════════════════════════════════════════════════════════╝
"""

__all__ = [
    "Grid",
    "StructuredGrid",
    "TPFASolver",
    "FluidProperties",
    "RockProperties",
    "SinglePhaseFlow",
    "ThermalFlow",
    "GARUDA_LOGO",
    "__version__",
]
