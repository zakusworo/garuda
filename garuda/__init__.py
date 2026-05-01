"""GARUDA - Geothermal And Reservoir Understanding with Data-driven Analytics

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

from garuda.core.fluid_properties import FluidProperties
from garuda.core.grid import Grid, StructuredGrid
from garuda.core.iapws_properties import IAPWSFluidProperties, WaterSteamProperties
from garuda.core.rock_properties import RockProperties
from garuda.core.tpfa_solver import TPFASolver
from garuda.physics.capillary_pressure import BrooksCoreyPc, CapillaryPressureModel, VanGenuchtenPc
from garuda.physics.multiphase import MultiphaseFlow, MultiphaseState
from garuda.physics.relative_permeability import (
    CoreyRelativePermeability,
    LinearRelativePermeability,
    RelativePermeabilityModel,
    StoneIRelativePermeability,
    VanGenuchtenMualem,
)
from garuda.physics.single_phase import SinglePhaseFlow
from garuda.physics.thermal import ThermalFlow
from garuda.physics.well_models import PeacemanWell, WellManager, WellOperatingConditions, WellParameters
from garuda.core.dual_porosity import BlockGeometry, DualPorosityModel, DualPorosityParams, TransferModel
from garuda.core.region_thermodynamics import (
    FluidThermoState,
    RegionThermodynamics,
    SaturationCurve,
    SupercriticalRegion,
    SteamRegion,
    ThermodynamicsRegion,
    WaterRegion,
)
from garuda.core.source_network import (
    Reinjector,
    Separator,
    SourceGroup,
    SourceNetwork,
    SourceNode,
)

__version__ = "0.2.0"

__all__ = [
    "Grid",
    "StructuredGrid",
    "TPFASolver",
    "FluidProperties",
    "RockProperties",
    "IAPWSFluidProperties",
    "WaterSteamProperties",
    "SinglePhaseFlow",
    "ThermalFlow",
    "MultiphaseFlow",
    "MultiphaseState",
    "PeacemanWell",
    "WellManager",
    "WellParameters",
    "WellOperatingConditions",
    "RelativePermeabilityModel",
    "CoreyRelativePermeability",
    "VanGenuchtenMualem",
    "LinearRelativePermeability",
    "StoneIRelativePermeability",
    "CapillaryPressureModel",
    "BrooksCoreyPc",
    "VanGenuchtenPc",
    "TransferModel",
    "BlockGeometry",
    "DualPorosityModel",
    "DualPorosityParams",
    "ThermodynamicsRegion",
    "WaterRegion",
    "SteamRegion",
    "SupercriticalRegion",
    "SaturationCurve",
    "RegionThermodynamics",
    "FluidThermoState",
    "SourceNode",
    "Separator",
    "Reinjector",
    "SourceGroup",
    "SourceNetwork",
    "GARUDA_LOGO",
    "__version__",
]

# Optional PETSc solver backend
has_petsc = False

try:
    from garuda.solvers.petsc_solver import PETScDMSolver, PETScTPFASolver

    has_petsc = True
    __all__.extend(["PETScTPFASolver", "PETScDMSolver", "has_petsc"])
except ImportError:
    pass

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
    "IAPWSFluidProperties",
    "WaterSteamProperties",
    "SinglePhaseFlow",
    "ThermalFlow",
    "MultiphaseFlow",
    "MultiphaseState",
    "PeacemanWell",
    "WellManager",
    "WellParameters",
    "WellOperatingConditions",
    "RelativePermeabilityModel",
    "CoreyRelativePermeability",
    "VanGenuchtenMualem",
    "LinearRelativePermeability",
    "StoneIRelativePermeability",
    "CapillaryPressureModel",
    "BrooksCoreyPc",
    "VanGenuchtenPc",
    "TransferModel",
    "BlockGeometry",
    "DualPorosityModel",
    "DualPorosityParams",
    "ThermodynamicsRegion",
    "WaterRegion",
    "SteamRegion",
    "SupercriticalRegion",
    "SaturationCurve",
    "RegionThermodynamics",
    "FluidThermoState",
    "SourceNode",
    "Separator",
    "Reinjector",
    "SourceGroup",
    "SourceNetwork",
    "GARUDA_LOGO",
    "__version__",
]
