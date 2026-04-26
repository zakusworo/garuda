# GARUDA

## Geothermal And Reservoir Understanding with Data-driven Analytics

[![CI](https://github.com/zakusworo/garuda/actions/workflows/ci.yml/badge.svg)](https://github.com/zakusworo/garuda/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 50+](https://img.shields.io/badge/tests-50%2B%20passing-brightgreen.svg)](https://github.com/zakusworo/garuda/actions/workflows/ci.yml)
[![Coverage: 49%](https://img.shields.io/badge/coverage-49%25-yellow.svg)](./htmlcov)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with ❤️ in Indonesia](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F%20Indonesia-red)](https://indonesia.id)

```
╔═══════════════════════════════════════════════════════════════╗
║   GARUDA - Geothermal And Reservoir Understanding            ║
║          with Data-driven Analytics                          ║
║                                                              ║
║          🇮🇩  Powered by Indonesia  🇮🇩                       ║
╚═══════════════════════════════════════════════════════════════╝
```

**Acronym:** **G**eothermal **A**nd **R**eservoir **U**nderstanding + **D**ata-driven **A**nalytics  

**GARUDA** is a modern, open-source reservoir simulator for **petroleum** and **geothermal** energy systems, with special focus on Indonesian volcanic geothermal resources and AI/ML integration.

> Named after **Garuda**, the mythical bird king of Indonesian mythology - representing speed, power, and vision in energy simulation.

---

## Features

### Core Capabilities
- ✅ **Single-phase flow** with TPFA (Two-Point Flux Approximation)
- ✅ **Non-isothermal flow** for geothermal applications
- ✅ **IAPWS-IF97** thermophysical properties — saturation pressure, density, viscosity, enthalpy, specific heat, thermal conductivity
- ✅ **Well models** — pressure-constrained (BHP) or rate-constrained wells with automatic switching
- ✅ **Structured grids** (1D, 2D, 3D Cartesian) with heterogeneous permeability and porosity
- ✅ **Numba-accelerated** solvers for performance
- ✅ **Pure Python** implementation (no C++ compilation needed)
- ✅ **50+ unit & integration tests** with pytest and coverage reporting

### Geothermal Extensions
- 🌡️ Temperature-dependent fluid properties (IAPWS-IF97 water/steam)
- 🔥 Coupled heat transport (conduction + convection)
- 💧 Reinjection modeling for sustainable production
- 🌋 Indonesian geothermal reservoir templates (volcanic, high-T)

### Petroleum Extensions
- 🛢️ Single-phase oil/gas (currently) with extension points for black-oil/compositional
- ⛽ Compositional modeling (planned)
- 📊 History matching tools (planned)
- 🎯 Well optimization (planned)

### AI/ML Integration (Planned)
- 🤖 ML-based permeability upscaling (CNN on heterogeneity)
- 🧠 Neural surrogate models (1000× faster than numerical)
- 📊 Bayesian history matching (MCMC parameter inversion)
- 🎯 RL for well control optimization
- 🔗 Multi-agent AI assistant integration (Ollama LLM)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/zakusworo/garuda.git
cd garuda

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Optional: ML capabilities
pip install -e ".[ml]"

# Optional: GPU acceleration
pip install -e ".[gpu]"

# Optional: Geothermal-specific tools
pip install -e ".[geothermal]"
```

### Requirements
- Python 3.10+
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Numba ≥ 0.56

---

## Quick Start

### 1D Single-Phase Flow

```python
from garuda import StructuredGrid, TPFASolver

# Create a 1D grid (10 cells, 100 m each, 1 m² cross-section)
grid = StructuredGrid(nx=10, ny=1, nz=1, dx=100.0, dy=1.0, dz=1.0)

# Set rock properties directly on the grid
import numpy as np
grid.set_porosity(np.full(grid.num_cells, 0.2))
grid.set_permeability(np.full(grid.num_cells, 1e-12))  # ~1 Darcy in m²

# Create TPFA solver — uses water viscosity from IAPWS-IF97
solver = TPFASolver(grid, mu=1e-3, rho=998.0)

# Define boundary conditions (Dirichlet: p_left=200 bar, p_right=100 bar)
bc_values = [200e5, 100e5]  # Pa

# Solve for pressure
pressure = solver.solve(
    source_terms=[0] * grid.num_cells,
    bc_type='dirichlet',
    bc_values=bc_values,
)

print(f"Pressure range: {pressure.min()/1e5:.1f} - {pressure.max()/1e5:.1f} bar")
```

### Well Model (BHP or Rate Constraint)

```python
from garuda.physics.well_models import WellModel
from garuda.core.fluid_properties import IAPWSFluidProperties

fluid = IAPWSFluidProperties()

# Producer well — pressure-controlled
well = WellModel(
    name="PROD-1",
    coordinates=(500.0, 500.0, 1000.0),
    radius=0.1,           # wellbore radius [m]
    skin=0.0,             # skin factor
    perf_top=900.0,
    perf_bottom=1100.0,
    grid=grid,
)

# Well is controlled by bottom-hole pressure (negative = producer)
well.set_operating_constraint(
    constraint_type = "pressure",   # or "rate"
    target_value    = -150e5,      # 150 bar BHP, producer
    max_rate        = 50.0,        # max 50 kg/s
    min_pressure    = 80e5,        # shut-in below this
)

# During simulation — returns mass rate [kg/s]
rate = well.compute_rate(
    pressure=np.full(grid.num_cells, 200e5),
    density=fluid.density(pressure=200e5, temperature=523.15),
    viscosity=fluid.viscosity(pressure=200e5, temperature=523.15),
)
print(f"Well rate: {rate:.2f} kg/s")
```

### IAPWS-IF97 Thermophysical Properties

```python
from garuda.core.iapws_properties import IAPWSFluidProperties

props = IAPWSFluidProperties()

# Single properties
rho = props.density(pressure=15.0, temperature=550.0)   # [MPa], [K] → kg/m³
mu  = props.viscosity(pressure=15.0, temperature=550.0)    # → Pa·s
h   = props.enthalpy(pressure=15.0, temperature=550.0)    # → kJ/kg

# Get all at once
all_props = props.get_all_properties(1.0, 293.15)
# → {
#     'density': 998.14,
#     'viscosity': 0.001003,
#     'enthalpy': 84.01,
#     'specific_heat_cp': 4.182,
#     'thermal_conductivity': 0.598,
#     'phase': 'liquid'
# }

# Saturation curve
Tsat = props.saturation_temperature(pressure=10.0)   # MPa → K
Psat = props.saturation_pressure(temperature=523.15)  # K → MPa
phase = props.get_region(pressure=10.0, temperature=523.15)  # 1=liquid, 2=vapor
```

### Geothermal Simulation (Non-Isothermal)

```python
from garuda.core.grid import StructuredGrid
from garuda.core.iapws_properties import IAPWSFluidProperties
from garuda.physics.thermal import ThermalFlow
from garuda.core.rock_properties import RockProperties

# 3D reservoir grid
grid = StructuredGrid(nx=20, ny=20, nz=10, dx=50.0, dy=50.0, dz=20.0)

# Indonesian volcanic reservoir rock
rock = RockProperties(
    porosity=0.12,
    permeability=150.0,       # md
    permeability_unit='md',
    lambda_rock=2.5,          # W/(m·K) thermal conductivity
    cp=840.0,               # J/(kg·K)
)
grid.set_porosity(np.full(grid.num_cells, rock.porosity))
grid.set_permeability(np.full(grid.num_cells, rock.permeability_m2))

# Geothermal fluid + thermal model
fluid = IAPWSFluidProperties()
thermal = ThermalFlow(grid, rock, fluid)

# Geothermal gradient initialization (30 °C/km)
T_init = thermal.compute_geothermal_gradient(
    surface_temp=298.15,   # 25 °C
    gradient=0.03,         # 30 °C/km
)

# Injection well (positive rate = injector)
source_terms = [0.0] * grid.num_cells
source_terms[grid.num_cells // 2] = 50.0   # 50 kg/s injection

# Time-stepping
dt = 3600  # 1 hour
for step in range(100):
    result = thermal.step_coupled(
        dt=dt,
        source_terms=source_terms,
        heat_sources=[0.0] * grid.num_cells,
        bc_type='dirichlet',
        bc_values={'pressure': [250e5, 250e5]},
        flow_solver=TPFASolver(grid, mu=1e-3, rho=998.0),
    )

    if step % 10 == 0:
        print(f"Step {step}: T_max={thermal.temperature.max()-273.15:.1f}°C")
```

---

## Heterogeneous Permeability

```python
# Channelized (fractured) permeability field
rock = RockProperties()
rock.set_channelized_permeability(
    nx=50, ny=50, nz=10,
    channel_orientation='x',
    channel_fraction=0.2,     # 20% of grid is high-perm channel
    k_channel=1000.0,         # md
    k_background=10.0,        # md
)
grid.set_permeability(rock.permeability_m2)

# Or Gaussian random field (synthetic geology)
rock.set_gaussian_permeability(nx=50, ny=50, mean=100.0, std=30.0)
grid.set_permeability(rock.permeability_m2)
```

---

## Run the Demos

```bash
# 1D single-phase flow demo
python demo.py

# Geothermal field simulation (Indonesian volcanic reservoir)
python demo_geothermal.py
```

> These standalone demos work with NumPy only — no heavy dependencies needed.

---

## Architecture

```
garuda/
├── garuda/
│   ├── core/
│   │   ├── grid.py              # Structured Cartesian grids (1D/2D/3D)
│   │   ├── tpfa_solver.py       # TPFA finite volume solver (Numba JIT)
│   │   ├── fluid_properties.py  # Basic PVT properties
│   │   ├── iapws_properties.py  # IAPWS-IF97 water/steam properties
│   │   └── rock_properties.py   # Permeability, porosity, thermal
│   ├── physics/
│   │   ├── single_phase.py      # Single-phase mass conservation
│   │   ├── thermal.py           # Coupled non-isothermal flow
│   │   └── well_models.py       # BHP/rate-constrained well models
│   ├── ml/                      # (Planned)
│   │   ├── upscaling_cnn.py     # ML permeability upscaling
│   │   └── surrogate_model.py   # Neural network emulator
│   ├── agents/                  # (Planned)
│   │   └── integration.py       # Multi-agent AI assistant
│   └── utils/
│       └── visualization.py     # Plotting utilities
├── examples/
├── tests/                       # 50+ unit & integration tests
├── docs/
└── pyproject.toml
```

---

## Roadmap

### Phase 1: Core Development ✅ (v0.1.0)
- [x] Modern Python packaging (pyproject.toml)
- [x] Pure Python TPFA solver
- [x] Structured grid generation (1D/2D/3D, heterogeneous)
- [x] Basic single-phase flow
- [x] **IAPWS-IF97** water/steam properties
- [x] **Well models** (BHP + rate constraints)
- [x] Thermal flow module
- [x] **50+ unit and integration tests**
- [ ] 2D/3D solver completion (pending)
- [ ] Documentation (Sphinx + ReadTheDocs)

### Phase 2: Domain Extensions
- [ ] Multiphase flow (water/steam two-phase for geothermal)
- [ ] Black oil model (for petroleum)
- [ ] History matching tools
- [ ] TOUGH2 comparison benchmarks

### Phase 3: AI/ML Integration
- [ ] CNN for permeability upscaling
- [ ] Neural surrogate models
- [ ] Bayesian parameter inversion
- [ ] RL for well optimization
- [ ] Ollama LLM multi-agent assistant

---

## Comparison with Other Simulators

| Feature | GARUDA | TOUGH2 | MRST | tNavigator |
|---------|--------|--------|------|------------|
| **License** | MIT (Open) | Proprietary | GPL | Proprietary |
| **Cost** | Free | $50k+ | Free | $100k+ |
| **Language** | Python | Fortran | MATLAB | C++ |
| **Geothermal** | ✅ Yes | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| **Petroleum** | 🔄 Single-phase | ✅ Yes | ✅ Yes | ✅ Yes |
| **AI/ML** | ✅ Planned | ❌ No | ⚠️ Basic | ⚠️ Basic |
| **Indonesian Focus** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Installation** | `pip install -e .` | Manual | MATLAB req. | Installer |

---

## Contributing

Contributions welcome! Areas needing help:

1. **2D/3D solver completion** — Finish face connectivity and flux assembly
2. **Test suite** — Add more unit and integration tests toward 80% coverage
3. **Documentation** — Expand API docs and tutorials
4. **Multiphase flow** — Implement black oil / compositional two-phase
5. **ML integration** — Build surrogate models and upscaling

### Development Setup

```bash
# Fork and clone
git clone https://github.com/zakusworo/garuda.git
cd garuda

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=garuda --cov-report=html

# Lint
ruff check garuda/
black garuda/
```

---

## Citation

If you use GARUDA in your research, please cite:

```bibtex
@software{garuda2026,
  author = {Kusworo, Zulfikar Aji},
  title = {GARUDA: Geothermal And Reservoir Understanding with Data-driven Analytics},
  year = {2026},
  url = {https://github.com/zakusworo/garuda},
  doi = {10.5281/zenodo.19653501},
}
```

---

## Acknowledgments

- Inspired by the deprecated [PRESTO](https://github.com/padmec-reservoir/PRESTO) project
- Built on NumPy, SciPy, and Numba
- IAPWS-IF97 implementation for industrial-grade water/steam thermodynamics

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Author**: Zulfikar Aji Kusworo  
**Email**: greataji13@gmail.com  
**GitHub**: [@zakusworo](https://github.com/zakusworo)

---

```
╔═══════════════════════════════════════════════════════════════╗
║         Soaring Above Energy Challenges                      ║
║                  🇮🇩 GARUDA 🇮🇩                               ║
╚═══════════════════════════════════════════════════════════════╝
```
