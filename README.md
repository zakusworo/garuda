# GARUDA

## Geothermal And Reservoir Understanding with Data-driven Analytics

[![CI](https://github.com/zakusworo/garuda/actions/workflows/ci.yml/badge.svg)](https://github.com/zakusworo/garuda/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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
*(GARU + DA = GARUDA)*

**GARUDA** is a modern, open-source reservoir simulator for **petroleum** and **geothermal** energy systems, with special focus on Indonesian volcanic geothermal resources and AI/ML integration.

> Named after **Garuda**, the mythical bird king of Indonesian mythology - representing speed, power, and vision in energy simulation.

---

## Features

### Core Capabilities
- ✅ **Single-phase flow** with TPFA (Two-Point Flux Approximation)
- ✅ **Non-isothermal flow** for geothermal applications
- ✅ **Petroleum reservoir** simulation (oil & gas)
- ✅ **Structured grids** (1D, 2D, 3D Cartesian)
- ✅ **Heterogeneous permeability** (channelized, Gaussian random fields)
- ✅ **Numba-accelerated** solvers for performance
- ✅ **Pure Python** implementation (no C++ compilation needed)

### Geothermal Extensions
- 🌡️ Temperature-dependent fluid properties (density, viscosity)
- 🔥 Coupled heat transport (conduction + convection)
- 🌋 Indonesian geothermal reservoir templates (volcanic, high-T)
- 💧 Reinjection modeling for sustainable production

### Petroleum Extensions
- 🛢️ Black oil formulation (planned)
- ⛽ Compositional modeling (planned)
- 📊 History matching tools (planned)
- 🎯 Well optimization (planned)

### AI/ML Integration (Planned)
- 🤖 ML-based permeability upscaling (CNN on heterogeneity)
- 🧠 Neural surrogate models (1000x faster than numerical)
- 📊 Bayesian history matching (MCMC parameter inversion)
- 🎯 RL for well control optimization
- 🔗 Integration with geothermal-agents system

---

## Installation

```bash
# Clone the repository
git clone https://github.com/zakusworo/garuda.git
cd garuda

# Install in development mode
pip install -e ".[dev]"

# Optional: ML capabilities
pip install -e ".[ml]"

# Optional: GPU acceleration
pip install -e ".[gpu]"

# Optional: Geothermal-specific tools
pip install -e ".[geothermal]"
```

---

## Quick Start

### 1D Single-Phase Flow

```python
from garuda import StructuredGrid, TPFASolver, FluidProperties, RockProperties

# Create a 1D grid (10 cells, 100m each)
grid = StructuredGrid(nx=10, ny=1, nz=1, dx=100, dy=1, dz=1)

# Set rock properties (permeability in millidarcy)
rock = RockProperties(porosity=0.2, permeability=100, permeability_unit='md')
grid.set_permiability(rock.permiability_m2)
grid.set_porosity(rock.porosity)

# Set fluid properties (water)
fluid = FluidProperties(fluid_type='water')

# Create TPFA solver
solver = TPFASolver(grid, mu=fluid.mu, rho=fluid.rho)

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

### Geothermal Simulation (Non-Isothermal)

```python
from garuda import StructuredGrid, TPFASolver
from garuda.core import FluidProperties, RockProperties
from garuda.physics import ThermalFlow

# Create 3D grid
grid = StructuredGrid(nx=20, ny=20, nz=10, dx=50, dy=50, dz=20)

# Indonesian geothermal reservoir properties
rock = RockProperties(
    porosity=0.12,
    permeability=150,  # md (fractured volcanic rock)
    permeability_unit='md',
    lambda_rock=2.5,  # W/(m·K)
)

fluid = FluidProperties(fluid_type='geothermal')

# Initialize with geothermal gradient (30°C/km)
thermal = ThermalFlow(grid, rock, fluid)
T_init = thermal.compute_geothermal_gradient(
    surface_temp=298.15,  # 25°C (tropical)
    gradient=0.03,  # 30°C/km
)

# Production well (injection would be negative)
source_terms = [0] * grid.num_cells
source_terms[grid.num_cells // 2] = -50.0  # 50 kg/s production

# Time-stepping
dt = 3600  # 1 hour
for t in range(100):
    result = thermal.step_coupled(
        dt=dt,
        source_terms=source_terms,
        heat_sources=[0] * grid.num_cells,
        bc_type='dirichlet',
        bc_values={'pressure': [250e5, 250e5]},
        flow_solver=TPFASolver(grid, mu=fluid.mu, rho=fluid.rho),
    )
    
    if t % 10 == 0:
        print(f"Step {t}: T_max={thermal.temperature.max()-273.15:.1f}°C")
```

---

## Run the Demos

GARUDA includes standalone demos that work without installing dependencies:

```bash
# 1D single-phase flow demo
python demo.py

# Geothermal field simulation (Indonesian volcanic reservoir)
python demo_geothermal.py
```

---

## Architecture

```
garuda/
├── garuda/
│   ├── core/
│   │   ├── grid.py              # Structured/unstructured grids
│   │   ├── tpfa_solver.py       # TPFA finite volume solver (Numba JIT)
│   │   ├── fluid_properties.py  # PVT and transport properties
│   │   └── rock_properties.py   # Permeability, porosity, thermal
│   ├── physics/
│   │   ├── single_phase.py      # Mass conservation equation
│   │   └── thermal.py           # Coupled heat transport
│   ├── ml/                      # (Coming soon)
│   │   ├── upscaling_cnn.py     # ML permeability upscaling
│   │   └── surrogate_model.py   # Neural network emulator
│   ├── agents/                  # (Coming soon)
│   │   └── integration.py       # AI agent integration
│   └── utils/
│       └── visualization.py     # Plotting utilities
├── examples/
├── tests/
├── docs/
└── pyproject.toml
```

---

## Roadmap

### Phase 1: Core Development ✅ (In Progress)
- [x] Modern Python packaging (pyproject.toml)
- [x] Pure Python TPFA solver
- [x] Structured grid generation
- [x] Basic single-phase flow
- [x] Thermal flow module
- [ ] 2D/3D solver completion
- [ ] Comprehensive test suite
- [ ] Documentation (Sphinx + ReadTheDocs)

### Phase 2: Domain Extensions
- [ ] Multiphase flow (water/steam for geothermal)
- [ ] Black oil model (for petroleum)
- [ ] Real gas EOS (IAPWS-97)
- [ ] Well models (pressure/rate constraints)
- [ ] History matching tools
- [ ] TOUGH2 comparison benchmarks

### Phase 3: AI/ML Integration
- [ ] CNN for permeability upscaling
- [ ] Neural surrogate models
- [ ] Bayesian parameter inversion
- [ ] RL for well optimization
- [ ] Ollama LLM integration

---

## Comparison with Other Simulators

| Feature | GARUDA | TOUGH2 | MRST | tNavigator |
|---------|--------|--------|------|------------|
| **License** | MIT (Open) | Proprietary | GPL | Proprietary |
| **Cost** | Free | $50k+ | Free | $100k+ |
| **Language** | Python | Fortran | MATLAB | C++ |
| **Geothermal** | ✅ Yes | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| **Petroleum** | 🔄 Planned | ✅ Yes | ✅ Yes | ✅ Yes |
| **AI/ML** | ✅ Planned | ❌ No | ⚠️ Basic | ⚠️ Basic |
| **Indonesian Focus** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Installation** | pip | Manual | MATLAB req. | Installer |

---

## Contributing

Contributions welcome! Areas needing help:

1. **2D/3D solver completion** - Finish face connectivity and flux assembly
2. **Test suite** - Add more unit and integration tests
3. **Documentation** - Expand API docs and tutorials
4. **Multiphase flow** - Implement black oil / compositional
5. **ML integration** - Build surrogate models and upscaling

### Development Setup

```bash
# Fork and clone
git clone https://github.com/zakusworo/garuda.git
cd garuda

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

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

---

## License

MIT License - see [LICENSE](LICENSE) for details.

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
