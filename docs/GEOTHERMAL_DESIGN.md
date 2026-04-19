# Geothermal Extension Design for GARUDA

## Overview

This document outlines the design for extending GARUDA with **non-isothermal multiphase flow** capabilities specifically optimized for **Indonesian geothermal reservoirs** (volcanic, high-temperature, tropical climate).

---

## 1. Indonesian Geothermal Context

### 1.1 Typical Reservoir Characteristics

| Parameter | Range | Typical Value | Notes |
|-----------|-------|---------------|-------|
| **Depth** | 1500-3500 m | 2500 m | Deep volcanic systems |
| **Temperature** | 200-350°C | 280°C | High enthalpy (volcanic) |
| **Pressure** | 150-400 bar | 250 bar | Overpressured possible |
| **Permeability** | 10-500 md | 150 md | Fractured volcanic rock |
| **Porosity** | 0.05-0.20 | 0.12 | Low (dense rock) |
| **Fluid type** | Liquid-dominated | Water/steam | Two-phase at production |

### 1.2 Key Challenges

1. **Phase change**: Water flashes to steam during production
2. **Temperature-dependent properties**: Viscosity drops 10x from 25°C to 250°C
3. **Reinjection cooling**: Cold injectate creates thermal fronts
4. **Natural recharge**: Tropical rainfall maintains reservoir pressure
5. **Mineral scaling**: Silica/calcite precipitation at changing T,P

---

## 2. Governing Equations

### 2.1 Mass Conservation (Multiphase)

```
∂/∂t (φ Σᵅ ρᵅ Sᵅ) + ∇·(Σᵅ ρᵅ uᵅ) = Σᵅ qᵅ

where:
    ᵅ = phase (w = water, s = steam)
    Sᵅ = saturation (S_w + S_s = 1)
    uᵅ = -(k·kᵣᵅ/μᵅ)·(∇pᵅ - ρᵅg∇z)  [Darcy's law]
```

### 2.2 Energy Conservation

```
∂/∂t [(ρCp)_bulk T] + ∇·(Σᵅ ρᵅ hᵅ uᵅ) - ∇·(λ_eff ∇T) = Q_T

where:
    (ρCp)_bulk = (1-φ)·ρ_r·Cp_r + φ·Σᵅ Sᵅ·ρᵅ·Cpᵅ
    hᵅ = specific enthalpy of phase ᵅ
    λ_eff = effective thermal conductivity
```

### 2.3 Phase Equilibrium

```
At phase boundary (saturation conditions):
    T = T_sat(p)
    p = p_sat(T)

Using IAPWS-97 formulation for water/steam properties
```

---

## 3. Implementation Plan

### 3.1 Module Structure

```
presto/
├── physics/
│   ├── thermal.py              # ✅ Existing (single-phase thermal)
│   ├── multiphase.py           # NEW: Black oil / water-steam
│   └── geothermal.py           # NEW: High-T specific models
├── eos/
│   ├── iapws97.py              # NEW: Water/steam properties
│   └── real_gas.py             # NEW: Non-condensable gases
├── well/
│   ├── productivity_index.py   # NEW: PI models
│   └── thermal_wellbore.py     # NEW: Wellbore heat loss
└── utils/
    └── geothermal_plots.py     # NEW: T-h diagrams, etc.
```

### 3.2 Water-Steam Properties (IAPWS-97)

```python
# presto/eos/iapws97.py

from scipy.optimize import brentq
import numpy as np

class IAPWS97:
    """
    IAPWS-97 formulation for water and steam properties.
    
    Industrial formulation accurate to 0.1% for:
    - Temperature: 273.15 K to 1073.15 K
    - Pressure: 0 to 100 MPa
    """
    
    # Region boundaries
    T_min = 273.15  # K
    T_max = 1073.15  # K
    p_min = 0  # Pa
    p_max = 100e6  # Pa
    
    @staticmethod
    def saturation_temperature(pressure: float) -> float:
        """
        Compute saturation temperature at given pressure.
        
        Parameters
        ----------
        pressure : float
            Pressure [Pa]
        
        Returns
        -------
        T_sat : float
            Saturation temperature [K]
        """
        # Simplified Antoine equation for demo
        # Full IAPWS-97 has 100+ coefficients
        p_bar = pressure / 1e5
        T_sat = 1 / (1/373.15 - np.log(p_bar) / 4894.0)
        return T_sat
    
    @staticmethod
    def saturation_pressure(temperature: float) -> float:
        """Compute saturation pressure at given temperature."""
        T_bar = temperature / 373.15
        p_sat = np.exp(4894.0 * (1/373.15 - 1/temperature)) * 1e5
        return p_sat
    
    @staticmethod
    def liquid_density(pressure: float, temperature: float) -> float:
        """Liquid water density [kg/m³]."""
        # Simplified correlation
        T_sat = IAPWS97.saturation_temperature(pressure)
        if temperature >= T_sat:
            # At saturation
            rho = 1000 * (1 - (temperature - 273.15) / 374.0) ** 0.35
        else:
            # Compressed liquid
            rho = 1000 * (1 - (temperature - 273.15) / 374.0) ** 0.35
            rho *= 1 + 4.4e-10 * (pressure - 1e5)
        return max(rho, 50)  # Clamp to reasonable range
    
    @staticmethod
    def vapor_density(pressure: float, temperature: float) -> float:
        """Steam density [kg/m³]."""
        # Ideal gas approximation (good for low pressure)
        R = 461.5  # J/(kg·K) for steam
        return pressure / (R * temperature)
    
    @staticmethod
    def liquid_enthalpy(pressure: float, temperature: float) -> float:
        """Liquid water enthalpy [J/kg]."""
        Cp = 4182  # J/(kg·K)
        h = Cp * (temperature - 273.15)
        # Pressure correction
        h += (pressure - 1e5) / 1000
        return h
    
    @staticmethod
    def vapor_enthalpy(pressure: float, temperature: float) -> float:
        """Steam enthalpy [J/kg]."""
        T_sat = IAPWS97.saturation_temperature(pressure)
        h_sat_v = 2.676e6  # Saturation vapor enthalpy at 100°C [J/kg]
        Cp_v = 2000  # Steam Cp [J/(kg·K)]
        h = h_sat_v + Cp_v * (temperature - T_sat)
        return h
    
    @staticmethod
    def latent_heat(pressure: float) -> float:
        """Latent heat of vaporization [J/kg]."""
        T_sat = IAPWS97.saturation_temperature(pressure)
        # Simplified: decreases with T, zero at critical point
        h_fg = 2.257e6 * ((647.0 - T_sat) / (647.0 - 373.15)) ** 0.38
        return max(h_fg, 0)
    
    @staticmethod
    def liquid_viscosity(temperature: float) -> float:
        """Liquid water viscosity [Pa·s]."""
        # Andrade equation
        T_ref = 293.15
        mu_ref = 1e-3
        mu = mu_ref * np.exp(0.02 * (T_ref - temperature))
        return mu
    
    @staticmethod
    def vapor_viscosity(temperature: float) -> float:
        """Steam viscosity [Pa·s]."""
        # Increases with T for gases
        T_ref = 373.15
        mu_ref = 1.2e-5
        mu = mu_ref * (temperature / T_ref) ** 0.7
        return mu
    
    @staticmethod
    def phase_fraction(pressure: float, enthalpy: float) -> tuple:
        """
        Compute steam quality (vapor mass fraction) from enthalpy.
        
        Returns
        -------
        (phase, x) : tuple
            phase: 'liquid', 'two-phase', or 'vapor'
            x: vapor mass fraction (0-1 in two-phase region)
        """
        T_sat = IAPWS97.saturation_temperature(pressure)
        h_l = IAPWS97.liquid_enthalpy(pressure, T_sat)
        h_v = IAPWS97.vapor_enthalpy(pressure, T_sat)
        
        if enthalpy <= h_l:
            return 'liquid', 0.0
        elif enthalpy >= h_v:
            return 'vapor', 1.0
        else:
            x = (enthalpy - h_l) / (h_v - h_l)
            return 'two-phase', x
```

### 3.3 Multiphase Flow Solver

```python
# presto/physics/multiphase.py

import numpy as np
from presto.core.tpfa_solver import TPFASolver
from presto.eos.iapws97 import IAPWS97

class GeothermalMultiphaseSolver:
    """
    Two-phase (water/steam) geothermal reservoir simulator.
    
    Uses sequential implicit method (SIM):
    1. Solve pressure equation (total mass conservation)
    2. Solve saturation equation (phase mass balance)
    3. Solve energy equation
    4. Update fluid properties
    5. Iterate until convergence
    """
    
    def __init__(self, grid, rock):
        self.grid = grid
        self.rock = rock
        
        # State variables
        self.pressure = None      # [Pa]
        self.saturation = None    # Steam saturation S_s
        self.temperature = None   # [K]
        self.enthalpy = None      # [J/kg]
        
        # Relative permeability parameters
        self.swr = 0.2  # Irreducible water saturation
        self.ssr = 0.0  # Residual steam saturation
        self.krw0 = 1.0  # End-point water relperm
        self.krs0 = 1.0  # End-point steam relperm
        self.nw = 2.0   # Water exponent
        self.ns = 2.0   # Steam exponent
    
    def relative_permiability(self, S_w: float, S_s: float) -> tuple:
        """
        Compute relative permeabilities using Corey model.
        
        Returns
        -------
        (krw, krs) : tuple
            Water and steam relative permeabilities
        """
        # Normalize saturations
        S_we = (S_w - self.swr) / (1 - self.swr - self.ssr)
        S_se = (S_s - self.ssr) / (1 - self.swr - self.ssr)
        
        S_we = np.clip(S_we, 0, 1)
        S_se = np.clip(S_se, 0, 1)
        
        # Corey model
        krw = self.krw0 * S_we ** self.nw
        krs = self.krs0 * S_se ** self.ns
        
        return krw, krs
    
    def capillary_pressure(self, S_w: float) -> float:
        """
        Compute capillary pressure (steam-water).
        
        Using Leverett J-function approximation.
        """
        # For geothermal, Pc is often negligible at high T
        # But can be important for reinjection
        S_we = (S_w - self.swr) / (1 - self.swr - self.ssr)
        S_we = np.clip(S_we, 0.01, 0.99)
        
        # Simplified: Pc increases as water saturation decreases
        J = 1.0 / S_we ** 2
        sigma = 0.05  # Surface tension [N/m] (decreases at high T)
        sqrt_k_phi = np.sqrt(self.rock.permiability_m2[0] / self.rock.porosity)
        
        Pc = J * sigma / sqrt_k_phi
        return Pc
    
    def step_sim(self, dt, source_terms, heat_sources, bc, max_iter=20):
        """
        Sequential implicit time step.
        
        Parameters
        ----------
        dt : float
            Time step [s]
        source_terms : dict
            {'water': [...], 'steam': [...]} [kg/s]
        heat_sources : ndarray
            Heat sources [W/m³]
        bc : dict
            Boundary conditions
        max_iter : int
            Maximum coupling iterations
        
        Returns
        -------
        result : dict
            Convergence information
        """
        for iteration in range(max_iter):
            # Store old values
            p_old = self.pressure.copy()
            S_s_old = self.saturation.copy()
            T_old = self.temperature.copy()
            
            # === Step 1: Pressure equation ===
            # Total mass: ∂(φ Σ ρᵅSᵅ)/∂t + ∇·(Σ ρᵅuᵅ) = Σ qᵅ
            # Mobility-weighted pressure solve
            
            # Compute total mobility
            S_w = 1 - self.saturation
            krw, krs = self.relative_permiability(S_w, self.saturation)
            
            rho_w = IAPWS97.liquid_density(self.pressure, self.temperature)
            rho_s = IAPWS97.vapor_density(self.pressure, self.temperature)
            
            mu_w = IAPWS97.liquid_viscosity(self.temperature)
            mu_s = IAPWS97.vapor_viscosity(self.temperature)
            
            lambda_w = krw * rho_w / mu_w
            lambda_s = krs * rho_s / mu_s
            lambda_total = lambda_w + lambda_s
            
            # Build and solve pressure equation
            # (Implementation continues...)
            
            # === Step 2: Saturation equation ===
            # Steam mass: ∂(φ ρ_s S_s)/∂t + ∇·(ρ_s u_s) = q_s
            # Includes phase change (boiling/condensation)
            
            # === Step 3: Energy equation ===
            # Same as ThermalFlow but with phase enthalpies
            
            # === Step 4: Update properties ===
            # Recompute densities, viscosities, etc.
            
            # === Check convergence ===
            # ...
        
        return {'converged': True, 'iterations': iteration + 1}
```

### 3.4 Well Models

```python
# presto/well/productivity_index.py

class GeothermalWell:
    """
    Production/injection well model for geothermal reservoirs.
    
    Supports:
    - Pressure-constrained production
    - Rate-constrained production
    - Enthalpy-limited production (steam quality)
    - Reinjection with thermal breakthrough
    """
    
    def __init__(self, name, position, well_type='production'):
        self.name = name
        self.position = position  # (ix, iy, iz) or (x, y, z)
        self.well_type = well_type
        
        # Well parameters
        self.rw = 0.15  # Wellbore radius [m]
        self.skin = 0.0  # Skin factor
        self.PI = None  # Productivity index [kg/(s·Pa)]
        
        # Operating constraints
        self.p_wf_target = None  # Target flowing pressure [Pa]
        self.q_target = None  # Target mass rate [kg/s]
        self.h_max = None  # Max enthalpy [J/kg] (for steam quality)
    
    def compute_productivity_index(self, grid, rock, fluid):
        """
        Compute PI using Peaceman formula.
        
        PI = 2π·h·k / (μ·(ln(re/rw) + skin))
        """
        # Grid block dimensions
        dx = grid.dx if hasattr(grid, 'dx') else 100
        dy = grid.dy if hasattr(grid, 'dy') else 100
        dz = grid.dz if hasattr(grid, 'dz') else 10
        
        # Effective radius (Peaceman)
        re = 0.28 * np.sqrt(dx * dy) / (
            np.sqrt(dy/dx) ** 0.5 + np.sqrt(dx/dy) ** 0.5
        )
        
        # Permeability (assume isotropic for now)
        k = rock.permiability_m2[0] if np.isscalar(rock.permiability_m2) else np.mean(rock.permiability_m2)
        
        # Reservoir thickness
        h = dz
        
        # Fluid properties (evaluated at reservoir conditions)
        mu = fluid.mu
        
        # PI
        self.PI = 2 * np.pi * h * k / (mu * (np.log(re / self.rw) + self.skin))
        
        return self.PI
    
    def compute_rate(self, p_reservoir, p_wf=None):
        """
        Compute well mass flow rate.
        
        q = PI · (p_reservoir - p_wf)
        
        Parameters
        ----------
        p_reservoir : float
            Reservoir pressure at well block [Pa]
        p_wf : float, optional
            Flowing bottomhole pressure [Pa]
        
        Returns
        -------
        q : float
            Mass flow rate [kg/s] (positive = production)
        """
        if self.well_type == 'production':
            if p_wf is None:
                p_wf = self.p_wf_target if self.p_wf_target else p_reservoir * 0.5
            
            q = self.PI * (p_reservoir - p_wf)
            return max(0, q)  # No injection for production well
        
        else:  # injection
            if p_wf is None:
                p_wf = self.p_wf_target if self.p_wf_target else p_reservoir * 1.5
            
            q = self.PI * (p_reservoir - p_wf)
            return min(0, q)  # No production for injection well
```

---

## 4. Example: Indonesian Geothermal Field

```python
# examples/geothermal_field.py

from presto import StructuredGrid
from presto.core import FluidProperties, RockProperties
from presto.physics import GeothermalMultiphaseSolver
from presto.well import GeothermalWell

# === Create reservoir grid ===
# 2km x 2km x 500m, discretized to 20x20x10 cells
grid = StructuredGrid(nx=20, ny=20, nz=10, dx=100, dy=100, dz=50)

# === Indonesian volcanic reservoir properties ===
rock = RockProperties(
    porosity=0.12,
    permeability=150,  # md (fractured andesite)
    permeability_unit='md',
    k_ratio=(1.0, 1.0, 0.1),  # Vertical permeability lower
    lambda_rock=2.8,  # W/(m·K) (volcanic rock)
)

fluid = FluidProperties(fluid_type='geothermal')

# === Initialize reservoir ===
solver = GeothermalMultiphaseSolver(grid, rock)

# Initial conditions: 280°C, 250 bar, liquid-dominated
solver.pressure = np.full(grid.num_cells, 250e5)
solver.temperature = np.full(grid.num_cells, 553.15)  # 280°C
solver.saturation = np.full(grid.num_cells, 0.05)  # 5% initial steam

# Apply geothermal gradient
surface_temp = 298.15  # 25°C (tropical)
gradient = 0.06  # 60°C/km (high heat flow volcanic)
depth = -grid.cell_centroids[:, 2]
solver.temperature = surface_temp + gradient * depth

# === Define wells ===
# Production well at center
prod_well = GeothermalWell('PROD-01', (10, 10, 5), 'production')
prod_well.p_wf_target = 100e5  # 100 bar flowing pressure
prod_well.compute_productivity_index(grid, rock, fluid)

# Reinjection wells at corners
inj_well_1 = GeothermalWell('INJ-01', (2, 2, 5), 'injection')
inj_well_1.p_wf_target = 280e5  # 280 bar injection pressure
inj_well_1.compute_productivity_index(grid, rock, fluid)

inj_well_2 = GeothermalWell('INJ-02', (18, 18, 5), 'injection')
inj_well_2.p_wf_target = 280e5
inj_well_2.compute_productivity_index(grid, rock, fluid)

# === Simulation ===
dt = 86400  # 1 day time steps
n_steps = 3650  # 10 years

print("Starting geothermal field simulation...")
print(f"Production: {prod_well.name} at {prod_well.p_wf_target/1e5:.0f} bar")
print(f"Reinjection: {inj_well_1.name}, {inj_well_2.name}")
print()

for t in range(n_steps):
    # Compute well rates
    cell_idx = grid.get_cell_index(*prod_well.position)
    q_prod = prod_well.compute_rate(solver.pressure[cell_idx])
    
    # Apply to source terms
    source_terms = {
        'water': np.zeros(grid.num_cells),
        'steam': np.zeros(grid.num_cells),
    }
    source_terms['water'][cell_idx] = -q_prod * 0.9  # 90% liquid
    source_terms['steam'][cell_idx] = -q_prod * 0.1  # 10% steam
    
    # Reinjection (cold water at 25°C)
    inj_idx = grid.get_cell_index(*inj_well_1.position)
    q_inj = inj_well_1.compute_rate(solver.pressure[inj_idx])
    source_terms['water'][inj_idx] = -q_inj
    
    # Time step
    result = solver.step_sim(
        dt=dt,
        source_terms=source_terms,
        heat_sources=np.zeros(grid.num_cells),
        bc={},
    )
    
    # Output every year
    if t % 365 == 0:
        year = t / 365
        T_avg = solver.temperature.mean() - 273.15
        p_avg = solver.pressure.mean() / 1e5
        S_s_avg = solver.saturation.mean() * 100
        print(f"Year {year:4.0f}: T={T_avg:6.1f}°C, p={p_avg:6.1f} bar, "
              f"S_steam={S_s_avg:5.1f}%, converged={result['converged']}")

print("\nSimulation completed!")
```

---

## 5. Validation Plan

### 5.1 Analytical Solutions

1. **Theis solution** (transient radial flow)
2. **Line source with heat transport** (convection-conduction)
3. **Buckley-Leverett** (two-phase frontal advance)

### 5.2 Benchmark Cases

1. **SPE Geothermal Benchmark** (if available)
2. **TOUGH2 comparison** (same grid, same properties)
3. **Field data** (Wayang Windu, Salak, or other Indonesian fields)

### 5.3 Test Suite

```python
# tests/test_geothermal.py

def test_iapws97_saturation():
    """Test IAPWS-97 saturation properties."""
    from presto.eos.iapws97 import IAPWS97
    
    # At 1 atm, T_sat should be 100°C
    T_sat = IAPWS97.saturation_temperature(101325)
    assert abs(T_sat - 373.15) < 1.0
    
    # Latent heat at 100°C should be ~2257 kJ/kg
    h_fg = IAPWS97.latent_heat(101325)
    assert abs(h_fg - 2.257e6) < 1e5

def test_productivity_index():
    """Test Peaceman PI formula."""
    from presto import StructuredGrid
    from presto.core import RockProperties, FluidProperties
    from presto.well import GeothermalWell
    
    grid = StructuredGrid(nx=10, ny=10, nz=1, dx=100, dy=100, dz=50)
    rock = RockProperties(permeability=100, permeability_unit='md')
    fluid = FluidProperties()
    
    well = GeothermalWell('TEST', (5, 5, 0))
    PI = well.compute_productivity_index(grid, rock, fluid)
    
    # PI should be positive and reasonable
    assert PI > 0
    assert PI < 100  # kg/(s·Pa) - sanity check
```

---

## 6. Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1**: IAPWS-97 implementation | 2 weeks | Water/steam properties module |
| **Phase 2**: Multiphase solver | 4 weeks | Water-steam flow with phase change |
| **Phase 3**: Well models | 2 weeks | PI-based wells, constraints |
| **Phase 4**: Validation | 2 weeks | Test suite, benchmarks |
| **Phase 5**: Documentation | 1 week | API docs, tutorials |

**Total**: ~11 weeks for MVP geothermal extension

---

## 7. References

1. **IAPWS-97**: International Association for the Properties of Water and Steam, "Revised Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam", 2012.

2. **Peaceman, D.W.**: "Interpretation of Well-Block Pressures in Numerical Reservoir Simulation", SPE Journal, 1978.

3. **Pruess, K. et al.**: "TOUGH2 User's Guide", Lawrence Berkeley National Laboratory, 1999.

4. **Grant, M.A. & Donaldson, I.G.**: "Geothermal Reservoir Engineering", 2nd ed., Academic Press, 2015.

5. **Horne, R.N.**: "Modern Well Test Analysis", Petroway, 1995.

6. **Indonesian Geothermal Case Studies**:
   - Wayang Windu Field (Star Energy)
   - Salak Field (Chevron)
   - Ulubelu Field (PLTP)
