# GARUDA + Geothermal-Agents Integration Plan

## Overview

This document describes how to integrate the modernized **GARUDA** reservoir simulator with the existing **geothermal-agents** AI system, creating a hybrid workflow that combines:

- **Fast analytical models** (geothermal-agents) for agent decision loops
- **High-fidelity numerical simulation** (GARUDA) for validation and calibration

---

## 1. Architecture

### 1.1 Current Systems

```
┌─────────────────────────────────────────┐
│         geothermal-agents               │
│  ┌─────────────────────────────────┐    │
│  │  ReAct Agent (LLM-based)        │    │
│  │  - Fast reasoning (ms)          │    │
│  │  - Analytical physics models    │    │
│  │  - Multi-agent coordination     │    │
│  └─────────────────────────────────┘    │
│              │                          │
│              ▼                          │
│  ┌─────────────────────────────────┐    │
│  │  Analytical Reservoir Model     │    │
│  │  - Material balance             │    │
│  │  - Heat balance                 │    │
│  │  - Decline curves               │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│              GARUDA                     │
│  ┌─────────────────────────────────┐    │
│  │  TPFA Solver (Numerical)        │    │
│  │  - Full PDE solution            │    │
│  │  - 3D heterogeneous grids       │    │
│  │  - Non-isothermal multiphase    │    │
│  └─────────────────────────────────┘    │
│              │                          │
│              ▼                          │
│  ┌─────────────────────────────────┐    │
│  │  High-Fidelity Simulation       │    │
│  │  - Accurate but slow (s-min)    │    │
│  │  - Spatial resolution           │    │
│  │  - Complex physics              │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### 1.2 Integrated Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID WORKFLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐  │
│  │   LLM Agent  │─────▶│  Analytical  │─────▶│ Decision │  │
│  │  (Ollama)    │      │   Model      │      │  (fast)  │  │
│  └──────────────┘      └──────────────┘      └──────────┘  │
│         │                                        │          │
│         │                                        │          │
│         │         ┌──────────────────┐           │          │
│         └────────▶│    GARUDA        │◀──────────┘          │
│                   │    Simulator     │                      │
│                   │  (Validation)    │                      │
│                   └──────────────────┘                      │
│                            │                                │
│                            ▼                                │
│                   ┌──────────────────┐                      │
│                   │  Update Analytical│                      │
│                   │  Model Parameters │                      │
│                   └──────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Operational Modes:
1. REAL-TIME: Agent + Analytical (ms response)
2. PERIODIC: GARUDA validation (hourly/daily)
3. CALIBRATION: History matching (weekly/monthly)
```

---

## 2. Integration Points

### 2.1 Shared Data Models

```python
# presto/agents/integration.py

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np
import json

@dataclass
class ReservoirState:
    """Shared state representation between agents and GARUDA."""
    
    # Identification
    reservoir_id: str
    timestamp: str
    
    # Bulk properties (analytical model)
    pressure_avg: float      # [Pa]
    temperature_avg: float   # [K]
    steam_fraction: float    # [0-1]
    
    # Well states
    wells: Dict[str, WellState]
    
    # Performance metrics
    total_production: float  # [kg/s]
    total_injection: float   # [kg/s]
    power_thermal: float     # [MW]
    
    # Spatial fields (GARUDA output)
    pressure_field: Optional[np.ndarray] = None
    temperature_field: Optional[np.ndarray] = None
    saturation_field: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert numpy arrays to lists
        if self.pressure_field is not None:
            d['pressure_field'] = self.pressure_field.tolist()
        if self.temperature_field is not None:
            d['temperature_field'] = self.temperature_field.tolist()
        if self.saturation_field is not None:
            d['saturation_field'] = self.saturation_field.tolist()
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ReservoirState':
        """Reconstruct from dict."""
        if data.get('pressure_field') is not None:
            data['pressure_field'] = np.array(data['pressure_field'])
        if data.get('temperature_field') is not None:
            data['temperature_field'] = np.array(data['temperature_field'])
        if data.get('saturation_field') is not None:
            data['saturation_field'] = np.array(data['saturation_field'])
        return cls(**data)


@dataclass
class WellState:
    """Well operating state."""
    name: str
    type: str  # 'production' or 'injection'
    flow_rate: float  # [kg/s], negative for injection
    flowing_pressure: float  # [Pa]
    enthalpy: float  # [J/kg]
    temperature: float  # [K]
    status: str  # 'online', 'offline', 'maintenance'
```

### 2.2 Adapter Classes

```python
# presto/agents/presto_adapter.py

import numpy as np
from presto import StructuredGrid, TPFASolver
from presto.core import FluidProperties, RockProperties
from presto.physics import ThermalFlow
from presto.agents.integration import ReservoirState, WellState

class PrestoAdapter:
    """
    Adapter class to connect geothermal-agents with GARUDA simulator.
    
    Converts between:
    - Analytical model state (lumped parameters)
    - Numerical model state (spatial fields)
    """
    
    def __init__(self, config: dict):
        """
        Initialize GARUDA adapter.
        
        Parameters
        ----------
        config : dict
            Configuration including:
            - grid: Grid parameters
            - rock: Rock properties
            - fluid: Fluid properties
            - initial_conditions: Initial state
        """
        self.config = config
        
        # Initialize grid
        self.grid = StructuredGrid(
            nx=config['grid']['nx'],
            ny=config['grid']['ny'],
            nz=config['grid']['nz'],
            dx=config['grid']['dx'],
            dy=config['grid']['dy'],
            dz=config['grid']['dz'],
        )
        
        # Initialize rock
        self.rock = RockProperties(
            porosity=config['rock']['porosity'],
            permeability=config['rock']['permeability'],
            permeability_unit='md',
        )
        
        # Initialize fluid
        self.fluid = FluidProperties(fluid_type='geothermal')
        
        # Initialize thermal solver
        self.thermal = ThermalFlow(self.grid, self.rock, self.fluid)
        
        # Initialize state
        self._initialize_state(config.get('initial_conditions', {}))
    
    def _initialize_state(self, init: dict):
        """Set initial conditions."""
        # Pressure
        p_init = init.get('pressure', 250e5)
        self.thermal.pressure = np.full(self.grid.num_cells, p_init)
        
        # Temperature (with geothermal gradient)
        if 'temperature' in init:
            T_init = init['temperature']
            if isinstance(T_init, (int, float)):
                self.thermal.temperature = np.full(self.grid.num_cells, T_init)
            elif T_init.get('type') == 'gradient':
                surface_temp = T_init.get('surface', 298.15)
                gradient = T_init.get('gradient', 0.03)
                depth = -self.grid.cell_centroids[:, 2]
                self.thermal.temperature = surface_temp + gradient * depth
        
        # Store initial state for resets
        self.p_initial = self.thermal.pressure.copy()
        self.T_initial = self.thermal.temperature.copy()
    
    def state_to_analytical(self) -> dict:
        """
        Convert GARUDA spatial state to lumped analytical parameters.
        
        Returns
        -------
        params : dict
            Lumped parameters for analytical model:
            - pressure_avg, temperature_avg
            - productivity_index, decline_rate
            - heat_content, energy_production_rate
        """
        p = self.thermal.pressure
        T = self.thermal.temperature
        
        # Volume-weighted averages
        V = self.grid.cell_volumes
        p_avg = np.average(p, weights=V)
        T_avg = np.average(T, weights=V)
        
        # Compute derived parameters
        phi = self.rock.porosity if np.isscalar(self.rock.porosity) else np.mean(self.rock.porosity)
        k = np.mean(self.rock.permiability_m2)
        
        # Productivity index estimate
        h = self.grid.dz if np.isscalar(self.grid.dz) else np.mean(self.grid.dz)
        mu = self.fluid.viscosity(T_avg)
        PI_est = 2 * np.pi * h * k / (mu * 10)  # Simplified
        
        # Heat content
        rhoCp = self.rock.heat_capacity_bulk(self.fluid.cp, self.fluid.rho)
        heat_content = np.sum(V * rhoCp * (T - 298.15))  # J above 25°C
        
        return {
            'pressure_avg': p_avg,
            'temperature_avg': T_avg,
            'productivity_index': PI_est,
            'decline_rate': 0.05,  # Would compute from history
            'heat_content': heat_content,
            'reservoir_volume': np.sum(V),
            'permeability': k,
            'porosity': phi,
        }
    
    def analytical_to_state(self, params: dict):
        """
        Update GARUDA state from analytical model parameters.
        
        Used for history matching: adjust spatial fields to match
        lumped parameter behavior.
        
        Parameters
        ----------
        params : dict
            Lumped parameters from analytical model
        """
        # Simple approach: scale entire field
        p_target = params.get('pressure_avg')
        T_target = params.get('temperature_avg')
        
        if p_target is not None:
            p_current = np.average(self.thermal.pressure, weights=self.grid.cell_volumes)
            scale = p_target / p_current
            self.thermal.pressure *= scale
        
        if T_target is not None:
            T_current = np.average(self.thermal.temperature, weights=self.grid.cell_volumes)
            offset = T_target - T_current
            self.thermal.temperature += offset
    
    def run_simulation(self, wells: Dict[str, WellState], dt: float, n_steps: int) -> List[ReservoirState]:
        """
        Run GARUDA simulation with given well configuration.
        
        Parameters
        ----------
        wells : dict
            Well states (rates, pressures)
        dt : float
            Time step [s]
        n_steps : int
            Number of time steps
        
        Returns
        -------
        states : list
            Reservoir state at each time step
        """
        states = []
        
        # Convert wells to source terms
        source_terms = self._wells_to_sources(wells)
        
        for step in range(n_steps):
            # Run time step
            result = self.thermal.step_coupled(
                dt=dt,
                source_terms=source_terms,
                heat_sources=np.zeros(self.grid.num_cells),
                bc_type='neumann',  # Wells handled as sources
                bc_values=None,
                flow_solver=TPFASolver(self.grid, mu=self.fluid.mu, rho=self.fluid.rho),
            )
            
            # Extract state
            state = self._extract_state(step * dt, wells)
            states.append(state)
            
            # Check convergence
            if not result['converged']:
                print(f"Warning: Simulation did not converge at step {step}")
                break
        
        return states
    
    def _wells_to_sources(self, wells: Dict[str, WellState]) -> np.ndarray:
        """Convert well states to grid source terms."""
        sources = np.zeros(self.grid.num_cells)
        
        for well in wells.values():
            # Find nearest cell
            if hasattr(well, 'cell_index'):
                idx = well.cell_index
            else:
                # Use position if available
                idx = self.grid.num_cells // 2  # Default to center
            
            sources[idx] += well.flow_rate  # Negative for production
        
        return sources
    
    def _extract_state(self, time: float, wells: Dict[str, WellState]) -> ReservoirState:
        """Extract ReservoirState from current simulation state."""
        from datetime import datetime
        
        # Compute bulk metrics
        V = self.grid.cell_volumes
        p_avg = np.average(self.thermal.pressure, weights=V)
        T_avg = np.average(self.thermal.temperature, weights=V)
        
        # Total production
        total_prod = sum(w.flow_rate for w in wells.values() if w.flow_rate > 0)
        total_inj = sum(abs(w.flow_rate) for w in wells.values() if w.flow_rate < 0)
        
        # Thermal power
        rhoCp = self.rock.heat_capacity_bulk(self.fluid.cp, self.fluid.rho)
        power = total_prod * self.fluid.cp * (T_avg - 298.15) / 1e6  # MW
        
        return ReservoirState(
            reservoir_id=self.config.get('id', 'default'),
            timestamp=datetime.now().isoformat(),
            pressure_avg=p_avg,
            temperature_avg=T_avg,
            steam_fraction=0.0,  # Would compute from multiphase
            wells={w.name: w for w in wells.values()},
            total_production=total_prod,
            total_injection=total_inj,
            power_thermal=power,
            pressure_field=self.thermal.pressure,
            temperature_field=self.thermal.temperature,
        )
```

### 2.3 Agent Tool Integration

```python
# geothermal-agents/tools/presto_tools.py

from presto.agents.presto_adapter import PrestoAdapter
from presto.agents.integration import ReservoirState, WellState
from typing import Dict, List

class PrestoTools:
    """
    Tools for geothermal agents to interact with GARUDA simulator.
    
    These tools enable agents to:
    - Run high-fidelity simulations
    - Validate analytical model predictions
    - Perform history matching
    - Generate training data for surrogates
    """
    
    def __init__(self, adapter: PrestoAdapter):
        self.adapter = adapter
    
    def run_scenario(self, 
                     scenario_name: str,
                     wells: Dict[str, dict],
                     duration_days: float,
                     dt_days: float = 1.0) -> dict:
        """
        Run a production scenario simulation.
        
        Parameters
        ----------
        scenario_name : str
            Name for this scenario
        wells : dict
            Well configurations: {name: {type, rate, pressure}}
        duration_days : float
            Simulation duration [days]
        dt_days : float
            Time step [days]
        
        Returns
        -------
        result : dict
            Simulation results including:
            - states: Time series of reservoir states
            - metrics: Performance metrics
            - comparison: vs analytical model
        """
        # Convert well configs to WellState objects
        well_states = {}
        for name, config in wells.items():
            well_states[name] = WellState(
                name=name,
                type=config.get('type', 'production'),
                flow_rate=config.get('rate', 0),
                flowing_pressure=config.get('pressure', 0),
                enthalpy=1.2e6,  # Default
                temperature=553.15,
                status='online',
            )
        
        # Run simulation
        dt = dt_days * 86400  # Convert to seconds
        n_steps = int(duration_days / dt_days)
        
        states = self.adapter.run_simulation(well_states, dt, n_steps)
        
        # Compute metrics
        metrics = self._compute_metrics(states)
        
        # Compare with analytical
        analytical = self.adapter.state_to_analytical()
        
        return {
            'scenario_name': scenario_name,
            'states': [s.to_dict() for s in states],
            'metrics': metrics,
            'analytical_comparison': analytical,
        }
    
    def _compute_metrics(self, states: List[ReservoirState]) -> dict:
        """Compute performance metrics from simulation."""
        if not states:
            return {}
        
        # Extract time series
        pressures = [s.pressure_avg for s in states]
        temperatures = [s.temperature_avg for s in states]
        productions = [s.total_production for s in states]
        powers = [s.power_thermal for s in states]
        
        return {
            'pressure_drop': (pressures[0] - pressures[-1]) / 1e5,  # bar
            'temperature_drop': temperatures[0] - temperatures[-1],  # K
            'avg_production': sum(productions) / len(productions),
            'avg_power': sum(powers) / len(powers),
            'total_energy': sum(powers) * 24 * 3600 / len(powers),  # MWh
            'final_steam_fraction': states[-1].steam_fraction,
        }
    
    def calibrate_analytical(self, 
                             historical_data: List[dict],
                             parameters: List[str]) -> dict:
        """
        Calibrate analytical model parameters using GARUDA + history.
        
        Parameters
        ----------
        historical_data : list
            Historical production data
        parameters : list
            Parameters to calibrate: ['permeability', 'porosity', ...]
        
        Returns
        -------
        result : dict
            Calibrated parameters and fit quality
        """
        # This would use optimization (scipy.optimize) to find
        # parameters that minimize mismatch between:
        # - GARUDA simulation with those parameters
        # - Historical data
        
        from scipy.optimize import minimize
        
        def objective(param_values):
            # Update model with trial parameters
            for param, value in zip(parameters, param_values):
                if hasattr(self.adapter.rock, param):
                    setattr(self.adapter.rock, param, value)
            
            # Run simulation
            # Compare with history
            # Return mismatch
            
            return mismatch
        
        # Initial guess
        x0 = [getattr(self.adapter.rock, p) for p in parameters]
        
        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B')
        
        return {
            'calibrated_parameters': dict(zip(parameters, result.x)),
            'fit_quality': 1 - result.fun,
            'convergence': result.success,
        }
    
    def generate_training_data(self,
                               parameter_ranges: Dict[str, tuple],
                               n_samples: int) -> List[dict]:
        """
        Generate training data for ML surrogate model.
        
        Parameters
        ----------
        parameter_ranges : dict
            Parameter sampling ranges: {param: (min, max)}
        n_samples : int
            Number of samples to generate
        
        Returns
        -------
        data : list
            Training samples (inputs + outputs)
        """
        import numpy as np
        from scipy.stats import qmc
        
        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=len(parameter_ranges))
        samples = sampler.random(n=n_samples)
        
        # Scale to parameter ranges
        param_names = list(parameter_ranges.keys())
        scaled_samples = qmc.scale(samples, 
                                   [parameter_ranges[p][0] for p in param_names],
                                   [parameter_ranges[p][1] for p in param_names])
        
        training_data = []
        
        for sample in scaled_samples:
            # Set parameters
            params = dict(zip(param_names, sample))
            
            # Run simulation
            # Store input-output pair
            
            training_data.append({
                'inputs': params,
                'outputs': {},  # Would fill from simulation
            })
        
        return training_data
```

---

## 3. Usage Examples

### 3.1 Agent Workflow with GARUDA Validation

```python
# geothermal-agents/examples/presto_integration.py

from agents.react_agent import ReActAgent
from tools.geothermal_tools import GeothermalTools
from tools.presto_tools import PrestoTools
from presto.agents.presto_adapter import PrestoAdapter

# Initialize both systems
adapter = PrestoAdapter(config={
    'id': 'wayang-windu',
    'grid': {'nx': 20, 'ny': 20, 'nz': 10, 'dx': 100, 'dy': 100, 'dz': 50},
    'rock': {'porosity': 0.12, 'permeability': 150},
    'initial_conditions': {
        'pressure': 250e5,
        'temperature': {'type': 'gradient', 'surface': 298.15, 'gradient': 0.06},
    },
})

# Create tools
analytical_tools = GeothermalTools()  # Fast analytical
presto_tools = PrestoTools(adapter)   # High-fidelity

# Create agent with both tool sets
agent = ReActAgent(
    tools={**analytical_tools, **presto_tools},
    verbose=True,
)

# Agent can now choose which tool to use
result = agent.execute("""
Optimize production for Wayang Windu field.

1. First, use analytical model for quick optimization
2. Then, validate with GARUDA simulation (1 year)
3. If GARUDA shows >10% deviation, recalibrate analytical model
4. Report final recommendation
""")

print(result['final_summary'])
```

### 3.2 Periodic Validation Cron Job

```python
# ~/.hermes/cron/presto_validation.py

"""
Daily GARUDA validation job.

Runs every night to:
1. Compare analytical predictions vs GARUDA
2. Flag significant deviations
3. Update analytical parameters if needed
"""

from presto.agents.presto_adapter import PrestoAdapter
from tools.presto_tools import PrestoTools

adapter = PrestoAdapter.load('wayang-windu')
presto_tools = PrestoTools(adapter)

# Get current state from analytical model
analytical_state = adapter.state_to_analytical()

# Run GARUDA for next 30 days
result = presto_tools.run_scenario(
    scenario_name='forecast_30d',
    wells=get_current_well_config(),
    duration_days=30,
    dt_days=1,
)

# Compare
deviation = (
    abs(result['metrics']['avg_production'] - analytical_state['predicted_production'])
    / analytical_state['predicted_production']
)

if deviation > 0.1:
    # Flag for recalibration
    alert(f"Production deviation {deviation*100:.1f}% - recalibration needed")
    
    # Auto-recalibrate
    calibration = presto_tools.calibrate_analytical(
        historical_data=load_recent_history(),
        parameters=['permeability', 'productivity_index'],
    )
    
    # Update analytical model
    update_analytical_params(calibration['calibrated_parameters'])
```

---

## 4. API Reference

### 4.1 REST API (Optional)

```python
# presto/api/server.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GARUDA API")

class SimulationRequest(BaseModel):
    grid: dict
    rock: dict
    fluid: dict
    wells: dict
    duration_days: float

@app.post("/simulate")
async def run_simulation(request: SimulationRequest):
    adapter = PrestoAdapter(request.dict())
    result = adapter.run_simulation(...)
    return result

@app.get("/state/{reservoir_id}")
async def get_state(reservoir_id: str):
    return get_current_state(reservoir_id)

# Run with: uvicorn presto.api.server:app --host 0.0.0.0 --port 8000
```

---

## 5. Performance Considerations

| Operation | Analytical | GARUDA | Use Case |
|-----------|------------|--------|----------|
| Single evaluation | <1 ms | 1-60 s | Real-time vs batch |
| 1000 evaluations | <1 s | 15 min - 10 hr | Optimization |
| History matching | Seconds | Hours-days | Calibration |
| Uncertainty quantification | Minutes | Days-weeks | Risk analysis |

### Strategies:

1. **Analytical first**: Use for all real-time decisions
2. **GARUDA periodic**: Validate daily/weekly
3. **Surrogate ML**: Train neural net on GARUDA outputs for 1000x speedup
4. **Adaptive fidelity**: Use GARUDA only when analytical uncertainty is high

---

## 6. Implementation Checklist

- [ ] Create `presto/agents/integration.py` with shared data models
- [ ] Create `presto/agents/presto_adapter.py` with adapter class
- [ ] Create `geothermal-agents/tools/presto_tools.py` with agent tools
- [ ] Add import to `geothermal-agents/tools/__init__.py`
- [ ] Test end-to-end workflow
- [ ] Document in both repositories
- [ ] Create example notebook
- [ ] Add unit tests

---

## 7. Future Enhancements

1. **Neural Surrogate**: Train CNN/Transformer on GARUDA outputs
2. **Bayesian Calibration**: MCMC for uncertainty quantification
3. **Real-time Dashboard**: Streamlit/Plotly for monitoring
4. **Multi-fidelity Optimization**: Combine analytical + GARUDA in optimization loop
5. **Digital Twin**: Continuous calibration with live field data
