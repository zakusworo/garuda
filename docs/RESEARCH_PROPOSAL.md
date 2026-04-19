# Research Proposal

## AI-Enhanced Reservoir Simulation for Geothermal Energy Optimization

### Application to Indonesian Volcanic Geothermal Systems

---

**Principal Investigator:** Zulfikar Aji Kusworo  
**Affiliation:** Politeknik Energi dan Pertambangan Bandung, Indonesia  
**Email:** greataji13@gmail.com  
**GitHub:** @zakusworo  
**DOI:** 10.5281/zenodo.19653501

**Date:** April 2026

---

## Abstract

Indonesia possesses an estimated 29 GW of geothermal potential—approximately 40% of global reserves—yet only 2.3 GW has been developed. A critical bottleneck is the lack of accessible, high-fidelity reservoir simulation tools optimized for Indonesian volcanic geothermal systems. This proposal presents **GARUDA-AI**, a next-generation open-source reservoir simulator that combines physics-based numerical modeling with machine learning and AI agent systems for real-time geothermal reservoir management.

The project has three objectives: (1) modernize the deprecated GARUDA reservoir simulator with non-isothermal multiphase flow capabilities; (2) integrate AI/ML methods including neural surrogate models and Bayesian history matching; and (3) develop an agentic AI system for autonomous well control optimization. The resulting platform will enable 1000× faster simulation than traditional tools while maintaining accuracy, making it suitable for real-time decision support in geothermal operations.

This research directly supports Indonesia's target of 7.3 GW geothermal capacity by 2030 under the National Energy Policy (KEN) and contributes to SDG 7 (Affordable and Clean Energy).

**Keywords:** Geothermal energy, reservoir simulation, machine learning, AI agents, Indonesia, renewable energy, TPFA, surrogate modeling

---

## 1. Introduction

### 1.1 Background

Geothermal energy represents Indonesia's largest renewable energy resource, with an estimated potential of 29 GW spread across 336 identified sites (ESDM, 2023). The majority of these resources are associated with volcanic systems characterized by:

- High reservoir temperatures (250-350°C)
- Complex fracture networks in andesitic/dacitic rocks
- Two-phase (water-steam) flow conditions
- Significant non-condensable gas content

Despite this potential, Indonesia has developed only 2.3 GW of geothermal capacity as of 2025. Key barriers include:

1. **High exploration risk**: Drilling success rates below 60%
2. **Reservoir management challenges**: Premature reservoir decline, inadequate reinjection
3. **Limited simulation capacity**: Dependence on expensive commercial software (TOUGH2, CMG) with steep learning curves
4. **Skills gap**: Shortage of reservoir engineers trained in numerical simulation

### 1.2 Problem Statement

Current reservoir simulation tools face three fundamental limitations for Indonesian geothermal applications:

**Computational Cost:** Traditional finite-volume simulators require minutes to hours per simulation run, making them unsuitable for:
- Real-time production optimization
- Uncertainty quantification (requiring 1000+ runs)
- History matching (iterative parameter calibration)

**Accessibility Barriers:** Commercial simulators cost $50,000-200,000 in licensing fees, placing them beyond reach of:
- Small geothermal developers
- Academic institutions
- Government agencies in developing countries

**Static Models:** Conventional simulators do not learn from operational data, requiring manual recalibration that delays decision-making.

### 1.3 Proposed Solution

This research proposes **GARUDA-AI**, an open-source AI-enhanced reservoir simulation platform that addresses these limitations through:

1. **Modern Physics Engine**: Pure Python implementation with Numba JIT acceleration, eliminating compilation barriers
2. **Machine Learning Surrogates**: Neural networks trained on high-fidelity simulations for 1000× speedup
3. **Agentic AI System**: Autonomous agents for continuous reservoir monitoring and optimization
4. **Indonesian Context**: Templates and property models optimized for volcanic geothermal systems

---

## 2. Research Objectives

### 2.1 Primary Objectives

**Objective 1: Develop Non-Isothermal Multiphase Reservoir Simulator**

- Implement TPFA (Two-Point Flux Approximation) finite volume method
- Support water-steam phase change with IAPWS-97 property formulation
- Validate against analytical solutions and TOUGH2 benchmarks
- Target accuracy: <5% deviation from reference solutions

**Objective 2: Integrate Machine Learning Methods**

- Develop CNN-based permeability upscaling from fine to coarse grids
- Train neural surrogate models for rapid pressure/temperature prediction
- Implement Bayesian history matching with MCMC sampling
- Target speedup: 100-1000× vs numerical simulation

**Objective 3: Build Agentic AI System for Reservoir Management**

- Create ReAct (Reasoning + Acting) agents for production optimization
- Implement multi-agent coordination for field-scale management
- Integrate with large language models (Ollama) for natural language interface
- Target: Autonomous well control with 10% improvement in recovery factor

### 2.2 Secondary Objectives

- Create educational materials for Indonesian universities
- Establish open-source community for ongoing development
- Publish 3-5 peer-reviewed papers in Q1 journals
- Present at international geothermal conferences (IGC, GRC)

---

## 3. Methodology

### 3.1 Phase 1: Core Simulator Development (Months 1-6)

#### 3.1.1 Numerical Methods

The simulator uses the **TPFA (Two-Point Flux Approximation)** finite volume method, which offers:

- **Monotonicity**: No spurious oscillations in pressure/saturation
- **Local Conservation**: Mass balance enforced at cell level
- **Efficiency**: Sparse linear systems amenable to iterative solvers

Governing equations for non-isothermal two-phase flow:

**Mass Conservation:**
```
∂/∂t (φ Σᵅ ρᵅ Sᵅ) + ∇·(Σᵅ ρᵅ uᵅ) = Σᵅ qᵅ
```

**Energy Conservation:**
```
∂/∂t [(ρCp)_bulk T] + ∇·(Σᵅ ρᵅ hᵅ uᵅ) - ∇·(λ_eff ∇T) = Q_T
```

**Darcy's Law (multiphase):**
```
uᵅ = -(k·kᵣᵅ/μᵅ)·(∇pᵅ - ρᵅg∇z)
```

#### 3.1.2 Implementation Strategy

```python
# Core solver architecture
presto/
├── core/
│   ├── grid.py              # Structured/unstructured meshes
│   ├── tpfa_solver.py       # Flux assembly, matrix construction
│   ├── fluid_properties.py  # IAPWS-97 water/steam properties
│   └── rock_properties.py   # Heterogeneous permeability
├── physics/
│   ├── single_phase.py      # Pressure equation
│   ├── multiphase.py        # Water-steam with phase change
│   └── thermal.py           # Heat transport (conduction + convection)
├── ml/
│   ├── upscaling.py         # CNN for permeability homogenization
│   ├── surrogate.py         # Neural network emulator
│   └── history_match.py     # Bayesian parameter inversion
└── agents/
    ├── react_agent.py       # Reasoning + Acting agent
    └── multi_agent.py       # Field-scale coordination
```

#### 3.1.3 Validation Plan

| Test Case | Analytical/Numerical | Acceptance Criteria |
|-----------|---------------------|---------------------|
| 1D linear flow | Theis solution | <2% pressure error |
| Radial flow | Peaceman well model | <5% PI error |
| Heat conduction | Line source solution | <3% temperature error |
| Buckley-Leverett | Frontal advance theory | <5% saturation error |
| SPE Benchmark | TOUGH2 reference | <10% all variables |

### 3.2 Phase 2: Machine Learning Integration (Months 7-12)

#### 3.2.1 Permeability Upscaling with CNN

Fine-grid heterogeneity cannot be resolved in field-scale simulations. Traditional upscaling methods (arithmetic, harmonic, geometric means) fail to capture connectivity effects.

**Proposed Approach:**

```
Input: Fine-grid permeability field (64×64×64)
       → 3D Convolutional Layers (ResNet architecture)
       → Global Average Pooling
       → Dense Layers
Output: Upscaled permeability tensor (coarse grid)
```

Training data: 10,000 synthetic permeability fields with known effective permeability (computed from fine-grid flow simulations).

**Expected Performance:**
- MAE < 10% vs fine-grid reference
- Inference time: <100 ms per upscaling operation

#### 3.2.2 Neural Surrogate Models

For optimization and uncertainty quantification, we replace the numerical solver with a neural network emulator.

**Architecture:**

```
Input: [permeability, porosity, well_rates, time]
       → Transformer Encoder (temporal attention)
       → MLP Head
Output: [pressure_field, temperature_field, production_rate]
```

**Training Strategy:**
1. Generate 100,000 simulation samples using Latin Hypercube Sampling
2. Train surrogate on 80% (validation: 10%, test: 10%)
3. Use physics-informed loss: add PDE residual as regularization

**Expected Performance:**
- Inference time: <10 ms (vs 1-60 s for numerical)
- Accuracy: <5% error on test set

#### 3.2.3 Bayesian History Matching

Calibrating reservoir models to production data is an inverse problem with non-unique solutions.

**Method:** Markov Chain Monte Carlo (MCMC) with surrogate acceleration

```
Algorithm: Adaptive Metropolis-Hastings
1. Propose new parameters θ' from proposal distribution
2. Evaluate likelihood: L = exp(-||data - model(θ')||² / 2σ²)
3. Accept/reject based on Metropolis criterion
4. Adapt proposal covariance based on acceptance rate

Acceleration: Use neural surrogate for 99% of proposals,
              validate with numerical solver every 100 steps
```

**Output:** Posterior distribution of reservoir parameters with uncertainty quantification.

### 3.3 Phase 3: Agentic AI System (Months 13-18)

#### 3.3.1 ReAct Agent Architecture

The agent uses the **ReAct (Reasoning + Acting)** pattern for reservoir management:

```
Loop:
  1. OBSERVE: Current reservoir state (pressure, temperature, rates)
  2. REASON: LLM generates thought process
  3. ACT: Call tools (simulate, optimize, query database)
  4. REFLECT: Evaluate outcome, update strategy
```

**Example Interaction:**

```
[OBSERVATION] PROD-01: p=180 bar, T=260°C, q=45 kg/s
              INJ-01: p=290 bar, T=120°C, q=-40 kg/s
              Reservoir avg: p=245 bar (declining 2 bar/month)

[THOUGHT] Pressure decline exceeds target (1 bar/month).
          Need to increase reinjection or reduce production.
          Check if INJ-01 can accept more flow.

[ACTION] simulate_scenario(wells={'INJ-01': {'rate': -55}})

[OBSERVATION] Simulation: p_stable at 240 bar after 6 months

[THOUGHT] Increasing injection to 55 kg/s stabilizes pressure.
          Recommend operational change.

[ACTION] submit_recommendation("Increase INJ-01 to 55 kg/s")
```

#### 3.3.2 Multi-Agent Coordination

For field-scale management, multiple specialized agents coordinate:

```
┌─────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                     │
│  - Field-level optimization                             │
│  - Conflict resolution                                  │
│  - Economic evaluation                                  │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ RESERVOIR     │ │ PRODUCTION    │ │ MAINTENANCE   │
│ AGENT         │ │ AGENT         │ │ AGENT         │
│ - Pressure    │ │ - Well rates  │ │ - Equipment   │
│ - Temperature │ │ - Power gen   │ │ - Scheduling  │
│ - Reinjection │ │ - Constraints │ │ - Predictions │
└───────────────┘ └───────────────┘ └───────────────┘
```

#### 3.3.3 Integration with Ollama LLM

The system uses local LLM inference via Ollama for:
- Data privacy (no cloud API calls)
- Low-latency responses (<1 s)
- Customization for geothermal domain

**Supported Models:**
- Llama 3.1 8B (default, balanced)
- Mistral 7B (fast, lower quality)
- Llama 3.1 70B (high quality, slower)

---

## 4. Expected Outcomes and Deliverables

### 4.1 Software Deliverables

| Component | Description | Target Users |
|-----------|-------------|--------------|
| GARUDA Core | Open-source reservoir simulator | Researchers, engineers |
| GARUDA-ML | Machine learning extensions | Data scientists |
| GARUDA-Agents | AI agent system | Operators, managers |
| Documentation | API docs, tutorials, examples | All users |

### 4.2 Scientific Outputs

**Publications (Target: 3-5 papers):**

1. "GARUDA: A Modern Open-Source Reservoir Simulator for Geothermal Applications" (Computers & Geosciences)
2. "Neural Surrogate Models for Real-Time Geothermal Reservoir Simulation" (Geothermics)
3. "Agentic AI for Autonomous Geothermal Production Optimization" (Applied Energy)
4. "Bayesian History Matching of Indonesian Geothermal Fields Using MCMC" (Renewable Energy)
5. "Machine Learning Permeability Upscaling for Heterogeneous Volcanic Reservoirs" (Water Resources Research)

**Conference Presentations:**
- World Geothermal Congress 2026
- GRC Annual Meeting 2026
- NeurIPS AI for Science Workshop 2026

### 4.3 Capacity Building

- **Workshop Series**: Train 100+ Indonesian engineers/students
- **University Partnerships**: Integrate GARUDA into curriculum at ITB, UGM, UI
- **Open-Source Community**: Establish contributor base for ongoing development

### 4.4 Impact Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Simulation speedup | 100-1000× | Benchmark vs TOUGH2 |
| Recovery factor improvement | 10% | Field case studies |
| Cost reduction | 90% | vs commercial software |
| Users trained | 500+ | Workshop attendance |
| GitHub stars | 1000+ | Repository metrics |
| Citations | 50+ | Google Scholar |

---

## 5. Timeline and Milestones

```
2026 Q2 (Apr-Jun)          2026 Q3 (Jul-Sep)          2026 Q4 (Oct-Dec)
│                          │                          │
├─ M1: Core TPFA solver    ├─ M4: Multiphase flow     ├─ M7: CNN upscaling
├─ M2: Thermal module      ├─ M5: IAPWS-97 props      ├─ M8: Surrogate model
├─ M3: 1D/2D validation    ├─ M6: Well models         ├─ M9: History matching
│                          │                          │
2027 Q1 (Jan-Mar)          2027 Q2 (Apr-Jun)          2027 Q3 (Jul-Sep)
│                          │                          │
├─ M10: ReAct agent        ├─ M13: Multi-agent        ├─ M16: Field validation
├─ M11: Ollama integration ├─ M14: Dashboard          ├─ M17: Documentation
├─ M12: Agent tools        ├─ M15: User workshop      ├─ M18: Final release
```

**Key Milestones:**
- M3 (Jun 2026): Single-phase thermal solver complete
- M6 (Sep 2026): Two-phase flow with phase change
- M9 (Dec 2026): ML surrogates operational
- M12 (Mar 2027): Agent system functional
- M15 (Jun 2027): First user workshop
- M18 (Sep 2027): Production release v1.0

---

## 6. Budget and Resources

### 6.1 Personnel

| Role | FTE | Duration | Cost (USD) |
|------|-----|----------|------------|
| Principal Investigator | 0.2 | 18 months | 30,000 |
| PhD Student | 1.0 | 18 months | 45,000 |
| Software Engineer | 0.5 | 12 months | 35,000 |
| **Total** | | | **110,000** |

### 6.2 Computational Resources

| Item | Specification | Cost (USD) |
|------|---------------|------------|
| GPU Server | RTX 4090, 64GB RAM | 5,000 |
| Cloud Credits | AWS/Azure for large runs | 3,000 |
| **Total** | | **8,000** |

### 6.3 Other Costs

| Item | Cost (USD) |
|------|------------|
| Conference travel (2 trips) | 6,000 |
| Workshop organization | 5,000 |
| Publication fees (OA) | 4,000 |
| **Total** | **15,000** |

**Grand Total: 133,000 USD**

### 6.4 Funding Sources

Potential funding agencies:
- **LPDP** (Indonesia Endowment Fund for Education)
- **BRIN** (National Research and Innovation Agency)
- **ASEAN Centre for Energy**
- **World Bank Geothermal Development Facility**
- **IRENA** (International Renewable Energy Agency)

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Technical: Solver convergence issues | Medium | High | Start with implicit formulations; use robust linear solvers (PETSc) |
| Technical: ML model accuracy insufficient | Medium | Medium | Hybrid approach: surrogate + numerical fallback |
| Personnel: Student turnover | Low | Medium | Document thoroughly; cross-train team members |
| External: Field data access delayed | Medium | Low | Use synthetic data for development; partner with industry early |
| External: Competition from commercial tools | Low | Low | Focus on open-source advantage; target underserved markets |

---

## 8. Ethical Considerations

### 8.1 Open-Source Licensing

- Software released under **MIT License** (permissive, commercial-friendly)
- Documentation under **CC-BY 4.0** (attribution required)
- Training data under **ODC-BY** (Open Data Commons)

### 8.2 Dual-Use Considerations

While designed for geothermal energy, the technology could potentially be adapted for:
- Carbon sequestration monitoring (positive dual-use)
- Groundwater extraction optimization (requires responsible use guidelines)

Mitigation: Include ethical use guidelines in documentation; restrict access to sensitive features if needed.

### 8.3 Environmental Impact

Positive environmental impact through:
- Enabling faster geothermal development (low-carbon baseload power)
- Reducing need for physical drilling through better simulation
- Open access reduces duplication of effort globally

---

## 9. Conclusion

This proposal presents a transformative approach to geothermal reservoir simulation that combines physics-based modeling with modern AI/ML techniques. By developing an open-source platform optimized for Indonesian volcanic systems, this research addresses critical barriers to geothermal development while building local capacity in computational geoscience.

The expected outcomes—100-1000× faster simulation, 10% improvement in recovery factors, and 90% cost reduction vs commercial tools—have the potential to accelerate Indonesia's geothermal development and serve as a model for other geothermal-rich developing countries.

With 18 months of focused development and strategic partnerships with industry and government, GARUDA-AI can become the standard tool for geothermal reservoir management in Indonesia and beyond.

---

## 10. References

1. ESDM (2023). "Handbook of Geothermal Energy Statistics 2023". Ministry of Energy and Mineral Resources, Republic of Indonesia.

2. IRENA (2023). "Renewable Energy Market Analysis: Southeast Asia". International Renewable Energy Agency.

3. Pruess, K., Oldenburg, C., & Moridis, G. (1999). "TOUGH2 User's Guide, Version 2". Lawrence Berkeley National Laboratory.

4. Aarnes, J. E., Kippe, V., & Lie, K. A. (2005). "Modeling of flow in porous media using multiscale methods". Computers & Geosciences, 31(8), 953-962.

5. Lie, K. A. (2019). "An Introduction to Reservoir Simulation Using MATLAB/GNU Octave: User Guide for the MATLAB Reservoir Simulation Toolbox (MRST)". Cambridge University Press.

6. Zhu, Y., et al. (2021). "Physics-informed neural networks for solving forward and inverse flow problems in porous media". Advances in Water Resources, 158, 104058.

7. Yao, J., et al. (2022). "Machine learning for subsurface characterization: A review". Journal of Petroleum Science and Engineering, 208, 109268.

8. Grant, M. A., & Donaldson, I. G. (2015). "Geothermal Reservoir Engineering" (2nd ed.). Academic Press.

9. Horne, R. N. (2020). "Machine Learning Applied to Geothermal Reservoir Management". Proceedings, 45th Stanford Geothermal Workshop.

10. Kusworo, Z. A. (2026). "Geothermal Reservoir Simulation Agents". GitHub Repository. https://github.com/zakusworo/geothermal-agents

11. PADMEC Reservoir (2019). "GARUDA: The Python Reservoir Simulation Toolbox". GitHub Repository. https://github.com/padmec-reservoir/GARUDA

12. IAPWS (2012). "Revised Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam". International Association for the Properties of Water and Steam.

---

## Appendix A: Preliminary Results

### A.1 Existing geothermal-agents System

The PI has already developed a functional AI agent system for geothermal reservoir management (Kusworo, 2026) with:

- Analytical physics models (material balance, heat balance)
- ReAct agent with tool-calling capability
- Multi-agent coordination framework
- Integration with Ollama LLM

**Performance:**
- Decision latency: <100 ms
- Accuracy: Within 15% of numerical simulators for simple cases
- Limitation: Cannot capture spatial heterogeneity

GARUDA-AI will extend this foundation with high-fidelity numerical simulation.

### A.2 Code Repository Structure

Initial development has created the following structure:

```
garuda/
├── presto/
│   ├── core/
│   │   ├── grid.py (9.7 KB)
│   │   ├── tpfa_solver.py (17 KB)
│   │   ├── fluid_properties.py (5.7 KB)
│   │   └── rock_properties.py (8.8 KB)
│   └── physics/
│       ├── single_phase.py (4.5 KB)
│       └── thermal.py (12 KB)
├── examples/
│   └── example_1d_single_phase.py (5.3 KB)
├── docs/
│   ├── GEOTHERMAL_DESIGN.md (21 KB)
│   └── INTEGRATION_PLAN.md (28 KB)
├── pyproject.toml (3.1 KB)
└── README.md (7.7 KB)
```

Total: ~120 KB of production-ready code with documentation.

---

## Appendix B: Letters of Support

*(To be obtained from:)*
- Politeknik Energi dan Pertambangan Bandung
- PT Pertamina Geothermal Energy
- Indonesian Geothermal Association (Asosiasi Panas Bumi Indonesia)
- University partners (ITB, UGM)

---

**End of Proposal**
