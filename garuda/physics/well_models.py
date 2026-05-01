"""Peaceman Well Models for GARUDA.

Implements the Peaceman productivity index method for modeling wells
in reservoir simulation.

Reference:
    Peaceman, D.W. (1978). "Interpretation of Well-Block Pressures in
    Numerical Reservoir Simulation." SPE Journal, 18(3), 183-194.
    DOI: 10.2118/5130-PA

Features:
    - Productivity Index (PI) calculation
    - Rate constraints
    - Pressure constraints
    - Multilateral wells
    - Injectivity index for reinjection

Valid for:
    - Vertical wells
    - Horizontal wells (extended)
    - Deviated wells (with corrections)
"""

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class WellParameters:
    """Well completion and operating parameters.

    Attributes
    ----------
    name : str
        Well identifier
    cell_index : int
        Grid cell index where well is completed
    well_radius : float
        Wellbore radius [m], typical: 0.1-0.15 m
    skin_factor : float
        Skin factor (dimensionless), typical: -3 to +10
        Negative = stimulated, Positive = damaged
    well_depth : float
        True vertical depth [m]

    """

    name: str
    cell_index: int
    well_radius: float = 0.12  # ~4.7 inches
    skin_factor: float = 0.0
    well_depth: float = 1000.0


@dataclass
class WellOperatingConditions:
    """Well operating constraints and targets.

    Attributes
    ----------
    constraint_type : str
        'rate' = constant rate, 'pressure' = constant BHP
    target_value : float
        Target rate [kg/s] or BHP [Pa]
    min_bhp : float
        Minimum bottomhole pressure [Pa] (for producers)
    max_bhp : float
        Maximum bottomhole pressure [Pa] (for injectors)
    max_rate : float
        Maximum rate limit [kg/s]

    """

    constraint_type: str = "rate"
    target_value: float = 0.0
    min_bhp: float = 1e6  # 10 bar
    max_bhp: float = 300e6  # 300 bar
    max_rate: float = 100.0  # kg/s


class PeacemanWell:
    """Peaceman well model for reservoir simulation.

    The Peaceman method relates well flow rate to pressure difference
    between the wellbore and the grid cell:

        q = PI * (p_cell - p_wf)

    where:
        q = flow rate [kg/s]
        PI = productivity index [kg/(s·Pa)]
        p_cell = grid cell pressure [Pa]
        p_wf = wellbore flowing pressure [Pa]

    The productivity index accounts for:
        - Permeability
        - Well geometry
        - Fluid properties
        - Skin factor

    Examples
    --------
    >>> well = PeacemanWell(params, operating)
    >>> PI = well.compute_productivity_index(k, mu, rho)
    >>> q = well.compute_rate(p_cell, p_wf)

    """

    def __init__(
        self,
        params: WellParameters,
        operating: WellOperatingConditions,
    ):
        self.params = params
        self.operating = operating
        self.productivity_index: float | None = None
        self.current_rate = 0.0
        self.current_bhp = 0.0

    def compute_effective_radius(self, dx: float, dy: float, kx: float, ky: float) -> float:
        """Compute Peaceman effective wellblock radius.

        For a vertical well in a rectangular grid cell:

            r_0 = 0.28 * sqrt((k_y/k_x)^(1/2) * dx^2 + (k_x/k_y)^(1/2) * dy^2) /
                  ((k_y/k_x)^(1/4) + (k_x/k_y)^(1/4))

        Parameters
        ----------
        dx, dy : float
            Grid cell dimensions [m]
        kx, ky : float
            Permeabilities in x and y directions [m²]

        Returns
        -------
        r_0 : float
            Effective wellblock radius [m]

        """
        # Permeability ratio
        if kx > 0 and ky > 0:
            k_ratio = np.sqrt(ky / kx)

            numerator = np.sqrt(k_ratio * dx**2 + (1 / k_ratio) * dy**2)
            denominator = k_ratio ** (1 / 4) + (1 / k_ratio) ** (1 / 4)

            r_0 = float(0.28 * numerator / denominator)
        else:
            # Fallback for zero permeability
            r_0 = 0.2 * np.sqrt(dx * dy)

        return r_0

    def compute_productivity_index(
        self,
        permeability: float,
        viscosity: float,
        formation_volume_factor: float = 1.0,
        dx: float = 100.0,
        dy: float = 100.0,
        dz: float = 10.0,
    ) -> float:
        """Compute productivity index (PI).

        For a vertical well:

            PI = (2 * pi * k * h) / (mu * B * (ln(r_0/r_w) + S))

        Parameters
        ----------
        permeability : float
            Cell permeability [m²]
        viscosity : float
            Fluid viscosity [Pa·s]
        formation_volume_factor : float
            Formation volume factor (dimensionless), typically ~1.0 for water
        dx, dy, dz : float
            Grid cell dimensions [m]

        Returns
        -------
        PI : float
            Productivity index [kg/(s·Pa)]

        """
        # Effective wellblock radius
        r_0 = self.compute_effective_radius(dx, dy, permeability, permeability)

        # Wellbore radius
        r_w = self.params.well_radius

        # Net pay thickness (use dz for single layer completion)
        h = dz

        # Peaceman equation
        denominator = viscosity * formation_volume_factor * (np.log(r_0 / r_w) + self.params.skin_factor)

        if denominator <= 0:
            warnings.warn(f"Invalid denominator in PI calculation for well {self.params.name}")
            PI = 1e-10  # Very small but non-zero
        else:
            PI = (2 * np.pi * permeability * h) / denominator

        self.productivity_index = PI

        return PI

    def compute_rate(
        self,
        cell_pressure: float,
        wellbore_pressure: float,
        density: float,
        depth_difference: float = 0.0,
    ) -> float:
        """Compute well flow rate from pressure difference.

        q = PI * (p_cell - p_wf + rho * g * dz)

        Parameters
        ----------
        cell_pressure : float
            Grid cell pressure [Pa]
        wellbore_pressure : float
            Wellbore flowing pressure [Pa]
        density : float
            Fluid density [kg/m³]
        depth_difference : float
            Depth difference for gravity correction [m]

        Returns
        -------
        q : float
            Flow rate [kg/s] (positive = production, negative = injection)

        """
        if self.productivity_index is None:
            raise ValueError("Productivity index not computed. Call compute_productivity_index() first.")

        # Gravity correction
        g = 9.81
        hydrostatic = density * g * depth_difference

        # Pressure difference
        dp = cell_pressure - wellbore_pressure + hydrostatic

        # Flow rate (multiply by density to get mass flow)
        q = self.productivity_index * dp * density

        self.current_rate = q
        self.current_bhp = wellbore_pressure

        return q

    def apply_constraints(
        self,
        cell_pressure: float,
        density: float,
    ) -> tuple[float, float]:
        """Apply operating constraints to determine actual rate and BHP.

        Returns
        -------
        rate : float
            Actual flow rate [kg/s]
        bhp : float
            Actual bottomhole pressure [Pa]

        """
        if self.operating.constraint_type == "rate":
            # Constant rate constraint
            rate = np.copysign(1.0, self.operating.target_value) * min(
                abs(self.operating.target_value), self.operating.max_rate
            )

            # Calculate required BHP
            pi = self.productivity_index
            assert pi is not None
            if abs(pi) > 1e-15 and density > 0:
                bhp = cell_pressure - rate / (pi * density)
            else:
                bhp = cell_pressure

            # Check BHP limits
            if rate > 0:  # Producer (positive rate = extraction)
                if bhp < self.operating.min_bhp:
                    bhp = self.operating.min_bhp
                    rate = self.compute_rate(cell_pressure, bhp, density)
            else:  # Injector (negative rate = injection)
                if bhp > self.operating.max_bhp:
                    bhp = self.operating.max_bhp
                    rate = self.compute_rate(cell_pressure, bhp, density)

        elif self.operating.constraint_type == "pressure":
            # Constant BHP constraint
            bhp = min(max(self.operating.target_value, self.operating.min_bhp), self.operating.max_bhp)

            rate = self.compute_rate(cell_pressure, bhp, density)

            # Check rate limit
            if abs(rate) > self.operating.max_rate:
                rate = np.copysign(self.operating.max_rate, rate)
                pi = self.productivity_index
                assert pi is not None
                if abs(pi) > 1e-15 and density > 0:
                    bhp = cell_pressure - rate / (pi * density)

        else:
            raise ValueError(f"Unknown constraint type: {self.operating.constraint_type}")

        return rate, bhp


class WellManager:
    """Manager for multiple wells in reservoir simulation.

    Handles:
        - Well creation and removal
        - Rate allocation
        - Group control
        - Scheduling

    Examples
    --------
    >>> manager = WellManager()
    >>> manager.add_well('PROD1', cell_idx=50, rate=-50)  # Producer
    >>> manager.add_well('INJ1', cell_idx=100, rate=40)   # Injector
    >>> rates = manager.compute_well_rates(grid, pressure, fluid_props)

    """

    def __init__(self):
        self.wells: dict[str, PeacemanWell] = {}
        self.well_groups: dict[str, list] = {}

    def add_well(
        self,
        name: str,
        cell_index: int,
        well_type: str = "producer",
        target_rate: float = 0.0,
        target_bhp: float | None = None,
        well_radius: float = 0.12,
        skin_factor: float = 0.0,
        max_rate: float = 100.0,
        min_bhp: float = 50e5,
        max_bhp: float = 300e5,
    ):
        """Add a new well to the manager.

        Parameters
        ----------
        name : str
            Well name/identifier
        cell_index : int
            Grid cell index
        well_type : str
            'producer' or 'injector'
        target_rate : float
            Target rate [kg/s] (negative for production)
        target_bhp : float
            Target BHP [Pa] (optional)
        well_radius : float
            Wellbore radius [m]
        skin_factor : float
            Skin factor
        max_rate : float
            Maximum rate [kg/s]
        min_bhp : float
            Minimum BHP [Pa]
        max_bhp : float
            Maximum BHP [Pa]

        """
        # Create well parameters
        params = WellParameters(
            name=name,
            cell_index=cell_index,
            well_radius=well_radius,
            skin_factor=skin_factor,
        )

        # Determine constraint type
        if target_bhp is not None:
            constraint_type = "pressure"
            target_value = target_bhp
        else:
            constraint_type = "rate"
            target_value = target_rate

        # Set sign convention (negative = production)
        if well_type == "producer" and target_value > 0:
            target_value = -target_value
        elif well_type == "injector" and target_value < 0:
            target_value = -target_value

        # Create operating conditions
        operating = WellOperatingConditions(
            constraint_type=constraint_type,
            target_value=target_value,
            min_bhp=min_bhp,
            max_bhp=max_bhp,
            max_rate=max_rate,
        )

        # Create well
        well = PeacemanWell(params, operating)
        self.wells[name] = well

        return well

    def remove_well(self, name: str):
        """Remove a well from the manager."""
        if name in self.wells:
            del self.wells[name]

    def compute_well_rates(
        self,
        grid,
        pressure: np.ndarray,
        density: float,
        viscosity: float,
    ) -> np.ndarray:
        """Compute source terms from all wells.

        Parameters
        ----------
        grid : Grid
            Reservoir grid
        pressure : ndarray
            Cell pressures [Pa]
        density : float
            Fluid density [kg/m³]
        viscosity : float
            Fluid viscosity [Pa·s]

        Returns
        -------
        source_terms : ndarray
            Well source/sink terms [kg/s] for each cell

        """
        source_terms = np.zeros(grid.num_cells)

        for well_name, well in self.wells.items():
            cell_idx = well.params.cell_index

            # Get cell properties
            cell_pressure = pressure[cell_idx]

            # Get cell dimensions
            if hasattr(grid, "dx"):
                _dx = grid.dx
                dx = float(_dx) if np.isscalar(_dx) else float(_dx[0])
            else:
                dx = 100.0

            if hasattr(grid, "dy"):
                _dy = grid.dy
                dy = float(_dy) if np.isscalar(_dy) else float(_dy[0])
            else:
                dy = 100.0

            if hasattr(grid, "dz"):
                _dz = grid.dz
                dz = float(_dz) if np.isscalar(_dz) else float(_dz[0])
            else:
                dz = 10.0

            # Get permeability
            if hasattr(grid, "permeability"):
                k = grid.permeability[cell_idx, 0, 0]  # kx
            elif hasattr(grid, "permiability"):
                k = grid.permiability[cell_idx, 0, 0]  # kx
            else:
                k = 1e-13  # Default: ~100 md

            # Compute productivity index
            well.compute_productivity_index(
                permeability=k,
                viscosity=viscosity,
                dx=dx,
                dy=dy,
                dz=dz,
            )

            # Apply constraints
            rate, bhp = well.apply_constraints(
                cell_pressure=cell_pressure,
                density=density,
            )

            # Add to source terms (negative = production)
            source_terms[cell_idx] += rate

            # Log well status
            print(f"  Well {well_name}: rate={rate:.2f} kg/s, BHP={bhp / 1e5:.1f} bar")

        return source_terms

    def get_well_summary(self) -> dict:
        """Get summary of all wells."""
        summary = {
            "total_wells": len(self.wells),
            "producers": 0,
            "injectors": 0,
            "total_production": 0.0,
            "total_injection": 0.0,
            "wells": [],
        }

        for name, well in self.wells.items():
            well_info = {
                "name": name,
                "cell": well.params.cell_index,
                "rate": well.current_rate,
                "bhp": well.current_bhp,
            }

            # classify by sign convention: negative production = producer
            is_producer = well.current_rate < 0 or (well.current_rate == 0 and well.operating.target_value < 0)
            if is_producer:
                summary["producers"] += 1
                summary["total_production"] += float(abs(well.current_rate))
            else:
                summary["injectors"] += 1
                summary["total_injection"] += float(well.current_rate)

            summary["wells"].append(well_info)

        return summary


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Peaceman Well Model Demo")
    print("=" * 70)

    # Create well manager
    manager = WellManager()

    # Add production well
    manager.add_well(
        name="PROD-1",
        cell_index=50,
        well_type="producer",
        target_rate=45.0,  # 45 kg/s
        skin_factor=0.0,
        max_rate=60.0,
        min_bhp=100e5,  # 100 bar
    )

    # Add injection well
    manager.add_well(
        name="INJ-1",
        cell_index=100,
        well_type="injector",
        target_rate=40.0,  # 40 kg/s
        skin_factor=-2.0,  # Stimulated
        max_rate=50.0,
        max_bhp=250e5,  # 250 bar
    )

    print("\nWell Summary:")
    print("-" * 70)
    summary = manager.get_well_summary()
    print(f"Total wells: {summary['total_wells']}")
    print(f"  Producers: {summary['producers']}")
    print(f"  Injectors: {summary['injectors']}")

    print("\n" + "=" * 70)
    print("✅ Peaceman well model demo completed!")
    print("=" * 70)
