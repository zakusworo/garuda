"""Source network module for GARUDA.

Manages reservoir source/sink terms, well groups, surface separators,
and reinjection networks for coupled geothermal and petroleum simulation.

Design
------
* ``SourceNode``    – point source/sink attached to a grid cell.
* ``Separator``     – surface facility that splits produced fluids into phases.
* ``Reinjector``    – injection well fed by a separator stream.
* ``SourceGroup``   – collection of ``SourceNode`` objects with group-level constraints.
* ``SourceNetwork`` – top-level container that wires groups, separators, and reinjectors
together and computes net source terms for the reservoir simulator.

The module is written to match the style of ``garuda.physics.well_models`` so that
a ``WellManager`` can be synchronised into a ``SourceNetwork`` seamlessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Import existing well infrastructure for interoperability
from garuda.physics.well_models import WellManager


# ---------------------------------------------------------------------------
# SourceNode
# ---------------------------------------------------------------------------


@dataclass
class SourceNode:
    """A point source or sink term in the reservoir grid.

    Parameters
    ----------
    name : str
        Unique identifier.
    cell_index : int
        Grid cell where the source/sink is applied.
    rate : float
        Mass flow rate [kg/s].  Positive = injection, negative = production.
    phase : str
        Dominant fluid phase: ``'water'``, ``'steam'``, ``'oil'``, ``'gas'``,
        or ``'total'``.
    enthalpy : float or None
        Specific enthalpy [J/kg] of the injected / produced fluid.
    active : bool
        Whether the source is currently active.

    """

    name: str
    cell_index: int
    rate: float = 0.0
    phase: str = "total"
    enthalpy: float | None = None
    active: bool = True


# ---------------------------------------------------------------------------
# Separator
# ---------------------------------------------------------------------------


class Separator:
    """Surface separator that splits produced fluid into output streams.

    Typically sits between production well groups and surface discharge /
    reinjection.  The ``separation_curve`` defines how an incoming mass
    flow is partitioned by phase.

    Parameters
    ----------
    name : str
        Separator identifier.
    separation_curve : dict[str, float] or None
        Mapping ``{phase: mass_fraction}``.  Fractions are normalised
        internally so they need not sum to 1.0.  If *None* the separator
        acts as a pass-through.

    Attributes
    ----------
    incoming_rate : float
        Last mass rate entering the separator [kg/s].
    outgoing_rates : dict[str, float]
        Mass rate per output stream [kg/s] after the last ``separate()`` call.

    """

    def __init__(
        self,
        name: str,
        separation_curve: dict[str, float] | None = None,
    ):
        self.name = name
        self.separation_curve: dict[str, float] = separation_curve or {}
        self.incoming_rate: float = 0.0
        self.outgoing_rates: dict[str, float] = {}

    # ---------------------------------------------------------------

    def separate(self, incoming_rate: float, incoming_enthalpy: float | None = None) -> dict[str, float]:
        """Split incoming mass flow into output phases.

        Parameters
        ----------
        incoming_rate : float
            Total mass flow entering the separator [kg/s].
        incoming_enthalpy : float, optional
            Enthalpy of the incoming stream [J/kg] (stored for later
            inspection but not used in the default partition).

        Returns
        -------
        outgoing : dict[str, float]
            Mass flow per phase / stream [kg/s].

        """
        self.incoming_rate = incoming_rate

        if not self.separation_curve or incoming_rate <= 0.0:
            self.outgoing_rates = {"total": incoming_rate}
            return self.outgoing_rates

        total_fraction = sum(self.separation_curve.values())
        if total_fraction <= 0.0:
            self.outgoing_rates = {"total": incoming_rate}
            return self.outgoing_rates

        self.outgoing_rates = {
            phase: incoming_rate * (fraction / total_fraction) for phase, fraction in self.separation_curve.items()
        }
        return self.outgoing_rates

    def get_stream(self, phase: str) -> float:
        """Return the mass rate of a named output stream.

        Parameters
        ----------
        phase : str
            Name of the output stream (e.g. ``'water'``).

        Returns
        -------
        rate : float
            Mass flow rate [kg/s].  Zero if the stream does not exist.

        """
        return self.outgoing_rates.get(phase, 0.0)


# ---------------------------------------------------------------------------
# Reinjector
# ---------------------------------------------------------------------------


class Reinjector:
    """Reinjection well that takes a separator stream and injects into the grid.

    Attributes
    ----------
    name : str
        Reinjector identifier.
    cell_index : int
        Grid cell for injection.
    target_rate : float
        Desired injection rate [kg/s] (positive = injection).
    inlet_stream : str
        Name of the separator output stream that feeds this reinjector.
    injection_temperature : float
        Temperature of the injected fluid [K].
    active : bool
        Whether the reinjector is currently active.
    current_rate : float
        Actual injection rate after constraints [kg/s].

    """

    def __init__(
        self,
        name: str,
        cell_index: int,
        target_rate: float = 0.0,
        inlet_stream: str = "",
        injection_temperature: float = 298.15,
    ):
        self.name = name
        self.cell_index = cell_index
        self.target_rate = target_rate
        self.inlet_stream = inlet_stream
        self.injection_temperature = injection_temperature
        self.active = True
        self.current_rate = 0.0

    # ---------------------------------------------------------------

    def compute_rate(self, available_mass: float) -> float:
        """Determine the actual injection rate given available mass.

        Parameters
        ----------
        available_mass : float
            Mass available from upstream [kg/s].

        Returns
        -------
        rate : float
            Actual injection rate [kg/s].  Cannot exceed *available_mass*.

        """
        if not self.active:
            self.current_rate = 0.0
            return 0.0

        rate = min(self.target_rate, max(0.0, available_mass))
        self.current_rate = rate
        return rate

    def injection_enthalpy(self) -> float:
        """Approximate enthalpy of cold-water injection.

        Uses a simple correlation for liquid water:

        .. math::
            h \\approx c_p \\cdot (T_{inj} - 273.15)

        Returns
        -------
        h : float
            Specific enthalpy [J/kg].

        """
        cp_water = 4182.0  # J/(kg·K)
        return cp_water * (self.injection_temperature - 273.15)


# ---------------------------------------------------------------------------
# SourceGroup
# ---------------------------------------------------------------------------


class SourceGroup:
    """Collection of ``SourceNode`` objects with group-level constraints.

    A group is the natural unit for surface / gathering-network control:
    e.g. "all production wells in the eastern sector" or "all reinjectors".

    Parameters
    ----------
    name : str
        Group identifier.
    group_rate_target : float or None
        Target total mass rate for the group [kg/s].
        Positive = injection target, negative = production target.
    min_bhp : float
        Minimum bottomhole pressure constraint [Pa] (for producers).
    max_bhp : float
        Maximum bottomhole pressure constraint [Pa] (for injectors).

    """

    def __init__(
        self,
        name: str,
        group_rate_target: float | None = None,
        min_bhp: float = 1e6,
        max_bhp: float = 300e6,
    ):
        self.name = name
        self.nodes: dict[str, SourceNode] = {}
        self.group_rate_target = group_rate_target
        self.min_bhp = min_bhp
        self.max_bhp = max_bhp

    # ---------------------------------------------------------------

    def add_node(self, node: SourceNode) -> None:
        """Add a ``SourceNode`` to the group."""
        self.nodes[node.name] = node

    def remove_node(self, name: str) -> None:
        """Remove a ``SourceNode`` by name."""
        self.nodes.pop(name, None)

    def get_node(self, name: str) -> SourceNode | None:
        """Retrieve a node by name."""
        return self.nodes.get(name)

    def compute_group_rate(self) -> float:
        """Sum of current rates across active nodes.

        Returns
        -------
        q_total : float
            Total mass rate [kg/s].

        """
        return sum(n.rate for n in self.nodes.values() if n.active)

    def allocate_rates(self, target_total: float, method: str = "uniform") -> None:
        """Distribute a target total rate among active group members.

        Parameters
        ----------
        target_total : float
            Total rate to allocate [kg/s].
        method : {'uniform', 'proportional'}
            - ``'uniform'``      – split evenly among active nodes.
            - ``'proportional'`` – scale existing rates proportionally.

        Raises
        ------
        ValueError
            If ``method`` is unknown.

        """
        if method not in {"uniform", "proportional"}:
            raise ValueError(f"Unknown allocation method: {method}")

        active_nodes = [n for n in self.nodes.values() if n.active]
        if not active_nodes:
            return

        if method == "uniform":
            per_node = target_total / len(active_nodes)
            for node in active_nodes:
                node.rate = per_node
        elif method == "proportional":
            current_sum = sum(abs(n.rate) for n in active_nodes)
            if current_sum == 0.0:
                per_node = target_total / len(active_nodes)
                for node in active_nodes:
                    node.rate = per_node
            else:
                scale = target_total / current_sum
                for node in active_nodes:
                    node.rate *= scale
        else:
            raise ValueError(f"Unknown allocation method: {method}")

    def get_source_terms(self, num_cells: int) -> np.ndarray:
        """Build source/sink array for the entire group.

        Parameters
        ----------
        num_cells : int
            Number of grid cells.

        Returns
        -------
        q : ndarray
            Mass rate per cell [kg/s].  Length = *num_cells*.

        """
        q = np.zeros(num_cells)
        for node in self.nodes.values():
            if not node.active:
                continue
            if 0 <= node.cell_index < num_cells:
                q[node.cell_index] += node.rate
        return q

    def get_enthalpy_terms(self, num_cells: int) -> np.ndarray:
        """Build enthalpy-weighted source term.

        Parameters
        ----------
        num_cells : int
            Number of grid cells.

        Returns
        -------
        h_dot : ndarray
            Enthalpy flux per cell [W].  Length = *num_cells*.

        """
        h = np.zeros(num_cells)
        for node in self.nodes.values():
            if not node.active or node.enthalpy is None:
                continue
            if 0 <= node.cell_index < num_cells:
                h[node.cell_index] += node.rate * node.enthalpy
        return h

    def __repr__(self) -> str:
        return f"SourceGroup(name={self.name!r}, nodes={len(self.nodes)}, rate={self.compute_group_rate():.3e})"


# ---------------------------------------------------------------------------
# SourceNetwork
# ---------------------------------------------------------------------------


class SourceNetwork:
    """Top-level reservoir source network.

    Manages ``SourceGroup``, ``Separator``, and ``Reinjector`` instances,
    wires them together via connections, and computes net source/sink
    terms for the grid solver.

    Connections are stored as 5-tuples::

        (upstream_type, upstream_name, downstream_type, downstream_name, stream_phase)

    where ``upstream_type`` is ``'group'`` or ``'separator'``, and
    ``downstream_type`` is ``'separator'``, ``'reinjector'``, or ``'group'``.

    """

    def __init__(self):
        self.groups: dict[str, SourceGroup] = {}
        self.separators: dict[str, Separator] = {}
        self.reinjectors: dict[str, Reinjector] = {}
        self.connections: list[tuple[str, str, str, str, str]] = []

    # ---------------------------------------------------------------

    def add_group(self, group: SourceGroup) -> None:
        """Register a ``SourceGroup``."""
        self.groups[group.name] = group

    def add_separator(self, separator: Separator) -> None:
        """Register a ``Separator``."""
        self.separators[separator.name] = separator

    def add_reinjector(self, reinjector: Reinjector) -> None:
        """Register a ``Reinjector``."""
        self.reinjectors[reinjector.name] = reinjector

    def connect(
        self,
        upstream_type: str,
        upstream_name: str,
        downstream_type: str,
        downstream_name: str,
        stream_phase: str = "total",
    ) -> None:
        """Add a directional connection in the network.

        Parameters
        ----------
        upstream_type : {'group', 'separator'}
            Type of the upstream entity.
        upstream_name : str
            Name of the upstream entity.
        downstream_type : {'separator', 'reinjector', 'group'}
            Type of the downstream entity.
        downstream_name : str
            Name of the downstream entity.
        stream_phase : str
            Which phase/stream to transfer (e.g. ``'water'``,
            ``'steam'``, ``'total'``).

        Raises
        ------
        ValueError
            If an invalid type string is supplied.

        """
        valid_upstream = {"group", "separator"}
        valid_downstream = {"separator", "reinjector", "group"}
        if upstream_type not in valid_upstream:
            raise ValueError(f"upstream_type must be one of {valid_upstream}, got {upstream_type!r}")
        if downstream_type not in valid_downstream:
            raise ValueError(f"downstream_type must be one of {valid_downstream}, got {downstream_type!r}")
        self.connections.append((upstream_type, upstream_name, downstream_type, downstream_name, stream_phase))

    def remove_connection(self, upstream_name: str, downstream_name: str) -> None:
        """Remove all connections between two named entities."""
        self.connections = [c for c in self.connections if not (c[1] == upstream_name and c[3] == downstream_name)]

    # ---------------------------------------------------------------

    def get_group(self, name: str) -> SourceGroup | None:
        """Retrieve a group by name."""
        return self.groups.get(name)

    def get_separator(self, name: str) -> Separator | None:
        """Retrieve a separator by name."""
        return self.separators.get(name)

    def get_reinjector(self, name: str) -> Reinjector | None:
        """Retrieve a reinjector by name."""
        return self.reinjectors.get(name)

    # ---------------------------------------------------------------

    def compute_source_terms(self, num_cells: int) -> dict[str, np.ndarray]:
        """Compute net source/sink terms from the whole network.

        The evaluation order is:

        1. Direct contributions from all ``SourceGroup`` (production +
           direct injection).
        2. Upstream mass flows into ``Separator`` instances.
        3. ``Reinjector`` objects drawing from separator streams and adding
           back to the grid.

        Parameters
        ----------
        num_cells : int
            Number of grid cells.

        Returns
        -------
        terms : dict
            Dictionary with keys ``'mass'`` [kg/s] and ``'enthalpy'`` [W],
            each an ``ndarray`` of length *num_cells*.

        """
        mass = np.zeros(num_cells)
        enthalpy = np.zeros(num_cells)

        # ------------------------------------------------------------------
        # Stage 1 – direct contributions from every group
        # ------------------------------------------------------------------
        for group in self.groups.values():
            mass += group.get_source_terms(num_cells)
            enthalpy += group.get_enthalpy_terms(num_cells)

        # ------------------------------------------------------------------
        # Stage 2 – accumulate separator feed rates from upstream connections
        # ------------------------------------------------------------------
        sep_mass_in: dict[str, float] = {name: 0.0 for name in self.separators}
        sep_h_in: dict[str, float] = {name: 0.0 for name in self.separators}

        for up_type, up_name, down_type, down_name, phase in self.connections:
            if up_type == "group" and down_type == "separator":
                group = self.groups.get(up_name)
                sep = self.separators.get(down_name)
                if group is None or sep is None:
                    continue
                # Production is represented by negative rates
                group_rate = group.compute_group_rate()
                prod_rate = -min(0.0, group_rate)  # absolute production
                sep_mass_in[sep.name] += prod_rate

                # Approximate average enthalpy from producing nodes
                producers = [n for n in group.nodes.values() if n.active and n.rate < 0]
                if producers:
                    avg_h = sum(n.enthalpy or 0.0 for n in producers) / len(producers)
                    sep_h_in[sep.name] += prod_rate * avg_h

        # Run all separators once inputs are known
        for sep in self.separators.values():
            rate_in = sep_mass_in.get(sep.name, 0.0)
            if rate_in > 0.0:
                h_in = sep_h_in.get(sep.name, 0.0)
                sep.separate(rate_in, h_in / rate_in if rate_in > 0 else None)

        # ------------------------------------------------------------------
        # Stage 3 – reinjectors consuming separator output streams
        # ------------------------------------------------------------------
        for up_type, up_name, down_type, down_name, phase in self.connections:
            if up_type == "separator" and down_type == "reinjector":
                sep = self.separators.get(up_name)
                reinj = self.reinjectors.get(down_name)
                if sep is None or reinj is None:
                    continue
                stream_rate = sep.get_stream(phase)
                reinj_rate = reinj.compute_rate(stream_rate)
                if reinj_rate > 0.0:
                    mass[reinj.cell_index] += reinj_rate
                    enthalpy[reinj.cell_index] += reinj_rate * reinj.injection_enthalpy()

        return {"mass": mass, "enthalpy": enthalpy}

    # ---------------------------------------------------------------

    def update_from_wells(self, well_manager: WellManager) -> None:
        """Synchronise the network with an existing ``WellManager``.

        Creates a default ``SourceGroup`` named ``'well_group'`` containing
        ``SourceNode`` copies of every active well in *well_manager*.
        Existing groups are **not** cleared; the new group is simply added.

        Note
        ----
        ``well.current_rate`` is only populated after the well has been
        evaluated (i.e. after ``apply_constraints`` or
        ``compute_well_rates``). Calling this method before the first
        evaluation copies a rate of 0 for every well — re-call after
        evaluating the wells to keep the network in sync.

        Parameters
        ----------
        well_manager : WellManager
            Manager holding ``PeacemanWell`` instances.

        """
        if not well_manager.wells:
            return

        default_group = SourceGroup(name="well_group")
        for name, well in well_manager.wells.items():
            node = SourceNode(
                name=name,
                cell_index=well.params.cell_index,
                rate=well.current_rate,  # negative = production in PeacemanWell
                phase="water",
                active=True,
            )
            default_group.add_node(node)

        self.groups[default_group.name] = default_group

    # ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SourceNetwork("
            f"groups={len(self.groups)}, "
            f"separators={len(self.separators)}, "
            f"reinjectors={len(self.reinjectors)}, "
            f"connections={len(self.connections)}"
            f")"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SourceNode",
    "Separator",
    "Reinjector",
    "SourceGroup",
    "SourceNetwork",
]
