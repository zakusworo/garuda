"""Multiphase flow module — Two-phase water/steam geothermal reservoir simulation.

Phase 2 implementation: robust SIM using tested TPFASolver for pressure,
explicit energy update, and IAPWS-97 phase equilibrium constraints.
"""

from dataclasses import dataclass

import numpy as np

from garuda.core.grid import Grid


@dataclass
class MultiphaseState:
    pressure: np.ndarray
    saturation: np.ndarray
    temperature: np.ndarray
    enthalpy: np.ndarray
    prev_pressure: np.ndarray
    prev_saturation: np.ndarray
    prev_temperature: np.ndarray
    prev_enthalpy: np.ndarray


class MultiphaseFlow:
    """Two-phase geothermal simulator using SIM + IAPWS-97."""

    def __init__(self, grid: Grid, rock: object, fluid: object, iapws: object = None):
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.iapws = iapws or self._make_iapws()

        self.swr, self.ssr = 0.2, 0.0
        self.krw0, self.krs0 = 1.0, 1.0
        self.nw, self.ns = 2.0, 2.0

        nc = grid.num_cells
        self.state = MultiphaseState(
            pressure=np.full(nc, 1e5),
            saturation=np.zeros(nc),
            temperature=np.full(nc, 293.15),
            enthalpy=np.zeros(nc),
            prev_pressure=np.full(nc, 1e5),
            prev_saturation=np.zeros(nc),
            prev_temperature=np.full(nc, 293.15),
            prev_enthalpy=np.zeros(nc),
        )
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _make_iapws():
        from garuda.core.iapws_properties import WaterSteamProperties

        return WaterSteamProperties()

    # ── Corey relative permeability ──────────────────────────────────────

    def relative_permeability(self, S_w):
        d = 1.0 - self.swr - self.ssr
        Se = np.clip((S_w - self.swr) / d, 0, 1) if d > 0 else np.clip(S_w, 0, 1)
        return self.krw0 * Se**self.nw, self.krs0 * (1 - Se) ** self.ns

    # ── Cache IAPWS properties ───────────────────────────────────────────

    def _refresh(self):
        p, T = self.state.pressure, self.state.temperature
        pm = p / 1e6
        # density_region1, viscosity_liquid, enthalpy_liquid, enthalpy_vapor
        # all broadcast over arrays — avoid the per-cell Python loop.
        rw = np.asarray(self.iapws.density_region1(pm, T), dtype=float)
        mw = np.asarray(self.iapws.viscosity_liquid(T), dtype=float)
        hw = np.asarray(self.iapws.enthalpy_liquid(T), dtype=float) * 1000.0
        hv = np.asarray(self.iapws.enthalpy_vapor(T), dtype=float) * 1000.0
        # saturation_temperature stays scalar-only (uses early-return logic).
        ts = np.fromiter((self.iapws.saturation_temperature(pi) for pi in pm),
                         dtype=float, count=pm.size)
        self._cache = {"rw": rw, "mw": mw, "hw": hw, "hv": hv, "Ts": ts}

    # ── IAPWS phase equilibrium ──────────────────────────────────────────

    def apply_phase_equilibrium(self):
        self._refresh()
        T, h, Ts = self.state.temperature, self.state.enthalpy, self._cache["Ts"]
        hw, hv = self._cache["hw"], self._cache["hv"]
        for i in range(self.grid.num_cells):
            if not np.isfinite(Ts[i]):
                self.state.saturation[i] = 0.0
                self.state.enthalpy[i] = hw[i]
                continue
            if T[i] < Ts[i]:
                self.state.saturation[i] = 0.0
                self.state.temperature[i] = max(T[i], 273.15)
            else:
                self.state.temperature[i] = Ts[i]
                hfg = max(hv[i] - hw[i], 1e-10)
                self.state.saturation[i] = np.clip((h[i] - hw[i]) / hfg, 0.0, 1.0)
            self.state.enthalpy[i] = (1 - self.state.saturation[i]) * hw[i] + self.state.saturation[i] * hv[i]

    # ── Time step ────────────────────────────────────────────────────────

    def step(self, dt, source_terms, heat_sources=None, bc_type="dirichlet", bc_values=None, max_iter=20, tol=1e-6):

        from garuda.core.tpfa_solver import TPFASolver

        if heat_sources is None:
            heat_sources = np.zeros(self.grid.num_cells)
        dts = max(dt, 1e-10)

        phi = (
            np.full(self.grid.num_cells, self.rock.porosity) if np.isscalar(self.rock.porosity) else self.rock.porosity
        )
        vol = self.grid.cell_volumes
        rho_r, cp_r = self.rock.rho_rock, self.rock.cp

        # Build the pressure solver once per time step. mu/rho here are
        # constants from FluidProperties; rebuilding inside the loop forced a
        # full transmissibility (and numba) recompute every Picard iteration.
        solver = TPFASolver(self.grid, mu=self.fluid.mu, rho=self.fluid.rho)

        converged = False
        for it in range(max_iter):
            po, To, So = self.state.pressure.copy(), self.state.temperature.copy(), self.state.saturation.copy()

            # ── Pressure solve via TPFASolver ──
            # If no BC provided, use far-field Dirichlet to keep system well-posed
            if bc_values is None:
                bc = np.array([self.state.pressure[0], self.state.pressure[-1]])
            else:
                bc = bc_values
            self.state.pressure = solver.solve(source_terms, bc_type, bc, "direct")

            # ── Explicit energy update (source-only, simplified) ──
            # (rho*Cp)_bulk * dT/dt = Q_src   [J/(m^3*K) * K/s = W/m^3]
            # Conduction is intentionally omitted in this lumped step; for a
            # proper face-based conduction discretisation use ThermalFlow.
            self._refresh()
            for i in range(self.grid.num_cells):
                rhoCp = (1 - phi[i]) * rho_r * cp_r + phi[i] * self._cache["rw"][i] * self.fluid.cp
                Q_src = heat_sources[i]  # W/m^3
                dT = Q_src * dts / max(rhoCp, 1.0)
                self.state.temperature[i] = self.state.temperature[i] + dT
                self.state.temperature[i] = np.clip(self.state.temperature[i], 273.15, 650.0)

            # ── Enthalpy from phase mix ──
            self._refresh()
            S_w = 1.0 - self.state.saturation
            S_s = self.state.saturation
            self.state.enthalpy = (
                S_w * self._cache["hw"]
                + S_s * self._cache["hv"]
                + heat_sources * dts / np.maximum(self._cache["rw"] * vol, 1.0)
            )

            # ── Phase constraints ──
            self.apply_phase_equilibrium()

            # ── Convergence ──
            dp = np.linalg.norm(self.state.pressure - po) / (np.linalg.norm(po) + 1e-10)
            dT = np.linalg.norm(self.state.temperature - To) / (np.linalg.norm(To) + 1e-10)
            dS = np.linalg.norm(self.state.saturation - So) / (max(np.linalg.norm(So), 1e-10))
            if max(dp, dT, dS) < tol:
                converged = True
                break

        self.state.prev_pressure = self.state.pressure.copy()
        self.state.prev_saturation = self.state.saturation.copy()
        self.state.prev_temperature = self.state.temperature.copy()
        self.state.prev_enthalpy = self.state.enthalpy.copy()

        return {"converged": converged, "iterations": it + 1, "dp_norm": dp, "dT_norm": dT, "dS_norm": dS}

    # ── Initialization ───────────────────────────────────────────────────

    def set_initial_state(self, pressure, temperature, saturation=None):
        # Accept Python scalars / lists / arrays alike — broadcast scalars to
        # the grid so callers don't have to wrap a single number in np.full.
        nc = self.grid.num_cells
        p = np.broadcast_to(np.asarray(pressure, dtype=float), (nc,)).copy()
        T = np.broadcast_to(np.asarray(temperature, dtype=float), (nc,)).copy()
        if saturation is None:
            S = np.zeros(nc)
        else:
            S = np.broadcast_to(np.asarray(saturation, dtype=float), (nc,)).copy()

        self.state.pressure = p
        self.state.temperature = T
        self.state.saturation = S
        self._refresh()
        Sw = 1.0 - self.state.saturation
        Ss = self.state.saturation
        self.state.enthalpy = Sw * self._cache["hw"] + Ss * self._cache["hv"]
        self.state.prev_pressure = p.copy()
        self.state.prev_saturation = S.copy()
        self.state.prev_temperature = T.copy()
        self.state.prev_enthalpy = self.state.enthalpy.copy()

    def compute_geothermal_gradient(self, surface_temp=298.15, gradient=0.06, depth=None):
        if depth is None:
            # Grid z grows upward from 0; interpret z as depth below the top
            # of the reservoir so temperature increases with z.
            depth = self.grid.cell_centroids[:, 2]
        Ti = surface_temp + gradient * depth
        self.state.temperature = Ti
        self.state.prev_temperature = Ti.copy()
        return Ti

    def get_summary(self):
        p = self.state.pressure / 1e5
        T = self.state.temperature - 273.15
        S = self.state.saturation * 100
        Savg = float(np.mean(self.state.saturation))
        return {
            "p_min": float(p.min()),
            "p_max": float(p.max()),
            "p_avg": float(p.mean()),
            "T_min": float(T.min()),
            "T_max": float(T.max()),
            "T_avg": float(T.mean()),
            "S_min": float(S.min()),
            "S_max": float(S.max()),
            "S_avg": float(S.mean()),
            "phase": "liquid-dominated" if Savg < 0.05 else "vapor-dominated" if Savg > 0.95 else "two-phase",
        }
