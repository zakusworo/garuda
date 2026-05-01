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
        nc = self.grid.num_cells
        pm = p / 1e6
        rw = np.zeros(nc)
        mw = np.zeros(nc)
        hw = np.zeros(nc)
        hv = np.zeros(nc)
        ts = np.zeros(nc)
        for i in range(nc):
            rw[i] = self.iapws.density_region1(pm[i], T[i])
            mw[i] = self.iapws.viscosity_liquid(T[i])
            hw[i] = self.iapws.enthalpy_liquid(T[i]) * 1000
            hv[i] = self.iapws.enthalpy_vapor(T[i]) * 1000
            ts[i] = self.iapws.saturation_temperature(pm[i])
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
        lam = self.rock.lambda_rock

        converged = False
        for it in range(max_iter):
            po, To, So = self.state.pressure.copy(), self.state.temperature.copy(), self.state.saturation.copy()

            # ── Pressure solve via TPFASolver ──
            solver = TPFASolver(self.grid, mu=self.fluid.mu, rho=self.fluid.rho)
            # If no BC provided, use far-field Dirichlet to keep system well-posed
            if bc_values is None:
                bc = np.array([self.state.pressure[0], self.state.pressure[-1]])
            else:
                bc = bc_values
            self.state.pressure = solver.solve(source_terms, bc_type, bc, "direct")

            # ── Explicit energy update ──
            self._refresh()
            for i in range(self.grid.num_cells):
                rhoCp = (1 - phi[i]) * rho_r * cp_r + phi[i] * self._cache["rw"][i] * self.fluid.cp
                Q_cond = lam * vol[i]  # simplified conduction
                Q_src = heat_sources[i] * vol[i]
                dT = (-Q_cond * (self.state.temperature[i] - self.state.prev_temperature[i]) + Q_src) * dts
                self.state.temperature[i] = self.state.temperature[i] + dT / max(rhoCp * vol[i], 1.0)
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
        self.state.pressure = pressure.copy()
        self.state.temperature = temperature.copy()
        self.state.saturation = saturation.copy() if saturation is not None else np.zeros_like(pressure)
        self._refresh()
        Sw = 1.0 - self.state.saturation
        Ss = self.state.saturation
        self.state.enthalpy = Sw * self._cache["hw"] + Ss * self._cache["hv"]
        self.state.prev_pressure = pressure.copy()
        self.state.prev_saturation = self.state.saturation.copy()
        self.state.prev_temperature = temperature.copy()
        self.state.prev_enthalpy = self.state.enthalpy.copy()

    def compute_geothermal_gradient(self, surface_temp=298.15, gradient=0.06, depth=None):
        if depth is None:
            depth = -self.grid.cell_centroids[:, 2]
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
