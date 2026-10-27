"""Real-time simulation loop: fixed physics step + wall-clock accumulator."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .aero import AeroModel, PlaceholderAeroModel
from .config import GliderConfig
from .dynamics import Controls, Environment, GliderDynamics
from .estimator import LiftingLineEstimator
from .geometry import GliderGeometry
from .integrators import INTEGRATORS
from .rigid_body import (
    POS, QUAT, RATE, STATE_SIZE, VEL,
    euler_from_quat, quat_from_euler, quat_normalize, quat_to_dcm,
)
from .trim import find_glide_trim

MAX_FRAME_DT = 0.1        # s of wall time accepted per advance() call (tab-switch guard)
MAX_SUBSTEPS = 60
SAFE_TOUCHDOWN_SPEED = 3.0  # m/s; faster than this counts as a crash
FALLBACK_LAUNCH_SPEED = 10.0  # m/s, used when the aero model has no glide trim point


class GliderSimulation:
    def __init__(
        self,
        config: GliderConfig | None = None,
        aero: AeroModel | None = None,
        env: Environment | None = None,
        dt: float = 0.01,
        integrator: str = "rk4",
    ):
        self.config = config or GliderConfig()
        self.aero = aero or PlaceholderAeroModel.from_config(self.config)
        self.env = env or Environment()
        self.dynamics = GliderDynamics(self.config, self.aero, self.env)
        self.dt = dt
        self._step_fn = INTEGRATORS[integrator]
        self.controls = Controls()
        self.reset()

    @classmethod
    def from_geometry(cls, geometry: GliderGeometry, **kwargs) -> "GliderSimulation":
        """Estimate the aerodynamics of ``geometry`` (stage 2) and build a simulation around it."""
        estimator = LiftingLineEstimator(geometry)
        sim = cls(config=geometry.to_config(), aero=estimator, **kwargs)
        sim.geometry = geometry
        return sim

    # ------------------------------------------------------------------ setup
    def reset(self, altitude: float = 120.0, heading_deg: float = 0.0,
              airspeed: float | None = None) -> None:
        """Start in a trimmed steady glide (or at a custom launch airspeed).

        If the aero model has no stable glide trim (e.g. a zero-load test model),
        fall back to a level start at ``airspeed`` and set ``self.trim = None``.
        """
        try:
            trim = find_glide_trim(self.config, self.aero, altitude)
            alpha, gamma = trim.alpha, trim.gamma
            speed = trim.airspeed if airspeed is None else airspeed
        except ValueError:
            trim, alpha, gamma = None, 0.0, 0.0
            speed = FALLBACK_LAUNCH_SPEED if airspeed is None else airspeed
        x = np.zeros(STATE_SIZE)
        x[2] = -altitude
        x[QUAT] = quat_from_euler(0.0, alpha + gamma, np.radians(heading_deg))
        x[VEL] = [speed * np.cos(alpha), 0.0, speed * np.sin(alpha)]
        self.x, self.t, self.status, self._accum = x, 0.0, "flying", 0.0
        self.trim = trim

    # ------------------------------------------------------------------ stepping
    def step(self) -> None:
        """Advance exactly one physics step of length ``dt``."""
        if self.status != "flying":
            return
        f = lambda s: self.dynamics.derivatives(s, self.controls)  # noqa: E731
        x_new = self._step_fn(f, self.x, self.dt)
        if not np.all(np.isfinite(x_new)):
            self.status = "diverged"
            return
        x_new[QUAT] = quat_normalize(x_new[QUAT])    # undo integration drift of |q|
        self.x, self.t = x_new, self.t + self.dt
        if -self.x[2] <= 0.0:
            self._touch_down()

    def _touch_down(self) -> None:
        impact = float(np.linalg.norm(self.x[VEL]))
        self.status = "landed" if impact <= SAFE_TOUCHDOWN_SPEED else "crashed"
        self.x[2] = 0.0
        self.x[VEL] = 0.0
        self.x[RATE] = 0.0

    def advance(self, wall_dt: float, time_scale: float = 1.0) -> None:
        """Consume ``wall_dt`` seconds of real time as whole physics steps."""
        self._accum += min(max(wall_dt, 0.0), MAX_FRAME_DT) * time_scale
        n = min(int(self._accum / self.dt), MAX_SUBSTEPS)
        self._accum -= n * self.dt
        for _ in range(n):
            self.step()

    def run(self, duration: float,
            controls_fn: Callable[[float, np.ndarray], Controls] | None = None,
            record_every: int = 1) -> dict:
        """Headless batch run. Returns {'t': (N,), 'state': (N, 13)}."""
        t_log, x_log = [], []
        for i in range(int(round(duration / self.dt))):
            if controls_fn is not None:
                self.controls = controls_fn(self.t, self.x)
            self.step()
            if i % record_every == 0:
                t_log.append(self.t)
                x_log.append(self.x.copy())
            if self.status != "flying":
                break
        state = np.vstack(x_log) if x_log else np.empty((0, STATE_SIZE))
        return {"t": np.asarray(t_log), "state": state}

    # ------------------------------------------------------------------ telemetry
    def snapshot(self) -> dict:
        """JSON-friendly view of the current state (plain floats / lists)."""
        s = self.dynamics.evaluate(self.x, self.controls)
        v_ned = s.dcm @ self.x[VEL]
        weight = self.config.weight
        return {
            "t": float(self.t),
            "status": self.status,
            "position_ned": self.x[POS].tolist(),
            "quat": self.x[QUAT].tolist(),
            "velocity_body": self.x[VEL].tolist(),
            "rates_body": self.x[RATE].tolist(),
            "euler_deg": np.degrees(euler_from_quat(self.x[QUAT])).tolist(),
            "altitude": float(-self.x[2]),
            "airspeed": float(s.airspeed),
            "groundspeed": float(np.hypot(v_ned[0], v_ned[1])),
            "climb_rate": float(-v_ned[2]),
            "alpha_deg": float(np.degrees(s.alpha)),
            "beta_deg": float(np.degrees(s.beta)),
            "cl": float(s.coeffs.CL),
            "cd": float(s.coeffs.CD),
            "ld": float(s.coeffs.CL / s.coeffs.CD),
            "load_factor": float(-s.force_aero_b[2] / weight),
            "controls": {
                "elevator": float(self.controls.elevator),
                "aileron": float(self.controls.aileron),
                "rudder": float(self.controls.rudder),
            },
        }
