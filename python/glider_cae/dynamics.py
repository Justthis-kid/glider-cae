"""Glider force & moment model: gravity + aerodynamics -> state derivative.

Aerodynamic forces are built in the wind axes of the *air-relative* velocity
v_rel = v_b - R^T wind:
    v_hat = v_rel / |v_rel|                   (direction of motion through the air)
    l_hat = (y_b x v_hat) / |y_b x v_hat|     (lift, perpendicular to v_hat)
    s_hat = v_hat x l_hat                     (side force, +right)
    F_aero = q_bar S ( -CD v_hat + CL l_hat + CY s_hat ),   q_bar = 1/2 rho V^2
    M_aero = q_bar S [ b Cl,  c Cm,  b Cn ]
    alpha = atan2(w_rel, u_rel),   beta = asin(v_rel / V)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aero import AeroCoefficients, AeroModel
from .atmosphere import isa_density, isa_viscosity
from .config import G0, GliderConfig
from .rigid_body import QUAT, RATE, VEL, cross3, quat_to_dcm, rigid_body_derivatives

V_MIN = 0.1                       # m/s, avoids 0/0 at (near) zero airspeed
_Y_BODY = np.array([0.0, 1.0, 0.0])


@dataclass
class Controls:
    """Normalised pilot commands in [-1, 1] (see aero.py for sign convention)."""

    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0


@dataclass
class Environment:
    gravity: float = G0
    wind_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s; D<0 = updraft


@dataclass
class LoadsSample:
    """Everything computed in one force/moment evaluation."""

    dcm: np.ndarray
    force_b: np.ndarray          # total (aero + gravity), body axes, N
    force_aero_b: np.ndarray     # aerodynamic part only, N
    moment_b: np.ndarray         # N m about the CG
    airspeed: float
    alpha: float
    beta: float
    q_bar: float
    coeffs: AeroCoefficients


class GliderDynamics:
    def __init__(self, config: GliderConfig, aero: AeroModel, env: Environment | None = None):
        self.config = config
        self.aero = aero
        self.env = env or Environment()
        self._inertia = np.asarray(config.inertia, dtype=float)
        self._inertia_inv = np.linalg.inv(self._inertia)

    def evaluate(self, x: np.ndarray, controls: Controls) -> LoadsSample:
        cfg = self.config
        dcm = quat_to_dcm(x[QUAT])
        v_b, omega = x[VEL], x[RATE]

        rho = float(isa_density(-x[2]))
        mu = float(isa_viscosity(-x[2]))
        v_rel = v_b - dcm.T @ self.env.wind_ned
        speed = max(float(np.linalg.norm(v_rel)), V_MIN)
        v_hat = v_rel / speed
        alpha = float(np.arctan2(v_rel[2], v_rel[0]))
        beta = float(np.arcsin(np.clip(v_rel[1] / speed, -1.0, 1.0)))
        q_bar = 0.5 * rho * speed * speed

        # non-dimensional body rates
        p_hat, q_hat, r_hat = omega * np.array([cfg.span, cfg.mean_chord, cfg.span]) / (2.0 * speed)
        reynolds = rho * speed * cfg.mean_chord / mu
        c = self.aero.coefficients(
            alpha, beta, p_hat, q_hat, r_hat,
            controls.elevator, controls.aileron, controls.rudder,
            reynolds=reynolds,
        )

        # wind-axes unit vectors
        l_raw = cross3(_Y_BODY, v_hat)
        l_hat = l_raw / max(float(np.linalg.norm(l_raw)), 1e-6)
        s_hat = cross3(v_hat, l_hat)

        qs = q_bar * cfg.wing_area
        f_aero = qs * (-c.CD * v_hat + c.CL * l_hat + c.CY * s_hat)
        f_grav = cfg.mass * self.env.gravity * dcm[2, :]        # R^T [0, 0, m g]
        m_aero = qs * np.array([cfg.span * c.Cl, cfg.mean_chord * c.Cm, cfg.span * c.Cn])

        return LoadsSample(dcm, f_aero + f_grav, f_aero, m_aero, speed, alpha, beta, q_bar, c)

    def derivatives(self, x: np.ndarray, controls: Controls) -> np.ndarray:
        s = self.evaluate(x, controls)
        return rigid_body_derivatives(
            x, s.force_b, s.moment_b, self.config.mass,
            self._inertia, self._inertia_inv, dcm=s.dcm,
        )
