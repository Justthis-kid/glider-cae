"""Aerodynamic model interface + a placeholder model for stage 1.

Stage 2 (aero estimator) replaces ``PlaceholderAeroModel`` with a model whose
coefficients are derived from the mesh. Anything with a ``coefficients`` method
of the same signature can be dropped into the simulator.

Control inputs are normalised pilot commands in [-1, 1]:
    elevator +1 = full nose-up (stick back)
    aileron  +1 = roll right
    rudder   +1 = yaw right

All inputs are NumPy-broadcastable, so a whole alpha sweep is one call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import GliderConfig


@dataclass
class AeroCoefficients:
    """Non-dimensional force / moment coefficients (about the CG)."""

    CL: np.ndarray | float   # lift          (perpendicular to relative wind)
    CD: np.ndarray | float   # drag          (along relative wind)
    CY: np.ndarray | float   # side force
    Cl: np.ndarray | float   # rolling moment
    Cm: np.ndarray | float   # pitching moment
    Cn: np.ndarray | float   # yawing moment


class AeroModel(Protocol):
    def coefficients(
        self, alpha, beta, p_hat, q_hat, r_hat, elevator, aileron, rudder, reynolds=None
    ) -> AeroCoefficients:
        """alpha/beta in rad; p_hat = p b / 2V, q_hat = q c / 2V, r_hat = r b / 2V.

        ``reynolds`` (based on the mean aerodynamic chord) is optional: models whose
        drag depends on Re use it, others ignore it. ``None`` means "design Re".
        """
        ...


@dataclass
class PlaceholderAeroModel:
    """Generic sailplane aerodynamics (hand-tuned, not derived from geometry).

    Lift slope (Helmbold finite-wing correction of thin-airfoil 2*pi):
        CLa = 2 pi AR / (2 + sqrt(4 + AR^2))
    Pre-stall:  CL = CLa (alpha - alpha_0L),   CD = CD0 + CL^2 / (pi e AR)
    Post-stall: flat plate, CL = CN90 sin(a) cos(a),  CD = CD0 + CN90 sin^2(a)
    Blended with a sigmoid sigma(alpha) (Beard & McLain) so the polar is smooth.
    """

    aspect_ratio: float = 11.76
    oswald: float = 0.90
    cd0: float = 0.020
    alpha_zero_lift: float = float(np.radians(-3.0))
    alpha_stall: float = float(np.radians(12.0))
    stall_sharpness: float = 30.0
    cn_flat_plate: float = 1.2
    alpha_trim_ref: float = float(np.radians(4.0))   # Cm = 0 here (elevator = 0)

    # Static / dynamic stability derivatives (per rad, rates non-dimensional)
    cy_beta: float = -0.30
    cl_beta: float = -0.06
    cl_p: float = -0.50
    cl_r: float = 0.06
    cm_alpha: float = -0.90
    cm_q: float = -12.0
    cn_beta: float = 0.06
    cn_p: float = -0.03
    cn_r: float = -0.08

    # Control effectiveness (per rad of surface) and travel limits
    cm_elevator: float = 0.45
    cl_aileron: float = 0.20
    cn_aileron: float = -0.02     # adverse yaw
    cn_rudder: float = 0.08
    elevator_max: float = float(np.radians(15.0))
    aileron_max: float = float(np.radians(20.0))
    rudder_max: float = float(np.radians(20.0))

    @classmethod
    def from_config(cls, config: GliderConfig) -> "PlaceholderAeroModel":
        return cls(aspect_ratio=config.aspect_ratio)

    @property
    def cl_alpha(self) -> float:
        ar = self.aspect_ratio
        return 2.0 * np.pi * ar / (2.0 + np.sqrt(4.0 + ar * ar))

    def coefficients(
        self, alpha, beta, p_hat, q_hat, r_hat, elevator, aileron, rudder, reynolds=None
    ) -> AeroCoefficients:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)

        # --- longitudinal: linear polar blended with a flat-plate post-stall polar
        m, a_s = self.stall_sharpness, self.alpha_stall
        e_pos = np.exp(-m * (alpha - a_s))
        e_neg = np.exp(m * (alpha + a_s))
        sigma = (1.0 + e_pos + e_neg) / ((1.0 + e_pos) * (1.0 + e_neg))

        cl_lin = self.cl_alpha * (alpha - self.alpha_zero_lift)
        k_induced = 1.0 / (np.pi * self.oswald * self.aspect_ratio)
        cd_lin = self.cd0 + k_induced * cl_lin ** 2

        sa, ca = np.sin(alpha), np.cos(alpha)
        cl_fp = self.cn_flat_plate * sa * ca
        cd_fp = self.cd0 + self.cn_flat_plate * sa * sa

        cl = (1.0 - sigma) * cl_lin + sigma * cl_fp
        cd = (1.0 - sigma) * cd_lin + sigma * cd_fp

        # --- moments and side force (linear stability-derivative model)
        cm0 = -self.cm_alpha * self.alpha_trim_ref
        d_e = elevator * self.elevator_max
        d_a = aileron * self.aileron_max
        d_r = rudder * self.rudder_max

        cy = self.cy_beta * beta
        c_roll = self.cl_beta * beta + self.cl_p * p_hat + self.cl_r * r_hat + self.cl_aileron * d_a
        c_pitch = cm0 + self.cm_alpha * alpha + self.cm_q * q_hat + self.cm_elevator * d_e
        c_yaw = (self.cn_beta * beta + self.cn_p * p_hat + self.cn_r * r_hat
                 + self.cn_rudder * d_r + self.cn_aileron * d_a)

        return AeroCoefficients(CL=cl, CD=cd, CY=cy, Cl=c_roll, Cm=c_pitch, Cn=c_yaw)
