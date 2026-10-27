"""Vehicle configuration (SI units everywhere).

Coordinate frames
-----------------
* Inertial frame "NED": x = North, y = East, z = Down  (right-handed).
* Body frame "FRD":     x = Forward (nose), y = Right wing, z = Down.

Positive body rates: p = roll right, q = pitch up, r = yaw right.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G0 = 9.80665  # standard gravity, m/s^2


@dataclass
class GliderConfig:
    """Mass properties and reference geometry of the glider.

    Stage 3 (mesh parser) will populate this from a mesh. Stage 1 ships with a
    ~2 m span RC sailplane so the simulator is usable immediately.
    """

    mass: float = 1.2                      # kg
    inertia: np.ndarray = field(           # kg m^2, about the CG, body axes
        default_factory=lambda: np.diag([0.12, 0.09, 0.20])
    )
    span: float = 2.0                      # b, m
    mean_chord: float = 0.17               # c-bar, m
    wing_area: float = 0.34                # S, m^2

    @property
    def aspect_ratio(self) -> float:
        """AR = b^2 / S."""
        return self.span ** 2 / self.wing_area

    @property
    def weight(self) -> float:
        """W = m g  (N)."""
        return self.mass * G0

    @property
    def inertia_inv(self) -> np.ndarray:
        return np.linalg.inv(self.inertia)
