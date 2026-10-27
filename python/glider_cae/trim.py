"""Steady-glide trim: where Cm = 0, then speed from lift = weight.

For a steady, wings-level glide at flight-path angle gamma (negative = descent):
    Cm(alpha_trim) = 0                         (pitch equilibrium)
    gamma = -atan(CD / CL)                     (force balance along the path)
    V = sqrt( 2 W cos(gamma) / (rho S CL) )    (force balance normal to the path)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aero import AeroModel
from .atmosphere import isa_density
from .config import GliderConfig


@dataclass(frozen=True)
class TrimSolution:
    alpha: float        # rad
    gamma: float        # rad (flight-path angle, negative when descending)
    airspeed: float     # m/s
    cl: float
    cd: float

    @property
    def glide_ratio(self) -> float:
        return self.cl / self.cd


def find_glide_trim(
    config: GliderConfig,
    aero: AeroModel,
    altitude: float = 0.0,
    elevator: float = 0.0,
) -> TrimSolution:
    # One vectorised sweep of the aero model replaces a scalar root-find loop.
    alphas = np.radians(np.linspace(-6.0, 11.0, 1701))
    sweep = aero.coefficients(alphas, 0.0, 0.0, 0.0, 0.0, elevator, 0.0, 0.0)
    cm = np.broadcast_to(np.asarray(sweep.Cm, dtype=float), alphas.shape)
    cl = np.broadcast_to(np.asarray(sweep.CL, dtype=float), alphas.shape)

    # Stable crossing (Cm falls with alpha), in attached flow only: past the CL peak a
    # stalled wing can create a spurious nose-down "trim" that is not a flyable glide.
    attached = np.arange(alphas.size - 1) < int(np.argmax(cl))
    crossings = np.flatnonzero((cm[:-1] > 0.0) & (cm[1:] <= 0.0) & attached)
    if crossings.size == 0:
        raise ValueError("No stable pitch-trim point found for alpha in [-6, 11] deg")
    i = crossings[0]
    a_trim = float(alphas[i] - cm[i] * (alphas[i + 1] - alphas[i]) / (cm[i + 1] - cm[i]))

    c = aero.coefficients(a_trim, 0.0, 0.0, 0.0, 0.0, elevator, 0.0, 0.0)
    cl, cd = float(c.CL), float(c.CD)
    if cl <= 0.0:
        raise ValueError("Trim lift coefficient is not positive; cannot sustain a glide")

    gamma = -float(np.arctan2(cd, cl))
    rho = float(isa_density(altitude))
    speed = float(np.sqrt(2.0 * config.weight * np.cos(gamma) / (rho * config.wing_area * cl)))
    return TrimSolution(alpha=a_trim, gamma=gamma, airspeed=speed, cl=cl, cd=cd)
