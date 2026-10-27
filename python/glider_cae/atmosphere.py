"""International Standard Atmosphere (troposphere only, h <= 11 km)."""
from __future__ import annotations

import numpy as np

T0 = 288.15        # sea-level temperature, K
RHO0 = 1.225       # sea-level density, kg/m^3
LAPSE = 0.0065     # temperature lapse rate, K/m
RHO_EXPONENT = 4.2559  # g/(R*L) - 1


def isa_density(altitude_m):
    """Air density rho(h) = rho0 * (1 - L h / T0) ** 4.2559.

    Works on scalars or NumPy arrays (vectorised).
    """
    h = np.clip(altitude_m, 0.0, 11_000.0)
    return RHO0 * (1.0 - LAPSE * h / T0) ** RHO_EXPONENT


def isa_temperature(altitude_m):
    """T(h) = T0 - L h  (K), troposphere."""
    return T0 - LAPSE * np.clip(altitude_m, 0.0, 11_000.0)


def isa_viscosity(altitude_m):
    """Dynamic viscosity from Sutherland's law: mu = 1.458e-6 T^1.5 / (T + 110.4)  (Pa s)."""
    t = isa_temperature(altitude_m)
    return 1.458e-6 * t ** 1.5 / (t + 110.4)
