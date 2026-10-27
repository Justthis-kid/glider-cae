"""Fixed-step explicit integrators: x_{k+1} = step(f, x_k, dt)."""
from __future__ import annotations

from typing import Callable

import numpy as np

Deriv = Callable[[np.ndarray], np.ndarray]


def euler_step(f: Deriv, x: np.ndarray, dt: float) -> np.ndarray:
    """Forward Euler, O(dt) - cheap, only for quick experiments."""
    return x + dt * f(x)


def rk4_step(f: Deriv, x: np.ndarray, dt: float) -> np.ndarray:
    """Classical 4th-order Runge-Kutta:
        x+ = x + dt/6 (k1 + 2 k2 + 2 k3 + k4)
    """
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


INTEGRATORS = {"euler": euler_step, "rk4": rk4_step}
