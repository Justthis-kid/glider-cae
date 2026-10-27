"""Glauert lifting-line theory for a symmetric, untwisted wing.

Circulation along the span is expanded in a Fourier sine series
    Gamma(theta) = 2 b V  sum_n A_n sin(n theta),     y = -(b/2) cos(theta)
and the lifting-line condition (Prandtl) at each collocation station reads
    mu(theta) (alpha - alpha_0L) sin(theta) = sum_n A_n sin(n theta) (n mu(theta) + sin(theta)),
    mu = a0 c(theta) / (4 b)
with a0 the 2-D section lift slope. Only odd n survive for a symmetric wing, so a
single N x N linear solve gives everything:
    CL      = pi AR A_1
    CL_a    = pi AR A_1          (solved for alpha - alpha_0L = 1 rad)
    CD_i    = pi AR sum n A_n^2
    e       = A_1^2 / sum n A_n^2     (span efficiency, CD_i = CL^2 / (pi e AR))
All of it is dense NumPy linear algebra - no loops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class LiftingLineResult:
    lift_slope: float          # dCL/dalpha, per rad, referenced to the wing area
    oswald: float              # span-efficiency factor e
    modes: np.ndarray          # odd harmonics n = 1, 3, 5, ...
    fourier: np.ndarray        # A_n for alpha - alpha_0L = 1 rad

    def circulation(self, theta: np.ndarray) -> np.ndarray:
        """Non-dimensional Gamma / (2 b V) at the given theta values."""
        return np.sin(np.outer(theta, self.modes)) @ self.fourier


def solve_lifting_line(
    span: float,
    area: float,
    chord_of_theta: Callable[[np.ndarray], np.ndarray],
    section_slope: float = 2.0 * np.pi,
    n_terms: int = 40,
) -> LiftingLineResult:
    """Solve the lifting-line system for ``n_terms`` odd harmonics.

    ``chord_of_theta`` maps theta in (0, pi/2] (root at pi/2, tip at 0) to chord in metres.
    """
    n = 2 * np.arange(n_terms) + 1                      # 1, 3, ..., 2N-1
    theta = (np.arange(1, n_terms + 1) * np.pi) / (2 * n_terms)
    sin_t = np.sin(theta)
    mu = section_slope * np.asarray(chord_of_theta(theta), dtype=float) / (4.0 * span)

    matrix = np.sin(np.outer(theta, n)) * (n[None, :] * mu[:, None] + sin_t[:, None])
    rhs = mu * sin_t                                    # alpha - alpha_0L = 1 rad
    fourier = np.linalg.solve(matrix, rhs)

    aspect_ratio = span ** 2 / area
    lift_slope = float(np.pi * aspect_ratio * fourier[0])
    oswald = float(fourier[0] ** 2 / np.sum(n * fourier ** 2))
    return LiftingLineResult(lift_slope, oswald, n, fourier)


def tapered_chord(root_chord: float, taper_ratio: float) -> Callable[[np.ndarray], np.ndarray]:
    """c(theta) = c_root (1 - (1 - lambda) |cos theta|)  - linear taper."""
    return lambda theta: root_chord * (1.0 - (1.0 - taper_ratio) * np.abs(np.cos(theta)))


def finite_wing_slope(aspect_ratio: float, section_slope: float = 2.0 * np.pi) -> float:
    """Helmbold / Diederich closed form for a straight wing:
        a = a0 / ( sqrt(1 + (a0/(pi AR))^2) + a0/(pi AR) )
    Used for the small tail surfaces where a full solve is unnecessary.
    """
    k = section_slope / (np.pi * aspect_ratio)
    return float(section_slope / (np.sqrt(1.0 + k * k) + k))
