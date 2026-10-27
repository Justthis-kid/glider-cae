"""Centroid and inertia of a triangle mesh, by exact integration over the triangles.

shell  : uniform mass per unit *area* (a hollow moulded airframe, works for open meshes)
volume : uniform density *inside* a closed mesh (a solid / foam model)

Both are vectorised sums over triangles. With vertices a, b, c and s = a + b + c:
    triangle   int x x^T dA = A/12 ( a a^T + b b^T + c c^T + s s^T )
    tetrahedron (0, a, b, c) int x x^T dV = V/20 ( a a^T + b b^T + c c^T + s s^T )
and the inertia tensor about the centroid is  I = tr(C) 1 - C  with C the second-moment
matrix about the centroid (parallel-axis shift  C - M c c^T).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MassProperties:
    model: str                 # 'shell' or 'volume'
    watertight: bool
    area: float
    volume: float | None
    centroid: np.ndarray       # (3,), same axes as the input vertices
    inertia_unit: np.ndarray   # (3, 3) about the centroid, per unit mass (m^2)


def is_watertight(faces: np.ndarray) -> bool:
    """Every undirected edge shared by exactly two triangles (vertices must already be merged)."""
    f = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges.sort(axis=1)
    n = int(f.max()) + 1
    _, counts = np.unique(edges[:, 0] * n + edges[:, 1], return_counts=True)
    return bool(np.all(counts == 2))


def _second_moment_terms(a, b, c):
    s = a + b + c
    outer = lambda u, v: u[:, :, None] * v[:, None, :]          # noqa: E731
    return outer(a, a) + outer(b, b) + outer(c, c) + outer(s, s)


def mass_properties(vertices: np.ndarray, faces: np.ndarray, model: str = "shell") -> MassProperties:
    tri = np.asarray(vertices, dtype=float)[faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    watertight = is_watertight(faces)
    area_each = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    area = float(area_each.sum())

    if model == "volume" and watertight:
        vol_each = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0           # signed tetrahedra
        signed = float(vol_each.sum())
        weight = vol_each * np.sign(signed)                                  # tolerate inward normals
        volume = total = abs(signed)
        centroid = (weight[:, None] * (a + b + c) / 4.0).sum(0) / total
        c0 = (weight[:, None, None] / 20.0 * _second_moment_terms(a, b, c)).sum(0)
        used = "volume"
    else:
        volume = None
        total = area
        centroid = (area_each[:, None] * (a + b + c) / 3.0).sum(0) / total
        c0 = (area_each[:, None, None] / 12.0 * _second_moment_terms(a, b, c)).sum(0)
        used = "shell"

    cc = c0 / total - np.outer(centroid, centroid)                           # per unit mass, about centroid
    inertia = np.trace(cc) * np.eye(3) - cc
    return MassProperties(used, watertight, area, volume, centroid, inertia)
