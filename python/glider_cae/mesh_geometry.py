"""Triangle-mesh geometry primitives and automatic orientation of a glider mesh.

``TriangleMesh`` caches per-triangle bounds so a plane section costs two comparisons per
triangle plus work on the few triangles that actually cross the plane.

Orientation (``detect_orientation``) finds which mesh axis is the span, which way the nose
points and which way is up, assuming the model is axis-aligned up to 90-degree rotations
and flips (the usual state of a CAD export). Cues, in order:
  * span axis      - the axis whose mirror image best overlaps the mesh (a glider is
                     left/right symmetric, nothing else about it is);
  * vertical axis  - of the remaining two, the shorter extent;
  * nose direction - the wing sits nearer the nose than the tail (outer-panel centroid);
  * up direction   - the fin and tail stand above the wing plane, the belly barely below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AXIS_NAMES = "XYZ"


class TriangleMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.tri = self.vertices[self.faces]                             # (M, 3, 3)
        self._lo = [self.tri[:, :, k].min(axis=1) for k in range(3)]
        self._hi = [self.tri[:, :, k].max(axis=1) for k in range(3)]
        cross = np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0])
        self.areas = 0.5 * np.linalg.norm(cross, axis=1)

    @property
    def extents(self) -> np.ndarray:
        return np.ptp(self.vertices, axis=0)

    def sample_surface(self, n: int = 100_000, seed: int = 0) -> np.ndarray:
        """Area-weighted random points on the surface (uniform barycentric sampling)."""
        rng = np.random.default_rng(seed)
        total = self.areas.sum()
        if total <= 0.0:
            return self.vertices.copy()
        pick = rng.choice(len(self.areas), size=n, p=self.areas / total)
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1.0
        u, v = np.where(flip, 1.0 - u, u), np.where(flip, 1.0 - v, v)
        t = self.tri[pick]
        return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])

    def slice_segments(self, axis: int, value: float) -> tuple[np.ndarray, np.ndarray]:
        """Intersect with the plane ``coordinate[axis] == value``: segment end points (K, 3) each."""
        idx = np.flatnonzero((self._lo[axis] <= value) & (self._hi[axis] > value))
        if idx.size == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        t = self.tri[idx]
        d = t[:, :, axis] - value
        positive = d > 0.0
        lone = np.where(positive.sum(axis=1) == 1, positive.argmax(axis=1), (~positive).argmax(axis=1))
        rows = np.arange(idx.size)
        i1, i2 = (lone + 1) % 3, (lone + 2) % 3
        a, b, c = t[rows, lone], t[rows, i1], t[rows, i2]
        da, db, dc = d[rows, lone], d[rows, i1], d[rows, i2]
        p1 = a + (b - a) * (da / (da - db))[:, None]
        p2 = a + (c - a) * (da / (da - dc))[:, None]
        return p1, p2

    def slice_points(self, axis: int, value: float, step: float | None = None) -> np.ndarray:
        """Section points. With ``step`` each segment is resampled at that spacing so the
        outline is dense even when the mesh uses a few large triangles."""
        p1, p2 = self.slice_segments(axis, value)
        if p1.size == 0:
            return p1
        if step is None:
            return np.concatenate([p1, p2])
        n = np.maximum(2, np.ceil(np.linalg.norm(p2 - p1, axis=1) / step).astype(int))
        n = np.minimum(n, 200)
        seg = np.repeat(np.arange(len(n)), n)
        within = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        u = (within / (n[seg] - 1))[:, None]
        return p1[seg] + (p2[seg] - p1[seg]) * u


# ---------------------------------------------------------------------------- orientation
@dataclass
class Orientation:
    rotation: np.ndarray                    # rows: forward, right, down, written in mesh axes
    forward: str
    up: str
    span_axis: str
    confidence: str                         # "high" | "medium" | "low"
    notes: list[str] = field(default_factory=list)
    symmetry: dict = field(default_factory=dict)


def parse_axis(spec: str) -> tuple[int, int]:
    """'+X' / '-y' -> (axis index, sign)."""
    s = spec.strip().upper()
    if len(s) != 2 or s[0] not in "+-" or s[1] not in AXIS_NAMES:
        raise ValueError(f"'{spec}' is not an axis like +X or -Z")
    return AXIS_NAMES.index(s[1]), (1 if s[0] == "+" else -1)


def _name(axis: int, sign: int) -> str:
    return ("+" if sign > 0 else "-") + AXIS_NAMES[axis]


def _symmetry_scores(points: np.ndarray, grid: int = 32) -> np.ndarray:
    """Per axis: fraction of occupied voxels whose mirror image (about the bbox centre) is occupied."""
    lo, hi = points.min(0), points.max(0)
    idx = np.clip(((points - lo) / np.maximum(hi - lo, 1e-12) * grid).astype(int), 0, grid - 1)
    occ = np.zeros((grid,) * 3, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    total = occ.sum()
    return np.array([(occ & np.flip(occ, axis=a)).sum() / total for a in range(3)])


def detect_orientation(mesh: TriangleMesh, forward: str | None = None, up: str | None = None) -> Orientation:
    pts = mesh.sample_surface(100_000)
    lo, hi = pts.min(0), pts.max(0)
    extent = hi - lo
    scores = _symmetry_scores(pts)
    notes: list[str] = []
    confidence = "high"

    f_axis, f_sign = parse_axis(forward) if forward else (None, None)
    u_axis, u_sign = parse_axis(up) if up else (None, None)
    if f_axis is not None and f_axis == u_axis:
        raise ValueError("Nose direction and up direction must be along different axes")

    # ---- which axis plays which role
    def best_symmetry(candidates):
        return max(candidates, key=lambda a: scores[a])

    if f_axis is not None and u_axis is not None:
        l_axis, v_axis = f_axis, u_axis
        s_axis = ({0, 1, 2} - {l_axis, v_axis}).pop()
    elif f_axis is not None:
        l_axis = f_axis
        s_axis = best_symmetry([a for a in range(3) if a != l_axis])
        v_axis = ({0, 1, 2} - {l_axis, s_axis}).pop()
    elif u_axis is not None:
        v_axis = u_axis
        s_axis = best_symmetry([a for a in range(3) if a != v_axis])
        l_axis = ({0, 1, 2} - {v_axis, s_axis}).pop()
    else:
        s_axis = int(np.argmax(scores))
        rest = [a for a in range(3) if a != s_axis]
        v_axis = min(rest, key=lambda a: extent[a])
        l_axis = ({0, 1, 2} - {s_axis, v_axis}).pop()
        ordered = np.sort(scores)
        if ordered[-2] > 0.85 * ordered[-1]:
            confidence = "low"
            notes.append("Left/right symmetry is not clear-cut; check the detected axes.")
        if extent[l_axis] < 1.3 * extent[v_axis]:
            confidence = "low"
            notes.append("The model is nearly as tall as it is long; check which way is up.")

    # ---- signs: nose direction from the wing position, up from the fin / tail height
    mid_s = 0.5 * (lo[s_axis] + hi[s_axis])
    half_span = 0.5 * extent[s_axis]
    outer = pts[np.abs(pts[:, s_axis] - mid_s) > 0.55 * half_span]
    if len(outer) < 50:
        outer = pts[np.abs(pts[:, s_axis] - mid_s) > 0.35 * half_span]

    if f_axis is None:
        along = (pts[:, l_axis] - lo[l_axis]) / extent[l_axis]
        half = np.abs(pts[:, s_axis] - mid_s)
        w_low = half[along < 0.15].max(initial=0.0)
        w_high = half[along > 0.85].max(initial=0.0)
        # Cue 1: the tail end carries the tailplane, the nose end only a slim fuselage. Valid
        # only while the wing itself stays out of both end zones (short nose, swept tips).
        wing_in_end_zone = max(w_low, w_high) >= 0.5 * half_span
        if not wing_in_end_zone and max(w_low, w_high) >= 2.0 * max(min(w_low, w_high), 1e-9):
            f_sign = 1 if w_low > w_high else -1            # tail at the low end => nose at the high end
        else:
            # Cue 2: the wing root sits nearer the nose than the tail (inner panels, so sweep
            # barely matters). Weak for flying wings, which centre the wing mid-length.
            root = pts[(np.abs(pts[:, s_axis] - mid_s) > 0.35 * half_span)
                       & (np.abs(pts[:, s_axis] - mid_s) < 0.60 * half_span)]
            frac = ((root if len(root) >= 50 else outer)[:, l_axis].mean() - lo[l_axis]) / extent[l_axis]
            f_sign = -1 if frac < 0.5 else 1
            confidence = "low" if abs(frac - 0.5) < 0.08 else "medium"
            notes.append("The nose direction comes from the wing position (no clear tailplane to go by).")
    if u_axis is None:
        # Wing plane at the root (regress out dihedral), then see how far the *tail end* of
        # the model reaches above and below it: the fin stands up, the boom barely hangs down.
        reach = np.abs(outer[:, s_axis] - mid_s)
        z_root = float(np.polyfit(reach, outer[:, v_axis], 1)[1]) if np.ptp(reach) > 1e-9 else float(outer[:, v_axis].mean())
        along = (pts[:, l_axis] - lo[l_axis]) / extent[l_axis]
        aft = pts[(along > 0.7) if f_sign < 0 else (along < 0.3)]
        if len(aft) < 20:
            aft = pts
        above, below = aft[:, v_axis].max() - z_root, z_root - aft[:, v_axis].min()
        u_sign = 1 if above >= below else -1
        ratio = max(above, below) / max(min(above, below), 1e-9)
        if ratio < 1.3:
            confidence = "low"
            notes.append("Nothing clearly stands above the wing at the tail, so 'up' is a guess.")
        elif ratio < 2.0 and confidence == "high":
            confidence = "medium"

    fwd = np.zeros(3); fwd[l_axis] = f_sign
    up_v = np.zeros(3); up_v[v_axis] = u_sign
    down = -up_v
    right = np.cross(down, fwd)                              # FRD is right-handed: right = down x forward
    return Orientation(
        rotation=np.stack([fwd, right, down]),
        forward=_name(l_axis, f_sign), up=_name(v_axis, u_sign),
        span_axis=AXIS_NAMES[s_axis], confidence=confidence, notes=notes,
        symmetry={AXIS_NAMES[a]: float(scores[a]) for a in range(3)},
    )
