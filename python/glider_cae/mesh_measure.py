"""Measure a glider from plane sections of its mesh (body frame: x forward, y right, z down).

Wing    - sections at constant y, tracked from the tip inward (the section nearest the previous
          one in chordwise position), so tailplane and fuselage are never mistaken for wing.
          Linear fits of chord, quarter-chord line and height give taper, sweep and dihedral;
          the planform is extended to the centreline because the fuselage hides the root.
Airfoil - mid-span sections, rotated so the chord line is horizontal: camber line and
          thickness against chord.
Tails   - everything behind the wing trailing edge: span and chord of the tailplane from
          sections at constant y, height and chord of the fin from the centre plane.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .mesh_geometry import TriangleMesh
from .mesh_io import MeshError

# np.trapezoid is NumPy >= 2.0; the browser's Pyodide still ships 1.26
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class WingMeasurement:
    span: float
    root_chord: float
    tip_chord: float
    area: float
    mac: float
    sweep_deg: float            # quarter-chord, + = swept back
    dihedral_deg: float
    incidence_deg: float
    camber: float
    thickness: float
    x_le_mac: float             # leading edge at the MAC station (body x, forward positive)
    x_qc_mac: float
    z_root: float               # wing plane height at the root (body z, down positive)
    x_te_min: float             # furthest-aft trailing edge
    stations: int
    notes: list[str] = field(default_factory=list)


@dataclass
class TailMeasurement:
    fuselage_length: float
    fuselage_diameter: float
    htail: dict | None          # span, chord, arm
    vtail: dict | None          # height, chord, arm, z_centre
    notes: list[str] = field(default_factory=list)


def _clusters(x: np.ndarray, gap: float) -> list[tuple[float, float]]:
    """Split sorted x values into groups separated by gaps larger than ``gap``: [(lo, hi), ...]."""
    xs = np.sort(x)
    breaks = np.flatnonzero(np.diff(xs) > gap)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(xs) - 1]])
    return [(float(xs[a]), float(xs[b])) for a, b in zip(starts, ends)]


def _section_metrics(x: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """(thickness/chord, camber/chord, incidence deg) of one dense airfoil section."""
    i_le, i_te = int(np.argmax(x)), int(np.argmin(x))
    dx, dz = x[i_te] - x[i_le], z[i_te] - z[i_le]            # LE -> TE (aft and, nose up, down)
    length = math.hypot(dx, dz)
    t = np.array([dx, dz]) / length                            # unit chord direction
    n = np.array([-dz, dx]) / length                           # chord normal ...
    if n[1] > 0:
        n = -n                                                 # ... pointing up (z is down)
    rel = np.stack([x - x[i_le], z - z[i_le]], 1)
    xi, eta = rel @ t / length, rel @ n / length               # chordwise / normal, in chord units
    bins = np.clip((xi * 20).astype(int), 0, 19)
    top = np.full(20, -np.inf)
    bottom = np.full(20, np.inf)
    np.maximum.at(top, bins, eta)
    np.minimum.at(bottom, bins, eta)
    mid = (np.arange(20) + 0.5) / 20
    ok = np.isfinite(top) & np.isfinite(bottom) & (mid > 0.04) & (mid < 0.96)
    if ok.sum() < 6:
        raise ValueError("section too sparse")
    thickness = float((top - bottom)[ok].max())
    camber = float(((top + bottom) / 2)[ok].max())
    incidence = math.degrees(math.atan2(dz, -dx))
    return thickness, max(camber, 0.0), incidence


def measure_wing(body: TriangleMesh) -> WingMeasurement:
    ys = body.vertices[:, 1]
    semi = 0.5 * float(ys.max() - ys.min())
    span = 2.0 * semi
    fracs = np.linspace(0.10, 0.95, 35)

    found: dict[float, list[tuple[float, float, float]]] = {}      # frac -> [(le, te, z_mean), ...]
    for side in (+1.0, -1.0):
        previous = None
        for frac in fracs[::-1]:                                   # tip -> root
            p = body.slice_points(1, side * frac * semi)
            if len(p) < 4:
                break
            x, z = p[:, 0], p[:, 2]
            if previous is None:
                le, te = float(x.max()), float(x.min())
            else:
                le0, te0 = previous
                c0 = le0 - te0
                best = None
                for lo, hi in _clusters(x, 0.35 * c0):
                    overlap = max(0.0, min(hi, le0) - max(lo, te0))
                    if overlap > 0.6 * c0 and 0.55 * c0 < hi - lo < 1.5 * c0:
                        if best is None or overlap > best[0]:
                            best = (overlap, lo, hi)
                if best is None:
                    break                                          # reached the fuselage
                te, le = best[1], best[2]
                x_sel = (x >= te - 1e-9) & (x <= le + 1e-9)
                z = z[x_sel]
            if le - te <= 1e-6:
                break
            found.setdefault(float(frac), []).append((le, te, float(z.mean())))
            previous = (le, te)

    if len(found) < 8:
        raise MeshError("Could not measure the wing: its sections could not be followed along the "
                        "span. Check the nose/up directions, or that the wing is a separate surface.")

    fr = np.array(sorted(found))
    y = fr * semi
    le = np.array([np.mean([s[0] for s in found[f]]) for f in fr])
    te = np.array([np.mean([s[1] for s in found[f]]) for f in fr])
    zm = np.array([np.mean([s[2] for s in found[f]]) for f in fr])
    chord = le - te

    # ---- planform: linear fit extended to the centreline, area from the measured chords
    slope_c, c0_fit = np.polyfit(y, chord, 1)
    c_tip_fit = max(slope_c * semi + c0_fit, 0.05 * c0_fit)
    inner = c0_fit * y[0] + slope_c * y[0] ** 2 / 2
    outer = c0_fit * (semi - y[-1]) + slope_c * (semi ** 2 - y[-1] ** 2) / 2
    area = 2.0 * (inner + _trapezoid(chord, y) + outer)
    k = area / (semi * (c0_fit + c_tip_fit))                        # area-preserving rescale
    root_chord, tip_chord = c0_fit * k, c_tip_fit * k
    taper = tip_chord / root_chord
    mac = 2.0 / 3.0 * root_chord * (1 + taper + taper ** 2) / (1 + taper)
    y_mac = span / 6.0 * (1 + 2 * taper) / (1 + taper)

    qc = le - 0.25 * chord
    sweep = math.degrees(math.atan(-np.polyfit(y, qc, 1)[0]))
    slope_z, z_root = np.polyfit(y, zm, 1)
    dihedral = math.degrees(math.atan(-slope_z))
    slope_le, le_root = np.polyfit(y, le, 1)
    x_le_mac = float(le_root + slope_le * y_mac)

    # ---- airfoil from mid-span sections
    metrics = []
    for f in fr[(fr >= 0.25) & (fr <= 0.70)][::3]:
        for side in (+1.0, -1.0):
            c_here = float(np.interp(f * semi, y, chord))
            pts = body.slice_points(1, side * f * semi, step=c_here / 60.0)
            lo, hi = float(np.interp(f * semi, y, te)) - 0.02 * c_here, float(np.interp(f * semi, y, le)) + 0.02 * c_here
            pts = pts[(pts[:, 0] >= lo) & (pts[:, 0] <= hi)]
            try:
                metrics.append(_section_metrics(pts[:, 0], pts[:, 2]))
            except (ValueError, IndexError):
                continue
    notes = []
    if metrics:
        thickness, camber, incidence = (float(np.median(c)) for c in zip(*metrics))
    else:
        thickness, camber, incidence = 0.09, 0.03, 0.0
        notes.append("The airfoil could not be measured; typical values (9% thick, 3% camber) were used.")

    return WingMeasurement(
        span=span, root_chord=float(root_chord), tip_chord=float(tip_chord), area=float(area),
        mac=float(mac), sweep_deg=sweep, dihedral_deg=dihedral, incidence_deg=incidence,
        camber=camber, thickness=thickness, x_le_mac=x_le_mac, x_qc_mac=x_le_mac - 0.25 * float(mac),
        z_root=float(z_root), x_te_min=float(te.min()), stations=len(fr), notes=notes)


def measure_tails_and_fuselage(body: TriangleMesh, wing: WingMeasurement) -> TailMeasurement:
    v = body.vertices
    x_min, x_max = float(v[:, 0].min()), float(v[:, 0].max())
    length = x_max - x_min
    notes: list[str] = []

    # ---- fuselage: height profile on the centre plane over the forward 60 % of the model
    centre = body.slice_points(1, 1e-4 * wing.span, step=length / 250.0)
    diameter = 0.07 * length
    if len(centre):
        bins = np.clip(((centre[:, 0] - x_min) / length * 60).astype(int), 0, 59)
        top = np.full(60, -np.inf)
        low = np.full(60, np.inf)
        np.maximum.at(top, bins, centre[:, 2])
        np.minimum.at(low, bins, centre[:, 2])
        height = top - low
        forward = (np.arange(60) + 0.5) / 60 > 0.4
        usable = height[forward & np.isfinite(height) & (height > 0)]
        if usable.size:
            diameter = float(np.percentile(usable, 90))
    diameter = float(np.clip(diameter, 0.02 * length, 0.25 * length))

    # ---- everything behind the wing trailing edge
    x_aft = wing.x_te_min - 0.10 * wing.mac
    aft = v[v[:, 0] < x_aft]
    htail = vtail = None
    if len(aft) >= 8:
        half_h = float(np.abs(aft[:, 1]).max())
        if 2 * half_h >= 0.12 * wing.span:
            chords, qcs = [], []
            for f in (0.3, 0.5, 0.7):
                for side in (+1.0, -1.0):
                    p = body.slice_points(1, side * f * half_h)
                    p = p[p[:, 0] < x_aft]
                    if len(p) >= 2:
                        chords.append(float(p[:, 0].max() - p[:, 0].min()))
                        qcs.append(float(p[:, 0].max()) - 0.25 * chords[-1])
            if chords:
                htail = {"span": 2 * half_h, "chord": float(np.mean(chords)),
                         "arm": wing.x_qc_mac - float(np.mean(qcs))}

        z_top = float(aft[:, 2].min())                                  # up is negative z
        h_v = wing.z_root - z_top
        if h_v >= 0.04 * wing.span:
            fin = centre[centre[:, 0] < x_aft]
            chords, qcs = [], []
            for f in (0.3, 0.5, 0.7):
                level = wing.z_root - f * h_v
                p = fin[np.abs(fin[:, 2] - level) < 0.06 * h_v]
                if len(p) >= 2:
                    chords.append(float(p[:, 0].max() - p[:, 0].min()))
                    qcs.append(float(p[:, 0].max()) - 0.25 * chords[-1])
            if len(chords) >= 2:
                vtail = {"height": h_v, "chord": float(np.mean(chords)),
                         "arm": wing.x_qc_mac - float(np.mean(qcs)), "z_centre": wing.z_root - 0.5 * h_v}

    if htail is None:
        notes.append("No horizontal tail found behind the wing; it was sized from tail-volume rules.")
    if vtail is None:
        notes.append("No fin found behind the wing; it was sized from tail-volume rules.")
    return TailMeasurement(length, diameter, htail, vtail, notes)
