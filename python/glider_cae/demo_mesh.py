"""Synthetic sailplane meshes with known dimensions, plus STL / OBJ writers.

Used by the tests (ground truth to measure the mesh parser against), by the sample-file
script, and handy for trying the web app without a model of your own.

Body frame while building: x forward, y right, z down (the simulator's FRD axes).
Wing root quarter-chord sits at the origin.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _naca4_ring(n: int, camber: float, thickness: float, camber_pos: float = 0.4):
    """Closed airfoil outline as (xi, z_up) in chord units: upper surface LE->TE, then lower."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    yt = 5.0 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                            + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    p = camber_pos
    if camber > 0.0:
        yc = np.where(x < p, camber / p ** 2 * (2 * p * x - x ** 2),
                      camber / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * camber / p ** 2 * (p - x), 2 * camber / (1 - p) ** 2 * (p - x))
    else:
        yc = dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu, zu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, zl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return (np.concatenate([xu, xl[-2:0:-1]]), np.concatenate([zu, zl[-2:0:-1]]))


def _loft(sections: np.ndarray, cap: bool = True):
    """Skin a stack of rings (S, R, 3) into triangles; optionally close both ends with fans."""
    s, r, _ = sections.shape
    verts = [sections.reshape(-1, 3)]
    ring = np.arange(r)
    nxt = (ring + 1) % r
    faces = []
    for k in range(s - 1):
        a, b = k * r + ring, k * r + nxt
        c, d = (k + 1) * r + nxt, (k + 1) * r + ring
        faces += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    count = s * r
    if cap:
        for k in (0, s - 1):
            verts.append(sections[k].mean(0, keepdims=True))
            faces.append(np.stack([k * r + ring, k * r + nxt, np.full(r, count)], 1))
            count += 1
    return np.concatenate(verts), np.concatenate(faces)


def _merge(parts):
    verts, faces, offset = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def make_demo_glider(
    span=2.0, root_chord=0.20, tip_chord=0.12, sweep_deg=4.0, dihedral_deg=5.0,
    incidence_deg=2.0, camber=0.03, thickness=0.09,
    fuselage_length=1.1, fuselage_diameter=0.09,
    htail_span=0.40, htail_chord=0.08, vtail_height=0.22, vtail_chord=0.14, tail_arm=0.68,
):
    """Return (vertices, faces, truth). ``truth`` holds the dimensions the mesh was built from."""
    semi, lam = span / 2.0, tip_chord / root_chord
    tan_sw, tan_di = math.tan(math.radians(sweep_deg)), math.tan(math.radians(dihedral_deg))
    inc = math.radians(incidence_deg)

    # ---- wing: 13 spanwise sections, root quarter-chord at the origin
    xi, zu = _naca4_ring(26, camber, thickness)
    secs = []
    for y in np.linspace(-semi, semi, 13):
        a = abs(y)
        c = root_chord * (1.0 - (1.0 - lam) * a / semi)
        le = -a * tan_sw + 0.25 * c                          # quarter-chord line swept aft
        u, w = -xi * c, -zu * c                              # aft / down offsets from the LE
        u2, w2 = u * math.cos(inc) + w * math.sin(inc), -u * math.sin(inc) + w * math.cos(inc)
        secs.append(np.stack([le + u2, np.full_like(u2, y), -a * tan_di + w2], 1))
    wing = _loft(np.array(secs))

    # ---- fuselage: nose ellipsoid, taper to a boom
    nose_x = 0.30
    s = np.linspace(0.004, 1.0, 50)
    r_max = fuselage_diameter / 2.0
    r_boom = 0.015
    rad = np.where(s < 0.30, r_max * np.sqrt(np.clip(1 - ((0.30 - s) / 0.30) ** 2, 0, 1)),
                   np.where(s < 0.55, r_max + (r_boom - r_max) * (s - 0.30) / 0.25, r_boom))
    phi = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    rings = [np.stack([np.full(20, nose_x - si * fuselage_length), ri * np.cos(phi), ri * np.sin(phi)], 1)
             for si, ri in zip(s, rad)]
    fuselage = _loft(np.array(rings))

    # ---- horizontal tail (symmetric section, on the boom axis)
    txi, tzu = _naca4_ring(14, 0.0, 0.08)
    x_tail_qc = -tail_arm
    ring_t = []
    for y in np.linspace(-htail_span / 2, htail_span / 2, 5):
        ring_t.append(np.stack([x_tail_qc + 0.25 * htail_chord - txi * htail_chord,
                                np.full_like(txi, y), -tzu * htail_chord], 1))
    htail = _loft(np.array(ring_t))

    # ---- fin: sections stacked upward, thickness in y, 30 % taper, 15 deg quarter-chord sweep
    fin_sections = []
    for h in np.linspace(0.0, vtail_height, 5):
        c = vtail_chord * (1.0 - 0.3 * h / vtail_height) / 0.85       # so mid-height chord = vtail_chord
        qc = x_tail_qc - (h - 0.5 * vtail_height) * math.tan(math.radians(15.0))
        fin_sections.append(np.stack([qc + 0.25 * c - txi * c, tzu * c, np.full_like(txi, -h)], 1))
    fin = _loft(np.array(fin_sections))

    verts, faces = _merge([wing, fuselage, htail, fin])

    y_mac = (span / 6.0) * (1 + 2 * lam) / (1 + lam)
    area = span * (root_chord + tip_chord) / 2.0
    truth = dict(
        span=span, root_chord=root_chord, tip_chord=tip_chord, area=area,
        mac=2 / 3 * root_chord * (1 + lam + lam ** 2) / (1 + lam),
        sweep_deg=sweep_deg, dihedral_deg=dihedral_deg, incidence_deg=incidence_deg,
        camber=camber, thickness=thickness,
        fuselage_length=fuselage_length, fuselage_diameter=fuselage_diameter,
        htail_span=htail_span, htail_chord=htail_chord, vtail_height=vtail_height,
        vtail_chord=vtail_chord,
        x_wing_qc_mac=-y_mac * tan_sw,
        htail_arm=-y_mac * tan_sw + tail_arm,
        vtail_arm=-y_mac * tan_sw + tail_arm,       # fin quarter-chord at mid-height is at -tail_arm
    )
    return verts, faces, truth


# ---------------------------------------------------------------------------- orientation helpers
def reorient(vertices: np.ndarray, perm=(0, 1, 2), signs=(1, 1, 1)) -> np.ndarray:
    """Re-express body coordinates in another axis arrangement: out[:, i] = signs[i] * v[:, perm[i]]."""
    return vertices[:, list(perm)] * np.asarray(signs, float)


# ---------------------------------------------------------------------------- file writers
def write_stl_binary(path, vertices, faces) -> None:
    tri = np.asarray(vertices, np.float32)[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = normal, tri
    with open(path, "wb") as fh:
        fh.write(b"glider_cae demo".ljust(80, b" "))
        fh.write(np.uint32(len(faces)).tobytes())
        fh.write(rec.tobytes())


def write_stl_ascii(path, vertices, faces) -> None:
    lines = ["solid demo"]
    for t in np.asarray(vertices)[faces]:
        lines.append("facet normal 0 0 0\n outer loop")
        lines += [f"  vertex {v[0]:.7g} {v[1]:.7g} {v[2]:.7g}" for v in t]
        lines.append(" endloop\nendfacet")
    lines.append("endsolid demo")
    Path(path).write_text("\n".join(lines))


def write_obj(path, vertices, faces) -> None:
    out = [f"v {v[0]:.7g} {v[1]:.7g} {v[2]:.7g}" for v in vertices]
    out += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    Path(path).write_text("\n".join(out))
