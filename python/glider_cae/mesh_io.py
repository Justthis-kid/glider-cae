"""Reading meshes (STL / OBJ and, through trimesh, many more), unit handling, preview packing.

``trimesh`` is the primary loader. If it is not installed (for example the browser could not
fetch it) a small built-in reader handles STL and OBJ so the app keeps working. Everything
downstream only sees plain NumPy arrays (``MeshData``).
"""
from __future__ import annotations

import base64
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_FACES = 500_000
UNIT_SCALES = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}


class MeshError(ValueError):
    """A problem with the user's mesh, worded so it can be shown directly in the UI."""


@dataclass
class MeshData:
    vertices: np.ndarray        # (N, 3) float64, in the file's own units
    faces: np.ndarray           # (M, 3) int64
    name: str = ""
    loader: str = ""


# ---------------------------------------------------------------------------- built-in readers
def _merge_vertices(tri_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(3M, 3) triangle-soup vertices -> shared vertex list and (M, 3) faces."""
    extent = float(np.ptp(tri_vertices, axis=0).max()) or 1.0
    key = np.round(tri_vertices / (extent * 1e-7)).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return tri_vertices[first], inverse.reshape(-1, 3)


def read_stl(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Binary or ASCII STL."""
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n:                                    # binary
            rec = np.frombuffer(data, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")],
                                count=n, offset=84)
            return _merge_vertices(rec["v"].reshape(-1, 3).astype(np.float64))
    text = data.decode("utf-8", errors="ignore")
    found = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not found or len(found) % 3:
        raise MeshError("This STL file has no readable triangles.")
    return _merge_vertices(np.array(found, dtype=np.float64))


def read_obj(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Wavefront OBJ: vertices and (triangulated) faces; textures and normals are ignored."""
    verts, faces = [], []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            verts.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("f "):
            idx = []
            for token in line.split()[1:]:
                i = int(token.split("/")[0])
                idx.append(i - 1 if i > 0 else len(verts) + i)         # negative = relative
            faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    if not verts or not faces:
        raise MeshError("This OBJ file has no vertices or faces.")
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


# ---------------------------------------------------------------------------- public loader
def load_mesh(path: str) -> MeshData:
    """Load a mesh file. Uses trimesh when available, else the built-in STL / OBJ readers."""
    p = Path(path)
    suffix = p.suffix.lower()
    try:
        import trimesh                                                  # noqa: PLC0415
    except ImportError:
        trimesh = None

    if trimesh is not None:
        try:
            loaded = trimesh.load(str(p), force="mesh")
            vertices = np.asarray(loaded.vertices, dtype=np.float64)
            faces = np.asarray(loaded.faces, dtype=np.int64)
            loader = f"trimesh {getattr(trimesh, '__version__', '')}".strip()
        except Exception as err:                                        # noqa: BLE001
            raise MeshError(f"Could not read this mesh: {err}") from err
    elif suffix in (".stl", ".obj"):
        try:
            data = p.read_bytes()
        except OSError as err:
            raise MeshError(f"Could not open the file: {err}") from err
        if suffix == ".stl":
            (vertices, faces), loader = read_stl(data), "built-in STL reader"
        else:
            (vertices, faces), loader = read_obj(data), "built-in OBJ reader"
    else:
        raise MeshError(f"'{suffix or 'this file type'}' needs the trimesh package; use STL or OBJ.")

    if len(faces) == 0 or len(vertices) < 4:
        raise MeshError("The mesh is empty: it has no triangles.")
    if len(faces) > MAX_FACES:
        raise MeshError(f"The mesh has {len(faces):,} triangles; the limit is {MAX_FACES:,}. "
                        "Simplify it in your modelling tool and try again.")
    if not np.all(np.isfinite(vertices)):
        raise MeshError("The mesh contains invalid (NaN or infinite) coordinates.")
    return MeshData(vertices, faces, p.name, loader)


# ---------------------------------------------------------------------------- units
def suggest_units(vertices: np.ndarray) -> str:
    """Guess the file's length unit from the largest dimension (a glider is roughly 0.3 to 6 m)."""
    largest = float(np.ptp(vertices, axis=0).max())
    if largest < 12.0:
        return "m"
    return "cm" if largest < 400.0 else "mm"


# ---------------------------------------------------------------------------- preview for the browser
def decimate(vertices: np.ndarray, faces: np.ndarray, max_faces: int):
    """Vertex-clustering decimation: snap vertices to a grid and merge, growing the cell until
    the face count fits. Only for display; analysis always uses the full mesh."""
    if len(faces) <= max_faces:
        return vertices, faces
    lo, hi = vertices.min(0), vertices.max(0)
    cell = float(np.linalg.norm(hi - lo)) / 220.0
    for _ in range(12):
        cells = np.floor((vertices - lo) / cell).astype(np.int64)
        _, cluster = np.unique(cells, axis=0, return_inverse=True)
        cluster = cluster.reshape(-1)
        count = np.bincount(cluster).astype(float)
        merged = np.stack([np.bincount(cluster, vertices[:, k]) / count for k in range(3)], 1)
        f = cluster[faces]
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        if keep.sum() <= max_faces:
            return merged, f[keep]
        cell *= 1.35
    return merged, f[keep]


def pack_preview(vertices: np.ndarray, faces: np.ndarray, max_faces: int = 30_000) -> dict:
    """Float32 vertices + uint32 faces as one base64 blob (plain JSON-safe string)."""
    v, f = decimate(vertices, faces, max_faces)
    blob = np.asarray(v, "<f4").tobytes() + np.asarray(f, "<u4").tobytes()
    return {"vertices": int(len(v)), "faces": int(len(f)), "data": base64.b64encode(blob).decode("ascii")}
