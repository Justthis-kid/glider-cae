"""Thin JSON bridge used by the browser (Pyodide): plain numbers in, JSON strings out.

Module state: the active simulation, the geometry behind it, and (after a mesh upload) the
parsed mesh so options such as units or nose direction can be changed without re-uploading.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from .dynamics import Controls
from .geometry import GliderGeometry
from .mesh_analysis import MeshAnalysis, MeshOptions, analyze_mesh
from .mesh_io import MeshData, MeshError, load_mesh as read_mesh_file, pack_preview
from .simulation import GliderSimulation

_sim: GliderSimulation | None = None
_geometry: GliderGeometry | None = None
_mesh: MeshData | None = None
_mesh_summary: dict | None = None
_mesh_preview: dict | None = None


# ---------------------------------------------------------------------------- analysis payloads
def _mesh_info(sim: GliderSimulation, include_preview: bool) -> dict | None:
    if _mesh_summary is None:
        return None
    d = sim.aero.d
    info = dict(_mesh_summary)
    # Where the CG sits in the mesh's body frame: behind the MAC leading edge by h chords.
    info["cg_body"] = [info["x_le_mac"] - d.cg * d.mac, 0.0, info["z_centroid"]]
    info["cg_chosen_frac"] = d.cg
    info["cg_note"] = None
    if abs(d.cg - info["cg_mesh_frac"]) > 0.08:
        info["cg_note"] = (f"The mesh's own centroid is at {100 * info['cg_mesh_frac']:.0f}% of the chord; "
                           f"the CG used is at {100 * d.cg:.0f}%. Real gliders carry nose ballast, "
                           "so a surface-mass centroid is usually too far aft.")
    if include_preview:
        info["preview"] = _mesh_preview
    return info


def _analysis(sim: GliderSimulation, include_preview: bool = False) -> dict:
    est = sim.aero
    return {
        "source": "mesh" if _mesh_summary else "default",
        "geometry": sim.geometry.to_dict(),
        "report": est.report(),
        "polar": {k: np.asarray(v).tolist() for k, v in est.polar().items()},
        "loading": {k: np.asarray(v).tolist() for k, v in est.spanwise_loading().items()},
        "mesh": _mesh_info(sim, include_preview),
    }


def _install(geometry: GliderGeometry, include_preview: bool = False) -> str:
    """Estimate aerodynamics for ``geometry`` and make it the active simulation."""
    global _sim, _geometry
    new = GliderSimulation.from_geometry(geometry)
    _sim, _geometry = new, geometry              # replaced only once estimation succeeded
    return json.dumps({"state": new.snapshot(), "analysis": _analysis(new, include_preview)})


# ---------------------------------------------------------------------------- lifecycle
def init() -> str:
    """Start (or go back to) the built-in glider."""
    global _mesh, _mesh_summary, _mesh_preview
    _mesh = _mesh_summary = _mesh_preview = None
    return _install(GliderGeometry().with_sized_tails())


use_default = init


def configure(params_json: str) -> str:
    """Edit the active geometry. Fields that are not sent (or are null) keep their value; mass
    changes scale a mesh-derived inertia. ``auto_tail`` re-sizes the tails for the new wing.
    Returns {"state", "analysis"} or {"error": message} - never raises into JavaScript."""
    try:
        params = json.loads(params_json)
        known = set(GliderGeometry.scalar_fields())
        updates = {k: v for k, v in params.items() if k in known and v is not None}
        new = replace(_geometry, **updates)
        if "mass" in updates and _geometry.inertia is not None and _geometry.mass > 0:
            new = replace(new, inertia=np.asarray(_geometry.inertia) * (new.mass / _geometry.mass))
        auto_tail = bool(params.get("auto_tail", False))
        problems = new.validate(tails=not auto_tail)
        if problems:
            raise ValueError("; ".join(problems))
        if auto_tail:
            new = new.with_sized_tails()
        return _install(new)
    except (ValueError, TypeError, KeyError) as err:
        return json.dumps({"error": str(err)})


# ---------------------------------------------------------------------------- mesh
def _apply_mesh(options: dict) -> str:
    global _mesh_summary, _mesh_preview
    result: MeshAnalysis = analyze_mesh(_mesh, MeshOptions.from_dict(options))
    summary = result.summary(_mesh.name, _mesh.loader)
    preview = pack_preview(result.body_vertices, result.faces)
    out = _install_mesh(result, summary, preview)
    return out


def _install_mesh(result: MeshAnalysis, summary: dict, preview: dict) -> str:
    global _mesh_summary, _mesh_preview
    previous = (_mesh_summary, _mesh_preview)
    _mesh_summary, _mesh_preview = summary, preview
    try:
        return _install(result.geometry, include_preview=True)
    except Exception:
        _mesh_summary, _mesh_preview = previous           # keep the old state if estimation fails
        raise


def load_mesh(path: str, options_json: str) -> str:
    """Read an uploaded file (already written to Pyodide's file system) and analyse it."""
    global _mesh
    try:
        options = json.loads(options_json) if options_json else {}
        mesh = read_mesh_file(path)
        mesh.name = options.get("name") or mesh.name
        previous, _mesh = _mesh, mesh
        try:
            return _apply_mesh(options)
        except Exception:
            _mesh = previous
            raise
    except (MeshError, ValueError) as err:
        return json.dumps({"error": str(err)})


def reanalyze(options_json: str) -> str:
    """Re-run the analysis of the loaded mesh with new options (units, axes, CG source...)."""
    if _mesh is None:
        return json.dumps({"error": "No mesh is loaded."})
    try:
        return _apply_mesh(json.loads(options_json) if options_json else {})
    except (MeshError, ValueError) as err:
        return json.dumps({"error": str(err)})


# ---------------------------------------------------------------------------- simulation
def reset(altitude: float, heading_deg: float, airspeed: float) -> str:
    """airspeed <= 0 means 'use the trimmed glide speed'."""
    _sim.reset(altitude, heading_deg, airspeed if airspeed > 0 else None)
    return snapshot()


def advance(dt, time_scale, elevator, aileron, rudder, wind_n, wind_e, wind_d) -> str:
    _sim.controls = Controls(float(elevator), float(aileron), float(rudder))
    _sim.env.wind_ned = np.array([wind_n, wind_e, wind_d], dtype=float)
    _sim.advance(float(dt), float(time_scale))
    return snapshot()


def snapshot() -> str:
    return json.dumps(_sim.snapshot())
