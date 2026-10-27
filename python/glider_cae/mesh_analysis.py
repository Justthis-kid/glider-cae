"""Stage 3 - from a mesh to a ``GliderGeometry``.

    load_mesh()  ->  units  ->  orientation  ->  wing / airfoil / fuselage / tails
                 ->  centroid and inertia  ->  GliderGeometry  (+ notes and warnings)

The result plugs straight into stage 2 (``LiftingLineEstimator``) and the simulator.
Everything it could not measure falls back to a documented rule and says so in ``notes``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .geometry import GliderGeometry
from .mesh_geometry import Orientation, TriangleMesh, detect_orientation
from .mesh_io import UNIT_SCALES, MeshData, MeshError, suggest_units
from .mesh_mass import MassProperties, mass_properties
from .mesh_measure import (TailMeasurement, WingMeasurement, measure_tails_and_fuselage,
                           measure_wing)

WING_LOADING_GUESS = 3.5        # kg/m^2, used when the user gives no mass


@dataclass
class MeshOptions:
    units: str = "auto"                     # auto | m | cm | mm | in | ft
    forward: str | None = None              # e.g. "-Y"; None = detect
    up: str | None = None                   # e.g. "+Z"; None = detect
    mass: float | None = None               # kg; None = estimate from wing area
    cg_mode: str = "static_margin"          # static_margin | mesh | manual
    cg_manual: float = 0.30                 # fraction of the MAC, for cg_mode "manual"
    static_margin_target: float = 0.10
    mass_model: str = "shell"               # shell | volume

    @classmethod
    def from_dict(cls, data: dict) -> "MeshOptions":
        known = cls.__dataclass_fields__
        clean = {k: v for k, v in data.items() if k in known and v not in (None, "", "auto")}
        if data.get("units") == "auto":
            clean["units"] = "auto"
        return cls(**clean)


@dataclass
class MeshAnalysis:
    geometry: GliderGeometry
    orientation: Orientation
    wing: WingMeasurement
    tails: TailMeasurement
    mass_props: MassProperties
    units: str
    units_assumed: bool
    mass_assumed: bool
    cg_mesh_frac: float                      # where the mesh's own centroid sits, fraction of the MAC
    body_vertices: np.ndarray                # oriented, centred, metres, FRD
    faces: np.ndarray
    extents: np.ndarray
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def centroid_body(self) -> np.ndarray:
        return self.mass_props.centroid

    def summary(self, name: str = "", loader: str = "") -> dict:
        o = self.orientation
        return {
            "name": name, "loader": loader, "units": self.units, "units_assumed": self.units_assumed,
            "vertices": int(len(self.body_vertices)), "faces": int(len(self.faces)),
            "watertight": self.mass_props.watertight, "mass_model": self.mass_props.model,
            "mass_assumed": self.mass_assumed, "extents": self.extents.tolist(),
            "orientation": {"forward": o.forward, "up": o.up, "span_axis": o.span_axis,
                            "confidence": o.confidence, "symmetry": o.symmetry},
            "cg_mesh_frac": self.cg_mesh_frac, "x_le_mac": self.wing.x_le_mac,
            "z_centroid": float(self.mass_props.centroid[2]),
            "notes": self.notes, "warnings": self.warnings,
        }


def _clip(name: str, value: float, lo: float, hi: float, notes: list[str]) -> float:
    if value < lo or value > hi:
        notes.append(f"{name} measured as {value:.3g}; limited to {lo:g} to {hi:g}.")
    return float(min(max(value, lo), hi))


def analyze_mesh(mesh: MeshData, options: MeshOptions | None = None) -> MeshAnalysis:
    opts = options or MeshOptions()
    notes: list[str] = []
    warnings: list[str] = []

    # ---- units
    units_assumed = opts.units == "auto"
    units = suggest_units(mesh.vertices) if units_assumed else opts.units
    if units not in UNIT_SCALES:
        raise MeshError(f"Unknown unit '{units}'. Use m, cm, mm, in or ft.")
    if units_assumed:
        notes.append(f"Assumed the model is in {units}; change the units if the size looks wrong.")
    vertices = mesh.vertices * UNIT_SCALES[units]

    # ---- orientation into body axes (x forward, y right, z down), centred on the bounding box
    try:
        orientation = detect_orientation(TriangleMesh(vertices, mesh.faces), opts.forward, opts.up)
    except ValueError as err:
        raise MeshError(str(err)) from err
    notes += orientation.notes
    centre = 0.5 * (vertices.min(0) + vertices.max(0))
    body_vertices = (vertices - centre) @ orientation.rotation.T
    body = TriangleMesh(body_vertices, mesh.faces)

    # ---- measurements
    wing = measure_wing(body)
    tails = measure_tails_and_fuselage(body, wing)
    notes += wing.notes + tails.notes
    mp = mass_properties(body_vertices, mesh.faces, opts.mass_model)
    if opts.mass_model == "volume" and mp.model != "volume":
        notes.append("The mesh is not watertight, so a hollow-shell mass model was used.")

    if not 0.15 <= wing.span <= 30.0:
        warnings.append(f"The wing span came out as {wing.span:.2f} m. Check the units setting.")

    # ---- assemble the geometry (everything clipped into the estimator's supported range)
    mass_assumed = opts.mass is None
    mass = opts.mass if opts.mass else WING_LOADING_GUESS * wing.area
    if mass_assumed:
        notes.append(f"Mass estimated at {mass:.2f} kg ({WING_LOADING_GUESS} kg/m2 wing loading); enter the real mass.")

    taper = _clip("Taper ratio", wing.tip_chord / wing.root_chord, 0.05, 1.5, notes)
    base = GliderGeometry(
        span=wing.span, root_chord=wing.root_chord, tip_chord=taper * wing.root_chord,
        sweep_deg=_clip("Sweep", wing.sweep_deg, -40, 40, notes),
        dihedral_deg=_clip("Dihedral", wing.dihedral_deg, -20, 20, notes),
        incidence_deg=wing.incidence_deg,
        camber=_clip("Camber", wing.camber, 0.0, 0.08, notes),
        thickness=_clip("Thickness", wing.thickness, 0.04, 0.20, notes),
        fuselage_length=tails.fuselage_length,
        fuselage_diameter=min(tails.fuselage_diameter, tails.fuselage_length / 2.6),
        mass=mass, static_margin_target=opts.static_margin_target,
        inertia=mass * mp.inertia_unit,
    )

    sized = base.with_sized_tails()
    h, v = tails.htail, tails.vtail
    if h is not None and h["arm"] < 1.0 * wing.mac:
        notes.append("The horizontal tail sits too close to the wing to be a tailplane; it was re-sized.")
        h = None
    if v is not None and v["arm"] < 1.0 * wing.mac:
        notes.append("The fin sits too close to the wing; it was re-sized.")
        v = None
    geometry = base
    if h is not None:
        geometry = replace(geometry, htail_span=h["span"], htail_chord=h["chord"], htail_arm=h["arm"])
    else:
        geometry = replace(geometry, htail_span=sized.htail_span, htail_chord=sized.htail_chord,
                           htail_arm=sized.htail_arm)
    if v is not None:
        z_cg = float(mp.centroid[2])
        geometry = replace(geometry, vtail_height=v["height"], vtail_chord=v["chord"], vtail_arm=v["arm"],
                           vtail_height_above_cg=max(z_cg - v["z_centre"], 0.25 * v["height"]))
    else:
        geometry = replace(geometry, vtail_height=sized.vtail_height, vtail_chord=sized.vtail_chord,
                           vtail_arm=sized.vtail_arm, vtail_height_above_cg=sized.vtail_height_above_cg)

    # ---- centre of gravity
    cg_mesh = float((wing.x_le_mac - mp.centroid[0]) / wing.mac)
    if opts.cg_mode == "mesh":
        geometry = replace(geometry, cg_frac_mac=_clip("Mesh CG position", cg_mesh, 0.0, 0.8, warnings))
    elif opts.cg_mode == "manual":
        geometry = replace(geometry, cg_frac_mac=_clip("CG position", opts.cg_manual, 0.0, 0.8, warnings))
    elif opts.cg_mode != "static_margin":
        raise MeshError(f"Unknown CG mode '{opts.cg_mode}'.")

    problems = geometry.validate()
    if problems:
        raise MeshError("The measured geometry is outside the supported range: " + "; ".join(problems) + ".")

    if orientation.confidence != "high":
        warnings.append(f"Orientation confidence is {orientation.confidence}: nose {orientation.forward}, "
                        f"up {orientation.up}. Set them by hand if this is wrong.")
    return MeshAnalysis(
        geometry=geometry, orientation=orientation, wing=wing, tails=tails, mass_props=mp,
        units=units, units_assumed=units_assumed, mass_assumed=mass_assumed, cg_mesh_frac=cg_mesh,
        body_vertices=body_vertices, faces=mesh.faces, extents=np.ptp(vertices, axis=0),
        notes=notes, warnings=warnings,
    )
