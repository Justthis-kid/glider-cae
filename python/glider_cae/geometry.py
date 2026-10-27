"""Geometric description of a glider - the contract between stage 3 and stage 2.

Stage 3 (mesh parser) will measure these quantities from a 3-D mesh; stage 2 only
needs them to estimate aerodynamics. ``GliderGeometry.from_bounding_box`` is the
bridge for when only the mesh extents are known: it assumes a conventional
tapered-wing / boom / T-tail sailplane and sizes the tail with classical tail
volume coefficients.

Lengths in metres, angles in degrees, mass in kg. Longitudinal arms are measured
from the wing quarter-chord point of the mean aerodynamic chord (MAC) to the
aerodynamic centre of the tail surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np

from .config import GliderConfig

TAIL_VOLUME_H = 0.40     # V_H = S_h l_h / (S c_mac)   - typical sailplane 0.35 - 0.55
TAIL_VOLUME_V = 0.035    # V_V = S_v l_v / (S b)        - typical sailplane 0.02 - 0.04
GYRATION_ROLL = 0.30     # radius of gyration / (b/2)  - roll
GYRATION_PITCH = 0.40    # radius of gyration / (L/2)  - pitch


@dataclass
class GliderGeometry:
    # ---- wing
    span: float = 2.0
    root_chord: float = 0.20
    tip_chord: float = 0.12
    sweep_deg: float = 0.0               # quarter-chord sweep, + = swept back
    dihedral_deg: float = 6.0            # per side
    incidence_deg: float = 0.0           # wing setting angle w.r.t. the body x-axis
    camber: float = 0.03                 # max camber / chord of the section
    thickness: float = 0.09              # t/c
    cl_max_section: float | None = None  # None -> estimated from camber

    # ---- fuselage (equivalent body of revolution)
    fuselage_length: float = 1.1
    fuselage_diameter: float = 0.09

    # ---- horizontal tail
    htail_span: float = 0.40
    htail_chord: float = 0.08
    htail_arm: float = 0.66

    # ---- vertical tail
    vtail_height: float = 0.22
    vtail_chord: float = 0.14
    vtail_arm: float = 0.66
    vtail_height_above_cg: float = 0.10

    # ---- control surfaces
    elevator_chord_ratio: float = 0.35
    aileron_chord_ratio: float = 0.25
    aileron_span_inner: float = 0.55     # as a fraction of the semi-span
    aileron_span_outer: float = 0.95
    rudder_chord_ratio: float = 0.35
    elevator_travel_deg: float = 5.0     # full-stick deflection (kept modest: a stable
                                         # glider pitches a long way per degree of elevator)
    aileron_travel_deg: float = 12.0
    rudder_travel_deg: float = 20.0

    # ---- mass properties and trim
    mass: float = 1.2
    cg_frac_mac: float | None = None     # CG station as a fraction of the MAC; None -> auto
    static_margin_target: float = 0.10   # used only when cg_frac_mac is None
    tail_incidence_deg: float | None = None  # None -> set so Cm = 0 at design_cl
    design_cl: float = 0.6
    inertia: np.ndarray | None = None    # 3x3 about the CG; None -> radius-of-gyration estimate

    # ------------------------------------------------------------------ derived
    @property
    def taper(self) -> float:
        return self.tip_chord / self.root_chord

    @property
    def area(self) -> float:
        """Planform area S = b (c_root + c_tip) / 2."""
        return self.span * (self.root_chord + self.tip_chord) / 2.0

    @property
    def aspect_ratio(self) -> float:
        return self.span ** 2 / self.area

    @property
    def mac(self) -> float:
        """c_mac = (2/3) c_root (1 + l + l^2) / (1 + l),  l = taper ratio."""
        lam = self.taper
        return 2.0 / 3.0 * self.root_chord * (1.0 + lam + lam * lam) / (1.0 + lam)

    @property
    def htail_area(self) -> float:
        return self.htail_span * self.htail_chord

    @property
    def vtail_area(self) -> float:
        return self.vtail_height * self.vtail_chord

    # ------------------------------------------------------------------ helpers
    def validate(self, tails: bool = True) -> list[str]:
        """Return human-readable problems (empty list = OK). ``tails=False`` skips the tail
        checks, for vetting user input before the tails are sized automatically."""
        g, bad = self, []
        positive = ("span", "root_chord", "tip_chord", "fuselage_length", "fuselage_diameter", "mass")
        if tails:
            positive += ("htail_span", "htail_chord", "htail_arm", "vtail_height", "vtail_chord", "vtail_arm")
        for name in positive:
            if not getattr(g, name) > 0:
                bad.append(f"{name} must be greater than zero")
        if bad:
            return bad
        if not 0.05 <= g.taper <= 1.5:
            bad.append("tip chord / root chord must be between 0.05 and 1.5")
        if not 3.0 <= g.aspect_ratio <= 40.0:
            bad.append(f"wing aspect ratio {g.aspect_ratio:.1f} is outside the supported 3 to 40")
        if abs(g.dihedral_deg) > 20 or abs(g.sweep_deg) > 40:
            bad.append("dihedral must be within 20 degrees and sweep within 40 degrees")
        if not 0.0 <= g.camber <= 0.08 or not 0.04 <= g.thickness <= 0.20:
            bad.append("camber must be 0 to 8% and thickness 4 to 20% of chord")
        if g.fuselage_length / g.fuselage_diameter < 2.5:
            bad.append("fuselage fineness ratio (length / diameter) must be at least 2.5")
        if not 0.2 <= g.design_cl <= 1.0:
            bad.append("design lift coefficient must be between 0.2 and 1.0")
        if g.cg_frac_mac is None and not 0.0 <= g.static_margin_target <= 0.4:
            bad.append("static margin target must be between 0% and 40%")
        if not 0.1 <= g.aileron_span_inner < g.aileron_span_outer <= 1.0:
            bad.append("aileron span range must satisfy 0.1 <= inner < outer <= 1.0")
        for name in ("elevator_chord_ratio", "aileron_chord_ratio", "rudder_chord_ratio"):
            if not 0.1 <= getattr(g, name) <= 0.6:
                bad.append(f"{name} must be between 0.1 and 0.6")
        for name in ("elevator_travel_deg", "aileron_travel_deg", "rudder_travel_deg"):
            if not 1.0 <= getattr(g, name) <= 40.0:
                bad.append(f"{name} must be between 1 and 40 degrees")
        if tails and (g.htail_arm < g.mac or g.vtail_arm < g.mac):
            bad.append("tail arms are shorter than the wing chord; the tails would sit on the wing")
        return bad

    def estimate_inertia(self) -> np.ndarray:
        """Principal inertia from radii of gyration (Raymer-style), Izz ~ Ixx + Iyy (thin body)."""
        ixx = self.mass * (GYRATION_ROLL * self.span / 2.0) ** 2
        iyy = self.mass * (GYRATION_PITCH * self.fuselage_length / 2.0) ** 2
        return np.diag([ixx, iyy, ixx + iyy])

    def to_config(self) -> GliderConfig:
        inertia = self.estimate_inertia() if self.inertia is None else np.asarray(self.inertia, float)
        return GliderConfig(mass=self.mass, inertia=inertia, span=self.span,
                            mean_chord=self.mac, wing_area=self.area)

    # ------------------------------------------------------------------ (de)serialisation

    @classmethod
    def scalar_fields(cls) -> list[str]:
        return [f.name for f in fields(cls) if f.name != "inertia"]

    @classmethod
    def from_dict(cls, data: dict) -> "GliderGeometry":
        """Build from a plain dict (e.g. JSON). Unknown keys and None values are ignored, so a
        blank form field simply keeps its default."""
        known = set(cls.scalar_fields())
        return cls(**{k: v for k, v in data.items() if k in known and v is not None})

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.scalar_fields()}

    # ------------------------------------------------------------------ mesh-extent constructor
    @classmethod
    def from_bounding_box(
        cls,
        span: float,
        length: float,
        height: float,
        wing_area: float | None = None,
        mass: float = 1.2,
        taper: float = 0.6,
        **overrides,
    ) -> "GliderGeometry":
        """Conventional sailplane inferred from the mesh bounding box.

        span / length / height : bounding-box extents (m)
        wing_area              : planform area if known (stage 3 can measure it),
                                 otherwise AR = 12 is assumed
        """
        area = wing_area if wing_area is not None else span ** 2 / 12.0
        root = 2.0 * area / (span * (1.0 + taper))
        base = cls(span=span, root_chord=root, tip_chord=taper * root,
                   fuselage_length=length,
                   fuselage_diameter=float(np.clip(0.5 * height, 0.04 * length, 0.12 * length)),
                   mass=mass)
        sized = base.with_sized_tails()
        return replace(sized, **overrides) if overrides else sized

    def with_sized_tails(self, v_h: float = TAIL_VOLUME_H, v_v: float = TAIL_VOLUME_V,
                         horizontal: bool = True, vertical: bool = True) -> "GliderGeometry":
        """Resize the tails from tail volume coefficients (arms = 0.6 x fuselage length):
            S_h = V_H S c_mac / l_h,    S_v = V_V S b / l_v
        with tail aspect ratio 5 and fin aspect ratio 1.6. ``horizontal`` / ``vertical`` choose
        which surfaces to resize (the mesh parser keeps the ones it measured)."""
        arm = 0.6 * self.fuselage_length
        s_h = v_h * self.area * self.mac / arm
        b_h = math.sqrt(5.0 * s_h)
        s_v = v_v * self.area * self.span / arm
        h_v = math.sqrt(1.6 * s_v)
        changes = {}
        if horizontal:
            changes.update(htail_span=b_h, htail_chord=s_h / b_h, htail_arm=arm)
        if vertical:
            changes.update(vtail_height=h_v, vtail_chord=s_v / h_v, vtail_arm=arm,
                           vtail_height_above_cg=0.5 * h_v)
        return replace(self, **changes)
