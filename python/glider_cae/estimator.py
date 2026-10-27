"""Stage 2 - aerodynamic estimator: geometry in, coefficients out.

Component build-up in the classical (Nelson / Etkin / Raymer / DATCOM) style, with the
wing solved by Glauert lifting-line theory (see lifting_line.py). It implements the
``AeroModel`` protocol, so it drops into the simulator in place of the placeholder.

Longitudinal  (a = lift slope, eps_a = d eps / d alpha = 2 a_w / (pi AR), eta = tail q-ratio)
    CL   = CL_w + eta (S_h/S) CL_t
    CL_w = a_w (alpha + i_w - alpha_0L)        (blended with a flat plate after stall)
    CL_t = a_h (alpha + i_t - eps - tau_e d_e),   eps = eps_a CL_w / a_w
    Cm   = Cm_ac + CL_w (h - h_ac) - eta V_H CL_t + Cm_a,fus alpha + Cm_q q_hat
    Cm_q = -1.1 * 2 eta a_h V_H (l_h / c)              V_H = S_h l_h / (S c)
    static margin = -dCm/dCL = h_np - h
Drag
    CD = CD0(Re) + CL_w^2 / (pi e AR) + (S_h/S) CL_t^2 / (pi e_h AR_h)
    CD0 = 1.1 * sum_i Cf(Re_i) FF_i Q_i S_wet,i / S
Lateral-directional (b = span, V_V = S_v l_v / (S b))
    Cn_b = eta_v a_v V_V - 1.3 Vol_f/(S b)           Cy_b = -eta_v a_v S_v/S + ...
    Cl_b = -a_w G (1+2l)/(6(1+l)) + Cy_b,v z_v/b - 0.42 CL tan(sweep)
    Cl_p = -a_w (1+3l)/(12(1+l))        Cn_r = -2 l_v^2 |Cy_b,v|/b^2 - CD0,w (1+3l)/(6(1+l))
    Cl_da = 2 a_w tau_a I_a / (S b),    I_a = int c(y) y dy over the aileron span
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .aero import AeroCoefficients
from .atmosphere import RHO0, isa_viscosity
from .config import G0, GliderConfig
from .geometry import GliderGeometry
from .lifting_line import finite_wing_slope, solve_lifting_line, tapered_chord
from .trim import find_glide_trim

ETA_H = 0.95            # tail dynamic-pressure ratio q_t / q
ETA_V_SIDEWASH = 0.90   # eta_v (1 + d sigma / d beta) of the fin
FUSELAGE_FILL = 0.55    # body volume / (pi/4 d^2 L)
LAMINAR_FRACTION = 0.20  # share of the surface with laminar skin friction
CN90 = 1.2              # flat-plate normal-force coefficient (post-stall)
STALL_SHARPNESS = 50.0  # 1/rad, steepness of the stall blend (~1 deg wide)
TAIL_OSWALD = 0.90
H_AC = 0.25             # wing aerodynamic centre, fraction of the MAC


# ------------------------------------------------------------------ small physics helpers
def flap_effectiveness(chord_ratio: float) -> float:
    """Thin-airfoil flap effectiveness tau = 1 - (theta - sin theta)/pi, theta = acos(2E - 1),
    reduced 10 % for viscous / gap losses."""
    theta = math.acos(2.0 * chord_ratio - 1.0)
    return 0.9 * (1.0 - (theta - math.sin(theta)) / math.pi)


def skin_friction(reynolds: float) -> float:
    """Flat-plate Cf: laminar 1.328/sqrt(Re) and turbulent 0.455/log10(Re)^2.58, blended."""
    re = min(max(reynolds, 2.0e4), 5.0e7)
    return (LAMINAR_FRACTION * 1.328 / math.sqrt(re)
            + (1.0 - LAMINAR_FRACTION) * 0.455 / math.log10(re) ** 2.58)


def _stall_blend(alpha_w, centre, a_w, alpha_0l):
    """Stall blend: sigma = 0 attached, 1 fully stalled (flat-plate). A positive stall at
    ``centre`` and a mirrored negative stall about the zero-lift angle. Returns (sigma, CL_lin)."""
    m = STALL_SHARPNESS
    neg_centre = 2.0 * alpha_0l - centre
    up = 1.0 / (1.0 + np.exp(np.clip(-m * (alpha_w - centre), -60.0, 60.0)))
    down = 1.0 / (1.0 + np.exp(np.clip(m * (alpha_w - neg_centre), -60.0, 60.0)))
    return np.minimum(up + down, 1.0), a_w * (alpha_w - alpha_0l)


def _wing_cl(alpha_w, centre, a_w, alpha_0l):
    sigma, cl_lin = _stall_blend(alpha_w, centre, a_w, alpha_0l)
    return (1.0 - sigma) * cl_lin + sigma * CN90 * np.sin(alpha_w) * np.cos(alpha_w)


def _solve_stall_centre(a_w: float, alpha_0l: float, cl_target: float) -> float:
    """Choose the blend centre so the *blended* CL curve peaks at the estimated CLmax.
    One broadcast evaluation over (centre, alpha) replaces an iterative root find."""
    alpha_lin = alpha_0l + cl_target / a_w
    centres = alpha_lin + np.linspace(0.0, 0.2, 400)
    alphas = np.linspace(alpha_0l, alpha_lin + 0.35, 700)
    peaks = _wing_cl(alphas[None, :], centres[:, None], a_w, alpha_0l).max(axis=1)
    return float(centres[np.argmin(np.abs(peaks - cl_target))])


def zero_lift_drag(g: GliderGeometry, reynolds_mac: float) -> tuple[float, float]:
    """Return (CD0 total, CD0 of the wing alone) at the given MAC Reynolds number."""
    area, mac = g.area, g.mac

    def term(length: float, wet: float, form: float, interference: float = 1.0) -> float:
        return skin_friction(reynolds_mac * length / mac) * form * interference * wet / area

    tc = g.thickness
    wing = term(mac, 2.0 * area * (1.0 + 0.2 * tc), 1.0 + 2.0 * tc + 60.0 * tc ** 4)

    lam = g.fuselage_length / g.fuselage_diameter
    wet_f = (math.pi * g.fuselage_diameter * g.fuselage_length
             * (1.0 - 2.0 / lam) ** (2.0 / 3.0) * (1.0 + 1.0 / lam ** 2))
    fuselage = term(g.fuselage_length, wet_f, 1.0 + 60.0 / lam ** 3 + lam / 400.0)

    ff_tail = 1.0 + 2.0 * 0.08 + 60.0 * 0.08 ** 4
    htail = term(g.htail_chord, 2.04 * g.htail_area, ff_tail, 1.05)
    vtail = term(g.vtail_chord, 2.04 * g.vtail_area, ff_tail, 1.05)
    return 1.10 * (wing + fuselage + htail + vtail), 1.10 * wing   # +10 % for leakage, joints


# ------------------------------------------------------------------ derived constants
@dataclass(frozen=True)
class AeroDerivatives:
    """Every scalar the estimator derives from the geometry (all SI, angles in rad)."""

    # planform and wing
    area: float; span: float; aspect_ratio: float; mac: float; taper: float
    cl_alpha_wing: float; oswald: float; alpha_zero_lift: float; incidence: float
    cm_ac_wing: float; cl_max_section: float; cl_max_linear: float
    stall_center: float                     # wing angle of attack of the stall blend
    # tails, fuselage, CG
    cl_alpha_tail: float; cl_alpha_fin: float; tail_area_ratio: float; tail_ar: float
    eps_alpha: float; cg: float; arm_h: float; arm_v: float; v_h: float; v_v: float
    tail_incidence: float; cm_alpha_fuselage: float
    # stability and damping derivatives
    cl_alpha_total: float; cm_alpha: float; cm_q: float
    static_margin: float; neutral_point: float
    cy_beta: float; cl_beta_fixed: float; cl_beta_sweep: float; cn_beta: float
    cl_p: float; cl_r_fin: float; cn_r: float
    # controls
    tau_e: float; tau_a: float; tau_r: float
    elevator_max: float; aileron_max: float; rudder_max: float
    cl_aileron: float; cy_rudder: float; cn_rudder: float; cl_rudder: float
    # drag
    cd0_design: float; cd0_wing_design: float; k_tail: float
    reynolds_design: float; speed_design: float


def _build(g: GliderGeometry) -> tuple[AeroDerivatives, "object"]:
    S, b, AR, lam, mac = g.area, g.span, g.aspect_ratio, g.taper, g.mac
    sweep, gamma = math.radians(g.sweep_deg), math.radians(g.dihedral_deg)
    tc, i_w = g.thickness, math.radians(g.incidence_deg)

    # ---- wing: lifting line
    a0 = 2.0 * math.pi * (1.0 + 0.77 * tc)                  # thickness-corrected section slope
    ll = solve_lifting_line(b, S, tapered_chord(g.root_chord, lam), a0 * math.cos(sweep))
    a_w, e_w = ll.lift_slope, ll.oswald
    alpha0l = -math.radians(105.0 * g.camber)               # ~ -1.05 deg per 1 % camber
    eps_alpha = 2.0 * a_w / (math.pi * AR)                  # downwash gradient
    cm_ac = (-2.3 * g.camber) * AR * math.cos(sweep) ** 2 / (AR + 2.0 * math.cos(sweep))
    cl_max_sec = g.cl_max_section if g.cl_max_section is not None else 0.95 + 9.0 * g.camber
    cl_max_lin = 0.9 * cl_max_sec * math.cos(sweep)
    stall_center = _solve_stall_centre(a_w, alpha0l, cl_max_lin)

    # ---- tails and fuselage
    s_h, s_v = g.htail_area, g.vtail_area
    ar_h = g.htail_span ** 2 / s_h
    a_h = finite_wing_slope(ar_h)
    a_v = finite_wing_slope(1.55 * g.vtail_height ** 2 / s_v)  # end-plate effect of the fuselage
    area_ratio = s_h / S
    vol_f = FUSELAGE_FILL * math.pi / 4.0 * g.fuselage_diameter ** 2 * g.fuselage_length
    cm_alpha_f = 2.0 * vol_f / (S * mac)                    # Munk moment, destabilising
    cn_beta_f = -1.3 * vol_f / (S * b)
    cy_beta_f = -2.0 * (math.pi / 4.0 * g.fuselage_diameter ** 2) / S
    tail_gain = ETA_H * area_ratio * a_h * (1.0 - eps_alpha)
    a_tot = a_w + tail_gain

    def arms(h: float) -> tuple[float, float, float, float]:
        shift = (h - H_AC) * mac                             # CG aft of the AC shortens the arms
        l_h, l_v = g.htail_arm - shift, g.vtail_arm - shift
        if min(l_h, l_v) <= 0.0:
            raise ValueError("CG lies behind the tail; check tail arms and CG position")
        return l_h, l_v, s_h * l_h / (S * mac), s_v * l_v / (S * b)

    if g.cg_frac_mac is None:                                # place the CG for the target margin
        h = H_AC + 0.05
        for _ in range(60):
            v_h = arms(h)[2]
            h_new = H_AC + (-g.static_margin_target * a_tot
                            + ETA_H * v_h * a_h * (1.0 - eps_alpha) - cm_alpha_f) / a_w
            converged = abs(h_new - h) < 1e-10
            h = h_new
            if converged:
                break
    else:
        h = g.cg_frac_mac
    l_h, l_v, v_h, v_v = arms(h)

    cm_alpha = a_w * (h - H_AC) - ETA_H * v_h * a_h * (1.0 - eps_alpha) + cm_alpha_f
    static_margin = -cm_alpha / a_tot

    # ---- tail incidence: Cm = 0 at the design lift coefficient (two linear equations)
    if g.tail_incidence_deg is None:
        k = (h - H_AC) + l_h / mac + cm_alpha_f / a_w
        x_w = ((l_h / mac) * g.design_cl - cm_ac - cm_alpha_f * (alpha0l - i_w)) / k   # CL_w
        cl_t = (g.design_cl - x_w) / (ETA_H * area_ratio)
        alpha_d = x_w / a_w + alpha0l - i_w
        i_t = cl_t / a_h - alpha_d + eps_alpha * x_w / a_w
    else:
        i_t = math.radians(g.tail_incidence_deg)

    # ---- controls
    tau_e = flap_effectiveness(g.elevator_chord_ratio)
    tau_a = flap_effectiveness(g.aileron_chord_ratio)
    tau_r = flap_effectiveness(g.rudder_chord_ratio)
    y1, y2 = g.aileron_span_inner * b / 2.0, g.aileron_span_outer * b / 2.0
    k_c = (1.0 - lam) * 2.0 / b                              # c(y) = c_r (1 - k_c y)
    i_a = g.root_chord * ((y2 ** 2 - y1 ** 2) / 2.0 - k_c * (y2 ** 3 - y1 ** 3) / 3.0)
    cl_da = 2.0 * a_w * tau_a * i_a / (S * b)

    # ---- lateral-directional derivatives
    cy_beta_v = -ETA_V_SIDEWASH * a_v * s_v / S
    z_v = g.vtail_height_above_cg
    cd0, cd0_w = _design_drag(g)
    cy_rudder = -ETA_H * a_v * tau_r * s_v / S               # +rudder = yaw right = force to the left

    d = AeroDerivatives(
        area=S, span=b, aspect_ratio=AR, mac=mac, taper=lam,
        cl_alpha_wing=a_w, oswald=e_w, alpha_zero_lift=alpha0l, incidence=i_w,
        cm_ac_wing=cm_ac, cl_max_section=cl_max_sec, cl_max_linear=cl_max_lin,
        stall_center=stall_center,
        cl_alpha_tail=a_h, cl_alpha_fin=a_v, tail_area_ratio=area_ratio, tail_ar=ar_h,
        eps_alpha=eps_alpha, cg=h, arm_h=l_h, arm_v=l_v, v_h=v_h, v_v=v_v,
        tail_incidence=i_t, cm_alpha_fuselage=cm_alpha_f,
        cl_alpha_total=a_tot, cm_alpha=cm_alpha,
        cm_q=-1.1 * 2.0 * ETA_H * a_h * v_h * l_h / mac,
        static_margin=static_margin, neutral_point=h + static_margin,
        cy_beta=cy_beta_v + cy_beta_f,
        cl_beta_fixed=(-a_w * gamma * (1.0 + 2.0 * lam) / (6.0 * (1.0 + lam))
                       + cy_beta_v * z_v / b),
        cl_beta_sweep=-0.42 * math.tan(sweep),
        cn_beta=-cy_beta_v * l_v / b + cn_beta_f,
        cl_p=-a_w * (1.0 + 3.0 * lam) / (12.0 * (1.0 + lam)),
        cl_r_fin=2.0 * l_v * z_v * abs(cy_beta_v) / b ** 2,
        cn_r=(-2.0 * l_v ** 2 * abs(cy_beta_v) / b ** 2
              - cd0_w * (1.0 + 3.0 * lam) / (6.0 * (1.0 + lam))),
        tau_e=tau_e, tau_a=tau_a, tau_r=tau_r,
        elevator_max=math.radians(g.elevator_travel_deg),
        aileron_max=math.radians(g.aileron_travel_deg),
        rudder_max=math.radians(g.rudder_travel_deg),
        cl_aileron=cl_da, cy_rudder=cy_rudder, cn_rudder=-cy_rudder * l_v / b,
        cl_rudder=cy_rudder * z_v / b,
        cd0_design=cd0, cd0_wing_design=cd0_w,
        k_tail=ETA_H * area_ratio / (math.pi * TAIL_OSWALD * ar_h),
        reynolds_design=_design_reynolds(g)[0], speed_design=_design_reynolds(g)[1],
    )
    return d, ll


def _design_reynolds(g: GliderGeometry) -> tuple[float, float]:
    """Sea-level Re (on the MAC) at the design lift coefficient, and the speed it implies."""
    speed = math.sqrt(2.0 * g.mass * G0 / (RHO0 * g.area * g.design_cl))
    return RHO0 * speed * g.mac / float(isa_viscosity(0.0)), speed


def _design_drag(g: GliderGeometry) -> tuple[float, float]:
    return zero_lift_drag(g, _design_reynolds(g)[0])


# ------------------------------------------------------------------ the AeroModel
class LiftingLineEstimator:
    """Geometry-driven ``AeroModel``. Build once, then evaluate cheaply every physics step."""

    def __init__(self, geometry: GliderGeometry):
        problems = geometry.validate()
        if problems:
            raise ValueError("; ".join(problems))
        self.geometry = geometry
        self.d, self._ll = _build(geometry)

    # ---- AeroModel interface
    def coefficients(self, alpha, beta, p_hat, q_hat, r_hat,
                     elevator, aileron, rudder, reynolds=None) -> AeroCoefficients:
        d = self.d
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        de, da, dr = elevator * d.elevator_max, aileron * d.aileron_max, rudder * d.rudder_max

        cl_w, cdi_w = self._wing(alpha + d.incidence)
        eps = d.eps_alpha * cl_w / d.cl_alpha_wing
        alpha_t = np.clip(alpha + d.tail_incidence - eps - d.tau_e * de, -0.35, 0.35)
        cl_t = d.cl_alpha_tail * alpha_t                     # +elevator (stick back) = TE up

        cd0 = d.cd0_design if reynolds is None else zero_lift_drag(self.geometry, reynolds)[0]
        cl = cl_w + ETA_H * d.tail_area_ratio * cl_t
        cd = cd0 + cdi_w + d.k_tail * cl_t ** 2

        cm = (d.cm_ac_wing + cl_w * (d.cg - H_AC) - ETA_H * d.v_h * cl_t
              + d.cm_alpha_fuselage * alpha + d.cm_q * q_hat)
        cy = d.cy_beta * beta + d.cy_rudder * dr
        cl_roll = ((d.cl_beta_fixed + d.cl_beta_sweep * cl_w) * beta + d.cl_p * p_hat
                   + (d.cl_r_fin + 0.25 * cl_w) * r_hat + d.cl_aileron * da + d.cl_rudder * dr)
        cn = (d.cn_beta * beta - 0.125 * cl_w * p_hat + d.cn_r * r_hat + d.cn_rudder * dr
              - 0.2 * cl_w * d.cl_aileron * da)               # adverse yaw from the ailerons
        return AeroCoefficients(CL=cl, CD=cd, CY=cy, Cl=cl_roll, Cm=cm, Cn=cn)

    # ---- wing lift / induced drag with a smooth stall
    def _wing(self, alpha_w):
        d = self.d
        sigma, cl_lin = _stall_blend(alpha_w, d.stall_center, d.cl_alpha_wing, d.alpha_zero_lift)
        sa, ca = np.sin(alpha_w), np.cos(alpha_w)
        cl = (1.0 - sigma) * cl_lin + sigma * CN90 * sa * ca
        cdi = ((1.0 - sigma) * cl_lin ** 2 / (math.pi * d.oswald * d.aspect_ratio)
               + sigma * CN90 * sa * sa)                    # induced drag -> flat-plate drag
        return cl, cdi

    # ---- analysis products for the UI / reports
    def polar(self, alpha_deg: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Vectorised sweep of CL, CD, Cm and L/D (zero sideslip and controls)."""
        if alpha_deg is None:
            stall = math.degrees(self.d.stall_center - self.d.incidence)
            alpha_deg = np.linspace(-6.0, stall + 3.0, 120)
        c = self.coefficients(np.radians(alpha_deg), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cl, cd = np.asarray(c.CL), np.asarray(c.CD)
        return {"alpha_deg": np.asarray(alpha_deg), "cl": cl, "cd": cd,
                "cm": np.asarray(c.Cm), "ld": cl / cd}

    def spanwise_loading(self, n: int = 61) -> dict[str, np.ndarray]:
        """Normalised circulation Gamma(y)/Gamma_max from the lifting-line solution."""
        theta = np.linspace(1e-3, np.pi - 1e-3, n)
        gamma = self._ll.circulation(theta)
        return {"y_over_semispan": -np.cos(theta), "gamma": gamma / gamma.max()}

    def report(self) -> dict:
        """Everything worth showing about the configuration, plus plain-language warnings."""
        d, g = self.d, self.geometry
        polar = self.polar(np.linspace(-4.0, math.degrees(d.stall_center) + 2.0, 400))
        cl, cd = polar["cl"], polar["cd"]
        weight = g.mass * G0
        ok = cl > 0.15
        speed = np.sqrt(2.0 * weight / (RHO0 * d.area * np.where(ok, cl, np.nan)))
        ld = np.where(ok, cl / cd, -np.inf)
        sink = np.where(ok, speed * cd / cl, np.inf)
        i_ld, i_sink, i_max = int(np.argmax(ld)), int(np.argmin(sink)), int(np.argmax(cl))

        try:
            trim = find_glide_trim(g.to_config(), self, altitude=0.0)
            trim_info = {"alpha_deg": math.degrees(trim.alpha), "speed": trim.airspeed,
                         "cl": trim.cl, "cd": trim.cd, "glide_ratio": trim.glide_ratio, "sink": trim.airspeed * math.sin(-trim.gamma)}
        except ValueError:
            trim_info = None

        warnings = []
        if d.static_margin < 0.02:
            warnings.append(f"Static margin is {d.static_margin:.0%}: pitch is nearly neutral or "
                            "unstable. Move the CG forward or enlarge the horizontal tail.")
        elif d.static_margin > 0.35:
            warnings.append(f"Static margin is {d.static_margin:.0%}: very nose-heavy, elevator "
                            "authority will be poor.")
        if d.cn_beta < 0.02:
            warnings.append("Weak weathercock stability: the fin is small for this wing.")
        if d.cl_beta_fixed + d.cl_beta_sweep * float(cl[i_ld]) > -0.015:
            warnings.append("Very little dihedral effect: expect a slow, divergent spiral.")
        if abs(math.degrees(d.tail_incidence)) > 8.0:
            warnings.append("Tail incidence needed for trim is large; the CG or tail size is unusual.")
        if trim_info is None:
            warnings.append("No stable glide trim exists for this configuration.")

        return {
            "span": d.span, "area": d.area, "aspect_ratio": d.aspect_ratio, "mac": d.mac,
            "taper": d.taper, "mass": g.mass, "wing_loading": g.mass / d.area,
            "lift_slope_wing": d.cl_alpha_wing, "lift_slope_total": d.cl_alpha_total,
            "oswald": d.oswald, "alpha_zero_lift_deg": math.degrees(d.alpha_zero_lift),
            "cl_max": float(cl[i_max]), "stall_alpha_deg": float(polar["alpha_deg"][i_max]),
            "cd0": d.cd0_design, "best_ld": float(ld[i_ld]), "best_ld_cl": float(cl[i_ld]),
            "best_ld_speed": float(speed[i_ld]), "min_sink": float(sink[i_sink]),
            "min_sink_speed": float(speed[i_sink]), "stall_speed": float(
                math.sqrt(2.0 * weight / (RHO0 * d.area * cl[i_max]))),
            "cg_frac_mac": d.cg, "neutral_point": d.neutral_point,
            "static_margin": d.static_margin, "tail_volume_h": d.v_h, "tail_volume_v": d.v_v,
            "tail_incidence_deg": math.degrees(d.tail_incidence),
            "cm_alpha": d.cm_alpha, "cm_q": d.cm_q, "cy_beta": d.cy_beta,
            "cl_beta": d.cl_beta_fixed, "cn_beta": d.cn_beta, "cl_p": d.cl_p, "cn_r": d.cn_r,
            "cl_aileron": d.cl_aileron, "cm_elevator": ETA_H * d.v_h * d.cl_alpha_tail * d.tau_e,
            "cn_rudder": d.cn_rudder,
            "trim": trim_info, "warnings": warnings,
        }

    def derivatives_dict(self) -> dict:
        return asdict(self.d)
