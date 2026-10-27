"""Stage 2 tests: lifting line, component build-up, and flight with estimated aerodynamics."""
import math
from dataclasses import replace

import numpy as np
import pytest

from glider_cae.dynamics import Controls
from glider_cae.estimator import LiftingLineEstimator
from glider_cae.geometry import GliderGeometry
from glider_cae.lifting_line import finite_wing_slope, solve_lifting_line, tapered_chord
from glider_cae.rigid_body import QUAT, euler_from_quat
from glider_cae.simulation import GliderSimulation
from glider_cae.trim import find_glide_trim

A0 = 2 * math.pi


# ---------------------------------------------------------------- lifting-line theory
def test_elliptic_wing_matches_closed_form():
    span, ar = 10.0, 10.0
    area = span * span / ar
    c_root = 4 * area / (math.pi * span)
    r = solve_lifting_line(span, area, lambda th: c_root * np.sin(th), A0)
    assert r.oswald == pytest.approx(1.0, abs=1e-4)
    assert r.lift_slope == pytest.approx(A0 / (1 + A0 / (math.pi * ar)), rel=1e-4)


def test_taper_improves_span_efficiency_with_optimum_near_0_4():
    area, span = 0.32, 2.0
    e = {}
    for lam in (1.0, 0.7, 0.4, 0.2):
        root = 2 * area / (span * (1 + lam))
        e[lam] = solve_lifting_line(span, area, tapered_chord(root, lam), A0).oswald
    assert e[1.0] < e[0.7] < e[0.4]
    assert e[0.4] > 0.97 and e[0.2] < e[0.4]


def test_lifting_line_slope_close_to_helmbold_for_high_aspect_ratio():
    area, span = 0.32, 2.0
    r = solve_lifting_line(span, area, tapered_chord(2 * area / (span * 1.6), 0.6), A0)
    assert r.lift_slope == pytest.approx(finite_wing_slope(span ** 2 / area), rel=0.06)


# ---------------------------------------------------------------- geometry
def test_planform_formulas():
    g = GliderGeometry(span=2.0, root_chord=0.2, tip_chord=0.1)
    assert g.area == pytest.approx(0.3)
    assert g.aspect_ratio == pytest.approx(4 / 0.3)
    assert g.mac == pytest.approx(2 / 3 * 0.2 * (1 + 0.5 + 0.25) / 1.5)


def test_validation_rejects_nonsense():
    assert GliderGeometry().validate() == []
    assert GliderGeometry(span=-1.0).validate()
    assert GliderGeometry(tip_chord=0.9).validate()                    # taper > 1.5
    with pytest.raises(ValueError):
        LiftingLineEstimator(GliderGeometry(fuselage_diameter=0.9))


def test_bounding_box_constructor_sizes_tails_by_volume_coefficient():
    g = GliderGeometry.from_bounding_box(span=3.0, length=1.5, height=0.3, mass=3.0)
    assert g.validate() == []
    d = LiftingLineEstimator(g).d
    assert g.aspect_ratio == pytest.approx(12.0, rel=1e-6)
    assert d.v_h == pytest.approx(0.40, rel=0.15)
    assert d.v_v == pytest.approx(0.035, rel=0.15)


# ---------------------------------------------------------------- component build-up
def test_estimated_derivatives_have_stabilising_signs():
    d = LiftingLineEstimator(GliderGeometry()).d
    assert d.cm_alpha < 0 and d.cm_q < 0
    assert d.cl_beta_fixed < 0 and d.cn_beta > 0 and d.cy_beta < 0
    assert d.cl_p < 0 and d.cn_r < 0
    assert d.cl_aileron > 0 and d.cn_rudder > 0


def test_cg_placement_hits_requested_static_margin():
    for target in (0.05, 0.10, 0.20):
        d = LiftingLineEstimator(GliderGeometry(static_margin_target=target)).d
        assert d.static_margin == pytest.approx(target, abs=1e-6)


def test_analytic_cm_alpha_matches_finite_difference():
    est = LiftingLineEstimator(GliderGeometry())
    h = 1e-4
    cm = lambda a: float(est.coefficients(a, 0, 0, 0, 0, 0, 0, 0).Cm)  # noqa: E731
    cl = lambda a: float(est.coefficients(a, 0, 0, 0, 0, 0, 0, 0).CL)  # noqa: E731
    a0 = math.radians(2.0)
    assert (cm(a0 + h) - cm(a0 - h)) / (2 * h) == pytest.approx(est.d.cm_alpha, rel=2e-3)
    assert (cl(a0 + h) - cl(a0 - h)) / (2 * h) == pytest.approx(est.d.cl_alpha_total, rel=2e-3)


def test_auto_tail_incidence_trims_at_design_cl():
    g = GliderGeometry(design_cl=0.7)
    est = LiftingLineEstimator(g)
    alphas = np.radians(np.linspace(-2, 8, 4001))
    c = est.coefficients(alphas, 0, 0, 0, 0, 0, 0, 0)
    i = int(np.argmin(np.abs(np.asarray(c.CL) - 0.7)))
    assert abs(float(np.asarray(c.Cm)[i])) < 2e-3


def test_polar_has_a_stall_and_realistic_performance():
    r = LiftingLineEstimator(GliderGeometry()).report()
    assert 0.9 < r["cl_max"] < 1.4
    assert 12.0 < r["best_ld"] < 30.0
    assert 0.2 < r["min_sink"] < 0.8
    assert r["stall_speed"] < r["trim"]["speed"] < 14.0
    assert r["warnings"] == []


def test_drag_depends_on_reynolds_number():
    est = LiftingLineEstimator(GliderGeometry())
    low = float(est.coefficients(0.05, 0, 0, 0, 0, 0, 0, 0, reynolds=5e4).CD)
    high = float(est.coefficients(0.05, 0, 0, 0, 0, 0, 0, 0, reynolds=5e5).CD)
    assert low > high


def test_controls_have_the_right_sign():
    est = LiftingLineEstimator(GliderGeometry())
    base = est.coefficients(0.05, 0, 0, 0, 0, 0, 0, 0)
    up = est.coefficients(0.05, 0, 0, 0, 0, 1.0, 0, 0)
    assert up.Cm > base.Cm and up.CL < base.CL                       # stick back: nose up, tail lifts less
    assert est.coefficients(0.05, 0, 0, 0, 0, 0, 1.0, 0).Cl > 0      # right aileron: roll right
    assert est.coefficients(0.05, 0, 0, 0, 0, 0, 1.0, 0).Cn < 0      # adverse yaw
    assert est.coefficients(0.05, 0, 0, 0, 0, 0, 0, 1.0).Cn > 0      # right rudder: yaw right


def test_unstable_configuration_is_reported():
    est = LiftingLineEstimator(GliderGeometry(cg_frac_mac=0.55))
    r = est.report()
    assert r["static_margin"] < 0 and r["trim"] is None
    assert any("pitch" in w for w in r["warnings"])


# ---------------------------------------------------------------- flight with estimated aerodynamics
def make_sim(geometry=None, altitude=400.0):
    sim = GliderSimulation.from_geometry(geometry or GliderGeometry())
    sim.reset(altitude=altitude)
    return sim


def test_estimated_glider_trims_and_glides_steadily():
    sim = make_sim(altitude=300.0)
    t = sim.trim
    assert math.degrees(t.alpha) == pytest.approx(math.degrees(0.05), abs=3.0)
    state = sim.run(30.0)["state"]
    assert sim.status == "flying" and np.all(np.isfinite(state))
    alpha = np.degrees(np.arctan2(state[:, 9], state[:, 7]))
    assert alpha.min() > -2.0 and alpha.max() < 8.0
    ground = math.hypot(state[-1, 0], state[-1, 1])
    assert ground / (300.0 + state[-1, 2]) == pytest.approx(t.glide_ratio, rel=0.08)


def test_estimated_glider_short_period_damped_and_lateral_bounded():
    sim = make_sim(altitude=600.0)
    sim.x[10:13] = [0.5, 0.5, 0.0]                                   # roll and pitch-rate kick
    state = sim.run(15.0)["state"]
    assert sim.status == "flying"
    roll = np.degrees([euler_from_quat(q)[0] for q in state[:, QUAT]])
    assert np.abs(roll).max() < 60.0
    assert np.abs(state[-300:, 11]).max() < 0.2                      # pitch rate settled


def test_controls_move_the_estimated_glider_the_right_way():
    base = make_sim(altitude=800.0)
    pull = make_sim(altitude=800.0)
    a_base = np.degrees(np.arctan2(*base.run(5.0)["state"][-1, [9, 7]]))
    a_pull = np.degrees(np.arctan2(*pull.run(5.0, controls_fn=lambda t, x: Controls(elevator=1.0))["state"][-1, [9, 7]]))
    assert a_pull > a_base + 3.0

    roll = make_sim(altitude=800.0).run(2.0, controls_fn=lambda t, x: Controls(aileron=1.0))["state"]
    assert roll[-1, 10] > 0.3                                        # p > 0: right wing down
    yaw = make_sim(altitude=800.0).run(3.0, controls_fn=lambda t, x: Controls(rudder=1.0))["state"]
    assert yaw[-1, 12] > 0.0                                         # r > 0: nose right


@pytest.mark.parametrize("span, length, mass", [(0.8, 0.5, 0.12), (1.5, 0.9, 0.7), (4.0, 2.0, 6.0)])
def test_bounding_box_gliders_of_any_size_fly(span, length, mass):
    g = GliderGeometry.from_bounding_box(span=span, length=length, height=0.2 * length, mass=mass)
    sim = make_sim(g, altitude=400.0)
    state = sim.run(20.0)["state"]
    assert sim.status == "flying" and np.all(np.isfinite(state))
    assert sim.trim is not None and sim.trim.glide_ratio > 8.0


def test_heavier_glider_trims_faster():
    light = LiftingLineEstimator(GliderGeometry(mass=1.0))
    heavy = LiftingLineEstimator(GliderGeometry(mass=2.0))
    v = lambda e, m: find_glide_trim(e.geometry.to_config(), e).airspeed  # noqa: E731
    assert v(heavy, 2.0) == pytest.approx(v(light, 1.0) * math.sqrt(2.0), rel=0.05)
