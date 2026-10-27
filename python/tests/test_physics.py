"""Sanity tests for the stage-1 physics engine."""
import numpy as np
import pytest

from glider_cae.aero import AeroCoefficients
from glider_cae.atmosphere import isa_density
from glider_cae.config import G0, GliderConfig
from glider_cae.dynamics import Controls, Environment, GliderDynamics
from glider_cae.integrators import rk4_step
from glider_cae.rigid_body import (
    QUAT, RATE, VEL, euler_from_quat, quat_from_euler, quat_to_dcm,
)
from glider_cae.simulation import GliderSimulation
from glider_cae.trim import find_glide_trim


class NullAero:
    """Zero aerodynamic loads - isolates gravity and rigid-body dynamics."""

    def coefficients(self, *args, **kwargs):
        return AeroCoefficients(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_isa_density():
    assert isa_density(0.0) == pytest.approx(1.225)
    assert isa_density(1000.0) == pytest.approx(1.1117, rel=1e-3)


def test_quaternion_roundtrip_and_orthonormal_dcm():
    q = quat_from_euler(0.3, -0.2, 1.1)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    r = quat_to_dcm(q)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.allclose(euler_from_quat(q), [0.3, -0.2, 1.1])


def test_free_fall_matches_closed_form():
    sim = GliderSimulation(aero=NullAero(), env=Environment())
    sim.reset(altitude=500.0)
    sim.x[VEL] = 0.0
    sim.x[QUAT] = quat_from_euler(0, 0, 0)
    out = sim.run(3.0)
    assert -out["state"][-1, 2] == pytest.approx(500.0 - 0.5 * G0 * 3.0 ** 2, abs=1e-6)


def test_torque_free_rotation_conserves_angular_momentum_and_energy():
    cfg = GliderConfig(inertia=np.diag([0.12, 0.09, 0.20]))
    dyn = GliderDynamics(cfg, NullAero(), Environment(gravity=0.0))
    x = np.zeros(13)
    x[QUAT] = [1, 0, 0, 0]
    x[RATE] = [1.0, 2.0, 0.5]

    def h_and_e(s):
        omega = s[RATE]
        h_inertial = quat_to_dcm(s[QUAT]) @ (cfg.inertia @ omega)
        return h_inertial, 0.5 * omega @ cfg.inertia @ omega

    h0, e0 = h_and_e(x)
    f = lambda s: dyn.derivatives(s, Controls())  # noqa: E731
    for _ in range(2000):                      # 20 s at dt = 0.01
        x = rk4_step(f, x, 0.01)
        x[QUAT] /= np.linalg.norm(x[QUAT])
    h1, e1 = h_and_e(x)
    assert np.allclose(h0, h1, atol=1e-6)
    assert e1 == pytest.approx(e0, rel=1e-6)


def test_trim_balances_forces_and_moments():
    sim = GliderSimulation()
    sim.reset(altitude=100.0)
    t = sim.trim
    s = sim.dynamics.evaluate(sim.x, Controls())
    assert abs(s.moment_b[1]) < 1e-3                       # pitch equilibrium
    assert t.glide_ratio == pytest.approx(1 / np.tan(-t.gamma), rel=1e-6)
    assert 5.0 < t.airspeed < 20.0


def test_trimmed_glide_is_steady():
    sim = GliderSimulation()
    sim.reset(altitude=300.0)
    out = sim.run(30.0)
    state = out["state"]
    assert sim.status == "flying" and np.all(np.isfinite(state))
    alpha = np.degrees(np.arctan2(state[:, 9], state[:, 7]))
    assert alpha.min() > 1.0 and alpha.max() < 8.0           # no stall, no wild phugoid
    horizontal = np.hypot(state[-1, 0], state[-1, 1])
    sink = 300.0 + state[-1, 2]
    assert horizontal / sink == pytest.approx(sim.trim.glide_ratio, rel=0.08)
    assert np.allclose(np.linalg.norm(state[:, QUAT], axis=1), 1.0)


def test_short_period_is_damped():
    sim = GliderSimulation()
    sim.reset(altitude=500.0)
    sim.x[RATE] = [0.0, 0.5, 0.0]                           # pitch-rate kick
    out = sim.run(20.0)
    state = out["state"]
    alpha = np.arctan2(state[:, 9], state[:, 7])
    assert np.degrees(np.abs(alpha[-500:] - alpha[-500:].mean()).max()) < 1.0
    assert np.abs(state[:, 11]).max() < 2.0                  # pitch rate stays bounded


def test_lateral_disturbance_stays_bounded():
    sim = GliderSimulation()
    sim.reset(altitude=600.0)
    sim.x[RATE] = [0.5, 0.0, 0.0]                           # roll-rate kick
    out = sim.run(15.0)
    assert sim.status == "flying"
    roll = np.degrees(np.array([euler_from_quat(q)[0] for q in out["state"][:, QUAT]]))
    assert np.abs(roll).max() < 60.0


def test_elevator_authority_and_stall_protection():
    sim = GliderSimulation()
    sim.reset(altitude=800.0)
    nose_up = sim.run(8.0, controls_fn=lambda t, x: Controls(elevator=1.0))["state"]
    a_up = np.degrees(np.arctan2(nose_up[:, 9], nose_up[:, 7]))
    sim.reset(altitude=800.0)
    nose_dn = sim.run(8.0, controls_fn=lambda t, x: Controls(elevator=-1.0))["state"]
    a_dn = np.degrees(np.arctan2(nose_dn[:, 9], nose_dn[:, 7]))
    assert a_up.max() > a_dn.max() + 3.0


def test_aileron_rolls_right():
    sim = GliderSimulation()
    sim.reset(altitude=600.0)
    out = sim.run(2.0, controls_fn=lambda t, x: Controls(aileron=1.0))
    assert out["state"][-1, 10] > 0.3                        # p > 0 : right wing down


def test_updraft_reduces_sink():
    base, lift = GliderSimulation(), GliderSimulation(env=Environment(wind_ned=np.array([0, 0, -2.0])))
    for s in (base, lift):
        s.reset(altitude=300.0)
        s.run(20.0)
    assert lift.x[2] < base.x[2] - 20.0                      # D is more negative => higher


def test_touchdown_freezes_vehicle():
    sim = GliderSimulation()
    sim.reset(altitude=5.0)
    sim.run(60.0)
    assert sim.status in ("landed", "crashed")
    assert sim.x[2] == 0.0 and np.allclose(sim.x[VEL], 0.0)
