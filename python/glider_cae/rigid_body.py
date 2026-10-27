"""Quaternion kinematics and the 6DOF rigid-body equations of motion.

State vector x (13,):
    [ N, E, D | qw, qx, qy, qz | u, v, w | p, q, r ]
      position   attitude body->NED   body velocity   body rates

Equations (body axes, mass m, inertia tensor I about the CG):
    p_dot = R(q) v_b                              (kinematics)
    q_dot = 1/2 * Omega(omega) q                  (quaternion kinematics)
    v_dot = F_b / m - omega x v_b                 (Newton, rotating frame)
    w_dot = I^-1 ( M_b - omega x (I omega) )      (Euler's rotation equation)
"""
from __future__ import annotations

import numpy as np

STATE_SIZE = 13
POS = slice(0, 3)
QUAT = slice(3, 7)
VEL = slice(7, 10)
RATE = slice(10, 13)


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3-vector cross product; ~10x faster than np.cross for single vectors (matters in Pyodide)."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R (body -> inertial) for q = [w, x, y, z]."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) Euler angles in radians -> quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def euler_from_quat(q: np.ndarray) -> np.ndarray:
    """Quaternion -> (roll, pitch, yaw) in radians."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def quat_rate(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """q_dot = 1/2 * Omega(omega) @ q, with omega = [p, q, r] in body axes."""
    p, qq, r = omega
    omega_matrix = np.array([
        [0.0, -p,  -qq, -r],
        [p,   0.0,  r,  -qq],
        [qq,  -r,   0.0, p],
        [r,   qq,  -p,   0.0],
    ])
    return 0.5 * omega_matrix @ q


def rigid_body_derivatives(
    x: np.ndarray,
    force_b: np.ndarray,
    moment_b: np.ndarray,
    mass: float,
    inertia: np.ndarray,
    inertia_inv: np.ndarray,
    dcm: np.ndarray | None = None,
) -> np.ndarray:
    """Time derivative of the 13-state vector for given body-axis loads."""
    q, v_b, omega = x[QUAT], x[VEL], x[RATE]
    if dcm is None:
        dcm = quat_to_dcm(q)

    dx = np.empty(STATE_SIZE)
    dx[POS] = dcm @ v_b
    dx[QUAT] = quat_rate(q, omega)
    dx[VEL] = force_b / mass - cross3(omega, v_b)
    dx[RATE] = inertia_inv @ (moment_b - cross3(omega, inertia @ omega))
    return dx
