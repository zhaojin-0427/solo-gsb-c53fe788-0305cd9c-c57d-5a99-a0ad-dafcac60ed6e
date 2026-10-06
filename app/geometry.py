"""2-D rigid-motion geometry helpers.

A pose is represented by a rotation angle ``theta`` (radians) and a
translation ``t`` (shape ``(2,)``).  The pose maps coordinates expressed in
the fragment's own image frame to the common world frame::

    p_world = R(theta) @ p_local + t
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidModel:
    """A rotation + translation fit and its residual statistics."""

    theta: float
    t: np.ndarray
    residual: np.ndarray
    rms: float
    max_residual: float

    @property
    def R(self) -> np.ndarray:
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    @property
    def matrix(self) -> np.ndarray:
        M = np.eye(3, dtype=np.float64)
        M[:2, :2] = self.R
        M[:2, 2] = self.t
        return M


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def fit_rigid(src: np.ndarray, dst: np.ndarray) -> RigidModel:
    """Estimate ``dst ~ R(theta) @ src + t`` by rotation-aware least squares.

    Uses the closed-form Kabsch/Procrustes solution in 2-D (a
    rotation-only polar factor, never a reflection), followed by the
    centroid translation.

    Raises ``ValueError`` when the configuration is degenerate, i.e. fewer
    than two pairs or all source / destination points coincide so the
    rotation is unobservable.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = src.shape[0]
    if dst.shape != (n, 2) or n < 2:
        raise ValueError("at least two point pairs are required")

    centroid_s = src.mean(axis=0)
    centroid_d = dst.mean(axis=0)
    ps = src - centroid_s
    pd = dst - centroid_d

    # All coincident points => rank deficient, rotation cannot be estimated.
    if np.linalg.norm(ps, axis=1).max() < 1e-9 or np.linalg.norm(pd, axis=1).max() < 1e-9:
        raise ValueError("point pairs are degenerate (collapsed to one point)")

    H = ps.T @ pd
    # SVD polar factor forces det(R) = +1 (proper rotation, no reflection).
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(2)
    D[1, 1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    if np.linalg.det(R) < 0:
        R = Vt.T @ np.diag([1.0, -1.0]) @ U.T
    theta = float(np.arctan2(R[1, 0], R[0, 0]))

    t = centroid_d - R @ centroid_s
    residual = (dst - (src @ R.T + t))
    norm = np.linalg.norm(residual, axis=1)
    rms = float(np.sqrt(np.mean(norm**2)))
    return RigidModel(theta=theta, t=t, residual=residual, rms=rms, max_residual=float(norm.max()))


def compose(theta_a: float, t_a: np.ndarray, theta_b: float, t_b: np.ndarray):
    """Return the pose ``b`` expressed after pose ``a`` (``a * b``)."""
    R = rotation_matrix(theta_a) @ rotation_matrix(theta_b)
    t = R @ t_b + t_a
    return float(np.arctan2(R[1, 0], R[0, 0])), t


def inverse_pose(theta: float, t: np.ndarray):
    R = rotation_matrix(theta).T
    return float(np.arctan2(R[1, 0], R[0, 0])), -R @ t


def relative_pose(pose_a, pose_b):
    """Relative pose mapping b-local coordinates into the a-local frame.

    Global poses map local -> world (``p_w = R p + t``), so the relative
    motion is ``Rr = R_a^T R_b``, ``tr = R_a^T (t_b - t_a)``.
    """
    Ra = rotation_matrix(pose_a[0])
    Rb = rotation_matrix(pose_b[0])
    Rr = Ra.T @ Rb
    tr = Ra.T @ (np.asarray(pose_b[1]) - np.asarray(pose_a[1]))
    return float(np.arctan2(Rr[1, 0], Rr[0, 0])), tr
