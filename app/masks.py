"""Mask rasterisation, physical-overlap measurement and PNG rendering."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from .geometry import rotation_matrix


def _affine(theta: float, t: np.ndarray) -> np.ndarray:
    """2x3 affine mapping local coordinates to target coordinates."""
    A = np.zeros((2, 3), dtype=np.float64)
    A[:2, :2] = rotation_matrix(theta)
    A[:2, 2] = np.asarray(t, dtype=np.float64).ravel()
    return A


def overlap_pixels(mask_a: np.ndarray, mask_b: np.ndarray,
                   theta_ba: float, t_ba: np.ndarray) -> int:
    """Count occupied pixels of fragment B landing on occupied pixels of A.

    ``p_a = R(theta_ba) p_b + t_ba`` maps B-local pixels into A's image
    frame.  B is warped once (nearest neighbour) onto A's canvas; pixels
    falling outside A's canvas cannot overlap A and are ignored.
    """
    h, w = mask_a.shape
    warped = cv2.warpAffine(
        mask_b.astype(np.uint8),
        _affine(theta_ba, t_ba),
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return int(np.count_nonzero((mask_a > 0) & (warped > 0)))


def _warped_corners(mask: np.ndarray, theta: float, t: np.ndarray):
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((0, 2))
    R = rotation_matrix(theta)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    return pts @ R.T + np.asarray(t).ravel()


def render_cluster(images: dict, poses: dict, pad: int = 8) -> bytes:
    """Render all fragments in a cluster under one common pose to PNG bytes.

    Images are RGBA.  Fragments are painted ordered by fragment id so
    output is deterministic; alpha blending is standard source-over.
    """
    ids = sorted(images)
    corners = []
    for fid in ids:
        c = _warped_corners(images[fid][:, :, 3] > 0, poses[fid][0], poses[fid][1])
        if c.shape[0]:
            corners.append(c)
    if not corners:
        canvas = np.zeros((1, 1, 4), dtype=np.uint8)
        ok, buf = cv2.imencode(".png", canvas)
        return buf.tobytes()

    allc = np.concatenate(corners, axis=0)
    x0 = int(np.floor(allc[:, 0].min())) - pad
    y0 = int(np.floor(allc[:, 1].min())) - pad
    x1 = int(np.ceil(allc[:, 0].max())) + pad
    y1 = int(np.ceil(allc[:, 1].max())) + pad
    W, H = x1 - x0, y1 - y0
    canvas = np.zeros((H, W, 4), dtype=np.uint8)

    for fid in ids:
        theta, t = poses[fid]
        A = _affine(theta, t + np.array([-x0, -y0]))
        warped = cv2.warpAffine(
            images[fid], A, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        alpha = warped[:, :, 3:4].astype(np.float64) / 255.0
        base_rgb = canvas[:, :, :3].astype(np.float64)
        base_a = canvas[:, :, 3:4].astype(np.float64) / 255.0
        out_rgb = warped[:, :, :3].astype(np.float64) * alpha + base_rgb * (1 - alpha) * base_a
        out_a = alpha + (1 - alpha) * base_a
        canvas[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        canvas[:, :, 3] = np.clip(out_a[:, :, 0] * 255.0, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def encode_png_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")
