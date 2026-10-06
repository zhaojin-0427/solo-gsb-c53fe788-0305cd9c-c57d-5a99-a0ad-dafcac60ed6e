"""Synthetic shard fixtures: split a plaque into two irregular pieces."""

from __future__ import annotations

import base64

import cv2
import numpy as np


def _png_b64(img_bgra: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgra)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def make_shard_pair(size=140, theta=0.3, tx=118.0, ty=-26.0, seed=0,
                    gap=0.0, noise=0.0):
    """Create two matching shards A and B.

    A is the left half of a plaque, B the right half, with a wavy vertical
    split.  B is extracted in its own image frame and the ground-truth
    relative pose mapping B-local -> A-local is returned together with
    boundary point pairs along the split.
    """
    rng = np.random.default_rng(seed)
    H = W = size
    color = np.zeros((H, W, 4), np.uint8)
    color[:, :, 0] = 90
    color[:, :, 1] = 150
    color[:, :, 2] = 210
    color[:, :, 3] = 255

    ys = np.arange(H)
    wave = (W / 2 + 14 * np.sin(ys / H * 3.0 * np.pi)
            + 6 * np.sin(ys / H * 9.0 * np.pi + 1.2))
    xA = np.floor(wave).astype(int)
    xB = np.ceil(wave + gap).astype(int)

    mA = np.zeros((H, W), bool)
    mB = np.zeros((H, W), bool)
    for y in ys:
        mA[y, :xA[y]] = True
        mB[y, xB[y]:] = True

    imgA = np.zeros_like(color)
    imgB = np.zeros_like(color)
    imgA[mA] = color[mA]
    imgB[mB] = color[mB]

    # B in its own tightly cropped frame (top-left = min occupied pixel).
    by, bx = np.nonzero(mB)
    y0, y1 = by.min(), by.max() + 1
    x0, x1 = bx.min(), bx.max() + 1
    cropB = imgB[y0:y1, x0:x1]

    # Ground-truth pose mapping B-local (crop coords) -> A image coords.
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Place B so that its split boundary matches A's; start identity pose
    # then rotate around the boundary midpoint to keep pieces from
    # intersecting badly.
    mid_y = H / 2
    mid_x = W / 2
    # B local origin at (x0, y0); choose translation so rotated boundary
    # sits against A: pick anchor point p0 on the seam.
    p0 = np.array([wave[int(mid_y)] - x0, mid_y - y0], dtype=np.float64)
    t = np.array([mid_x, mid_y]) - R @ p0 + np.array([tx - 118.0, ty + 26.0])

    pairs = []
    for y in range(10, H - 10, 12):
        pb = np.array([wave[y] - x0, y - y0], dtype=np.float64)
        pa = R @ pb + t
        if noise:
            pa = pa + rng.normal(scale=noise, size=2)
        pairs.append((pa, pb))

    return {
        "A": imgA,
        "B": cropB,
        "theta": theta,
        "t": t,
        "pairs": pairs,
    }


def fragment_payload(fid: str, img: np.ndarray) -> dict:
    return {"id": fid, "image_base64": _png_b64(img)}


def seam_payload(sid, fa, fb, pairs, score, pose_limit, overlap_limit):
    return {
        "id": sid,
        "fragment_a": fa,
        "fragment_b": fb,
        "score": score,
        "pose_error_limit": pose_limit,
        "overlap_limit": overlap_limit,
        "pairs": [{"a": list(map(float, pa)), "b": list(map(float, pb))}
                  for pa, pb in pairs],
    }
