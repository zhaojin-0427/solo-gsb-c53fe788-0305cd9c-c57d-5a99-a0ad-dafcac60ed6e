"""Generate a real, ready-to-send /assemblies request and run it.

Usage:
    python examples/make_example.py            # writes examples/request.json
    python examples/make_example.py --call     # also POSTs to a running server

The request contains three transparent-PNG potsherds (two match along a
wavy seam, the third is an isolated fragment), scored candidate seams with
per-seam pose-error and overlap-pixel limits.
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import cv2
import numpy as np


def png_b64(img_bgra: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgra)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def make_pieces(size=160):
    """Two irregular shards from one plaque along a wavy cut + one isolated."""
    H = W = size
    plaque = np.zeros((H, W, 4), np.uint8)
    plaque[:, :, :3] = (82, 120, 178)    # terracotta (B, G, R)
    plaque[:, :, 3] = 255

    ys = np.arange(H, dtype=np.float64)
    cut = (W / 2
           + 12.0 * np.sin(ys / H * 3.0 * np.pi)
           + 5.0 * np.sin(ys / H * 8.0 * np.pi + 0.7))
    xA = np.floor(cut).astype(int)

    mA = np.zeros((H, W), bool)
    for y in range(H):
        mA[y, :xA[y] + 1] = True
    imgA_full = np.zeros_like(plaque)
    imgA_full[mA] = plaque[mA]

    mB = np.zeros((H, W), bool)
    for y in range(H):
        mB[y, xA[y] + 1:] = True
    imgB_full = np.zeros_like(plaque)
    imgB_full[mB] = plaque[mB]

    # crop each to its occupied bounding box (independent image frames)
    def crop(img):
        yy, xx = np.nonzero(img[:, :, 3])
        return img[yy.min():yy.max() + 1, xx.min():xx.max() + 1], xx.min(), yy.min()

    imgA, ax0, ay0 = crop(imgA_full)
    imgB, bx0, by0 = crop(imgB_full)

    # Boundary point pairs expressed in each fragment's cropped frame.
    pairs = []
    for y in range(12, H - 12, 10):
        pa = [float(cut[y] - ax0), float(y - ay0)]
        pb = [float(cut[y] - bx0), float(y - by0)]
        pairs.append({"a": pa, "b": pb})

    # Isolated third fragment (a small rounded tile).
    iso = np.zeros((46, 46, 4), np.uint8)
    cv2.circle(iso, (23, 23), 20, (60, 110, 190, 255), -1)

    return imgA, imgB, iso, pairs


def build_request():
    imgA, imgB, iso, pairs = make_pieces()
    return {
        "fragments": [
            {"id": "shard-A", "image_base64": png_b64(imgA)},
            {"id": "shard-B", "image_base64": png_b64(imgB)},
            {"id": "shard-C", "image_base64": png_b64(iso)},
        ],
        "seams": [
            {
                "id": "seam-AB",
                "fragment_a": "shard-A",
                "fragment_b": "shard-B",
                "score": 12.5,
                "pose_error_limit": 3.0,
                "overlap_limit": 200,
                "pairs": pairs,
            },
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", action="store_true",
                    help="POST the request to a running service")
    ap.add_argument("--url", default="http://127.0.0.1:8000/assemblies")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    req = build_request()
    req_path = os.path.join(here, "request.json")
    with open(req_path, "w") as f:
        json.dump(req, f)
    print("wrote", req_path)

    if args.call:
        import urllib.request
        data = json.dumps(req).encode()
        http_req = urllib.request.Request(
            args.url, data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http_req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        out_path = os.path.join(here, "response.json")
        with open(out_path, "w") as f:
            json.dump(body, f, indent=2)
        # also write the rendered cluster PNG for inspection
        for cl in body["clusters"]:
            raw = base64.b64decode(cl["image_base64"])
            p = os.path.join(here, f"cluster_{cl['cluster_id']}.png")
            with open(p, "wb") as f:
                f.write(raw)
        print("wrote", out_path, "and cluster PNGs")


if __name__ == "__main__":
    main()
