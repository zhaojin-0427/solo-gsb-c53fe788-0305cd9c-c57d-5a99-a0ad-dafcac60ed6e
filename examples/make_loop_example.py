"""Generate a real /assemblies request exposing the rejection-attribution fix.

The candidate seam graph is a four-fragment chain with a chord::

    A -- B -- C -- D
    |               |
    +------ ad -----+

The accepted seams form the path ``ab`` ``bc`` ``cd``.  The rejected chord
``ad`` carries a 20px loop mismatch: a single global pose set cannot satisfy
all four seams within the 3px pose-error limit.  When re-checking the chord,
the service must refit the *whole* selected subgraph it touches (including
``bc`` -- not only the one-hop edges ``ab``/``cd``); otherwise the chord
plus its two incident edges is a tree that fits exactly and the seam is
wrongly reported as ``score_optimization_not_selected``.

The fixed service instead reports ``pose_residual_exceeds_limit`` (the
measured residual exceeds 3px) for ``ad``.

Usage:
    python examples/make_loop_example.py            # writes request JSON
    python examples/make_loop_example.py --call     # also POSTs it
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import cv2
import numpy as np


def square_png(fid: str, color) -> str:
    img = np.zeros((40, 40, 4), np.uint8)
    img[:, :, :3] = color
    img[:, :, 3] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return {"id": fid, "image_base64": base64.b64encode(buf).decode()}


def hseam(sid, fa, fb, gap, score, limit=3.0, overlap=10 ** 9):
    """Right-edge -> left-edge seam; ``gap`` is a vertical loop mismatch."""
    return {
        "id": sid, "fragment_a": fa, "fragment_b": fb, "score": score,
        "pose_error_limit": limit, "overlap_limit": overlap,
        "pairs": [
            {"a": [39.0, float(y)], "b": [0.0, float(y - gap)]}
            for y in range(4, 36, 4)
        ],
    }


def build_request():
    return {
        "fragments": [
            square_png("A", (200, 120, 90)),
            square_png("B", (210, 130, 110)),
            square_png("C", (220, 140, 130)),
            square_png("D", (230, 160, 150)),
        ],
        "seams": [
            hseam("ab", "A", "B", 0, score=8),
            hseam("bc", "B", "C", 0, score=8),
            hseam("cd", "C", "D", 0, score=8),
            # Chord with a 20px loop mismatch: forces the four-edge loop out.
            hseam("ad", "A", "D", 20, score=7),
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8000/assemblies")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    req = build_request()
    req_path = os.path.join(here, "loop_request.json")
    with open(req_path, "w") as f:
        json.dump(req, f)
    print("wrote", req_path)

    if args.call:
        import urllib.request
        http_req = urllib.request.Request(
            args.url, data=json.dumps(req).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http_req, timeout=60) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode())
        out_path = os.path.join(here, "loop_response.json")
        with open(out_path, "w") as f:
            json.dump(body, f, indent=2)
        with open(os.path.join(here, "loop_cluster.png"), "wb") as f:
            f.write(base64.b64decode(body["clusters"][0]["image_base64"]))
        print("wrote", out_path, "and loop_cluster.png")


if __name__ == "__main__":
    main()
