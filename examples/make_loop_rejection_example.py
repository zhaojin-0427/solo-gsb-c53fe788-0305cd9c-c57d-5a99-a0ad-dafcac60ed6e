"""Generate a real loop-closure-rejection /assemblies request and run it.

This is the rejection-reason scenario fixed in the service: the adopted
seams form the path A-B-C-D and the unselected candidate ``ad`` closes a
pose-inconsistent loop with it (its pairs imply the relative pose
(39, 12) for A->D while the chain implies (117, 0) -- a ~79 px mismatch
that no single global pose set can absorb under the 3 px limit).

The service re-judges ``ad`` against the *complete* adopted subgraph it
would augment (ab+bc+cd, re-fitted jointly with ad), so the response
reports the honest cause -- ``pose_residual_exceeds_limit`` with the
measured residuals -- instead of the old ``score_optimization_not_selected``
produced by re-fitting only the candidate's one-hop neighbours (which
drops b-c, turns the loop into an exactly-fittable tree, and hides the
real violation).

Usage:
    python examples/make_loop_rejection_example.py            # writes request JSON
    python examples/make_loop_rejection_example.py --call     # also POSTs it
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import cv2
import numpy as np


def rect(fid, color, w=40, h=40):
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = color
    img[:, :, 3] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return {"id": fid, "image_base64": base64.b64encode(buf).decode("ascii")}


def build_request():
    # +------+------+------+------+
    # |  A   |  B   |  C   |  D   |   identity-layout ground truth
    # +------+------+------+------+
    ys = list(range(4, 36, 4))
    chain = [{"a": [39.0, float(y)], "b": [0.0, float(y)]} for y in ys]
    # Self-consistent pairs (exact translation (39, 12), pairwise residual
    # 0) that contradict the chain-implied A->D translation (117, 0).
    ad = [{"a": [39.0, float(y)], "b": [0.0, float(y - 12)]}
          for y in range(16, 36, 4)]
    return {
        "fragments": [
            rect("A", (200, 120, 90)),
            rect("B", (210, 130, 110)),
            rect("C", (220, 140, 130)),
            rect("D", (230, 160, 150)),
        ],
        "seams": [
            {"id": "ab", "fragment_a": "A", "fragment_b": "B", "score": 8,
             "pose_error_limit": 3.0, "overlap_limit": 10 ** 9,
             "pairs": chain},
            {"id": "bc", "fragment_a": "B", "fragment_b": "C", "score": 8,
             "pose_error_limit": 3.0, "overlap_limit": 10 ** 9,
             "pairs": chain},
            {"id": "cd", "fragment_a": "C", "fragment_b": "D", "score": 8,
             "pose_error_limit": 3.0, "overlap_limit": 10 ** 9,
             "pairs": chain},
            {"id": "ad", "fragment_a": "A", "fragment_b": "D", "score": 5,
             "pose_error_limit": 3.0, "overlap_limit": 10 ** 9,
             "pairs": ad},
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
        for cl in body["clusters"]:
            with open(os.path.join(here,
                                   f"loop_cluster_{cl['cluster_id']}.png"),
                      "wb") as f:
                f.write(base64.b64decode(cl["image_base64"]))
        print("wrote", out_path, "and cluster PNGs")


if __name__ == "__main__":
    main()
