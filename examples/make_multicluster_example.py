"""Generate a real multi-cluster /assemblies request and run it.

This is the scenario that used to break the POST /assemblies endpoint
(HTTP 500, ``ValueError`` from the pose LP relaxation):

* the candidate seam graph is one connected graph (A-B-C plus D-E, linked
  by the bridge seam C-D);
* A-B-C contains a geometrically consistent *closed loop* (seams ab/ac/bc)
  whose chord ``bc`` violates its solid-mask overlap limit, so the full
  seam set is infeasible and the search must drop seams;
* while the temporary selection keeps the A-B-C cycle but drops the bridge,
  the subset has TWO independent components (a 3-node cyclic cluster and a
  2-node seam cluster).  Their pose-correction LP blocks have different
  numbers of variables; the service now solves them as one block-diagonal
  LP instead of merging rows of different widths.

Usage:
    python examples/make_multicluster_example.py            # writes request JSON
    python examples/make_multicluster_example.py --call     # also POSTs it
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import cv2
import numpy as np


def png_b64(w: int, h: int, color) -> str:
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = color
    img[:, :, 3] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf).decode("ascii")


def rect(fid, w, h, color):
    return {"id": fid, "image_base64": png_b64(w, h, color)}


def build_request():
    # +------+------+-----------+------+------+
    # |  A   |  B   |     C     |  D   |  E   |   identity-layout ground truth
    # +------+------+-----------+------+------+
    #   40     40        80         40     40
    ys = list(range(4, 36, 4))
    ab = [{"a": [39.0, float(y)], "b": [0.0, float(y)]} for y in ys]
    ac = [{"a": [float(x), 39.0], "b": [float(x), 0.0]} for x in ys]
    bc = [{"a": [float(x), 39.0], "b": [float(40 + x), 0.0]} for x in ys]
    cd = [{"a": [79.0, float(y)], "b": [0.0, float(y)]} for y in ys]
    de = [{"a": [79.0, float(y)], "b": [0.0, float(y)]} for y in ys]
    return {
        "fragments": [
            rect("A", 40, 40, (200, 120, 90)),
            rect("B", 40, 40, (210, 130, 110)),
            rect("C", 80, 40, (220, 140, 130)),
            rect("D", 40, 40, (230, 160, 150)),
            rect("E", 40, 40, (240, 180, 170)),
        ],
        "seams": [
            {"id": "ab", "fragment_a": "A", "fragment_b": "B", "score": 8,
             "pose_error_limit": 3.0, "overlap_limit": 10 ** 9, "pairs": ab},
            {"id": "ac", "fragment_a": "A", "fragment_b": "C", "score": 8,
             "pose_error_limit": 3.0, "overlap_limit": 10 ** 9, "pairs": ac},
            # Consistent loop chord, but its flush masks overlap 40 px > 0.
            {"id": "bc", "fragment_a": "B", "fragment_b": "C", "score": 9,
             "pose_error_limit": 3.0, "overlap_limit": 0, "pairs": bc},
            # Low-score bridge; may be temporarily dropped during the search.
            {"id": "cd", "fragment_a": "C", "fragment_b": "D", "score": 1,
             "pose_error_limit": 3.0, "overlap_limit": 10 ** 9, "pairs": cd},
            # Independent seam D-E inside the (connected) candidate graph.
            {"id": "de", "fragment_a": "D", "fragment_b": "E", "score": 7,
             "pose_error_limit": 3.0, "overlap_limit": 10 ** 9, "pairs": de},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8000/assemblies")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    req = build_request()
    req_path = os.path.join(here, "multi_request.json")
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
        out_path = os.path.join(here, "multi_response.json")
        with open(out_path, "w") as f:
            json.dump(body, f, indent=2)
        for cl in body["clusters"]:
            with open(os.path.join(here,
                                   f"multi_cluster_{cl['cluster_id']}.png"),
                      "wb") as f:
                f.write(base64.b64decode(cl["image_base64"]))
        print("wrote", out_path, "and cluster PNGs")


if __name__ == "__main__":
    main()
