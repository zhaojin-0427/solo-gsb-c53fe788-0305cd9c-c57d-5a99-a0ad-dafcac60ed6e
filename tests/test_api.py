import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from tests.synth import (
    fragment_payload,
    make_shard_pair,
    seam_payload,
)

client = TestClient(app)


def _decode_resp_png(s):
    return cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8),
                        cv2.IMREAD_UNCHANGED)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_basic_two_shard_assembly_and_render():
    d = make_shard_pair(theta=0.25, tx=118.0, ty=-26.0, seed=1)
    body = {
        "fragments": [fragment_payload("A", d["A"]),
                      fragment_payload("B", d["B"])],
        "seams": [seam_payload("s1", "A", "B", d["pairs"],
                               score=10.0, pose_limit=2.0, overlap_limit=10**9)],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200, r.text
    js = r.json()
    assert js["adopted_seam_ids"] == ["s1"]
    assert js["total_score"] == 10.0
    assert len(js["clusters"]) == 1
    cl = js["clusters"][0]
    assert cl["fragment_ids"] == ["A", "B"]
    assert len(cl["adopted_seams"]) == 1
    assert cl["adopted_seams"][0]["pair_residual_max"] <= 2.0
    assert cl["adopted_seams"][0]["loop_error"] <= 2.0
    img = _decode_resp_png(cl["image_base64"])
    assert img.ndim == 3 and img.shape[2] == 4
    assert int(img[:, :, 3].sum()) > 0

    poses = {p["fragment_id"]: p for p in cl["poses"]}
    assert abs(poses["A"]["theta"]) < 1e-6
    # recovered B pose should be near truth
    assert abs(poses["B"]["theta"] - d["theta"]) < 0.05
    t = np.array([poses["B"]["tx"], poses["B"]["ty"]])
    assert np.linalg.norm(t - d["t"]) < 3.0


def test_isolated_fragments_each_own_cluster():
    d = make_shard_pair(seed=2)
    extra = np.zeros((30, 30, 4), np.uint8)
    extra[5:25, 5:25] = (0, 0, 255, 255)
    body = {
        "fragments": [fragment_payload("A", d["A"]),
                      fragment_payload("B", d["B"]),
                      fragment_payload("C", extra)],
        "seams": [],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200
    js = r.json()
    assert [c["fragment_ids"] for c in js["clusters"]] == [["A"], ["B"], ["C"]]
    for c in js["clusters"]:
        assert c["adopted_seams"] == []
        img = _decode_resp_png(c["image_base64"])
        assert img[:, :, 3].max() == 255


def test_empty_mask_is_422():
    blank = np.zeros((20, 20, 4), np.uint8)
    d = make_shard_pair(seed=3)
    body = {
        "fragments": [fragment_payload("A", d["A"]),
                      fragment_payload("BLANK", blank)],
        "seams": [],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 422
    assert "empty" in r.text.lower()


def test_degenerate_pairs_is_422():
    d = make_shard_pair(seed=4)
    body = {
        "fragments": [fragment_payload("A", d["A"]),
                      fragment_payload("B", d["B"])],
        "seams": [seam_payload("s1", "A", "B",
                               [(d["pairs"][0][0], d["pairs"][0][1]),
                                (d["pairs"][0][0] + [0, 0],
                                 d["pairs"][0][1] + [0, 0])],
                               score=5, pose_limit=5, overlap_limit=10**9)],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 422
    assert "degenerate" in r.text.lower()


def test_single_pair_is_422():
    d = make_shard_pair(seed=5)
    seam = seam_payload("s1", "A", "B", d["pairs"][:1], 5, 5, 10**9)
    body = {"fragments": [fragment_payload("A", d["A"]),
                          fragment_payload("B", d["B"])],
            "seams": [seam]}
    r = client.post("/assemblies", json=body)
    assert r.status_code == 422


def test_bad_base64_is_422():
    body = {"fragments": [{"id": "A", "image_base64": "!!!notb64!!!"}],
            "seams": []}
    r = client.post("/assemblies", json=body)
    assert r.status_code == 422


def test_overlap_limit_rejects_seam_not_rendered():
    d = make_shard_pair(theta=-0.4, seed=6)
    # Force the pieces far apart via wrong pairs is hard to overlap; instead
    # use tight overlap_limit = 0 which the touching/rotated cut cannot meet.
    body = {
        "fragments": [fragment_payload("A", d["A"]),
                      fragment_payload("B", d["B"])],
        "seams": [seam_payload("s1", "A", "B", d["pairs"],
                               score=9, pose_limit=5.0, overlap_limit=0)],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200
    js = r.json()
    assert js["adopted_seam_ids"] == []
    reasons = {x["id"]: x["reason"] for x in js["rejected_seams"]}
    assert reasons["s1"] == "mask_overlap_exceeds_limit"
    # Two singleton clusters, seam attached to cluster of fragment A
    assert [c["fragment_ids"] for c in js["clusters"]] == [["A"], ["B"]]
    cl_a = js["clusters"][0]
    assert len(cl_a["rejected_seams"]) == 1
    # unqualified seam never participates in an image: singleton renders
    ia = _decode_resp_png(cl_a["image_base64"])
    ib = _decode_resp_png(js["clusters"][1]["image_base64"])
    # combined shards would span ~250px; singletons stay near their own size
    assert ia.shape[1] <= 104 and ib.shape[0] <= 156


def test_pose_limit_rejection_pairwise():
    d = make_shard_pair(seed=7, noise=0.0)
    # non-rigid distortion of the pairs: alternating radial offset cannot
    # be absorbed by any rotation+translation, so the pairwise fit fails
    pairs = [(pa + (np.array([30.0, 0]) if i % 2 else np.array([-30.0, 0])),
              pb) for i, (pa, pb) in enumerate(d["pairs"])]
    body = {
        "fragments": [fragment_payload("A", d["A"]),
                      fragment_payload("B", d["B"])],
        "seams": [seam_payload("s1", "A", "B", pairs, 7, 1.0, 10**9)],
    }
    r = client.post("/assemblies", json=body)
    js = r.json()
    assert js["adopted_seam_ids"] == []
    assert js["rejected_seams"][0]["reason"] == \
        "pairwise_pose_residual_exceeds_limit"
