"""Rejection-reason attribution tests.

The service must judge every rejected candidate against the *complete*
selected subgraph it would augment (the whole connected component(s) of
the adopted-seam graph touching its endpoints), re-fitted jointly under
one set of global poses -- never against one-hop edges only, which drops
the middle of a selected path and hides the closed loop that actually
forbids the candidate.
"""

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from tests.synth import fragment_payload, seam_payload

client = TestClient(app)


def _rect(fid, w=40, h=40, color=(90, 120, 200)):
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = color
    img[:, :, 3] = 255
    return fragment_payload(fid, img)


def _chain_pairs():
    """Consistent horizontal-neighbour pairs: b-local -> a-local is (39, 0)."""
    return [([39.0, float(y)], [0.0, float(y)]) for y in range(4, 36, 4)]


def _path_abcd_body(ad_pairs, ad_limit=3.0, chain_limit=3.0, ad_score=5):
    """Selected seams form the path A-B-C-D; candidate ``ad`` closes a loop.

    Ground truth is the identity layout of four 40px squares in a row, so
    the chain implies the A->D relative translation (117, 0).  ``ad_pairs``
    encode a different relative pose, making the closed loop A-B-C-D-A
    pose-inconsistent under any single global pose set.
    """
    chain = _chain_pairs()
    return {
        "fragments": [
            _rect("A", color=(90, 120, 200)),
            _rect("B", color=(110, 130, 210)),
            _rect("C", color=(130, 140, 220)),
            _rect("D", color=(150, 160, 230)),
        ],
        "seams": [
            seam_payload("ab", "A", "B", chain, 8, chain_limit, 10 ** 9),
            seam_payload("bc", "B", "C", chain, 8, chain_limit, 10 ** 9),
            seam_payload("cd", "C", "D", chain, 8, chain_limit, 10 ** 9),
            seam_payload("ad", "A", "D", ad_pairs, ad_score, ad_limit, 10 ** 9),
        ],
    }


# Candidate A-D pairs: self-consistent (exact translation (39, 12)), so the
# pairwise pre-screen passes, but the implied A->D pose contradicts the
# chain's (117, 0) by ~79 px -- no global pose set meets a 3 px limit.
AD_LOOP_PAIRS = [([39.0, float(y)], [0.0, float(y - 12)])
                 for y in range(16, 36, 4)]


def test_path_candidate_closing_inconsistent_loop_gets_pose_reason():
    """A-B-C-D adopted; A-D must not be blamed on score optimisation."""
    r = client.post("/assemblies", json=_path_abcd_body(AD_LOOP_PAIRS))
    assert r.status_code == 200, r.text
    js = r.json()

    # The optimal feasible set is the chain itself.
    assert js["adopted_seam_ids"] == ["ab", "bc", "cd"]
    assert js["total_score"] == 24.0
    assert [c["fragment_ids"] for c in js["clusters"]] == [["A", "B", "C", "D"]]

    rejected = {s["id"]: s for s in js["rejected_seams"]}
    rec = rejected["ad"]
    # The augmented subgraph closes the loop and is infeasible, so the
    # reason must be the actual pose/loop violation -- never the
    # score-optimisation placeholder produced by the old one-hop refit.
    assert rec["reason"] in ("pose_residual_exceeds_limit",
                             "loop_closure_exceeds_limit")
    # Diagnostics must corroborate the stated reason (limit is 3.0 px).
    if rec["reason"] == "pose_residual_exceeds_limit":
        assert rec["pair_residual_max"] > 3.0
    else:
        assert rec["loop_error"] > 3.0
    assert rec["pair_residual_max"] is not None
    assert rec["loop_error"] is not None


def test_augmented_failure_on_selected_edge_is_not_misreported_as_overlap():
    """Candidate within its own (loose) limit, selected edges pushed over.

    The augmented fit distributes the loop inconsistency over all four
    seams: the tight chain seams exceed their 3 px limit while the
    candidate itself stays within its loose 30 px limit.  The record must
    be adjudicated by the actually failing item (pose/loop residual of a
    selected seam) and must not falsely claim a mask-overlap violation;
    the reported diagnostics must exceed the violated limit.
    """
    body = _path_abcd_body(AD_LOOP_PAIRS, ad_limit=30.0, chain_limit=3.0)
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200, r.text
    js = r.json()

    assert js["adopted_seam_ids"] == ["ab", "bc", "cd"]
    rejected = {s["id"]: s for s in js["rejected_seams"]}
    rec = rejected["ad"]
    assert rec["reason"] in ("pose_residual_exceeds_limit",
                             "loop_closure_exceeds_limit")
    assert rec["reason"] != "mask_overlap_exceeds_limit"
    assert rec["reason"] != "score_optimization_not_selected"
    # Diagnostics are the measured values of the constraint that actually
    # failed (a chain seam with a 3 px limit), so they corroborate the
    # reason instead of contradicting it.
    if rec["reason"] == "pose_residual_exceeds_limit":
        assert rec["pair_residual_max"] > 3.0
    else:
        assert rec["loop_error"] > 3.0


def test_bridge_candidate_between_two_clusters_checks_both_sides():
    """A candidate bridging two adopted clusters is re-judged jointly.

    The bridge ``bd`` is pairwise-consistent but stacks D exactly onto B,
    blowing its zero-pixel overlap budget; both adopted clusters must be
    included in the augmented fit while the unrelated isolated fragment
    and the other cluster keep their poses and rejection attribution.
    """
    chain = _chain_pairs()
    body = {
        "fragments": [
            _rect("A", color=(90, 120, 200)),
            _rect("B", color=(110, 130, 210)),
            _rect("C", color=(130, 140, 220)),
            _rect("D", color=(150, 160, 230)),
            _rect("E", color=(170, 180, 240)),
        ],
        "seams": [
            seam_payload("ab", "A", "B", chain, 8, 3.0, 10 ** 9),
            seam_payload("cd", "C", "D", chain, 8, 3.0, 10 ** 9),
            # Pairwise-exact identity fit: D lands squarely on B (1600 px
            # of solid-mask overlap) once both clusters are re-fitted.
            seam_payload("bd", "B", "D",
                         [([20.0, float(y)], [20.0, float(y)])
                          for y in range(4, 36, 4)],
                         5, 3.0, 0),
        ],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200, r.text
    js = r.json()

    assert js["adopted_seam_ids"] == ["ab", "cd"]
    assert [c["fragment_ids"] for c in js["clusters"]] == [
        ["A", "B"], ["C", "D"], ["E"]]

    rejected = {s["id"]: s for s in js["rejected_seams"]}
    rec = rejected["bd"]
    assert rec["reason"] == "mask_overlap_exceeds_limit"
    assert rec["overlap_pixels"] > 0

    # Attribution: the bridge is reported on the cluster of its
    # fragment_a (B -> the A-B cluster); the unrelated clusters carry no
    # rejection and keep their identity/chain poses.
    cl_ab, cl_cd, cl_e = js["clusters"]
    assert [s["id"] for s in cl_ab["rejected_seams"]] == ["bd"]
    assert cl_cd["rejected_seams"] == []
    assert cl_e["rejected_seams"] == []
    poses_e = cl_e["poses"][0]
    assert poses_e["theta"] == 0.0 and poses_e["tx"] == 0.0 \
        and poses_e["ty"] == 0.0
    poses_cd = {p["fragment_id"]: p for p in cl_cd["poses"]}
    assert abs(poses_cd["C"]["theta"]) < 1e-9
    assert abs(poses_cd["D"]["tx"] - 39.0) < 1e-6


def test_rejected_loop_candidate_not_in_render_and_png_stays_transparent():
    """The rejected A-D seam must not participate in the cluster image."""
    r = client.post("/assemblies", json=_path_abcd_body(AD_LOOP_PAIRS))
    js = r.json()
    cl = js["clusters"][0]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(cl["image_base64"]),
                                     np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None and img.shape[2] == 4
    # Four 40px squares in a row plus padding: the bad A-D placement
    # (which would stack D near A) must not appear.
    assert img.shape[1] > 150  # full row of four shards is rendered
    assert int(img[:, :, 3].sum()) > 0
