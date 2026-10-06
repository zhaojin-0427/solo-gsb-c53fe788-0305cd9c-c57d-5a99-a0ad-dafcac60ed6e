"""End-to-end multi-fragment tests with loop closure and rejected seams."""

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.assembler import build_model
from app.main import app
from app.assembler import build_model, evaluate_subset
from app.relax import lp_relaxation_feasible
from tests.synth import fragment_payload, make_shard_pair, seam_payload

client = TestClient(app)


def _png(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _square(color_bgr, size=40):
    img = np.zeros((size, size, 4), np.uint8)
    img[:, :, :3] = color_bgr
    img[:, :, 3] = 255
    return img


def test_loop_closure_inconsistent_candidate_is_reported():
    """Three fragments: a consistent chain plus a contradictory shortcut.

    Any two edges form a tree and fit exactly; the contradiction only
    exists once the shortcut closes the loop (three edges, one cycle).
    The exact solver must refuse the 3-edge loop (no single global pose
    set meets the tight limit) and report the excluded shortcut with an
    honest pose/loop reason instead of silently rendering a bad assembly.
    """
    A = _square((90, 120, 200))
    B = _square((110, 130, 210))
    C = _square((130, 140, 220))

    def pairs(ax, bx, gap):
        return [([ax, float(y)], [bx, float(y - gap)])
                for y in range(4, 36, 6)]

    ab = pairs(39, 0, 0)
    bc = pairs(39, 0, 0)
    # shortcut A-C closing a loop with a 12px mismatch (true chain gives 0)
    ac = pairs(39, 0, 12)

    body = {
        "fragments": [fragment_payload("A", A), fragment_payload("B", B),
                      fragment_payload("C", C)],
        "seams": [
            seam_payload("ab", "A", "B", ab, 8, 3.0, 10**9),
            seam_payload("bc", "B", "C", bc, 8, 3.0, 10**9),
            seam_payload("ac", "A", "C", ac, 9, 3.0, 10**9),
        ],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200, r.text
    js = r.json()

    # The three-edge loop is infeasible (4px/edge compromise > 3px limit),
    # so exactly two edges are adopted and the winner is a spanning tree.
    adopted = set(js["adopted_seam_ids"])
    assert len(adopted) == 2
    assert adopted <= {"ab", "bc", "ac"}
    assert js["total_score"] == max(8 + 8, 8 + 9, 8 + 9)

    # The excluded contradictory seam is reported with an honest reason and
    # the measured (non-zero) pose/loop diagnostics.
    rejected = {s["id"]: s for s in js["rejected_seams"]}
    excluded_id = ({"ab", "bc", "ac"} - adopted).pop()
    ex = rejected[excluded_id]
    assert ex["reason"] in ("pose_residual_exceeds_limit",
                            "loop_closure_exceeds_limit",
                            "score_optimization_not_selected")
    if ex["reason"] != "score_optimization_not_selected":
        assert ex["pair_residual_max"] is not None

    # One connected cluster; all rendered under a single common pose.
    assert len(js["clusters"]) == 1
    cl = js["clusters"][0]
    assert cl["fragment_ids"] == ["A", "B", "C"]
    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(cl["image_base64"]), np.uint8),
        cv2.IMREAD_UNCHANGED)
    assert img.shape[2] == 4


def test_full_inconsistent_loop_is_infeasible_at_engine_level():
    """Directly assert the closed loop fails the global feasibility check."""
    from app.assembler import build_model, evaluate_subset
    from app.schemas import AssemblyRequest

    A = _square((90, 120, 200))
    B = _square((110, 130, 210))
    C = _square((130, 140, 220))

    def pairs(ax, bx, gap):
        # list of (pa, pb) coordinate lists for seam_payload
        return [([ax, float(y)], [bx, float(y - gap)])
                for y in range(4, 36, 6)]

    payload = AssemblyRequest.model_validate({
        "fragments": [fragment_payload("A", A), fragment_payload("B", B),
                      fragment_payload("C", C)],
        "seams": [
            seam_payload("ab", "A", "B", pairs(39, 0, 0), 8, 3.0, 10**9),
            seam_payload("bc", "B", "C", pairs(39, 0, 0), 8, 3.0, 10**9),
            seam_payload("ac", "A", "C", pairs(39, 0, 12), 9, 3.0, 10**9),
        ],
    })
    _, masks, edges = build_model(payload)
    ok, metrics, _, _ = evaluate_subset(edges, masks, {})
    assert not ok
    assert any(m.loop_error > 3.0 or m.pair_max > 3.0 for m in metrics.values())


def test_rejected_seam_does_not_join_clusters():
    """Two fragments with only an overlap-violating seam stay separate."""
    A = _square((80, 100, 180))
    B = _square((80, 100, 180))
    pairs_ = [([39.0, float(y)], [0.0, float(y)]) for y in range(4, 36, 6)]
    body = {
        "fragments": [fragment_payload("A", A), fragment_payload("B", B)],
        "seams": [seam_payload("ab", "A", "B", pairs_, 5, 5.0, 0)],
    }
    r = client.post("/assemblies", json=body)
    js = r.json()
    assert js["adopted_seam_ids"] == []
    assert [c["fragment_ids"] for c in js["clusters"]] == [["A"], ["B"]]
    assert js["clusters"][0]["rejected_seams"][0]["reason"] == \
        "mask_overlap_exceeds_limit"


def test_unknown_fragment_reference_is_422():
    d = make_shard_pair(seed=11)
    body = {
        "fragments": [fragment_payload("A", d["A"])],
        "seams": [seam_payload("s", "A", "GHOST", d["pairs"], 3, 5, 0)],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 422
    assert "GHOST" in r.text


def test_duplicate_ids_are_422():
    d = make_shard_pair(seed=12)
    body = {
        "fragments": [fragment_payload("A", d["A"]), fragment_payload("A", d["B"])],
        "seams": [],
    }
    assert client.post("/assemblies", json=body).status_code == 422


# --------------------------------------------------------------------------- #
# Multi-cluster subset search: a cyclic component plus an independent seam
# --------------------------------------------------------------------------- #
def _rect(fid, w, h, color):
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = color
    img[:, :, 3] = 255
    return fragment_payload(fid, img)


def _triangle_plus_chain_body():
    """Five fragments, one connected *candidate* seam graph.

    Layout (identity poses): A and B are 40x40 squares, C is an 80x40
    rectangle, D and E are 40x40 squares::

        +----+----+---------+----+----+
        | A  | B  |    C    | D  | E  |
        +----+----+---------+----+----+

    ``ab``, ``ac`` and ``bc`` form a geometrically consistent triangle
    (the bc chord closes within a 1px loop tolerance) whose solid masks
    overlap flush, so ``bc`` alone fails its ``overlap_limit=0`` while its
    *pose* LP stays feasible.  ``cd`` is a low-score bridge and ``de`` an
    independent seam: while the search tries the A-B-C cycle with ``cd``
    unselected, the temporary subset splits into a cyclic component
    (3 free pose variables) and an independent seam component (1 free
    variable) -- constraint rows of different widths must be solved as one
    block-diagonal LP instead of crashing with ``ValueError``.
    """
    ys = list(range(4, 36, 4))
    ab = [([39.0, float(y)], [0.0, float(y)]) for y in ys]
    ac = [([float(x), 39.0], [float(x), 0.0]) for x in ys]
    bc = [([float(x), 39.0], [float(40 + x), 0.0]) for x in ys]
    cd = [([79.0, float(y)], [0.0, float(y)]) for y in ys]
    de = [([79.0, float(y)], [0.0, float(y)]) for y in ys]
    return {
        "fragments": [
            _rect("A", 40, 40, (90, 120, 200)),
            _rect("B", 40, 40, (110, 130, 210)),
            _rect("C", 80, 40, (130, 140, 220)),
            _rect("D", 40, 40, (150, 160, 230)),
            _rect("E", 40, 40, (170, 180, 240)),
        ],
        "seams": [
            seam_payload("ab", "A", "B", ab, 8, 3.0, 10 ** 9),
            seam_payload("ac", "A", "C", ac, 8, 3.0, 10 ** 9),
            seam_payload("bc", "B", "C", bc, 9, 3.0, 0),
            seam_payload("cd", "C", "D", cd, 1, 3.0, 10 ** 9),
            seam_payload("de", "D", "E", de, 7, 3.0, 10 ** 9),
        ],
    }


def test_lp_relaxation_handles_multi_component_subset():
    """Cyclic component + independent seam must give a boolean, never raise."""
    from app.schemas import AssemblyRequest

    _, _, edges = build_model(
        AssemblyRequest.model_validate(_triangle_plus_chain_body()))
    by_id = {e.id: e for e in edges}

    cycle = [by_id[i] for i in ("bc", "ab", "ac")]
    # The cycle fits a single pose set under the linearised LP ...
    assert lp_relaxation_feasible(cycle) is True
    # ... and so does the temporary subset cycle + independent seam, which
    # used to raise "inhomogeneous shape" ValueError from ragged LP rows.
    subset = cycle + [by_id["de"]]
    assert lp_relaxation_feasible(subset) is True


def test_lp_relaxation_is_sound_across_components():
    """One contradictory component must make the joint block LP infeasible."""
    from app.schemas import AssemblyRequest

    _, _, edges = build_model(
        AssemblyRequest.model_validate(_triangle_plus_chain_body()))
    by_id = {e.id: e for e in edges}

    # Corrupt bc into a contradictory shortcut: the A-B-C pose loop cannot
    # close under a 3px limit, even though de alone is perfectly feasible.
    bad = by_id["bc"]
    bad.b_pts = bad.b_pts.copy()
    bad.b_pts[:, 1] += 12.0  # 12px loop mismatch
    from app.geometry import fit_rigid
    fit = fit_rigid(bad.b_pts, bad.a_pts)
    bad.local_theta, bad.local_t = fit.theta, fit.t
    bad.local_max = fit.max_residual
    assert lp_relaxation_feasible([bad, by_id["ab"], by_id["ac"],
                                   by_id["de"]]) is False


def test_multi_cluster_subset_search_returns_optimal_assembly():
    """Full POST: the ragged-LP node must not abort the assembly (was 500)."""
    r = client.post("/assemblies", json=_triangle_plus_chain_body())
    assert r.status_code == 200, r.text
    js = r.json()

    # Best feasible set drops the overlap-failing chord bc (score 9) and
    # keeps ab+ac (16) plus the bridge chain cd+de (8) => total 24.
    assert js["adopted_seam_ids"] == ["ab", "ac", "cd", "de"]
    assert js["total_score"] == 24.0

    assert len(js["clusters"]) == 1
    cl = js["clusters"][0]
    assert cl["fragment_ids"] == ["A", "B", "C", "D", "E"]
    adopted = {s["id"]: s for s in cl["adopted_seams"]}
    for sid in js["adopted_seam_ids"]:
        assert adopted[sid]["pair_residual_max"] <= 3.0 + 1e-6
        assert adopted[sid]["loop_error"] <= 3.0 + 1e-6

    # The excluded chord gets an honest reason and measured overlap count.
    rejected = {s["id"]: s for s in js["rejected_seams"]}
    assert rejected["bc"]["reason"] == "mask_overlap_exceeds_limit"
    assert rejected["bc"]["overlap_pixels"] > 0

    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(cl["image_base64"]), np.uint8),
        cv2.IMREAD_UNCHANGED)
    assert img.ndim == 3 and img.shape[2] == 4
    assert int(img[:, :, 3].sum()) > 0


# --------------------------------------------------------------------------- #
# Rejection attribution: the whole affected selected subgraph is re-fitted
# --------------------------------------------------------------------------- #
def _square_png(fid, color=(90, 120, 200), size=40):
    return fragment_payload(fid, _square(color, size))


def _vseam(sid, fa, fb, gap, score, limit=3.0, overlap=10 ** 9):
    """Horizontal seam: ``a``'s right edge to ``b``'s left edge.

    ``gap`` shifts every B-side point down by ``gap`` pixels, i.e. it is a
    mismatch along the chain direction that only manifests once the seam is
    constrained by a closed loop.
    """
    pairs = [([39.0, float(y)], [0.0, float(y - gap)])
             for y in range(4, 36, 4)]
    return seam_payload(sid, fa, fb, pairs, score, limit, overlap)


def test_path_plus_inconsistent_chord_gets_loop_reason():
    """Selected A-B-C-D path + rejected chord A-D closing a bad loop.

    The chord is perfectly consistent with a tree re-fit when only the
    incident edges (ab, cd) are retried, which is why the old attribution
    reported ``score_optimization_not_selected``.  The honest trial must
    include the whole path (bc included); the four-edge loop admits no
    single global pose set inside the 3px limit.
    """
    body = {
        "fragments": [_square_png("A", (90, 120, 200)),
                      _square_png("B", (110, 130, 210)),
                      _square_png("C", (130, 140, 220)),
                      _square_png("D", (150, 160, 230))],
        "seams": [_vseam("ab", "A", "B", 0, 8),
                  _vseam("bc", "B", "C", 0, 8),
                  _vseam("cd", "C", "D", 0, 8),
                  _vseam("ad", "A", "D", 20, 7)],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200, r.text
    js = r.json()

    # The path (24) beats any chord-containing tree; the 4-edge loop fails.
    assert js["adopted_seam_ids"] == ["ab", "bc", "cd"]
    assert js["total_score"] == 24.0
    rejected = {s["id"]: s for s in js["rejected_seams"]}
    rec = rejected["ad"]
    # Must NOT be blamed on score optimisation: the augmented set is infeasible.
    assert rec["reason"] in ("pose_residual_exceeds_limit",
                             "loop_closure_exceeds_limit")
    # Diagnostics come from the failing augmented fit and exceed the limit.
    assert rec["pair_residual_max"] is not None
    assert rec["loop_error"] > 3.0
    assert rec["pair_residual_max"] > 3.0

    # Engine-level ground truth: the full four-edge set really is infeasible
    # and every tree subset (the old trial was one) is feasible.
    from app.schemas import AssemblyRequest
    _, masks, edges = build_model(AssemblyRequest.model_validate(body))
    by_id = {e.id: e for e in edges}
    ok_full, metrics, _, _ = evaluate_subset(
        [by_id[i] for i in ("ab", "bc", "cd", "ad")], masks, {})
    assert not ok_full
    assert any(m.pair_max > 3.0 or m.loop_error > 3.0
               for m in metrics.values())
    # The old, too-small trial (chord + only its one-hop edges) is a tree and
    # therefore feasible -- reproducing the missed B-C pathology.
    ok_tree, _, _, _ = evaluate_subset(
        [by_id[i] for i in ("ab", "cd", "ad")], masks, {})
    assert ok_tree


def test_augmented_failure_uses_actual_failing_edge_limit():
    """The candidate's own metrics pass its generous limit, but an accepted
    seam with a tighter limit fails once the joint poses compromise.

    The verdict must follow the edge that actually fails (a pose/loop
    failure), never fall through to ``mask_overlap_exceeds_limit``.
    """
    body = {
        "fragments": [_square_png("A"), _square_png("B"), _square_png("C")],
        "seams": [_vseam("ab", "A", "B", 0, score=8, limit=15.0),
                  _vseam("bc", "B", "C", 0, score=8, limit=20.0),
                  _vseam("ac", "A", "C", 6, score=7, limit=20.0)],
    }
    from app.schemas import AssemblyRequest
    _, masks, edges = build_model(AssemblyRequest.model_validate(body))
    by_id = {e.id: e for e in edges}
    ok_aug, metrics, _, _ = evaluate_subset(
        [by_id["ab"], by_id["bc"], by_id["ac"]], masks, {})
    assert not ok_aug
    # The candidate chord itself stays within its own generous limit ...
    assert metrics["ac"].pair_max <= 20.0 + 1e-9
    assert metrics["ac"].loop_error <= 20.0 + 1e-9
    assert metrics["ac"].overlap <= by_id["ac"].overlap_limit
    # ... while a previously accepted tight edge is pushed past its limit.
    assert metrics["ab"].pair_max > 15.0 or metrics["ab"].loop_error > 15.0

    r = client.post("/assemblies", json=body)
    js = r.json()
    assert set(js["adopted_seam_ids"]) == {"ab", "bc"}
    rec = {s["id"]: s for s in js["rejected_seams"]}["ac"]
    assert rec["reason"] in ("pose_residual_exceeds_limit",
                             "loop_closure_exceeds_limit")
    assert rec["reason"] != "mask_overlap_exceeds_limit"
    assert rec["pair_residual_max"] > 15.0 or rec["loop_error"] > 15.0
    # The reported overlap number never exceeds the candidate's limit, so the
    # mask diagnostic is not fabricated into a false overlap violation.
    assert rec["overlap_pixels"] <= by_id["ac"].overlap_limit


def test_candidate_bridging_two_clusters_checks_both_sides():
    """Accepted seams rigidify two groups joined by an accepted bridge;
    the rejected candidate is a second, contradictory bridge whose loop
    spans *both* groups.  The augmented trial must pull in the constraints
    on both sides, and unrelated fragments keep singleton poses.
    """
    xs = list(range(4, 36, 4))
    ys = xs
    body = {
        "fragments": [_rect("A", 40, 40, (90, 120, 200)),
                      _rect("B", 40, 40, (110, 130, 210)),
                      _rect("C", 80, 40, (130, 140, 220)),
                      _rect("D", 40, 40, (150, 160, 230)),
                      _rect("E", 40, 40, (170, 180, 240)),
                      _rect("F", 80, 40, (190, 200, 250)),
                      _rect("Z", 40, 40, (40, 200, 120))],
        "seams": [
            # rigid triangle A-B-C
            seam_payload("ab", "A", "B",
                         [([39.0, float(y)], [0.0, float(y)]) for y in ys],
                         8, 3.0, 10 ** 9),
            seam_payload("ac", "A", "C",
                         [([float(x), 39.0], [float(x), 0.0]) for x in xs],
                         8, 3.0, 10 ** 9),
            seam_payload("bc", "B", "C",
                         [([float(x), 39.0], [float(40 + x), 0.0]) for x in xs],
                         8, 3.0, 10 ** 9),
            # rigid triangle D-E-F
            seam_payload("de", "D", "E",
                         [([39.0, float(y)], [0.0, float(y)]) for y in ys],
                         8, 3.0, 10 ** 9),
            seam_payload("df", "D", "F",
                         [([float(x), 39.0], [float(x), 0.0]) for x in xs],
                         8, 3.0, 10 ** 9),
            seam_payload("ef", "E", "F",
                         [([float(x), 39.0], [float(40 + x), 0.0]) for x in xs],
                         8, 3.0, 10 ** 9),
            # accepted bridge B-D joining the two groups
            seam_payload("bd", "B", "D",
                         [([39.0, float(y)], [0.0, float(y)]) for y in ys],
                         7, 3.0, 10 ** 9),
            # contradictory shortcut A-E: closes a loop that spans both groups
            seam_payload("ae", "A", "E",
                         [([39.0, float(y)], [0.0, float(y - 20)]) for y in ys],
                         6, 3.0, 10 ** 9),
        ],
    }
    from app.schemas import AssemblyRequest
    _, masks, edges = build_model(AssemblyRequest.model_validate(body))
    by_id = {e.id: e for e in edges}
    chosen_ids = ["ab", "ac", "bc", "de", "df", "ef", "bd"]
    ok_chosen, _, _, _ = evaluate_subset(
        [by_id[i] for i in chosen_ids], masks, {})
    assert ok_chosen
    # Adding the shortcut closes an infeasible cross-group loop ...
    ok_aug, metrics_aug, _, _ = evaluate_subset(
        [by_id[i] for i in chosen_ids + ["ae"]], masks, {})
    assert not ok_aug
    assert any(m.pair_max > 3.0 or m.loop_error > 3.0
               for m in metrics_aug.values())
    # ... whereas the shortcut alone with only one group is a perfectly fine
    # tree -- the one-hop trial would therefore miss the failure.
    ok_one_side, _, _, _ = evaluate_subset(
        [by_id["ab"], by_id["ac"], by_id["bc"], by_id["ae"]], masks, {})
    assert ok_one_side

    r = client.post("/assemblies", json=body)
    assert r.status_code == 200, r.text
    js = r.json()
    assert js["adopted_seam_ids"] == sorted(chosen_ids)
    rec = {s["id"]: s for s in js["rejected_seams"]}["ae"]
    assert rec["reason"] in ("pose_residual_exceeds_limit",
                             "loop_closure_exceeds_limit")
    assert rec["pair_residual_max"] > 3.0 or rec["loop_error"] > 3.0

    # One merged cluster of the six connected fragments ...
    assert len(js["clusters"]) == 2
    cl, zcl = js["clusters"]
    assert cl["fragment_ids"] == ["A", "B", "C", "D", "E", "F"]
    # ... and the unrelated singleton Z keeps its identity pose and no
    # rejected-seam attribution.
    assert zcl["fragment_ids"] == ["Z"]
    (zp,) = zcl["poses"]
    assert zp["theta"] == 0.0 and zp["tx"] == 0.0 and zp["ty"] == 0.0
    assert zcl["rejected_seams"] == []


def test_mask_reason_diagnostics_match_failure_kind():
    """Chord failing only the mask limit inside an affected subgraph:
    residual and loop stay within limits while overlap exceeds it, and the
    reported diagnostics stay consistent with the overlap reason.
    """
    body = {
        "fragments": [_rect("A", 40, 40, (90, 120, 200)),
                      _rect("B", 40, 40, (110, 130, 210)),
                      _rect("C", 80, 40, (130, 140, 220))],
        "seams": [
            seam_payload("ab", "A", "B",
                         [([39.0, float(y)], [0.0, float(y)])
                          for y in range(4, 36, 4)], 8, 3.0, 10 ** 9),
            seam_payload("ac", "A", "C",
                         [([float(x), 39.0], [float(x), 0.0])
                          for x in range(4, 36, 4)], 8, 3.0, 10 ** 9),
            # consistent loop chord (pose/loop pass) but flush masks overlap
            seam_payload("bc", "B", "C",
                         [([float(x), 39.0], [float(40 + x), 0.0])
                          for x in range(4, 36, 4)], 9, 3.0, 0),
        ],
    }
    r = client.post("/assemblies", json=body)
    js = r.json()
    rec = {s["id"]: s for s in js["rejected_seams"]}["bc"]
    assert rec["reason"] == "mask_overlap_exceeds_limit"
    assert rec["overlap_pixels"] > 0
    # Pose/loop diagnostics are the chord's actual measured values and pass.
    assert rec["pair_residual_max"] <= 3.0 + 1e-6
    assert rec["loop_error"] <= 3.0 + 1e-6
