"""Tests for the exact global optimizer (branch & bound, never greedy)."""

import numpy as np

from app.assembler import Edge, evaluate_subset, optimize
from app.geometry import fit_rigid, rotation_matrix


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mask():
    m = np.zeros((120, 120), bool)
    m[40:80, 10:50] = True
    return m


def _triangle_masks():
    m = np.zeros((120, 120), bool)
    m[40:80, 10:50] = True
    m2 = np.zeros((120, 120), bool)
    m2[40:80, 0:40] = True
    m3 = np.zeros((120, 120), bool)
    m3[40:80, 0:40] = True
    return {"A": m, "B": m2, "C": m3}


def _edge(eid, a, b, rel, src_pts, score, pose_limit, overlap_limit=10**9):
    Rr = rotation_matrix(rel[0])
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = (src_pts - rel[1]) @ Rr
    fit = fit_rigid(dst_pts, src_pts)
    return Edge(eid, a, b, float(score), float(pose_limit), int(overlap_limit),
                src_pts, dst_pts, fit.theta, fit.t, fit.max_residual, fit.rms)


def _stub_edge(eid, score=1.0):
    return Edge(eid, "x", "y", score, 1.0, 0,
                np.zeros((2, 2)), np.zeros((2, 2)),
                0.0, np.zeros(2), 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Real geometry: loop closure forces an edge out
# --------------------------------------------------------------------------- #
def test_loop_inconsistent_edge_is_rejected():
    masks = _triangle_masks()
    pts = np.array([[10.0, y] for y in range(42, 78, 6)])
    ab = _edge("ab", "A", "B", (0.0, np.array([-50.0, 0.0])), pts, 10, 4)
    bc = _edge("bc", "B", "C", (0.0, np.array([-50.0, 0.0])),
               pts + np.array([-50.0, 0.0]), 10, 4)
    # A-C with a 15px loop mismatch; joint LS leaves every edge at 5px > 4
    ac = _edge("ac", "A", "C", (0.0, np.array([-115.0, 0.0])), pts, 10, 4)

    # The full loop is infeasible ...
    ok_full, metrics, _, _ = evaluate_subset([ab, bc, ac], masks, {})
    assert not ok_full
    assert all(m.loop_error > 4.0 for m in metrics.values())

    # ... while each two-edge chain (a tree, no closed loop) is feasible, so
    # the exact solver keeps a score-optimal two-edge combination instead of
    # ever accepting the inconsistent three-edge loop.
    chosen, cache = optimize([ab, bc, ac], masks)
    assert len(chosen) == 2
    ok, _, _, _ = evaluate_subset(chosen, masks, cache)
    assert ok


# --------------------------------------------------------------------------- #
# Search semantics with an injected feasibility oracle
# --------------------------------------------------------------------------- #
def test_optimizer_is_exact_not_greedy():
    # Scores: hi is the best single edge; combining the two lower edges
    # beats it, but any set containing hi is infeasible together with
    # either lower edge. Greedy-by-score returns just hi (score 10); the
    # exact solver returns mid+lo (score 12).
    hi = _stub_edge("hi", score=10)
    mid = _stub_edge("mid", score=6)
    lo = _stub_edge("lo", score=6)

    def feasible(subset):
        ids = {e.id for e in subset}
        if "hi" in ids and (ids & {"mid", "lo"}):
            return False
        return True

    chosen, _ = optimize([hi, mid, lo], {}, feasible=feasible)
    assert sorted(e.id for e in chosen) == ["lo", "mid"]


def test_optimizer_tie_break_lexicographic_id_list():
    e1, e2, e3 = _stub_edge("z"), _stub_edge("a"), _stub_edge("m")
    # all singletons feasible; pairs infeasible -> exactly one chosen
    chosen, _ = optimize([e1, e2, e3], {},
                         feasible=lambda s: len(s) <= 1)
    assert [e.id for e in chosen] == ["a"]


def test_optimizer_tie_break_on_sorted_id_tuple():
    a, b, c = _stub_edge("a"), _stub_edge("b"), _stub_edge("c")
    # maximum feasible size is 2; only {a,b} is infeasible, so the two
    # optimal equal-score sets are {a,c} and {b,c} -> {a,c} wins on
    # lexicographic order of the sorted seam-id list
    chosen, _ = optimize(
        [a, b, c], {},
        feasible=lambda s: len(s) <= 2
        and {e.id for e in s} != {"a", "b"})
    assert sorted(e.id for e in chosen) == ["a", "c"]


def test_optimizer_full_set_shortcut_and_empty():
    masks = _triangle_masks()
    pts = np.array([[10.0, y] for y in range(42, 78, 6)])
    ab = _edge("ab", "A", "B", (0.0, np.array([-50.0, 0.0])), pts, 10, 5)
    chosen, _ = optimize([ab], masks)
    assert [e.id for e in chosen] == ["ab"]
    assert optimize([], masks)[0] == []
