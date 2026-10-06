"""Linear-programming relaxation of the seam feasibility test.

The exact question is: *does there exist a single set of global fragment
poses for which every selected seam's point residuals, loop-closure error
and (handled separately, non-linear) mask overlap stay within limits?*

Pose corrections are linearised around the spanning-forest propagated poses
(where every tree edge has zero residual).  Variables per non-anchor
fragment: ``(dx, dy, dtheta)``.  Every Euclidean residual
``|r + J @ delta| <= L`` is outer-approximated with a 12-gon
(``n_d . (r + J delta) <= L`` for 12 evenly spaced unit directions),
turning feasibility into an LP.

A candidate seam graph may be connected while a DFS *subset* is not: when
the seam joining two pieces is temporarily unselected, the subset is a
forest of several components, each with its own anchor and therefore its
own number of correction variables.  The LP must not couple those
independent components -- each component's pose correction is free around
its own anchor.  We therefore number variables over *every* non-anchor
node of *every* component once, in one global vector, and let each
component's rows touch only its own columns.  The resulting matrix is
block-diagonal, so the single LP is exactly the conjunction of the
independent per-component LPs (and has one uniform row width -- stacking
per-component rows of differing widths into one array would crash).

Because this is an *outer* approximation (linearised geometry, polygonal
norm), ``LP infeasible`` guarantees the true problem is infeasible as
well, so pruning on it can never discard a valid seam set.  LP-feasible
nodes still get the exact Gauss-Newton + rasterised-overlap check.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog

from .geometry import rotation_matrix
from .posegraph import forest_poses

_NDIR = 12
_DIRS = np.column_stack([
    np.cos(np.linspace(0.0, 2.0 * np.pi, _NDIR, endpoint=False)),
    np.sin(np.linspace(0.0, 2.0 * np.pi, _NDIR, endpoint=False)),
])  # (12, 2)
J90 = np.array([[0.0, -1.0], [1.0, 0.0]])


def lp_relaxation_feasible(subset) -> bool:
    """Conservative necessary-feasibility LP over one seam subset.

    The subset may span several mutually disconnected components; each is
    anchored at its smallest-id fragment and gets an independent block of
    correction variables in one block-diagonal LP.
    """
    if not subset:
        return True

    # One global variable layout: every non-anchor node of every component.
    col: Dict[str, int] = {}
    for nodes, _pose, _tree_ids in forest_poses(subset):
        anchor = nodes[0]
        for n in nodes:
            if n != anchor:
                col[n] = 3 * len(col)
    nv = 3 * len(col)

    A_rows: List[np.ndarray] = []
    b_rows: List[float] = []

    for nodes, pose, tree_ids in forest_poses(subset):
        comp_set = set(nodes)

        def add_cone(constant: np.ndarray, coefs: List[Tuple[str, np.ndarray]],
                     limit: float):
            """``|constant + sum coefs_fid @ delta_fid|_2 <= limit`` (relaxed)."""
            for d in range(_NDIR):
                nxy = _DIRS[d]
                row = np.zeros(nv)
                for fid, M in coefs:
                    c = col.get(fid)
                    if c is not None:
                        # M is the 2x3 Jacobian block of this residual;
                        # anchors have no columns, their correction is fixed 0.
                        row[c:c + 3] += nxy @ M
                A_rows.append(row)
                b_rows.append(limit - float(nxy @ constant))

        for e in subset:
            if e.a not in comp_set:
                continue
            theta_a, t_a = pose[e.a]
            theta_b, t_b = pose[e.b]
            Ra, Rb = rotation_matrix(theta_a), rotation_matrix(theta_b)

            # ---- point-pair residual in world frame -------------------- #
            gap = e.b_pts @ Rb.T + t_b - e.a_pts @ Ra.T - t_a
            # Jacobian blocks (2x3): w.r.t. correction of b and of a.
            for k in range(len(e.a_pts)):
                coefs = [
                    (e.b, np.column_stack([np.eye(2), J90 @ (Rb @ e.b_pts[k])])),
                    (e.a, np.column_stack([-np.eye(2), -J90 @ (Ra @ e.a_pts[k])])),
                ]
                add_cone(gap[k], coefs, e.pose_limit)

            # ---- loop closure: implied relative motion vs seam's fit ---- #
            if e.id not in tree_ids:
                Rr = Ra.T @ Rb
                # The seam's pairwise fit predicts b points as
                #   p_a^fit = R(local) p_b + t(local)
                # implied under tree poses: Rr p_b + Ra^T(t_b-t_a).
                rel_t = Ra.T @ (t_b - t_a)
                loop_gap_local = (
                    e.b_pts @ Rr.T + rel_t
                    - (e.b_pts @ rotation_matrix(e.local_theta).T + e.local_t)
                )  # (k, 2), in fragment-a local frame
                # Derivative of Ra^T (Rb pb + tb - ta) w.r.t. corrections:
                #   d tb : Ra^T ; d ta : -Ra^T ;
                #   d theta_b: Ra^T J90 Rb pb ; d theta_a: -J90 Rr pb
                for k in range(len(e.a_pts)):
                    Mb = np.zeros((2, 3))
                    Mb[:, :2] = Ra.T
                    Mb[:, 2] = Ra.T @ (J90 @ (Rb @ e.b_pts[k]))
                    Ma = np.zeros((2, 3))
                    Ma[:, :2] = -Ra.T
                    Ma[:, 2] = -J90 @ (Rr @ e.b_pts[k])
                    add_cone(loop_gap_local[k], [(e.b, Mb), (e.a, Ma)],
                             e.pose_limit)

    if not A_rows:
        # A single edge per component (a forest): tree poses fit exactly.
        return True
    res = linprog(np.zeros(nv),
                  A_ub=np.asarray(A_rows), b_ub=np.asarray(b_rows),
                  bounds=(None, None), method="highs")
    return res.status in (0, 3)  # optimal or unbounded -> feasible
