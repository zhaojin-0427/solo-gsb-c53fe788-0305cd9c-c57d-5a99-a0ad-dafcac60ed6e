"""Global rigid pose estimation on a seam graph.

Conventions
-----------
Fragment pose ``T_f`` maps fragment-local coordinates to the world frame::

    p_world = R(theta_f) p_local + t_f

A seam between fragments a (``src``) and b (``dst``) stores the relative
pose ``(Rr, tr)`` mapping b-local coordinates into the a-local frame::

    p_a = Rr p_b + tr,   Rr = R_a^T R_b,  tr = R_a^T (t_b - t_a)

This module solves for *one* set of global fragment poses fitting every
accepted seam simultaneously (Gauss-Newton), so a seam is never accepted
from a pairwise fit alone.  Residuals are the stacked 2-D point
discrepancies in the world frame; each seam is weighted ``1 / num_pairs``
so seams with many boundary points cannot dominate seams with few.
"""

from __future__ import annotations

import numpy as np

from .geometry import rotation_matrix

J90 = np.array([[0.0, -1.0], [1.0, 0.0]])


def _spanning_tree_poses(nodes, edges, anchor):
    """Initialise global poses by walking a spanning tree from ``anchor``."""
    adj = {n: [] for n in nodes}
    for e in edges:
        Rr, tr = rotation_matrix(e["_rel"][0]), np.asarray(e["_rel"][1])
        # b -> a : Rr = R_a^T R_b, tr = R_a^T(t_b - t_a)
        adj[e["src"]].append((e["dst"], Rr, tr))
        # a -> b : inverse relative pose
        adj[e["dst"]].append((e["src"], Rr.T, -Rr.T @ tr))

    poses = {anchor: (0.0, np.zeros(2))}
    queue = [anchor]
    while queue:
        cur = queue.pop(0)
        for nxt, Rrel, trel in adj[cur]:
            if nxt not in poses:
                theta_a, t_a = poses[cur]
                Ra = rotation_matrix(theta_a)
                Rnew = Ra @ Rrel
                tnew = t_a + Ra @ trel
                theta_new = float(np.arctan2(Rnew[1, 0], Rnew[0, 0]))
                poses[nxt] = (theta_new, tnew)
                queue.append(nxt)
    for n in nodes:  # disconnected node (defensive) -> identity
        poses.setdefault(n, (0.0, np.zeros(2)))
    return poses


def forest_poses(subset):
    """BFS spanning forest of a seam subset.

    Returns a list of ``(nodes, pose, tree_edge_ids)`` per connected
    component, where ``pose`` maps node -> world pose (``(theta, t)``)
    propagated from the component's anchor (smallest id, held at identity)
    using each edge's own pairwise relative pose, and ``tree_edge_ids`` is
    the set of edges chosen for the spanning tree.  Non-tree (chord) edges
    close loops and may carry a non-zero closure gap.
    """
    import math

    nodes_set = sorted({n for e in subset for n in (e.a, e.b)})
    adj = {n: [] for n in nodes_set}
    for e in subset:
        adj[e.a].append((e.b, e))
        adj[e.b].append((e.a, e))

    result = []
    seen = set()
    for root in nodes_set:
        if root in seen:
            continue
        pose = {root: (0.0, np.zeros(2))}
        tree_ids = set()
        queue = [root]
        seen.add(root)
        while queue:
            cur = queue.pop(0)
            for nxt, e in adj[cur]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                tree_ids.add(e.id)
                if cur == e.a:
                    Rrel = rotation_matrix(e.local_theta)
                    trel = np.asarray(e.local_t)
                else:
                    Rrel = rotation_matrix(e.local_theta).T
                    trel = -rotation_matrix(e.local_theta).T @ np.asarray(e.local_t)
                theta_a, t_a = pose[cur]
                Ra = rotation_matrix(theta_a)
                Rnew = Ra @ Rrel
                tnew = t_a + Ra @ trel
                pose[nxt] = (math.atan2(Rnew[1, 0], Rnew[0, 0]), tnew)
                queue.append(nxt)
        comp = sorted(pose)
        result.append((comp, pose, tree_ids))
    return result


def solve_global_poses(nodes, edges, anchor, max_iter=30):
    """Find global poses minimising all seam residuals jointly.

    Returns a dict ``node -> (theta, t)`` with ``anchor`` fixed at identity.
    ``edges`` need keys ``src``, ``dst``, ``src_pts``, ``dst_pts`` and
    precomputed ``_rel`` (relative theta, t) and ``_w``.
    """
    if not edges:
        return {n: (0.0, np.zeros(2)) for n in nodes}

    poses = _spanning_tree_poses(nodes, edges, anchor)
    order = [n for n in nodes if n != anchor]
    index = {n: i for i, n in enumerate(order)}
    dof = 3 * len(order)
    if dof == 0:
        return poses

    def residuals_and_jacobian():
        # Gather every point residual across all edges.
        r_blocks = []
        J_blocks = []
        for e in edges:
            ta, ta_t = poses[e["src"]]
            tb, tb_t = poses[e["dst"]]
            pa = np.asarray(e["src_pts"], dtype=np.float64)
            pb = np.asarray(e["dst_pts"], dtype=np.float64)
            Ra, Rb = rotation_matrix(ta), rotation_matrix(tb)
            sw = np.sqrt(e["_w"])
            k = len(pa)

            Rbpb = pb @ Rb.T
            Rapa = pa @ Ra.T
            x = Rbpb + tb_t - Rapa - ta_t           # (k, 2) world residual
            r_blocks.append(sw * x)

            # Jacobian blocks (k, 2, 3) per incident node.
            Jfull = np.zeros((k, 2, dof))
            if e["dst"] != anchor:
                j = 3 * index[e["dst"]]
                Jfull[:, :, j:j + 2] += np.broadcast_to(
                    np.eye(2), (k, 2, 2))
                Jfull[:, :, j + 2] = (J90 @ Rbpb.T).T
            if e["src"] != anchor:
                i = 3 * index[e["src"]]
                Jfull[:, :, i:i + 2] += -np.broadcast_to(
                    np.eye(2), (k, 2, 2))
                Jfull[:, :, i + 2] = -(J90 @ Rapa.T).T
            J_blocks.append(sw * Jfull)

        r = np.concatenate(r_blocks, axis=0).reshape(-1)
        J = np.concatenate(J_blocks, axis=0).reshape(-1, dof)
        return r, J

    prev_cost = None
    for _ in range(max_iter):
        r, J = residuals_and_jacobian()
        delta, *_ = np.linalg.lstsq(J.T @ J, -J.T @ r, rcond=None)
        for n in order:
            i = 3 * index[n]
            theta, t = poses[n]
            poses[n] = (theta + float(delta[i + 2]), t + delta[i:i + 2])
        cost = float(r @ r)
        if prev_cost is not None and abs(prev_cost - cost) <= 1e-10 * max(1.0, prev_cost):
            break
        prev_cost = cost
    return poses
