"""End-to-end assembly orchestration: builds the response payload."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .assembler import (
    Edge,
    EdgeMetrics,
    build_model,
    evaluate_subset,
    optimize,
)
from .masks import encode_png_base64, render_cluster


def _rejection_record(edge: Edge, reason: str, m: EdgeMetrics = None):
    rec = {
        "id": edge.id,
        "fragment_a": edge.a,
        "fragment_b": edge.b,
        "score": edge.score,
        "reason": reason,
    }
    if m is not None:
        rec.update({
            "pair_residual_max": m.pair_max,
            "loop_error": m.loop_error,
            "overlap_pixels": m.overlap,
        })
    return rec


def _violations(edge: Edge, m: EdgeMetrics) -> List[str]:
    """Constraint violations of one edge under the joint fit.

    Returned in the canonical reason-priority order: point-pair residual,
    then loop closure, then solid-mask overlap.
    """
    out: List[str] = []
    if m.pair_max > edge.pose_limit + 1e-9:
        out.append("pose_residual_exceeds_limit")
    if m.loop_error > edge.pose_limit + 1e-9:
        out.append("loop_closure_exceeds_limit")
    if m.overlap > edge.overlap_limit:
        out.append("mask_overlap_exceeds_limit")
    return out


def _affected_selected_subgraph(chosen: List[Edge], seed_nodes) -> List[Edge]:
    """Every selected edge in the component(s) of the selected-seam graph
    touched by ``seed_nodes`` (the candidate's two endpoints).

    A rejected candidate must be re-judged against the *complete* selected
    subgraph it would augment.  Taking only the one-hop selected edges at
    its endpoints drops the middle of a selected path (A-B-C-D loses B-C),
    turning a pose-inconsistent closed loop into an exactly-fittable tree
    and hiding the real rejection cause.  A candidate may also bridge two
    previously separate clusters; both sides' constraints must be
    re-checked, while unrelated clusters are left out entirely.
    """
    adj: Dict[str, List[Edge]] = {}
    for x in chosen:
        adj.setdefault(x.a, []).append(x)
        adj.setdefault(x.b, []).append(x)
    seen_nodes = set()
    seen_ids = set()
    out: List[Edge] = []
    stack = [n for n in seed_nodes if n in adj]
    while stack:
        cur = stack.pop()
        if cur in seen_nodes:
            continue
        seen_nodes.add(cur)
        for x in adj.get(cur, ()):
            if x.id not in seen_ids:
                seen_ids.add(x.id)
                out.append(x)
            nxt = x.b if x.a == cur else x.a
            if nxt not in seen_nodes:
                stack.append(nxt)
    return out


def _augmented_failure(trial: List[Edge], cand: Edge,
                       metrics: Dict[str, EdgeMetrics]):
    """Adjudicate why ``selected subgraph + candidate`` is infeasible.

    Returns ``(reason, diagnostics)``.  The candidate's own violations
    take priority.  Otherwise the augmented fit pushed an originally
    selected edge over its limit while the candidate itself is within
    limits: the reason is then taken from the edge that *actually* fails
    (never a blanket mask-overlap claim), and the reported diagnostics are
    that edge's measured values, so the numbers always corroborate the
    stated reason.
    """
    m_cand = metrics[cand.id]
    own = _violations(cand, m_cand)
    if own:
        return own[0], m_cand
    for x in sorted(trial, key=lambda e: e.id):
        if x.id == cand.id:
            continue
        v = _violations(x, metrics[x.id])
        if v:
            return v[0], metrics[x.id]
    # evaluate_subset only reports infeasible when some edge violates, so
    # this is unreachable; stay honest if it ever happens.
    return "pose_residual_exceeds_limit", m_cand


def assemble(payload) -> dict:
    images, masks, edges = build_model(payload)
    edge_by_id: Dict[str, Edge] = {e.id: e for e in edges}

    # Pairwise pre-screen: global fit cannot repair an already-failing edge.
    candidates: List[Edge] = []
    prereject: Dict[str, Edge] = {}
    for e in edges:
        if e.local_max > e.pose_limit + 1e-9:
            prereject[e.id] = e
        else:
            candidates.append(e)

    chosen, overlap_cache = optimize(candidates, masks)
    chosen_set = {e.id for e in chosen}
    total_score = float(sum(e.score for e in chosen))

    # Final authoritative metrics under the optimal global poses.
    ok, metrics, poses, comp_of = evaluate_subset(chosen, masks, overlap_cache)
    assert ok, "optimizer returned an infeasible subset"

    all_nodes = sorted(images)
    # union-find over fragments; accepted edges merge components
    parent = {fid: fid for fid in all_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for e in chosen:
        union(e.a, e.b)

    groups: Dict[str, List[str]] = {}
    for fid in all_nodes:
        groups.setdefault(find(fid), []).append(fid)
    ordered_groups = [sorted(v) for _, v in sorted(groups.items())]

    # Rejected seam assignment to clusters + honest per-seam reasons.
    rejected_by_cluster: Dict[str, List[dict]] = {g[0]: [] for g in ordered_groups}
    rejected_flat: List[dict] = []

    for e in edges:
        if e.id in chosen_set:
            continue
        if e.id in prereject:
            rec = _rejection_record(e, "pairwise_pose_residual_exceeds_limit")
            rec["pair_residual_max"] = e.local_max
            rec["loop_error"] = 0.0
            rec["overlap_pixels"] = None
        else:
            # Test this seam honestly against the *complete* selected
            # subgraph it would augment: every selected edge in the
            # connected component(s) touching its endpoints, re-fitted
            # jointly with the candidate under one set of global poses.
            # (One-hop edges alone would drop the middle of a selected
            # path and hide the closed loop that forbids the candidate;
            # a candidate bridging two clusters re-checks both sides.)
            trial = _affected_selected_subgraph(chosen, (e.a, e.b)) + [e]
            ok_aug, m_aug, _, _ = evaluate_subset(trial, masks, overlap_cache)
            if ok_aug:
                # The augmented set is feasible under one global pose set,
                # so the exact search left this seam out purely for score.
                rec = _rejection_record(e, "score_optimization_not_selected",
                                        m_aug[e.id])
            else:
                reason, diag = _augmented_failure(trial, e, m_aug)
                rec = _rejection_record(e, reason, diag)
        ra = find(e.a)
        rejected_by_cluster.setdefault(ra, []).append(rec)
        rejected_flat.append(rec)

    clusters_out = []
    for cid, members in enumerate(ordered_groups, start=1):
        anchor = members[0]
        cluster_poses = {fid: poses[fid] if fid in poses
                         else (0.0, np.zeros(2)) for fid in members}
        adopted = []
        # Accumulate point-level statistics per fragment for accurate RMS
        # (RMS must be computed over all individual point residuals, not by
        # averaging edge RMS values).
        sq_sum: Dict[str, float] = {fid: 0.0 for fid in members}
        pair_count: Dict[str, int] = {fid: 0 for fid in members}
        per_frag_max: Dict[str, float] = {fid: 0.0 for fid in members}
        for e in chosen:
            if find(e.a) != anchor:
                continue
            m = metrics[e.id]
            npts = len(e.a_pts)
            adopted.append({
                "id": e.id,
                "fragment_a": e.a,
                "fragment_b": e.b,
                "score": e.score,
                "pair_residual_max": m.pair_max,
                "loop_error": m.loop_error,
                "overlap_pixels": m.overlap,
            })
            for fid in (e.a, e.b):
                sq_sum[fid] += (m.pair_rms ** 2) * npts
                pair_count[fid] += npts
                per_frag_max[fid] = max(per_frag_max[fid], m.pair_max)

        adopted.sort(key=lambda r: r["id"])
        poses_out = []
        for fid in members:
            theta, t = cluster_poses[fid]
            poses_out.append({
                "fragment_id": fid,
                "theta": float(theta),
                "tx": float(t[0]),
                "ty": float(t[1]),
                "pair_residual_rms": float(np.sqrt(sq_sum[fid] / pair_count[fid]))
                if pair_count[fid] else 0.0,
                "pair_residual_max": float(per_frag_max[fid]),
            })

        cluster_images = {fid: images[fid] for fid in members}
        png = render_cluster(cluster_images, cluster_poses)
        cluster_score = float(sum(edge_by_id[a["id"]].score for a in adopted))
        clusters_out.append({
            "cluster_id": cid,
            "fragment_ids": members,
            "poses": poses_out,
            "adopted_seams": adopted,
            "rejected_seams": sorted(rejected_by_cluster[anchor], key=lambda r: r["id"]),
            "total_score": cluster_score,
            "image_base64": encode_png_base64(png),
        })

    rejected_flat.sort(key=lambda r: r["id"])
    return {
        "clusters": clusters_out,
        "adopted_seam_ids": sorted(chosen_set),
        "rejected_seams": rejected_flat,
        "total_score": total_score,
    }
