"""End-to-end assembly orchestration: builds the response payload.

Rejected seams are re-checked against the accepted set by fitting the
**complete** selected subgraph the candidate reaches (both sides when it
bridges two chosen clusters), not just its one-hop incident edges.  The
rejection reason follows whatever constraint actually fails under that
augmented joint fit; a feasible augmentation is the only case reported as
``score_optimization_not_selected``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .assembler import (
    EPS,
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


def _affected_subset(candidate: Edge, chosen: List[Edge]):
    """Complete selected subgraph reached by the candidate's endpoints.

    When re-checking a rejected candidate against the accepted seams, the
    trial must contain **every** accepted edge in the chosen connected
    component(s) the candidate touches -- not merely the edges one hop from
    its endpoints.  If the selected seams form the path A-B-C-D and the
    candidate is the chord A-D, closing the loop only exists once B-C is
    present; dropping it turns the augmented graph into a tree that fits
    exactly and hides the real residual / loop-closure failure.  A
    candidate may also bridge two previously separate chosen clusters, in
    which case both components are pulled in and checked jointly.
    """
    nodes = {n for e in chosen for n in (e.a, e.b)}
    adj: Dict[str, set] = {n: set() for n in nodes}
    for e in chosen:
        adj[e.a].add(e.b)
        adj[e.b].add(e.a)

    reached: set = set()
    stack = [candidate.a, candidate.b]
    while stack:
        cur = stack.pop()
        if cur in reached or cur not in adj:
            continue
        reached.add(cur)
        stack.extend(adj[cur] - reached)

    return [e for e in chosen if e.a in reached] + [candidate]


def _augmented_failure(candidate: Edge, trial: List[Edge],
                       metrics: Dict[str, EdgeMetrics]) -> tuple:
    """Classify why the augmented set is infeasible and build its record.

    The verdict follows the edges that actually fail under the joint
    global poses -- including previously accepted seams whose tighter
    per-seam limits are broken once the candidate forces a compromise --
    rather than the candidate alone.  Priority (residual, loop closure,
    mask overlap) matches the feasibility test order.  Each reported
    diagnostic is the worst measured value of that quantity over the
    edges genuinely violating the corresponding kind of limit, so the
    numbers are consistent with the reason and exceed the offending
    edge's limit; non-violating quantities fall back to the candidate's
    own measured value.
    """
    pair_offenders = [x for x in trial
                      if metrics[x.id].pair_max > x.pose_limit + EPS]
    loop_offenders = [x for x in trial
                      if metrics[x.id].loop_error > x.pose_limit + EPS]
    overlap_offenders = [x for x in trial
                         if metrics[x.id].overlap > x.overlap_limit]

    if pair_offenders:
        reason = "pose_residual_exceeds_limit"
    elif loop_offenders:
        reason = "loop_closure_exceeds_limit"
    else:
        reason = "mask_overlap_exceeds_limit"

    me = metrics[candidate.id]
    if pair_offenders:
        pair_val = max(metrics[x.id].pair_max for x in pair_offenders)
    else:
        pair_val = me.pair_max
    if loop_offenders:
        loop_val = max(metrics[x.id].loop_error for x in loop_offenders)
    else:
        loop_val = me.loop_error
    if overlap_offenders:
        overlap_val = max(metrics[x.id].overlap for x in overlap_offenders)
    else:
        overlap_val = me.overlap
    diag = EdgeMetrics(pair_val, me.pair_rms, loop_val, overlap_val)
    return reason, diag


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
            # Re-fit this candidate jointly with the *complete* selected
            # subgraph it touches (full chosen components, both sides when
            # it bridges two clusters) under one set of global poses.
            trial = _affected_subset(e, chosen)
            ok_aug, m_aug, _, _ = evaluate_subset(trial, masks, overlap_cache)
            if ok_aug:
                # The augmented set really is feasible; only then is the
                # candidate excluded purely for total-score maximisation.
                reason = "score_optimization_not_selected"
                rec = _rejection_record(e, reason, m_aug[e.id])
            else:
                # Infeasible: never blame score optimisation.  Classify by
                # the constraints that actually fail under the joint fit,
                # with diagnostics measured on the same fit.
                reason, diag = _augmented_failure(e, trial, m_aug)
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
