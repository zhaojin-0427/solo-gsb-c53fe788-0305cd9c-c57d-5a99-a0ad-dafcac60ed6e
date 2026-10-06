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


def _failure_reason(edge: Edge, m: EdgeMetrics) -> str:
    if m.pair_max > edge.pose_limit + 1e-9:
        return "pose_residual_exceeds_limit"
    if m.loop_error > edge.pose_limit + 1e-9:
        return "loop_closure_exceeds_limit"
    return "mask_overlap_exceeds_limit"


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
            # Test this seam honestly together with all accepted seams,
            # under the joint global poses of the augmented graph.  Include
            # every accepted edge incident on either endpoint (a candidate
            # may bridge two previously separate clusters).
            involved = {e.a, e.b}
            trial = [x for x in chosen
                     if x.a in involved or x.b in involved] + [e]
            ok_aug, m_aug, _, _ = evaluate_subset(trial, masks, overlap_cache)
            if ok_aug:
                reason = "score_optimization_not_selected"
                rec = _rejection_record(e, reason, m_aug[e.id])
            else:
                reason = _failure_reason(e, m_aug[e.id])
                rec = _rejection_record(e, reason, m_aug[e.id])
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
