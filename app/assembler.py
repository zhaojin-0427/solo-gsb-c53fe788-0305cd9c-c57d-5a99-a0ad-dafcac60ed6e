"""Global shard-assembly engine.

Pipeline
--------
1. Decode every transparent PNG and extract its solid (alpha > 0) mask.
2. Fit a rotation+translation to each candidate seam (closed-form
   Procrustes).  Too few / collapsed point pairs and empty masks are 422.
3. Discard seams whose *pairwise* pose residual already exceeds the limit
   (the global fit can only trade residuals for the worse, never better).
4. Exact global optimisation: enumerate seam subsets with branch & bound,
   maximising total score.  A subset is feasible only when **all** selected
   seams satisfy, under the *single* set of global poses, simultaneously:
     * per-pair residual <= the seam's pose_error_limit,
     * loop-closure error      <= the seam's pose_error_limit,
     * solid-mask overlap      <= the seam's overlap_limit.
   Ties are broken by the lexicographically smallest sorted seam-id list.
   This is global optimisation, never edge-by-edge greedy.
5. Connected fragments (incl. isolated singletons) form output clusters;
   each cluster is rendered to one transparent PNG under its common poses.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .geometry import fit_rigid, relative_pose
from .masks import overlap_pixels
from .posegraph import solve_global_poses

EPS = 1e-9


class AssemblyError(ValueError):
    """Input problem; mapped to HTTP 422 by the API layer."""


# --------------------------------------------------------------------------- #
# Image decoding
# --------------------------------------------------------------------------- #
def decode_png(data_url_or_b64: str, fragment_id: str) -> np.ndarray:
    raw = data_url_or_b64.strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        buf = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AssemblyError(f"fragment {fragment_id!r}: base64 decode failed ({exc})")
    arr = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise AssemblyError(f"fragment {fragment_id!r}: not a decodable PNG image")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        pass
    else:
        raise AssemblyError(f"fragment {fragment_id!r}: unsupported image shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


# --------------------------------------------------------------------------- #
# Internal types
# --------------------------------------------------------------------------- #
@dataclass
class Edge:
    id: str
    a: str
    b: str
    score: float
    pose_limit: float
    overlap_limit: int
    a_pts: np.ndarray
    b_pts: np.ndarray
    local_theta: float
    local_t: np.ndarray
    local_max: float
    local_rms: float
    pair_span: float = 0.0
    pair_rmax: float = 0.0

    @property
    def w(self) -> float:
        return 1.0 / len(self.a_pts)

    @property
    def angle_budget(self) -> float:
        """Conservative bound on absorbable relative-rotation correction.

        For any feasible correction ``(d_alpha, d_t)`` making every point
        residual ``|d_t + J90(R p_k) d_alpha| <= pose_limit`` we have
        ``|J90 R(p_i - p_j) d_alpha| = |(r_i-r_j)| <= 2L`` for every pair,
        hence ``|d_alpha| <= 2L / max_pair_separation``.
        """
        if self.pair_span <= 1e-9:
            return float("inf")
        return 2.0 * self.pose_limit / self.pair_span

    @property
    def trans_budget_coef(self) -> float:
        """Coefficient ``rho`` with ``|d_t| <= L + rho |d_alpha|``."""
        return self.pair_rmax

    def as_graph_edge(self, theta: float, t: np.ndarray) -> dict:
        return {
            "src": self.a,
            "dst": self.b,
            "src_pts": self.a_pts,
            "dst_pts": self.b_pts,
            "_rel": (theta, t),
            "_w": self.w,
        }


@dataclass
class EdgeMetrics:
    pair_max: float
    pair_rms: float
    loop_error: float
    overlap: int


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(payload) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Edge]]:
    images: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    for frag in payload.fragments:
        if frag.id in images:
            raise AssemblyError(f"duplicate fragment id {frag.id!r}")
        img = decode_png(frag.image_base64, frag.id)
        mask = img[:, :, 3] > 0
        if not np.any(mask):
            raise AssemblyError(f"fragment {frag.id!r}: alpha mask is empty")
        images[frag.id] = img
        masks[frag.id] = mask

    edges: List[Edge] = []
    seen_seam_ids = set()
    for seam in payload.seams:
        if seam.id in seen_seam_ids:
            raise AssemblyError(f"duplicate seam id {seam.id!r}")
        seen_seam_ids.add(seam.id)
        for ref in (seam.fragment_a, seam.fragment_b):
            if ref not in images:
                raise AssemblyError(
                    f"seam {seam.id!r}: unknown fragment {ref!r}")
        if seam.fragment_a == seam.fragment_b:
            raise AssemblyError(
                f"seam {seam.id!r}: fragment_a and fragment_b must differ")

        a_pts = np.array([p.a for p in seam.pairs], dtype=np.float64)
        b_pts = np.array([p.b for p in seam.pairs], dtype=np.float64)
        try:
            model = fit_rigid(b_pts, a_pts)  # b-frame -> a-frame
        except ValueError as exc:
            raise AssemblyError(f"seam {seam.id!r}: {exc}")

        # Geometry used by the conservative correction-budget relaxation:
        # largest separation of boundary points and their maximal distance
        # to the local origin (rotation sensitivity).
        rel_b = b_pts - b_pts.mean(axis=0)
        radii = np.linalg.norm(rel_b, axis=1)
        pair_span = float(
            np.linalg.norm(rel_b[:, None, :] - rel_b[None, :, :], axis=2).max()
        ) if len(rel_b) > 1 else 0.0

        edges.append(Edge(
            id=seam.id,
            a=seam.fragment_a,
            b=seam.fragment_b,
            score=float(seam.score),
            pose_limit=float(seam.pose_error_limit),
            overlap_limit=int(seam.overlap_limit),
            a_pts=a_pts,
            b_pts=b_pts,
            local_theta=model.theta,
            local_t=model.t,
            local_max=model.max_residual,
            local_rms=model.rms,
            pair_span=pair_span,
            pair_rmax=float(radii.max()),
        ))
    return images, masks, edges


# --------------------------------------------------------------------------- #
# Feasibility under a single global-pose configuration
# --------------------------------------------------------------------------- #
def _components(node_set, selected_edges):
    adj = {n: set() for n in node_set}
    for e in selected_edges:
        adj[e.a].add(e.b)
        adj[e.b].add(e.a)
    comps: List[List[str]] = []
    seen = set()
    for n in sorted(node_set):
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(sorted(comp))
    return comps


def _edge_metrics(edge: Edge, g_theta: float, g_t: np.ndarray,
                  masks, overlap_cache: dict) -> EdgeMetrics:
    gR = np.array([[np.cos(g_theta), -np.sin(g_theta)],
                   [np.sin(g_theta), np.cos(g_theta)]])
    moved = edge.b_pts @ gR.T + g_t
    res = moved - edge.a_pts
    nrm = np.linalg.norm(res, axis=1)
    pair_max = float(nrm.max())
    pair_rms = float(np.sqrt(np.mean(nrm**2)))

    local_R = np.array([[np.cos(edge.local_theta), -np.sin(edge.local_theta)],
                        [np.sin(edge.local_theta), np.cos(edge.local_theta)]])
    local_moved = edge.b_pts @ local_R.T + edge.local_t
    # Loop-closure error: how far the globally implied placement of each
    # boundary point is from the seam's own pairwise-fit placement (px).
    loop = np.linalg.norm(moved - local_moved, axis=1)
    loop_error = float(loop.max())

    key = (edge.id, round(float(g_theta), 6),
           round(float(g_t[0]), 4), round(float(g_t[1]), 4))
    if key not in overlap_cache:
        overlap_cache[key] = overlap_pixels(
            masks[edge.a], masks[edge.b], g_theta, g_t)
    return EdgeMetrics(pair_max, pair_rms, loop_error, overlap_cache[key])


def evaluate_subset(edges: List[Edge], masks, overlap_cache: dict
                    ) -> Tuple[bool, Dict[str, EdgeMetrics], Dict[str, Tuple[float, np.ndarray]], Dict[str, str]]:
    """Check feasibility of a seam subset under one global pose set.

    Returns ``(ok, metrics, poses, comp_of)`` where ``poses`` covers every
    node incident to ``edges`` and ``comp_of`` maps each such node to its
    component id (the component's smallest fragment id, also its anchor).
    """
    selected = list(edges)
    nodes = sorted({n for e in selected for n in (e.a, e.b)})
    comps = _components(nodes, selected)
    poses: Dict[str, Tuple[float, np.ndarray]] = {}
    comp_of: Dict[str, str] = {}
    for comp in comps:
        cid = comp[0]
        anchor = comp[0]
        sub_edges = [e.as_graph_edge(e.local_theta, e.local_t)
                     for e in selected if e.a in comp]
        comp_poses = solve_global_poses(comp, sub_edges, anchor)
        poses.update(comp_poses)
        for n in comp:
            comp_of[n] = cid

    metrics: Dict[str, EdgeMetrics] = {}
    ok = True
    for e in selected:
        g_theta, g_t = relative_pose(poses[e.a], poses[e.b])
        m = _edge_metrics(e, g_theta, g_t, masks, overlap_cache)
        metrics[e.id] = m
        if not (m.pair_max <= e.pose_limit + EPS
                and m.loop_error <= e.pose_limit + EPS
                and m.overlap <= e.overlap_limit):
            ok = False
    return ok, metrics, poses, comp_of


# --------------------------------------------------------------------------- #
# Exact global search (branch & bound over seam subsets)
# --------------------------------------------------------------------------- #
def _connected_components(edges: List[Edge]):
    """Connected components of the *candidate* seam graph."""
    nodes = {n for e in edges for n in (e.a, e.b)}
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        ra, rb = find(e.a), find(e.b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: Dict[str, List[Edge]] = {}
    for e in edges:
        groups.setdefault(find(e.a), []).append(e)
    return list(groups.values())


def _is_tree(subset: List[Edge]) -> bool:
    if not subset:
        return False
    nodes = {n for e in subset for n in (e.a, e.b)}
    comps = _components(nodes, subset)
    # acyclic iff edges = nodes - components
    return len(subset) == len(nodes) - len(comps)


def _tree_feasible(subset: List[Edge], masks, overlap_cache: dict) -> bool:
    """Exact feasibility for an acyclic subset (no Gauss-Newton needed).

    Propagating poses along the spanning tree reproduces every edge's own
    relative pose verbatim, so point residuals and loop error are zero;
    only the solid-mask overlap limit can be violated.
    """
    from .posegraph import forest_poses
    for comp, pose, _ in forest_poses(subset):
        comp_set = set(comp)
        for e in subset:
            if e.a not in comp_set:
                continue
            g_theta, g_t = relative_pose(pose[e.a], pose[e.b])
            key = (e.id, round(float(g_theta), 6),
                   round(float(g_t[0]), 4), round(float(g_t[1]), 4))
            ov = overlap_cache.get(key)
            if ov is None:
                ov = overlap_pixels(masks[e.a], masks[e.b], g_theta, g_t)
                overlap_cache[key] = ov
            if ov > e.overlap_limit:
                return False
    return True


def _pose_relax_ok(subset: List[Edge]) -> bool:
    """Monotone necessary pose-feasibility test (LP relaxation)."""
    from .relax import lp_relaxation_feasible
    return lp_relaxation_feasible(subset)


def _exact_feasible(subset, masks, overlap_cache):
    """Full exact feasibility: global GN poses + residual/loop/overlap."""
    if not subset:
        return True
    if _is_tree(subset):
        return _tree_feasible(subset, masks, overlap_cache)
    return evaluate_subset(subset, masks, overlap_cache)[0]


def optimize(candidates: List[Edge], masks, feasible=None):
    """Exact global search: highest-total-score feasible seam subset.

    Sound branch-and-bound over seam subsets -- never edge-by-edge greedy.
    With the default geometric oracle the search uses:

      * independent candidate-graph components optimised separately;
      * a monotone *pose* LP relaxation (linearised outer approximation)
        to prune every DFS node whose seam set cannot fit any single
        global pose set -- this can never discard a feasible subset;
      * an exact Gauss-Newton + rasterised-overlap check at leaves;
      * acyclic subsets take a closed-form tree path (only overlap binds).

    ``feasible`` may inject an arbitrary oracle for testing the pure
    subset-selection semantics.
    """
    overlap_cache: dict = {}
    if not candidates:
        return [], overlap_cache

    if feasible is not None:
        return _optimize_generic(candidates, feasible, overlap_cache)

    chosen: List[Edge] = []
    for comp in _connected_components(candidates):
        comp_chosen, overlap_cache = _optimize_component(
            comp, masks, overlap_cache)
        chosen.extend(comp_chosen)
    return chosen, overlap_cache


def _optimize_generic(candidates, feasible, overlap_cache):
    n = len(candidates)
    if feasible(candidates):
        return list(candidates), overlap_cache
    order = sorted(candidates, key=lambda e: (-e.score, e.id))
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + order[i].score
    best_score, best_ids = 0.0, []

    def dfs(i, chosen, score):
        nonlocal best_score, best_ids
        if score + suffix[i] < best_score - 1e-12:
            return
        if i == n:
            if feasible(chosen):
                ids = sorted(e.id for e in chosen)
                if (score > best_score + 1e-12
                        or (abs(score - best_score) <= 1e-12 and ids < best_ids)):
                    best_score, best_ids = score, ids
            return
        e = order[i]
        chosen.append(e)
        dfs(i + 1, chosen, score + e.score)
        chosen.pop()
        dfs(i + 1, chosen, score)

    dfs(0, [], 0.0)
    by_id = {e.id: e for e in candidates}
    return [by_id[i] for i in best_ids], overlap_cache


def _optimize_component(edges: List[Edge], masks, overlap_cache):
    n = len(edges)
    # Positive scores: the full set, if feasible, is trivially optimal.
    if _exact_feasible(edges, masks, overlap_cache):
        return list(edges), overlap_cache

    order = sorted(edges, key=lambda e: (-e.score, e.id))
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + order[i].score

    # Feasible incumbent (greedy); used solely as a pruning lower bound.
    inc: List[Edge] = []
    for e in order:
        trial = inc + [e]
        if not _is_tree(trial) and not _pose_relax_ok(trial):
            continue
        if _exact_feasible(trial, masks, overlap_cache):
            inc = trial
    best_score = sum(e.score for e in inc)
    best_ids = sorted(e.id for e in inc)

    def dfs(i, chosen, score):
        nonlocal best_score, best_ids
        if score + suffix[i] < best_score - 1e-12:
            return
        if i == n:
            if not _exact_feasible(chosen, masks, overlap_cache):
                return
            ids = sorted(e.id for e in chosen)
            if (score > best_score + 1e-12
                    or (abs(score - best_score) <= 1e-12 and ids < best_ids)):
                best_score, best_ids = score, ids
            return
        e = order[i]
        chosen.append(e)
        # Monotone pruning: if the current set cannot satisfy the pose
        # limits under any global poses, neither can any superset.  Tree
        # sets have zero pose error by construction, so skip the LP there.
        if _is_tree(chosen) or _pose_relax_ok(chosen):
            dfs(i + 1, chosen, score + e.score)
        chosen.pop()
        dfs(i + 1, chosen, score)

    dfs(0, [], 0.0)
    by_id = {e.id: e for e in edges}
    return [by_id[i] for i in best_ids], overlap_cache
