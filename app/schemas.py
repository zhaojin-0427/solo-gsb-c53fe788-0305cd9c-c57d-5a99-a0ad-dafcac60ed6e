"""Request / response models for the shard assembly service."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PointPair(BaseModel):
    a: List[float] = Field(..., min_length=2, max_length=2,
                           description="[x, y] on fragment A's boundary")
    b: List[float] = Field(..., min_length=2, max_length=2,
                           description="[x, y] on fragment B's boundary, matched to a")


class Seam(BaseModel):
    id: str
    fragment_a: str
    fragment_b: str
    score: float = Field(..., gt=0, description="positive fit score (higher is better)")
    pose_error_limit: float = Field(..., ge=0,
                                    description="max allowed point-pair / loop residual in px")
    overlap_limit: int = Field(..., ge=0,
                               description="max allowed overlapping solid-mask pixels")
    pairs: List[PointPair] = Field(..., min_length=2)


class AssemblyRequest(BaseModel):
    fragments: List["FragmentIn"] = Field(..., min_length=1)
    seams: List[Seam] = Field(default_factory=list)


class FragmentIn(BaseModel):
    id: str = Field(..., min_length=1)
    image_base64: str = Field(..., description="transparent PNG, base64 encoded (data URI ok)")


class PoseOut(BaseModel):
    fragment_id: str
    theta: float = Field(..., description="rotation in radians (image frame -> cluster frame)")
    tx: float
    ty: float
    pair_residual_rms: float = Field(..., description="global-fit RMS residual over its pairs, px")
    pair_residual_max: float


class AdoptedSeam(BaseModel):
    id: str
    fragment_a: str
    fragment_b: str
    score: float
    pair_residual_max: float
    loop_error: float
    overlap_pixels: int


class RejectedSeam(BaseModel):
    id: str
    fragment_a: str
    fragment_b: str
    score: float
    reason: str
    pair_residual_max: Optional[float] = None
    loop_error: Optional[float] = None
    overlap_pixels: Optional[int] = None


class Cluster(BaseModel):
    cluster_id: int
    fragment_ids: List[str]
    poses: List[PoseOut]
    adopted_seams: List[AdoptedSeam]
    rejected_seams: List[RejectedSeam]
    total_score: float
    image_base64: str


class AssemblyResponse(BaseModel):
    clusters: List[Cluster]
    adopted_seam_ids: List[str]
    rejected_seams: List[RejectedSeam]
    total_score: float


AssemblyRequest.model_rebuild()
