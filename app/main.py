"""FastAPI application: potsherd global seam assembly."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .assembler import AssemblyError
from .schemas import AssemblyRequest, AssemblyResponse
from .service import assemble

logger = logging.getLogger("shards")

app = FastAPI(
    title="陶片接缝全局拼合服务 / Potsherd Global Seam Assembly",
    version="1.0.0",
    description=(
        "Estimates rotation+translation fragment poses from paired boundary "
        "points and selects the **globally** highest-scoring feasible seam "
        "set (point residuals, loop-closure error and solid-mask overlap all "
        "within per-seam limits). Renders every connected cluster into one "
        "transparent PNG under a single common pose."
    ),
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assemblies", response_model=AssemblyResponse)
def create_assembly(req: AssemblyRequest):
    try:
        result = assemble(req)
    except AssemblyError as exc:
        logger.info("unprocessable assembly request: %s", exc)
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    return result
