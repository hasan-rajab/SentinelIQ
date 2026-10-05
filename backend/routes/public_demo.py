"""Read-only public inference sandbox, isolated from alerts and ingestion."""
from __future__ import annotations
import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[2]
router=APIRouter()
lock=threading.Lock()
slots=threading.BoundedSemaphore(2)
recent=deque()


class Sample(BaseModel):
    cpu_percent: float=Field(ge=0,le=100)
    mem_percent: float=Field(ge=0,le=100)
    disk_read_mbps: float=Field(ge=0,le=1000)
    disk_write_mbps: float=Field(ge=0,le=1000)
    net_in_mbps: float=Field(ge=0,le=1000)
    net_out_mbps: float=Field(ge=0,le=1000)
    open_connections: int=Field(ge=0,le=10000)
    process_count: int=Field(ge=0,le=2000)
    class Config:
        extra='forbid'


def limit(request):
    origin=request.headers.get('origin')
    allowed={o.strip() for o in os.getenv('SENTINELIQ_CORS_ORIGINS','http://localhost:3000,http://127.0.0.1:3000').split(',')}
    if origin and origin not in allowed:
        raise HTTPException(403,'Please use the demo on its own website.')
    peer=request.client.host if request.client else 'unknown'
    # This is a best-effort per-peer limit plus a strict process-wide bound.
    now=time.monotonic()
    with lock:
        while recent and recent[0][0]<now-60: recent.popleft()
        if len(recent)>=300 or sum(p==peer for _,p in recent)>=30:
            raise HTTPException(429,'Demo allowance reached. Please wait a minute.')
        recent.append((now,peer))


def model():
    from backend import main
    service=main.anomaly_service
    if not service or not service.if_metrics or service.if_metrics.threshold is None:
        raise HTTPException(503,'The live model is starting. Please try again shortly.')
    return service.if_metrics


@router.get('/demo')
def page():
    return FileResponse(ROOT/'public/demo.html',media_type='text/html')


@router.get('/api/demo/report')
def report():
    data=json.loads((ROOT/'docs/benchmarks/metric_report.json').read_text())
    return data


@router.post('/api/demo/score')
def score(body: Sample, request: Request):
    limit(request)
    values=body.model_dump() if hasattr(body,'model_dump') else body.dict()
    if not all(math.isfinite(v) for v in values.values()):
        raise HTTPException(422,'Values must be finite.')
    if not slots.acquire(blocking=False):
        raise HTTPException(429,'The model is busy. Please try again shortly.')
    try:
        fitted=model()
        frame=pd.DataFrame([values])
        raw=fitted.score(frame)
        threshold=float(fitted.threshold)
        result={'raw_score':float(raw[0]),'training_scaled_score':float(fitted.normalize_score(raw)[0]),
                'threshold':threshold,'flagged':bool(raw[0]>=threshold),
                'model':'deployed metric Isolation Forest',
                'interpretation':'Score is relative to this model and threshold; it is not a calibrated probability.',
                'boundary':'Read-only inference. No alert is created. This deployed model is separate from the standalone temporal benchmark below.'}
        if not all(math.isfinite(result[k]) for k in ['raw_score','training_scaled_score','threshold']):
            raise HTTPException(503,'Model returned an invalid score.')
        return result
    finally:
        slots.release()
