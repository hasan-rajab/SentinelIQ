from types import SimpleNamespace
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.routes import public_demo as demo

SAMPLE=dict(cpu_percent=30,mem_percent=55,disk_read_mbps=12,disk_write_mbps=8,
            net_in_mbps=24,net_out_mbps=18,open_connections=110,process_count=85)


@pytest.fixture
def client(monkeypatch):
    demo.recent.clear()
    fitted=SimpleNamespace(threshold=.5,score=lambda df:np.array([.6]),normalize_score=lambda raw:np.array([.8]))
    monkeypatch.setattr(demo,'model',lambda:fitted)
    app=FastAPI(); app.include_router(demo.router)
    return TestClient(app)


def test_public_read_only_scoring_and_report(client):
    assert client.get('/demo').status_code==200
    report=client.get('/api/demo/report').json()
    assert report['test']['records']==121 and report['test']['positive_records']==9
    result=client.post('/api/demo/score',json=SAMPLE)
    assert result.status_code==200 and result.json()['flagged'] is True
    assert 'No alert is created' in result.json()['boundary']
    assert client.post('/api/demo/score',json={**SAMPLE,'host':'private'}).status_code==422
    assert client.post('/api/demo/score',json={**SAMPLE,'cpu_percent':101}).status_code==422
    assert client.post('/api/demo/score',json=SAMPLE,headers={'Origin':'https://foreign.example'}).status_code==403


def test_scoring_is_bounded(client):
    for _ in range(30): assert client.post('/api/demo/score',json=SAMPLE).status_code==200
    assert client.post('/api/demo/score',json=SAMPLE).status_code==429
