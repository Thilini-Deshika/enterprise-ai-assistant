import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(offline_env):
    from app.main import app
    with TestClient(app) as c:
        yield c


def _login(client, who):
    r = client.post("/auth/login", json={"username": who, "password": f"{who}123"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _events(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]


def test_auth_required_and_bad_login(client):
    assert client.post("/chat/stream", json={"message": "hi there", "session_id": "abcdefgh"}).status_code == 401
    assert client.post("/auth/login", json={"username": "viewer", "password": "nope"}).status_code == 401


def test_input_validation(client):
    h = _login(client, "analyst")
    assert client.post("/chat/stream", headers=h, json={"message": "", "session_id": "abcdefgh"}).status_code == 422
    assert client.post("/chat/stream", headers=h, json={"message": "hello", "session_id": "bad id!"}).status_code == 422
    assert client.post("/chat/stream", headers=h, json={"message": "x" * 5000, "session_id": "abcdefgh"}).status_code == 422


def test_streaming_chat_returns_activity_and_tokens(client):
    h = _login(client, "analyst")
    r = client.post("/chat/stream", headers=h, json={"message": "What is the change freeze policy for payments?", "session_id": "api-session-1"})
    assert r.status_code == 200
    types = [e["type"] for e in _events(r)]
    assert types[0] == "status" and "token" in types and "retrieval" in types and types[-1] == "status" and "final" in types


def test_token_bucket_returns_429_with_retry_after(client):
    h = _login(client, "viewer")
    codes = [client.post("/chat/stream", headers=h, json={"message": "hello there", "session_id": "api-rate-001"}).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and 429 in codes[5:]
    r = client.post("/chat/stream", headers=h, json={"message": "hello there", "session_id": "api-rate-001"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_admin_endpoints_rbac(client):
    assert client.get("/admin/audit", headers=_login(client, "viewer")).status_code == 403
    r = client.get("/admin/audit", headers=_login(client, "admin"))
    assert r.status_code == 200 and isinstance(r.json()["items"], list)


def test_feedback_and_session_isolation(client):
    h = _login(client, "analyst")
    client.post("/chat/stream", headers=h, json={"message": "What is the change freeze policy?", "session_id": "api-session-2"})
    assert client.post("/feedback", headers=h, json={"session_id": "api-session-2", "message_id": "abc", "rating": 1}).json()["ok"]
    other = _login(client, "admin")
    assert client.get("/sessions/api-session-2/memory", headers=other).status_code == 403


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["checks"]["index_loaded"]
