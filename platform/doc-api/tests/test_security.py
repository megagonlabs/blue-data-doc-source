"""Server-policy + path-confinement tests for the doc API (#75 / DOCSRC-07).

Covers the shared ``blue_data.server`` policy — an API key is required on data
routes by default (401 without it), state-changing routes refuse form-style bodies
(415, CSRF), and the service binds loopback:8001 — plus ``BLUE_DOC_ROOT`` path
confinement of the ingest route: a server path outside the root is rejected, and a
filename inside it is accepted.
"""
import pytest
from fastapi.testclient import TestClient

from blue_data.server import resolve_bind
from doc_api.main import app


@pytest.fixture()
def _default_policy(monkeypatch):
    """No key configured and auth NOT disabled — the mandatory-key default."""
    monkeypatch.delenv("BLUE_DOC_API_KEY", raising=False)
    monkeypatch.delenv("BLUE_DOC_API_AUTH_DISABLED", raising=False)


@pytest.fixture()
def _auth_disabled(monkeypatch):
    monkeypatch.setenv("BLUE_DOC_API_AUTH_DISABLED", "true")
    monkeypatch.delenv("BLUE_DOC_ROOT", raising=False)


# ── server policy ─────────────────────────────────────────────────────────────

def test_missing_api_key_returns_401(_default_policy):
    with TestClient(app) as c:
        r = c.post("/documents/ingest", json={"path": "/etc/hostname"})
    assert r.status_code == 401


def test_health_is_open_without_key(_default_policy):
    with TestClient(app) as c:
        r = c.get("/documents/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_form_encoded_post_returns_415(_auth_disabled):
    # Auth disabled so the request clears the key guard and reaches the CSRF guard.
    with TestClient(app) as c:
        r = c.post("/documents/ingest", data={"path": "x"})
    assert r.status_code == 415


def test_default_bind_is_loopback_8001(monkeypatch):
    monkeypatch.delenv("BLUE_DOC_API_HOST", raising=False)
    monkeypatch.delenv("BLUE_DOC_API_PORT", raising=False)
    assert resolve_bind("BLUE_DOC_API", default_port=8001) == ("127.0.0.1", 8001)


def test_bind_honours_env_overrides(monkeypatch):
    monkeypatch.setenv("BLUE_DOC_API_HOST", "0.0.0.0")
    monkeypatch.setenv("BLUE_DOC_API_PORT", "9999")
    assert resolve_bind("BLUE_DOC_API", default_port=8001) == ("0.0.0.0", 9999)


# ── BLUE_DOC_ROOT path confinement ──────────────────────────────────────────────

def test_ingest_path_outside_root_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("BLUE_DOC_API_AUTH_DISABLED", "true")
    monkeypatch.setenv("BLUE_DOC_ROOT", str(tmp_path))
    # Absolute path outside the root → contain() refuses → 400, never read.
    with TestClient(app) as c:
        r = c.post("/documents/ingest", json={"path": "/etc/hostname"})
    assert r.status_code == 400


def test_ingest_traversal_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("BLUE_DOC_API_AUTH_DISABLED", "true")
    monkeypatch.setenv("BLUE_DOC_ROOT", str(tmp_path))
    with TestClient(app) as c:
        r = c.post("/documents/ingest", json={"path": "../secret.txt"})
    assert r.status_code == 400


def test_ingest_filename_inside_root_accepted(monkeypatch, tmp_path):
    monkeypatch.setenv("BLUE_DOC_API_AUTH_DISABLED", "true")
    monkeypatch.setenv("BLUE_DOC_ROOT", str(tmp_path))
    (tmp_path / "doc.txt").write_text("Confined document content about zebras.", encoding="utf-8")
    with TestClient(app) as c:
        r = c.post("/documents/ingest", json={"path": "doc.txt"})
    assert r.status_code == 200
    assert r.json()["file_name"] == "doc.txt"
