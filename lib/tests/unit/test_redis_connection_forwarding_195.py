"""Regression tests for issue #195.

A properties-built DocumentDataSource dropped the Redis password, username, URL and TLS
settings: _build_vector_source forwarded only host/port/scan_prefix, so a password-
protected or TLS Redis could not be reached, and injected host/port defaults even
overrode a connection.url. The whole connection (minus the registry protocol tag) is now
forwarded to RedisVectorSource. No server needed — we inspect the built source.
"""
import pytest

from blue_data_doc_source import DocumentDataSource, redis_vector_source_from_env


def _built_connection(conn):
    src = DocumentDataSource("d", properties={
        "embedding": "dummy", "vector": "redis", "collection": "d", "connection": conn,
    })
    return src._build_vector_source(src.properties).properties["connection"]


def test_password_username_ssl_are_forwarded():
    built = _built_connection({
        "host": "127.0.0.1", "port": 46380, "password": "pw",
        "username": "default", "ssl": False, "scan_prefix": "d:",
    })
    assert built["password"] == "pw"
    assert built["username"] == "default"
    assert built["ssl"] is False
    assert built["host"] == "127.0.0.1" and built["port"] == 46380
    assert built["scan_prefix"] == "d:"


def test_url_is_forwarded():
    built = _built_connection({"url": "redis://:pw@127.0.0.1:46380/0", "scan_prefix": "d:"})
    assert built["url"] == "redis://:pw@127.0.0.1:46380/0"


def test_scan_prefix_defaults_when_absent():
    built = _built_connection({"host": "127.0.0.1", "port": 6379})
    assert built["scan_prefix"] == "documents:"


def test_protocol_tag_is_dropped_and_port_coerced():
    built = _built_connection({"protocol": "vector/redis", "host": "h", "port": "6380", "scan_prefix": "d:"})
    assert built["port"] == 6380 and isinstance(built["port"], int)


# ---------- the shared env helper used by the doc CLI / API / MCP platforms ----------

@pytest.fixture(autouse=True)
def _clear_redis_env(monkeypatch):
    for key in ("URL", "HOST", "PORT", "USERNAME", "PASSWORD", "SSL", "PREFIX"):
        monkeypatch.delenv(f"BLUE_DOC_REDIS_{key}", raising=False)


def test_env_helper_returns_none_without_url_or_host():
    assert redis_vector_source_from_env("BLUE_DOC") is None


def test_env_helper_forwards_url(monkeypatch):
    monkeypatch.setenv("BLUE_DOC_REDIS_URL", "rediss://:pw@r.example:6380/0")
    conn = redis_vector_source_from_env("BLUE_DOC").properties["connection"]
    # The URL is forwarded and is what RedisVectorSource authenticates with (it uses
    # redis.Redis.from_url); RedisVectorSource's own host/port defaults coexist but are
    # ignored when a url is present.
    assert conn["url"] == "rediss://:pw@r.example:6380/0"
    assert conn["scan_prefix"] == "bluedoc"  # default


def test_env_helper_forwards_host_auth_tls(monkeypatch):
    monkeypatch.setenv("BLUE_DOC_REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("BLUE_DOC_REDIS_PORT", "46380")
    monkeypatch.setenv("BLUE_DOC_REDIS_USERNAME", "default")
    monkeypatch.setenv("BLUE_DOC_REDIS_PASSWORD", "pw")
    monkeypatch.setenv("BLUE_DOC_REDIS_SSL", "true")
    monkeypatch.setenv("BLUE_DOC_REDIS_PREFIX", "docs:")
    conn = redis_vector_source_from_env("BLUE_DOC").properties["connection"]
    assert conn["host"] == "127.0.0.1" and conn["port"] == 46380
    assert conn["username"] == "default" and conn["password"] == "pw" and conn["ssl"] is True
    assert conn["scan_prefix"] == "docs:"
