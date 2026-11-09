"""Build the doc platform's vector store and record store from ``{prefix}_REDIS_*`` /
``{prefix}_RECORD_STORE`` environment variables.

Shared by the doc CLI / API / MCP so they all support the SAME full connection — a URL,
or host/port with username, password and TLS — rather than each hand-rolling a host/port
block that dropped the password, username, URL and TLS settings (#195). It also backs the
``DocumentRecord`` store with the SAME Redis when one is configured, so FIND / record /
path-delete survive a restart alongside the chunks instead of living only in memory (#188).
Mirrors ``BLUE_DATA_REGISTRY_REDIS_URL`` on the registry side.
"""
from __future__ import annotations

import os
from typing import Optional

from blue_data.source.vector.vector_source import VectorSource

from .schema.schema_store import FileRecordStore, JsonSourceRecordStore, RecordStore


def _redis_connection_from_env(prefix: str) -> Optional[dict]:
    """Return the Redis connection dict from ``{prefix}_REDIS_*`` (URL, or host/port with
    username/password/TLS), or ``None`` when neither a URL nor a host is set. ``scan_prefix``
    / ``namespace`` are added by the caller — they differ between the vector and JSON stores.
    """
    url = os.getenv(f"{prefix}_REDIS_URL")
    host = os.getenv(f"{prefix}_REDIS_HOST")
    if not url and not host:
        return None

    connection: dict = {}
    if url:
        connection["url"] = url
    else:
        connection["host"] = host
        connection["port"] = int(os.getenv(f"{prefix}_REDIS_PORT", "6379"))

    username = os.getenv(f"{prefix}_REDIS_USERNAME")
    if username:
        connection["username"] = username
    password = os.getenv(f"{prefix}_REDIS_PASSWORD")
    if password:
        connection["password"] = password
    if (os.getenv(f"{prefix}_REDIS_SSL") or "").lower() in ("1", "true", "yes"):
        connection["ssl"] = True
    return connection


def redis_vector_source_from_env(prefix: str = "BLUE_DOC") -> Optional[VectorSource]:
    """Return a ``RedisVectorSource`` built from ``{prefix}_REDIS_*`` env vars, or ``None``
    when neither ``{prefix}_REDIS_URL`` nor ``{prefix}_REDIS_HOST`` is set.

    Variables (prefix ``BLUE_DOC`` by default):
      - ``_REDIS_URL``       full ``redis://`` / ``rediss://`` URL (host, auth, TLS, db)
      - ``_REDIS_HOST`` / ``_REDIS_PORT``   used when no URL is given (port default 6379)
      - ``_REDIS_USERNAME`` / ``_REDIS_PASSWORD``   auth (also honored alongside host/port)
      - ``_REDIS_SSL``       ``1``/``true``/``yes`` to enable TLS
      - ``_REDIS_PREFIX``    key namespace / scan prefix (default ``bluedoc``; required by
                             blue-data's create_index, so always set)
    """
    connection = _redis_connection_from_env(prefix)
    if connection is None:
        return None

    from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource

    # Required by blue-data's create_index; also namespaces keys (without it, keys are
    # prefixed with the literal "None"). See #70.
    connection["scan_prefix"] = os.getenv(f"{prefix}_REDIS_PREFIX", "bluedoc")
    return RedisVectorSource(name="redis", properties={"connection": connection})


def record_store_from_env(prefix: str = "BLUE_DOC") -> RecordStore:
    """Return the ``DocumentRecord`` store, persistent so FIND / record / path-delete
    survive a restart (#188):

      - the SAME Redis as the vector store when ``{prefix}_REDIS_*`` is configured (a
        ``JsonSourceRecordStore`` over Redis, namespaced ``{prefix}_RECORD_NAMESPACE``,
        default ``bluedoc:records``), so records persist exactly when the chunks do;
      - otherwise a ``FileRecordStore`` at ``{prefix}_RECORD_STORE`` (default
        ``~/.blue-doc/records.json``).

    The in-memory default lost every record on restart while the chunks stayed in Redis.
    """
    connection = _redis_connection_from_env(prefix)
    if connection is not None:
        from blue_data.source.json.redis.redis_json_source import RedisJSONSource

        namespace = os.getenv(f"{prefix}_RECORD_NAMESPACE", "bluedoc:records")
        return JsonSourceRecordStore(
            RedisJSONSource(name="doc_records",
                            properties={"namespace": namespace, "connection": connection})
        )

    path = os.getenv(f"{prefix}_RECORD_STORE", os.path.expanduser("~/.blue-doc/records.json"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return FileRecordStore(path)
