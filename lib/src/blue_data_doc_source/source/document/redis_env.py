"""Build a Redis vector store from ``{prefix}_REDIS_*`` environment variables.

Shared by the doc CLI / API / MCP so they all support the SAME full connection — a URL,
or host/port with username, password and TLS — rather than each hand-rolling a host/port
block that dropped the password, username, URL and TLS settings (#195). Mirrors
``BLUE_DATA_REGISTRY_REDIS_URL`` on the registry side.
"""
from __future__ import annotations

import os
from typing import Optional

from blue_data.source.vector.vector_source import VectorSource


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
    url = os.getenv(f"{prefix}_REDIS_URL")
    host = os.getenv(f"{prefix}_REDIS_HOST")
    if not url and not host:
        return None

    from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource

    connection: dict = {
        # Required by blue-data's create_index; also namespaces keys (without it, keys are
        # prefixed with the literal "None"). See #70.
        "scan_prefix": os.getenv(f"{prefix}_REDIS_PREFIX", "bluedoc"),
    }
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

    return RedisVectorSource(name="redis", properties={"connection": connection})
