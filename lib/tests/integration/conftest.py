import pytest
import redis


def _redis_available() -> bool:
    try:
        redis.Redis(host="localhost", port=6379).ping()
        return True
    except Exception:
        return False


requires_redis = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis Stack not available on localhost:6379",
)
