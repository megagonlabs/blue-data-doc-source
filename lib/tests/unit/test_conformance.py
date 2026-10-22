"""Adopt blue-data's DataSource conformance kit (#132, cross-repo XR-02).

Subclasses ``blue_data.testing.conformance.DataSourceConformance`` so this source
is verified against the DataSource contract (blue-data design/14): it is a
DataSource, accepts the registry's ``execute_query_nl`` signature, overrides no
sync base method with ``async def``, is reconstructible from ``(name, properties)``
alone, tolerates reserved/shared-optional properties, and returns ``list[dict]``.

The ``data_source`` fixture is built from ``properties`` ONLY (no Python-object
collaborators) — the same path ``DataRegistry.connect_source`` uses to rebuild a
source from its stored record. The ``sample_query`` is a read-only
``DOCUMENT.SEARCH``, so the behavioural checks confirm a properties-only source can
actually answer a query and return ``list[dict]``.

Allowed exceptions to §1 (list[dict]) — documented, not fixed (#132):
``DOCUMENT.ENSURE_INDEX`` returns ``None``, ``DOCUMENT.DELETE`` returns an ``int``
(deleted count), ``DOCUMENT.GET_RECORD`` returns a ``dict``, and
``DOCUMENT.SEARCH`` with ``rich=true`` returns a ``dict``. These are deliberate
ergonomic shapes on the *direct* API; none is reachable through the registry's NL
path, which is restricted to read-only ops and only ever yields the plain
``list[dict]`` ops exercised here.
"""
import pytest

from blue_data.testing.conformance import DataSourceConformance

from blue_data_doc_source import DocumentDataSource


class TestDocumentSourceConformance(DataSourceConformance):
    @pytest.fixture
    def data_source(self):
        source = DocumentDataSource(
            "t",
            properties={
                "embedding": "dummy",
                "vector": "memory",
                "collection": "t",
            },
        )
        # A properties-only source must be able to search; make the index exist.
        source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "t"})
        return source

    @pytest.fixture
    def sample_query(self):
        # Read-only: embeds the query and runs KNN over the (empty) index → [].
        return ({"op": "DOCUMENT.SEARCH", "query": "hello world", "collection": "t"}, {})
