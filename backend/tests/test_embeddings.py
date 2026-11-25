import math
import sys
import types

import pytest

from app.ingestion import embeddings as emb_module


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding (no model download in tests)."""

    init_kwargs: dict = {}

    def __init__(self, **kwargs):
        FakeTextEmbedding.init_kwargs = kwargs

    def embed(self, texts, batch_size=256):
        for t in texts:
            yield [float(len(t)), 3.0, 4.0]  # deliberately not unit length


@pytest.fixture
def fake_fastembed(monkeypatch):
    module = types.ModuleType("fastembed")
    module.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", module)
    emb_module.get_embeddings.cache_clear()
    yield
    emb_module.get_embeddings.cache_clear()


def test_vectors_are_normalised(fake_fastembed):
    e = emb_module.FastEmbedEmbeddings("BAAI/bge-small-en-v1.5", cache_dir="/tmp/x", threads=2)
    assert FakeTextEmbedding.init_kwargs == {
        "model_name": "BAAI/bge-small-en-v1.5",
        "cache_dir": "/tmp/x",
        "threads": 2,
    }
    docs = e.embed_documents(["a", "bbbb"])
    assert len(docs) == 2
    for v in docs:
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-5)
    q = e.embed_query("abc")
    assert len(q) == 3 and math.isclose(sum(x * x for x in q), 1.0, rel_tol=1e-5)
    assert e.embed_documents([]) == []


def test_get_embeddings_is_cached(fake_fastembed):
    a = emb_module.get_embeddings("m", "/tmp/c", 16, None)
    b = emb_module.get_embeddings("m", "/tmp/c", 16, None)
    assert a is b
