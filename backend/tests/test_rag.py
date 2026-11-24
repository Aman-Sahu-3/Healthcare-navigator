import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import SecretStr

from app.core.config import Settings
from app.ingestion.pipeline import build_store, ingest_path
from app.ingestion.search import RetrievedChunk
from app.rag.chain import NO_EVIDENCE_ANSWER, RAGService
from app.rag.citations import extract_citations, is_insufficient
from app.rag.confidence import score_confidence
from app.rag.llm import LLMNotConfiguredError, get_llm
from app.rag.prompt import build_prompt, format_context
from app.schemas.rag import Citation

from .ingestion_helpers import HashingEmbeddings, make_pdf
from .test_ingestion import ASTHMA, DM, HTN


def _chunk(i: int, sim: float = 0.8, source: str = "a.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        text=f"passage {i} " * 5,
        source=source,
        page=i,
        title=f"Doc {source}",
        chunk_id=f"x:p{i}:c0",
        similarity=sim,
    )


def _cit(sim: float, source: str = "a.pdf", i: int = 1) -> Citation:
    return Citation(id=i, source=source, title="T", page=1, snippet="s", similarity=sim)


# ---------- citations ----------


def test_extract_maps_markers_in_order_of_mention():
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    text, cits = extract_citations("Use thiazides [3]. Also ACE inhibitors [1].", chunks)
    assert [c.id for c in cits] == [3, 1]
    assert cits[0].page == 3 and cits[0].source == "a.pdf"
    assert text == "Use thiazides [3]. Also ACE inhibitors [1]."


@pytest.mark.parametrize(
    ("raw", "ids", "cleaned"),
    [
        ("A [1, 2].", [1, 2], "A [1][2]."),
        ("A [1-3].", [1, 2, 3], "A [1][2][3]."),
        ("A [2][2].", [2], "A [2][2]."),
        ("A [9].", [], "A."),  # out-of-range marker removed
        ("A [1, 9].", [1], "A [1]."),
        ("Year [2024] study.", [], "Year study."),
        # OpenAI-style / full-width brackets (gpt-oss on Groq emits these)
        ("Monotherapy is first-line【1】.", [1], "Monotherapy is first-line[1]."),
        ("A 【2†L10-L15】 and B 【1†source】.", [2, 1], "A [2] and B [1]."),
        ("A ［3］.", [3], "A [3]."),
        ("A 【1, 2】.", [1, 2], "A [1][2]."),
        ("A 【9】.", [], "A."),
    ],
)
def test_marker_formats(raw, ids, cleaned):
    text, cits = extract_citations(raw, [_chunk(1), _chunk(2), _chunk(3)])
    assert [c.id for c in cits] == ids
    assert text == cleaned


def test_snippet_truncated():
    long = _chunk(1)
    long.text = "x" * 1000
    _, cits = extract_citations("A [1].", [long])
    assert len(cits[0].snippet) <= 283 and cits[0].snippet.endswith("...")


def test_is_insufficient():
    assert is_insufficient("Insufficient evidence: no dosing data.")
    assert is_insufficient("  insufficient evidence - nothing")
    assert not is_insufficient("There is insufficient evidence in older trials, but [1]...")


# ---------- confidence ----------


@pytest.mark.parametrize(
    ("cits", "insufficient", "level"),
    [
        ([_cit(0.82), _cit(0.70, i=2)], False, "HIGH"),
        ([_cit(0.82)], False, "MEDIUM"),  # strong but single passage
        ([_cit(0.65), _cit(0.61, i=2)], False, "MEDIUM"),
        ([_cit(0.50), _cit(0.45, i=2)], False, "LOW"),
        ([], False, "LOW"),
        ([_cit(0.9), _cit(0.9, i=2)], True, "LOW"),
    ],
)
def test_confidence_levels(cits, insufficient, level):
    conf = score_confidence(cits, insufficient, high_threshold=0.75, medium_threshold=0.60)
    assert conf.level == level
    assert 0.0 <= conf.score <= 1.0
    assert conf.reasons


def test_confidence_score_increases_with_corroboration():
    one = score_confidence([_cit(0.8)], False, 0.75, 0.6).score
    three = score_confidence([_cit(0.8, i=i) for i in (1, 2, 3)], False, 0.75, 0.6).score
    assert three > one


# ---------- prompt ----------


def test_context_numbering_and_prompt_rules():
    ctx = format_context([_chunk(1), _chunk(2, source="b.pdf")])
    assert ctx.startswith("[1] (source: Doc a.pdf, page 1)")
    assert "[2] (source: Doc b.pdf, page 2)" in ctx
    msgs = build_prompt().format_messages(context=ctx, question="Q?")
    assert "ONLY" in msgs[0].content and "Insufficient evidence:" in msgs[0].content
    assert "Question: Q?" in msgs[1].content


# ---------- LLM factory ----------


def test_get_llm_requires_key():
    with pytest.raises(LLMNotConfiguredError):
        get_llm(Settings())


def test_get_llm_uses_configured_model():
    llm = get_llm(Settings(openai_api_key=SecretStr("sk-test"), llm_model="gpt-4o-mini"))
    assert llm.model_name == "gpt-4o-mini"
    assert llm.temperature == 0.0


def test_get_llm_supports_openai_compatible_providers():
    llm = get_llm(
        Settings(
            openai_api_key=SecretStr("gsk-test"),
            llm_base_url="https://api.groq.com/openai/v1",
            llm_model="openai/gpt-oss-20b",
        )
    )
    assert llm.model_name == "openai/gpt-oss-20b"
    assert llm.openai_api_base == "https://api.groq.com/openai/v1"


# ---------- end-to-end chain (fake LLM, real Chroma) ----------


@pytest.fixture
def rag_settings(tmp_path):
    return Settings(
        chroma_dir=tmp_path / "chroma",
        chunk_size=300,
        chunk_overlap=50,
        min_similarity=0.2,
        high_conf_threshold=0.5,
        medium_conf_threshold=0.3,
    )


@pytest.fixture
def store(tmp_path, rag_settings):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "htn_guideline.pdf", [HTN, ASTHMA], title="Hypertension Guideline 2024")
    make_pdf(raw / "diabetes_care.pdf", [DM])
    emb = HashingEmbeddings()
    ingest_path(raw, rag_settings, embeddings=emb)
    return build_store(rag_settings, emb)


def test_end_to_end_answer_with_citations(store, rag_settings):
    llm = FakeListChatModel(responses=["Thiazide diuretics are first-line [1]. Also ACEi [1]."])
    svc = RAGService(store, llm, rag_settings)
    res = svc.answer("first-line treatment for hypertension thiazide", k=3)

    assert res.citations and res.citations[0].source == "htn_guideline.pdf"
    assert res.citations[0].page == 1
    assert res.citations[0].title == "Hypertension Guideline 2024"
    assert res.insufficient_evidence is False
    assert res.retrieved_chunks >= 1
    assert res.confidence.level in {"MEDIUM", "HIGH"}
    assert res.latency_ms >= 0


def test_model_says_insufficient(store, rag_settings):
    llm = FakeListChatModel(responses=["Insufficient evidence: dosing not covered [1]."])
    res = RAGService(store, llm, rag_settings).answer("hypertension thiazide dose", k=3)
    assert res.insufficient_evidence is True
    assert res.citations == []
    assert res.confidence.level == "LOW"


def test_no_relevant_chunks_skips_llm(store, rag_settings):
    class ExplodingLLM(FakeListChatModel):
        def _call(self, *a, **k):
            raise AssertionError("LLM must not be called")

    s = rag_settings.model_copy(update={"min_similarity": 0.99})
    res = RAGService(store, ExplodingLLM(responses=["x"]), s).answer("quantum chromodynamics")
    assert res.answer == NO_EVIDENCE_ANSWER
    assert res.retrieved_chunks == 0
    assert res.confidence.level == "LOW"


def test_hallucinated_citation_is_dropped(store, rag_settings):
    llm = FakeListChatModel(responses=["Metformin is first-line [1][7]."])
    res = RAGService(store, llm, rag_settings).answer("metformin type 2 diabetes", k=2)
    assert [c.id for c in res.citations] == [1]
    assert "[7]" not in res.answer


def test_prompt_receives_retrieved_context(store, rag_settings):
    seen = {}

    class SpyLLM(FakeListChatModel):
        def _call(self, messages, *a, **k):
            seen["prompt"] = "\n".join(m.content for m in messages)
            return super()._call(messages, *a, **k)

    RAGService(store, SpyLLM(responses=["ok [1]"]), rag_settings).answer(
        "metformin type 2 diabetes", k=2
    )
    assert "[1] (source:" in seen["prompt"]
    assert "Metformin" in seen["prompt"]


def test_empty_question_rejected(store, rag_settings):
    with pytest.raises(ValueError):
        RAGService(store, FakeListChatModel(responses=["x"]), rag_settings).answer("   ")
