import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core.config import Settings, get_settings
from app.eval.__main__ import main
from app.eval.dataset import GoldenItem, load_golden, save_golden
from app.eval.generate import generate_golden, sample_chunks
from app.eval.judge import judge_answer
from app.eval.llm_json import parse_json_object
from app.eval.metrics import compute_metrics, suggest_min_similarity
from app.eval.report import render_report
from app.eval.runner import run_eval
from app.ingestion.pipeline import build_store, ingest_path
from app.rag.chain import RAGService

from .ingestion_helpers import HashingEmbeddings, make_pdf
from .test_ingestion import ASTHMA, DM, HTN

LONG_HTN = (HTN + " ") * 3
LONG_DM = (DM + " ") * 4


class RouterLLM(FakeListChatModel):
    """Answers like the generator, the judge or the RAG model depending on the prompt."""

    responses: list[str] = ["unused"]
    judge_reply: str = '{"faithfulness": "yes", "correctness": "yes", "reason": "ok"}'

    def _call(self, messages, *a, **k):
        system = messages[0].content
        human = messages[-1].content
        if "write evaluation questions" in system:
            if "Metformin" in human:
                return (
                    '```json\n{"question": "What is initial therapy for T2D?", '
                    '"answer": "Metformin."}\n```'
                )
            return '{"question": "What is first-line for hypertension?", "answer": "Thiazides."}'
        if "strict evaluator" in system:
            return self.judge_reply
        q = human.split("Question:")[-1]
        if any(w in q for w in ("France", "football", "bread")):
            return "Insufficient evidence: not covered."
        return "Per guidance, use the recommended therapy [1]."


@pytest.fixture
def env(tmp_path):
    settings = Settings(
        chroma_dir=tmp_path / "chroma",
        eval_dir=tmp_path / "eval",
        chunk_size=400,
        chunk_overlap=50,
        min_similarity=0.05,
        high_conf_threshold=0.5,
        medium_conf_threshold=0.2,
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "htn.pdf", [LONG_HTN, ASTHMA])
    make_pdf(raw / "dm.pdf", [LONG_DM])
    emb = HashingEmbeddings()
    ingest_path(raw, settings, embeddings=emb)
    store = build_store(settings, emb)
    return settings, store


GOLDEN = [
    GoldenItem(
        id="q1",
        question="first-line pharmacological treatment hypertension thiazide",
        expected_sources=["htn.pdf"],
        expected_pages=[1],
        reference_answer="Thiazides, ACEi, ARB or CCB.",
    ),
    GoldenItem(
        id="q2",
        question="metformin initial therapy type 2 diabetes",
        expected_sources=["dm.pdf"],
        expected_pages=[1],
    ),
    GoldenItem(id="n1", question="What is the capital of France?", answerable=False),
    GoldenItem(id="n2", question="Who won the football world cup?", answerable=False),
]


# ---------- dataset ----------


def test_golden_roundtrip_and_validation(tmp_path):
    p = tmp_path / "g.jsonl"
    save_golden(GOLDEN, p)
    assert [i.id for i in load_golden(p)] == ["q1", "q2", "n1", "n2"]

    p.write_text(p.read_text() + "# comment\n\n" + json.dumps({"id": "q1", "question": "dup?"}))
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(p)


def test_parse_json_object():
    assert parse_json_object('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object("no json here") is None
    assert parse_json_object("{broken") is None


# ---------- generation ----------


def test_sample_chunks_spread_across_sources(env):
    _, store = env
    picked = sample_chunks(store, 2, min_chars=100)
    assert {m["source"] for _, m in picked} == {"htn.pdf", "dm.pdf"}


def test_generate_golden(env):
    _, store = env
    items = generate_golden(store, RouterLLM(), n=2, min_chars=100)
    gen = [i for i in items if i.origin == "generated"]
    neg = [i for i in items if i.origin == "negative"]
    assert len(gen) == 2 and len(neg) >= 5
    by_src = {i.expected_sources[0]: i for i in gen}
    assert by_src["dm.pdf"].reference_answer == "Metformin."
    assert all(i.expected_pages for i in gen)
    assert all(not i.answerable for i in neg)


# ---------- judge ----------


def test_judge_parses_and_handles_garbage(env):
    j = judge_answer(RouterLLM(), "q", "a [1]", [], "ref")
    assert j.faithfulness == "yes" and j.correctness == "yes"
    bad = judge_answer(RouterLLM(judge_reply="I think it's fine"), "q", "a", [], None)
    assert bad.faithfulness is None and "unparseable" in bad.reason
    no_ref = judge_answer(RouterLLM(), "q", "a", [], None)
    assert no_ref.correctness is None


# ---------- run + metrics ----------


def test_run_eval_end_to_end(env):
    settings, store = env
    service = RAGService(store, RouterLLM(), settings)
    results = run_eval(service, GOLDEN, k=3, judge_llm=RouterLLM(), progress=False)
    by_id = {r.id: r for r in results}

    assert by_id["q1"].source_rank == 1 and by_id["q1"].page_hit is True
    assert by_id["q2"].cited_expected is True
    assert by_id["n1"].refused and by_id["n1"].refusal_correct
    assert by_id["q1"].judgement.faithfulness == "yes"
    assert by_id["n1"].judgement is None  # negatives aren't judged

    m = compute_metrics(results, k=3)
    assert m["counts"] == {"total": 4, "answerable": 2, "unanswerable": 2, "errors": 0, "judged": 2}
    assert m["retrieval"]["source_hit@3_pct"] == 100.0
    assert m["retrieval"]["mrr"] == 1.0
    assert m["generation"]["answered_when_answerable_pct"] == 100.0
    assert m["generation"]["refused_when_unanswerable_pct"] == 100.0
    assert m["generation"]["faithfulness"] == 1.0

    report = render_report(m, results, {"timestamp": "t", "model": "fake"})
    assert "Headline metrics" in report and "| Retrieval: MRR | 1.00 |" in report


def test_llm_errors_are_recorded_not_fatal(env):
    settings, store = env

    class Broken(RouterLLM):
        def _call(self, *a, **k):
            raise RuntimeError("rate limited")

    results = run_eval(RAGService(store, Broken(), settings), GOLDEN[:2], k=3, progress=False)
    assert all(r.error and "rate limited" in r.error for r in results)
    m = compute_metrics(results, k=3)
    assert m["counts"]["errors"] == 2
    report = render_report(m, results, {})
    assert "| error |" in report


def test_suggest_min_similarity_separates():
    s = suggest_min_similarity([0.8, 0.75, 0.7, 0.68], [0.5, 0.55, 0.62])
    assert 0.62 < s["min_similarity"] <= 0.68
    assert s["balanced_accuracy"] == 100.0
    assert suggest_min_similarity([0.8], []) is None


# ---------- CLI ----------


@pytest.fixture
def cli(env, monkeypatch):
    settings, _ = env
    monkeypatch.setenv("CHROMA_DIR", str(settings.chroma_dir))
    monkeypatch.setenv("EVAL_DIR", str(settings.eval_dir))
    monkeypatch.setenv("MIN_SIMILARITY", "0.05")
    monkeypatch.setattr(
        "app.ingestion.pipeline.get_embeddings", lambda *a, **k: HashingEmbeddings()
    )
    monkeypatch.setattr("app.rag.llm.get_llm", lambda s: RouterLLM())
    get_settings.cache_clear()
    return settings


def test_cli_generate_then_run(cli, capsys):
    assert main(["generate", "--n", "2", "--min-chars", "100"]) == 0
    draft = cli.eval_dir / "golden_draft.jsonl"
    assert draft.exists()
    draft.rename(cli.eval_dir / "golden.jsonl")

    assert main(["run", "--no-judge", "-k", "3"]) == 0
    out = capsys.readouterr().out
    assert "Headline metrics" in out
    reports = list((cli.eval_dir / "results").glob("report-*.md"))
    runs = list((cli.eval_dir / "results").glob("run-*.json"))
    assert reports and runs
    data = json.loads(runs[0].read_text())
    assert data["metrics"]["counts"]["judged"] == 0


def test_cli_without_api_key(env, monkeypatch, capsys):
    settings, _ = env
    monkeypatch.setenv("CHROMA_DIR", str(settings.chroma_dir))
    monkeypatch.setattr(
        "app.ingestion.pipeline.get_embeddings", lambda *a, **k: HashingEmbeddings()
    )
    get_settings.cache_clear()
    assert main(["generate", "--n", "1"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_cli_run_without_golden(cli, capsys):
    assert main(["run"]) == 2
    assert "Golden set not found" in capsys.readouterr().out
