import pytest

from app.core.config import Settings
from app.ingestion.loader import PDFLoadError, clean_text, load_pdf
from app.ingestion.pipeline import build_store, discover_pdfs, ingest_path
from app.ingestion.search import retrieve
from app.ingestion.splitter import split_documents
from app.ingestion.vectorstore import count, list_sources

from .ingestion_helpers import HashingEmbeddings, make_pdf

HTN = (
    "Hypertension management. First-line pharmacological treatment for adults with "
    "hypertension includes thiazide diuretics, ACE inhibitors, angiotensin receptor "
    "blockers and calcium channel blockers. Lifestyle modification including reduced "
    "sodium intake is recommended for all patients."
)
DM = (
    "Type 2 diabetes. Metformin is recommended as initial therapy for most adults with "
    "type 2 diabetes. HbA1c should be monitored every three to six months until glycaemic "
    "targets are achieved."
)
ASTHMA = (
    "Asthma care. Inhaled corticosteroids are the cornerstone of maintenance therapy for "
    "persistent asthma. Short-acting beta agonists are used for symptom relief."
)


@pytest.fixture
def settings(tmp_path):
    return Settings(chroma_dir=tmp_path / "chroma", chunk_size=300, chunk_overlap=50)


@pytest.fixture
def emb():
    return HashingEmbeddings()


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    make_pdf(d / "htn_guideline.pdf", [HTN, ASTHMA], title="Hypertension Guideline 2024")
    make_pdf(d / "diabetes_care.pdf", [DM])
    (d / "notes.txt").write_text("not a pdf")
    return d


# ---------- loader ----------


def test_clean_text_fixes_hyphenation_and_page_numbers():
    raw = "The treat-\nment works.\n\n\n\nPage 3 of 10\n  12  \nNext   paragraph."
    out = clean_text(raw)
    assert "treatment" in out
    assert "Page 3" not in out
    assert "12" not in out.split("\n")
    assert "Next paragraph." in out


def test_load_pdf_metadata(corpus):
    pages = load_pdf(corpus / "htn_guideline.pdf")
    assert len(pages) == 2
    m = pages[0].metadata
    assert m["source"] == "htn_guideline.pdf"
    assert m["page"] == 1 and pages[1].metadata["page"] == 2
    assert m["title"] == "Hypertension Guideline 2024"
    assert m["total_pages"] == 2
    assert len(m["file_hash"]) == 64
    assert "thiazide" in pages[0].page_content


def test_title_falls_back_to_filename(corpus):
    assert load_pdf(corpus / "diabetes_care.pdf")[0].metadata["title"] == "Diabetes Care"


def test_load_pdf_rejects_non_pdf(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_text("this is not a pdf")
    with pytest.raises(PDFLoadError):
        load_pdf(bad)


def test_blank_pdf_returns_no_pages(tmp_path):
    p = make_pdf(tmp_path / "blank.pdf", [""])
    assert load_pdf(p) == []


# ---------- splitter ----------


def test_chunks_keep_page_metadata_and_stable_ids(corpus):
    pages = load_pdf(corpus / "htn_guideline.pdf")
    a = split_documents(pages, chunk_size=150, chunk_overlap=30)
    b = split_documents(pages, chunk_size=150, chunk_overlap=30)
    assert len(a) > 2
    assert [c.id for c in a] == [c.id for c in b]  # deterministic
    assert len({c.id for c in a}) == len(a)  # unique
    for c in a:
        assert len(c.page_content) <= 150
        assert c.metadata["page"] in (1, 2)
        assert c.id.endswith(f"c{c.metadata['chunk_index']}")


def test_min_chunk_chars_drops_fragments(corpus):
    pages = load_pdf(corpus / "htn_guideline.pdf")
    chunks = split_documents(pages, chunk_size=150, chunk_overlap=30, min_chunk_chars=100)
    assert all(len(c.page_content) >= 100 for c in chunks)


# ---------- pipeline ----------


def test_discover_ignores_non_pdfs(corpus):
    assert [p.name for p in discover_pdfs(corpus)] == ["diabetes_care.pdf", "htn_guideline.pdf"]


def test_ingest_then_skip_unchanged(corpus, settings, emb):
    first = ingest_path(corpus, settings, embeddings=emb)
    assert {f.status for f in first.files} == {"ingested"}
    n = count(build_store(settings, emb))
    assert n == first.total_chunks > 0

    second = ingest_path(corpus, settings, embeddings=emb)
    assert {f.status for f in second.files} == {"skipped"}
    assert count(build_store(settings, emb)) == n  # no duplicates


def test_changed_file_is_replaced(corpus, settings, emb):
    ingest_path(corpus, settings, embeddings=emb)
    make_pdf(corpus / "diabetes_care.pdf", [DM + " Updated 2025 edition.", DM])
    report = ingest_path(corpus, settings, embeddings=emb)
    status = {f.source: f.status for f in report.files}
    assert status == {"diabetes_care.pdf": "replaced", "htn_guideline.pdf": "skipped"}
    store = build_store(settings, emb)
    srcs = {s["source"]: s for s in list_sources(store)}
    assert srcs["diabetes_care.pdf"]["total_pages"] == 2
    # only the new version's chunks remain
    hashes = {m["file_hash"] for m in store.get(where={"source": "diabetes_care.pdf"})["metadatas"]}
    assert len(hashes) == 1


def test_force_reingests(corpus, settings, emb):
    ingest_path(corpus, settings, embeddings=emb)
    n = count(build_store(settings, emb))
    report = ingest_path(corpus, settings, embeddings=emb, force=True)
    assert {f.status for f in report.files} == {"ingested"}
    assert count(build_store(settings, emb)) == n


def test_bad_file_does_not_stop_batch(corpus, settings, emb):
    (corpus / "corrupt.pdf").write_text("garbage")
    report = ingest_path(corpus, settings, embeddings=emb)
    status = {f.source: f.status for f in report.files}
    assert status["corrupt.pdf"] == "failed"
    assert status["htn_guideline.pdf"] == "ingested"


# ---------- retrieval ----------


@pytest.mark.parametrize(
    ("query", "source", "page"),
    [
        ("first-line treatment for hypertension thiazide", "htn_guideline.pdf", 1),
        ("metformin initial therapy type 2 diabetes", "diabetes_care.pdf", 1),
        ("inhaled corticosteroids asthma maintenance", "htn_guideline.pdf", 2),
    ],
)
def test_retrieval_finds_right_page(corpus, settings, emb, query, source, page):
    ingest_path(corpus, settings, embeddings=emb)
    hits = retrieve(build_store(settings, emb), query, k=3)
    assert hits[0].source == source and hits[0].page == page
    assert 0.0 <= hits[-1].similarity <= hits[0].similarity <= 1.0


def test_min_similarity_filters(corpus, settings, emb):
    ingest_path(corpus, settings, embeddings=emb)
    hits = retrieve(build_store(settings, emb), "quantum chromodynamics", k=5, min_similarity=0.5)
    assert hits == []


def test_file_with_only_tiny_fragments_is_empty(tmp_path, settings, emb):
    d = tmp_path / "tiny"
    d.mkdir()
    make_pdf(d / "stub.pdf", ["Short."])
    s = settings.model_copy(update={"min_chunk_chars": 80})
    report = ingest_path(d, s, embeddings=emb)
    assert report.files[0].status == "empty"
