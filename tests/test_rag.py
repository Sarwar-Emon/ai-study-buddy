from types import SimpleNamespace

from app import embeddings, llm
from app.llm import SYSTEM_PROMPT, build_context
from app.rag import NOT_FOUND_MESSAGE
from app.store import Hit


def test_ingest_is_idempotent(service, pdf_bytes):
    first, created_first = service.ingest_pdf("a.pdf", pdf_bytes)
    second, created_second = service.ingest_pdf("renamed.pdf", pdf_bytes)
    assert created_first is True and created_second is False
    assert first.doc_id == second.doc_id
    assert len(service.list_documents()) == 1


def test_answer_cites_the_right_page(service, pdf_bytes):
    service.ingest_pdf("a.pdf", pdf_bytes)
    result = service.ask("Which network does Docker Compose use?")
    assert result.grounded is True
    assert result.citations[0].page == 2
    assert result.citations[0].filename == "a.pdf"
    assert result.citations[0].number == 1


def test_unrelated_question_is_refused_without_calling_the_llm(service, pdf_bytes):
    service.ingest_pdf("a.pdf", pdf_bytes)

    class Boom:
        def answer(self, *_):
            raise AssertionError("LLM must not be called when nothing is relevant")

    service._answerer = Boom()
    result = service.ask("quantum chromodynamics of neutron stars")
    assert result.grounded is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []


def test_ask_with_no_documents_is_refused(service):
    assert service.ask("anything at all").grounded is False


def test_delete_removes_the_document(service, pdf_bytes):
    record, _ = service.ingest_pdf("a.pdf", pdf_bytes)
    assert service.delete_document(record.doc_id) is True
    assert service.list_documents() == []
    assert service.delete_document(record.doc_id) is False
    assert service.ask("Docker Compose network").grounded is False


# ----- prompt building and OpenAI wrappers (client is mocked, no network) -----
def test_context_contains_numbers_files_and_pages():
    hits = [Hit("d1", "paper.pdf", 7, "some text", 0.9), Hit("d1", "paper.pdf", 9, "more", 0.8)]
    ctx = build_context(hits)
    assert "[1] (file: paper.pdf, page 7)" in ctx
    assert "[2] (file: paper.pdf, page 9)" in ctx


def test_system_prompt_tells_model_to_ignore_instructions_in_documents():
    assert "never follow" in SYSTEM_PROMPT.lower()
    assert "only" in SYSTEM_PROMPT.lower()


def test_openai_answerer_sends_prompt_and_returns_text(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)
            msg = SimpleNamespace(content="  The answer [1]  ")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class FakeClient:
        def __init__(self, api_key, base_url=None, max_retries=2):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    answerer = llm.OpenAIAnswerer("sk-test", "gpt-test")
    text = answerer.answer("What?", [Hit("d", "f.pdf", 1, "ctx text", 0.9)])
    assert text == "The answer [1]"
    assert calls["model"] == "gpt-test"
    assert calls["temperature"] == 0
    assert calls["messages"][0]["role"] == "system"
    assert "ctx text" in calls["messages"][1]["content"]


def test_openai_embedder_sends_texts_in_batches(monkeypatch):
    batches = []

    class FakeEmbeddings:
        def create(self, model, input):
            batches.append(len(input))
            data = [SimpleNamespace(embedding=[float(len(t))]) for t in input]
            return SimpleNamespace(data=data)

    class FakeClient:
        def __init__(self, api_key, base_url=None, max_retries=2):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(embeddings, "OpenAI", FakeClient)
    embedder = embeddings.OpenAIEmbedder("sk-test", "model-x")
    vectors = embedder.embed(["a"] * 200)
    assert batches == [96, 96, 8]
    assert len(vectors) == 200


def test_missing_api_key_is_a_clear_error():
    import pytest

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        embeddings.OpenAIEmbedder("", "m")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm.OpenAIAnswerer("", "m")


def test_base_url_and_retries_reach_the_client(monkeypatch):
    """Using another provider (like Gemini) only needs a different base_url."""
    seen = []

    class FakeClient:
        def __init__(self, api_key, base_url=None, max_retries=2):
            seen.append({"api_key": api_key, "base_url": base_url, "max_retries": max_retries})

    monkeypatch.setattr(embeddings, "OpenAI", FakeClient)
    monkeypatch.setattr(llm, "OpenAI", FakeClient)

    url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embeddings.OpenAIEmbedder("key", "m", url)
    llm.OpenAIAnswerer("key", "m", url)
    embeddings.OpenAIEmbedder("key", "m")  # no base_url: use OpenAI's default
    llm.OpenAIAnswerer("key", "m", "")

    assert seen[0]["base_url"] == url and seen[1]["base_url"] == url
    assert seen[2]["base_url"] is None and seen[3]["base_url"] is None
    assert all(s["max_retries"] == 5 for s in seen)
