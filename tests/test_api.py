import httpx
import openai
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import make_pdf


def upload(client, data: bytes, name: str = "paper.pdf"):
    return client.post("/api/v1/documents", files={"file": (name, data, "application/pdf")})


def test_health_works_without_auth(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_list_ask_delete_flow(client, pdf_bytes):
    created = upload(client, pdf_bytes)
    assert created.status_code == 201
    doc = created.json()
    assert doc["pages"] == 2 and doc["filename"] == "paper.pdf"

    assert upload(client, pdf_bytes).status_code == 200  # same file again

    listing = client.get("/api/v1/documents").json()
    assert [d["doc_id"] for d in listing] == [doc["doc_id"]]

    answer = client.post("/api/v1/ask", json={"question": "Which network does Docker Compose use?"})
    body = answer.json()
    assert answer.status_code == 200
    assert body["grounded"] is True
    assert body["citations"][0]["page"] == 2

    assert client.delete(f"/api/v1/documents/{doc['doc_id']}").status_code == 204
    assert client.delete(f"/api/v1/documents/{doc['doc_id']}").status_code == 404
    assert client.get("/api/v1/documents").json() == []


def test_off_topic_question_is_not_grounded(client, pdf_bytes):
    upload(client, pdf_bytes)
    body = client.post("/api/v1/ask", json={"question": "quantum chromodynamics neutron stars"}).json()
    assert body["grounded"] is False and body["citations"] == []


def test_wrong_extension_is_rejected(client, pdf_bytes):
    assert upload(client, pdf_bytes, "notes.txt").status_code == 415


def test_fake_pdf_is_rejected(client):
    assert upload(client, b"not really a pdf", "x.pdf").status_code == 415


def test_pdf_without_text_is_422(client):
    assert upload(client, make_pdf([""]), "blank.pdf").status_code == 422


def test_oversized_file_is_413(service, tmp_path, pdf_bytes):
    small = Settings(provider="fake", data_dir=str(tmp_path), max_upload_mb=0, _env_file=None)
    client = TestClient(create_app(service=service, settings=small))
    assert upload(client, pdf_bytes).status_code == 413


def test_bad_questions_are_rejected(client):
    assert client.post("/api/v1/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/v1/ask", json={"question": "   "}).status_code == 422
    assert client.post("/api/v1/ask", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/api/v1/ask", json={"question": "ok", "top_k": 0}).status_code == 422
    assert client.post("/api/v1/ask", json={"question": "ok", "top_k": 99}).status_code == 422


def test_api_key_is_enforced_when_configured(service, tmp_path, pdf_bytes):
    secured = Settings(provider="fake", data_dir=str(tmp_path), api_key="s3cret", _env_file=None)
    client = TestClient(create_app(service=service, settings=secured))
    assert client.get("/api/v1/documents").status_code == 401
    assert client.get("/api/v1/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open


def test_rate_limit_kicks_in_after_20_questions(client):
    codes = [client.post("/api/v1/ask", json={"question": "anything"}).status_code for _ in range(22)]
    assert codes[:20] == [200] * 20
    assert codes[20:] == [429, 429]


def test_provider_failure_becomes_502_not_a_crash(service, settings, pdf_bytes):
    class Failing:
        def answer(self, *_):
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

    service._answerer = Failing()
    client = TestClient(create_app(service=service, settings=settings))
    upload(client, pdf_bytes)
    response = client.post("/api/v1/ask", json={"question": "Docker Compose network"})
    assert response.status_code == 502
    assert "provider" in response.json()["detail"].lower()


def test_missing_openai_key_gives_503_on_first_use(tmp_path):
    no_key = Settings(provider="openai", openai_api_key="", data_dir=str(tmp_path), _env_file=None)
    client = TestClient(create_app(settings=no_key))
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/documents").status_code == 503


def test_changing_embedding_model_is_caught_with_a_clear_message(tmp_path):
    from app.store import VectorStore

    # data folder was created earlier with a different embedding model
    VectorStore(str(tmp_path / "chroma"), embedding_id="old-model @ openai")
    changed = Settings(
        provider="openai",
        openai_api_key="key",
        embedding_model="gemini-embedding-2",
        openai_base_url="https://example.test/v1/",
        data_dir=str(tmp_path),
        _env_file=None,
    )
    client = TestClient(create_app(settings=changed))
    response = client.get("/api/v1/documents")
    assert response.status_code == 503
    assert "old-model @ openai" in response.json()["detail"]
    assert "docker compose down -v" in response.json()["detail"]


def test_same_embedding_model_reopens_fine(tmp_path):
    from app.store import VectorStore

    VectorStore(str(tmp_path / "chroma"), embedding_id="m @ openai")
    VectorStore(str(tmp_path / "chroma"), embedding_id="m @ openai")  # no error


def test_parallel_first_requests_build_the_service_only_once(tmp_path, monkeypatch):
    """The page loads documents and weak spots at the same moment. Both must share one service."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from app import main

    real_build, builds = main.build_service, []

    def slow_build(settings):
        builds.append(1)
        time.sleep(0.3)  # wide enough that unlocked threads would overlap
        return real_build(settings)

    monkeypatch.setattr(main, "build_service", slow_build)
    settings = Settings(provider="fake", data_dir=str(tmp_path), _env_file=None)
    client = TestClient(create_app(settings=settings))

    paths = ["/api/v1/documents", "/api/v1/gaps"] * 3
    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(lambda path: client.get(path).status_code, paths))

    assert codes == [200] * 6
    assert len(builds) == 1
