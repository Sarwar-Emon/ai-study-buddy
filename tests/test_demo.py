from datetime import date

from fastapi.testclient import TestClient

from app.config import Settings
from app.demo import DemoGuard
from app.main import create_app
from tests.conftest import make_pdf
from tests.test_api import upload


def demo_client(service, tmp_path, **over):
    cfg = Settings(provider="fake", data_dir=str(tmp_path), demo_mode=True, _env_file=None, **over)
    return TestClient(create_app(service=service, settings=cfg))


def test_guard_stops_at_the_limit_and_resets_next_day():
    day = [date(2026, 1, 1)]
    guard = DemoGuard(2, today=lambda: day[0])
    assert guard.take() and guard.take()
    assert not guard.take()
    day[0] = date(2026, 1, 2)
    assert guard.take()


def test_config_endpoint_reports_demo_mode(service, tmp_path, settings):
    off = TestClient(create_app(service=service, settings=settings))
    assert off.get("/api/v1/config").json()["demo"] is False
    on = demo_client(service, tmp_path, demo_max_docs=3)
    body = on.get("/api/v1/config").json()
    assert body["demo"] is True and body["max_documents"] == 3


def test_daily_budget_returns_a_friendly_429(service, tmp_path, pdf_bytes):
    client = demo_client(service, tmp_path, demo_daily_requests=2)
    assert upload(client, pdf_bytes).status_code == 201  # spends 1
    assert client.post("/api/v1/ask", json={"question": "What is phishing?"}).status_code == 200  # 2
    res = client.post("/api/v1/ask", json={"question": "What is phishing?"})
    assert res.status_code == 429 and "limit for today" in res.json()["detail"]


def test_document_cap_blocks_new_files_but_not_repeats(service, tmp_path, pdf_bytes):
    client = demo_client(service, tmp_path, demo_max_docs=1)
    assert upload(client, pdf_bytes).status_code == 201
    other = make_pdf(["A different document about gradient descent and learning rates."])
    assert upload(client, other, "other.pdf").status_code == 409
    assert upload(client, pdf_bytes).status_code == 200  # same file is not a new document


def test_sample_loads_once(service, tmp_path):
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(make_pdf(["Overfitting means a model memorizes its training data."]))
    cfg = Settings(provider="fake", data_dir=str(tmp_path), demo_mode=True,
                   sample_pdf=str(sample), _env_file=None)
    app = create_app(service=service, settings=cfg)
    app.state.load_sample()
    app.state.load_sample()  # a second call must not duplicate it
    assert len(TestClient(app).get("/api/v1/documents").json()) == 1


def test_built_frontend_is_served_without_hiding_the_api(service, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>hello</h1>")
    cfg = Settings(provider="fake", data_dir=str(tmp_path), static_dir=str(site), _env_file=None)
    client = TestClient(create_app(service=service, settings=cfg))
    assert "hello" in client.get("/").text
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/v1/documents").status_code == 200
