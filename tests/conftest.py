import io

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.config import Settings
from app.embeddings import FakeEmbedder
from app.llm import FakeAnswerer
from app.main import create_app
from app.rag import RagService
from app.store import VectorStore
from app.verify import FakeJudge


def make_pdf(pages: list[str]) -> bytes:
    """Build a small text PDF, one string per page (use \\n for new lines)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


PHISHING_PDF = [
    "Phishing detection uses URL length, hostname length and the number of dots.\n"
    "Random forests handle these tabular features well.",
    "Docker Compose runs the frontend and backend on an internal network.\n"
    "Only the frontend port is published to the host.",
]


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf(PHISHING_PDF)


@pytest.fixture
def service(tmp_path) -> RagService:
    return RagService(
        FakeEmbedder(),
        VectorStore(str(tmp_path / "chroma")),
        FakeAnswerer(),
        FakeJudge(),
        min_score=0.2,
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(provider="fake", data_dir=str(tmp_path), api_key="", _env_file=None)


@pytest.fixture
def client(service, settings) -> TestClient:
    return TestClient(create_app(service=service, settings=settings))
