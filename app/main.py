"""FastAPI application."""

import logging
import os
import secrets
import threading

import openai
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile

from app.chunking import PdfError
from app.config import Settings, get_settings
from app.embeddings import FakeEmbedder, OpenAIEmbedder
from app.llm import FakeAnswerer, OpenAIAnswerer
from app.ratelimit import RateLimiter
from app.rag import RagService
from app.progress import ProgressStore
from app.schemas import (
    AskIn,
    AskOut,
    AttemptsIn,
    CitationOut,
    ClaimOut,
    DocumentOut,
    FlashcardOut,
    FlashcardsOut,
    GapOut,
    GenerateIn,
    QuizOut,
    QuizQuestionOut,
    ReviewIn,
    ReviewOut,
    VerificationOut,
)
from app.store import VectorStore
from app.study import FakeStudyModel, LLMStudyModel, NotFoundError, StudyError, StudyService
from app.verify import FakeJudge, LLMJudge

log = logging.getLogger("rag")
if not log.handlers:  # uvicorn does not show INFO logs of our own logger by default
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     rag: %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False

PDF_MAGIC = b"%PDF-"


def build_service(settings: Settings) -> RagService:
    os.makedirs(settings.data_dir, exist_ok=True)
    if settings.provider == "fake":
        embedding_id = "fake"
    else:
        embedding_id = f"{settings.embedding_model} @ {settings.openai_base_url or 'openai'}"
    store = VectorStore(
        os.path.join(settings.data_dir, "chroma"), embedding_id=embedding_id
    )
    if settings.provider == "fake":
        embedder, answerer = FakeEmbedder(), FakeAnswerer()
        judge = FakeJudge() if settings.verify_answers else None
    else:
        embedder = OpenAIEmbedder(
            settings.openai_api_key, settings.embedding_model, settings.openai_base_url
        )
        answerer = OpenAIAnswerer(
            settings.openai_api_key,
            settings.chat_model,
            settings.openai_base_url,
            settings.reasoning_effort,
        )
        judge = (
            LLMJudge(
                settings.openai_api_key,
                settings.chat_model,
                settings.openai_base_url,
                settings.reasoning_effort,
            )
            if settings.verify_answers
            else None
        )
    return RagService(
        embedder,
        store,
        answerer,
        judge,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_score=settings.min_score,
    )


def build_study(settings: Settings, rag: RagService) -> StudyService:
    if settings.provider == "fake":
        model = FakeStudyModel()
    else:
        model = LLMStudyModel(
            settings.openai_api_key,
            settings.chat_model,
            settings.openai_base_url,
            settings.reasoning_effort,
        )
    progress = ProgressStore(os.path.join(settings.data_dir, "progress.db"))
    return StudyService(rag.store, model, rag.verifier, progress)


def create_app(
    service: RagService | None = None,
    settings: Settings | None = None,
    study: StudyService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="AI Study Buddy", version="2.0.0")
    ask_limiter = RateLimiter(max_requests=20, window_seconds=60)
    generate_limiter = RateLimiter(max_requests=10, window_seconds=60)
    state: dict = {"service": service, "study": study}
    # Requests run in parallel threads. Without this lock, two first requests would both
    # try to open the same Chroma database at once, and Chroma fails when that happens.
    build_lock = threading.Lock()

    def get_service() -> RagService:
        # Built on first use so the app can start (and /health works) without a key.
        if state["service"] is None:
            with build_lock:
                if state["service"] is None:
                    try:
                        state["service"] = build_service(settings)
                    except RuntimeError as exc:
                        raise HTTPException(status_code=503, detail=str(exc)) from exc
        return state["service"]

    def get_study(rag: RagService = Depends(get_service)) -> StudyService:
        if state["study"] is None:
            with build_lock:
                if state["study"] is None:
                    try:
                        state["study"] = build_study(settings, rag)
                    except RuntimeError as exc:
                        raise HTTPException(status_code=503, detail=str(exc)) from exc
        return state["study"]

    def caller_of(request: Request) -> str:
        return request.headers.get("x-api-key") or (request.client.host if request.client else "?")

    def generating(request: Request, work):
        """Rate-limit, run a study task, and turn its failures into clean HTTP errors."""
        if not generate_limiter.allow(caller_of(request)):
            raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
        try:
            return work()
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StudyError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except openai.OpenAIError as exc:
            log.exception("OpenAI error in study tools")
            raise HTTPException(status_code=502, detail="The AI provider request failed.") from exc

    def check_pages(body: GenerateIn) -> None:
        if body.page_start and body.page_end and body.page_start > body.page_end:
            raise HTTPException(status_code=422, detail="page_start must not be after page_end.")

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and not (
            x_api_key and secrets.compare_digest(x_api_key, settings.api_key)
        ):
            raise HTTPException(status_code=401, detail="Missing or invalid API key.")

    auth = [Depends(require_api_key)]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/documents", response_model=DocumentOut, dependencies=auth)
    def upload_document(
        response: Response,
        file: UploadFile = File(...),
        rag: RagService = Depends(get_service),
    ) -> DocumentOut:
        name = os.path.basename(file.filename or "")[:200]
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="Only .pdf files are accepted.")

        limit = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(
                status_code=413, detail=f"File is larger than {settings.max_upload_mb} MB."
            )
        if not data.startswith(PDF_MAGIC):
            raise HTTPException(status_code=415, detail="This file is not a real PDF.")

        try:
            record, created = rag.ingest_pdf(name, data)
        except PdfError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except openai.OpenAIError as exc:
            log.exception("OpenAI error while indexing")
            raise HTTPException(status_code=502, detail="The AI provider request failed.") from exc

        response.status_code = 201 if created else 200
        return DocumentOut(
            doc_id=record.doc_id, filename=record.filename, pages=record.pages, chunks=record.chunks
        )

    @app.get("/api/v1/documents", response_model=list[DocumentOut], dependencies=auth)
    def list_documents(rag: RagService = Depends(get_service)) -> list[DocumentOut]:
        return [
            DocumentOut(doc_id=d.doc_id, filename=d.filename, pages=d.pages, chunks=d.chunks)
            for d in rag.list_documents()
        ]

    @app.delete("/api/v1/documents/{doc_id}", status_code=204, dependencies=auth)
    def delete_document(
        doc_id: str,
        rag: RagService = Depends(get_service),
        learn: StudyService = Depends(get_study),
    ) -> Response:
        if not rag.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Document not found.")
        learn.forget(doc_id)
        return Response(status_code=204)

    @app.post("/api/v1/ask", response_model=AskOut, dependencies=auth)
    def ask(
        body: AskIn, request: Request, rag: RagService = Depends(get_service)
    ) -> AskOut:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question must not be empty.")
        if len(question) > settings.max_question_chars:
            raise HTTPException(
                status_code=422,
                detail=f"Question is longer than {settings.max_question_chars} characters.",
            )

        if not ask_limiter.allow(caller_of(request)):
            raise HTTPException(status_code=429, detail="Too many questions. Please slow down.")

        try:
            result = rag.ask(question, body.top_k or settings.top_k)
        except openai.OpenAIError as exc:
            log.exception("OpenAI error while answering")
            raise HTTPException(status_code=502, detail="The AI provider request failed.") from exc

        verification = None
        if result.verification is not None:
            verification = VerificationOut(
                trust_score=result.verification.trust_score,
                claims=[
                    ClaimOut(text=c.text, cites=c.cites, supported=c.supported)
                    for c in result.verification.claims
                ],
            )

        return AskOut(
            answer=result.answer,
            grounded=result.grounded,
            verification=verification,
            citations=[
                CitationOut(
                    number=c.number,
                    filename=c.filename,
                    page=c.page,
                    score=c.score,
                    snippet=c.snippet,
                )
                for c in result.citations
            ],
        )

    # ----- study tools -----
    @app.post("/api/v1/quiz", response_model=QuizOut, dependencies=auth)
    def make_quiz(
        body: GenerateIn, request: Request, learn: StudyService = Depends(get_study)
    ) -> QuizOut:
        check_pages(body)
        questions, dropped = generating(
            request, lambda: learn.make_quiz(body.doc_id, body.count, body.page_start, body.page_end)
        )
        return QuizOut(
            questions=[QuizQuestionOut(**vars(q)) for q in questions],
            requested=body.count,
            dropped=dropped,
        )

    @app.post("/api/v1/flashcards", response_model=FlashcardsOut, dependencies=auth)
    def make_flashcards(
        body: GenerateIn, request: Request, learn: StudyService = Depends(get_study)
    ) -> FlashcardsOut:
        check_pages(body)
        cards, dropped = generating(
            request,
            lambda: learn.make_flashcards(body.doc_id, body.count, body.page_start, body.page_end),
        )
        return FlashcardsOut(
            cards=[FlashcardOut(**vars(c)) for c in cards], requested=body.count, dropped=dropped
        )

    @app.post("/api/v1/attempts", status_code=204, dependencies=auth)
    def record_attempts(body: AttemptsIn, learn: StudyService = Depends(get_study)) -> Response:
        try:
            learn.record(body.doc_id, [(r.page, r.correct) for r in body.results])
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StudyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.get("/api/v1/gaps", response_model=list[GapOut], dependencies=auth)
    def knowledge_gaps(learn: StudyService = Depends(get_study)) -> list[GapOut]:
        return [
            GapOut(
                doc_id=s.doc_id,
                filename=s.filename,
                page=s.page,
                attempts=s.attempts,
                correct=s.correct,
                accuracy=s.accuracy,
                weak=s.weak,
            )
            for s in learn.gaps()
        ]

    @app.post("/api/v1/review", response_model=ReviewOut, dependencies=auth)
    def review_page(
        body: ReviewIn, request: Request, learn: StudyService = Depends(get_study)
    ) -> ReviewOut:
        filename, summary = generating(request, lambda: learn.review(body.doc_id, body.page))
        return ReviewOut(filename=filename, page=body.page, summary=summary)

    return app


app = create_app()
