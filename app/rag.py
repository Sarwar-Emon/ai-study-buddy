"""The RAG pipeline: index PDFs, then answer questions from them."""

import hashlib
import logging
from dataclasses import dataclass

from app.chunking import chunk_pdf
from app.embeddings import Embedder
from app.llm import Answerer
from app.store import DocumentRecord, Hit, VectorStore
from app.verify import Judge, Verification, VerifierError, verify_answer

log = logging.getLogger("rag")

NOT_FOUND_MESSAGE = (
    "I could not find this in the uploaded documents. "
    "Try rephrasing, or upload a document that covers it."
)


@dataclass(frozen=True)
class Citation:
    number: int
    filename: str
    page: int
    score: float
    snippet: str


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]
    grounded: bool  # False when we refused because nothing relevant was found
    verification: Verification | None = None  # None when the trust check did not run


class RagService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        answerer: Answerer,
        verifier: Judge | None = None,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_score: float = 0.2,
    ):
        self._embedder = embedder
        self._store = store
        self._answerer = answerer
        self._verifier = verifier
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_score = min_score

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def verifier(self) -> Judge | None:
        return self._verifier

    # ----- indexing -----
    @staticmethod
    def make_doc_id(data: bytes) -> str:
        """The same file always gets the same id, whatever it is named."""
        return hashlib.sha256(data).hexdigest()[:16]

    def ingest_pdf(self, filename: str, data: bytes) -> tuple[DocumentRecord, bool]:
        """Index a PDF. Returns (record, created). Same file twice is a no-op."""
        doc_id = self.make_doc_id(data)
        if self._store.has_document(doc_id):
            record = next(d for d in self._store.list_documents() if d.doc_id == doc_id)
            return record, False

        chunks = chunk_pdf(data, self._chunk_size, self._chunk_overlap)
        vectors = self._embedder.embed([c.text for c in chunks])
        self._store.add_document(doc_id, filename, chunks, vectors)
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            pages=max(c.page for c in chunks),
            chunks=len(chunks),
        )
        return record, True

    def list_documents(self) -> list[DocumentRecord]:
        return self._store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        return self._store.delete_document(doc_id)

    # ----- retrieval + answering -----
    def retrieve(self, question: str, top_k: int) -> list[Hit]:
        vector = self._embedder.embed([question])[0]
        return self._store.query(vector, top_k)

    def ask(self, question: str, top_k: int = 4) -> Answer:
        hits = [h for h in self.retrieve(question, top_k) if h.score >= self._min_score]
        if not hits:
            return Answer(answer=NOT_FOUND_MESSAGE, citations=[], grounded=False)

        text = self._answerer.answer(question, hits)
        citations = [
            Citation(
                number=i,
                filename=h.filename,
                page=h.page,
                score=round(h.score, 4),
                snippet=h.text[:240],
            )
            for i, h in enumerate(hits, start=1)
        ]

        verification = None
        if self._verifier is not None:
            try:
                verification = verify_answer(self._verifier, text, hits)
            except VerifierError:
                # The trust check is an extra. If it fails, still return the answer.
                log.warning("Trust check failed; returning the answer without it.", exc_info=True)
        return Answer(answer=text, citations=citations, grounded=True, verification=verification)
