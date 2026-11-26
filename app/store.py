"""Vector database layer (Chroma, stored on disk)."""

import os
from dataclasses import dataclass

import chromadb

from app.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    doc_id: str
    filename: str
    page: int
    text: str
    score: float  # cosine similarity, higher is better


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    filename: str
    pages: int
    chunks: int


class EmbeddingMismatchError(RuntimeError):
    """The saved vectors were made by a different embedding model than the one in use."""


def _check_embedding_id(path: str, embedding_id: str) -> None:
    """Vectors from different models cannot be compared, so refuse to mix them."""
    marker = os.path.join(path, "embedding_id.txt")
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as f:
            saved = f.read().strip()
        if saved != embedding_id:
            raise EmbeddingMismatchError(
                f"Your saved documents were indexed with '{saved}', but the app is now set to "
                f"'{embedding_id}'. Switch the settings back, or delete the saved data "
                "(with Docker: docker compose down -v) and upload your PDFs again."
            )
    else:
        with open(marker, "w", encoding="utf-8") as f:
            f.write(embedding_id)


class VectorStore:
    def __init__(self, path: str, collection: str = "papers", embedding_id: str | None = None):
        os.makedirs(path, exist_ok=True)
        if embedding_id is not None:
            _check_embedding_id(path, embedding_id)
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def has_document(self, doc_id: str) -> bool:
        found = self._col.get(where={"doc_id": doc_id}, limit=1, include=[])
        return bool(found["ids"])

    def add_document(
        self, doc_id: str, filename: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._col.add(
            ids=[f"{doc_id}:{c.index}" for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {"doc_id": doc_id, "filename": filename, "page": c.page} for c in chunks
            ],
        )

    def query(self, vector: list[float], top_k: int) -> list[Hit]:
        total = self._col.count()
        if total == 0:
            return []
        result = self._col.query(
            query_embeddings=[vector],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                Hit(
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    page=int(meta["page"]),
                    text=text,
                    score=1.0 - float(dist),  # chroma cosine distance = 1 - similarity
                )
            )
        return hits

    def get_chunks(self, doc_id: str) -> list[Hit]:
        """Every stored chunk of one document, in reading order (page, then position)."""
        data = self._col.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        rows = sorted(
            zip(data["ids"], data["documents"], data["metadatas"]),
            key=lambda r: (int(r[2]["page"]), int(r[0].rsplit(":", 1)[1])),
        )
        return [
            Hit(doc_id=m["doc_id"], filename=m["filename"], page=int(m["page"]), text=t, score=1.0)
            for _, t, m in rows
        ]

    def list_documents(self) -> list[DocumentRecord]:
        data = self._col.get(include=["metadatas"])
        grouped: dict[str, dict] = {}
        for meta in data["metadatas"]:
            entry = grouped.setdefault(
                meta["doc_id"],
                {"filename": meta["filename"], "pages": 0, "chunks": 0},
            )
            entry["pages"] = max(entry["pages"], int(meta["page"]))
            entry["chunks"] += 1
        return [
            DocumentRecord(doc_id=k, filename=v["filename"], pages=v["pages"], chunks=v["chunks"])
            for k, v in sorted(grouped.items(), key=lambda kv: kv[1]["filename"].lower())
        ]

    def delete_document(self, doc_id: str) -> bool:
        if not self.has_document(doc_id):
            return False
        self._col.delete(where={"doc_id": doc_id})
        return True
