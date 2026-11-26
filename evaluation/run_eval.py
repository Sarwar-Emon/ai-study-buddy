"""Measure how well retrieval finds the right page.

You write a small question set (see questions.example.json): each question says
which PDF and which page holds the answer. This script indexes your PDFs, asks
every question, and checks whether the right page shows up in the top results.

It only uses embeddings (no chat model), so it is cheap to run.

    python -m evaluation.run_eval --pdf-dir ./my_papers --questions evaluation/questions.json
"""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.chunking import PdfError
from app.config import get_settings
from app.embeddings import FakeEmbedder, OpenAIEmbedder
from app.llm import FakeAnswerer
from app.rag import RagService
from app.store import VectorStore


@dataclass(frozen=True)
class EvalItem:
    question: str
    filename: str
    page: int


@dataclass(frozen=True)
class EvalReport:
    total: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    k: int
    misses: list[str]


def load_items(path: str) -> list[EvalItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalItem(r["question"], r["filename"], int(r["page"])) for r in raw]


def evaluate(service: RagService, items: list[EvalItem], k: int) -> EvalReport:
    """Score retrieval. A question counts as a hit if the expected page is in the top k."""
    if not items:
        raise ValueError("The question set is empty.")
    hit1 = hitk = 0
    reciprocal_sum = 0.0
    misses: list[str] = []
    for item in items:
        hits = service.retrieve(item.question, k)
        rank = next(
            (i for i, h in enumerate(hits, start=1) if h.filename == item.filename and h.page == item.page),
            None,
        )
        if rank is None:
            misses.append(item.question)
            continue
        hitk += 1
        hit1 += rank == 1
        reciprocal_sum += 1.0 / rank
    n = len(items)
    return EvalReport(
        total=n, hit_at_1=hit1 / n, hit_at_k=hitk / n, mrr=reciprocal_sum / n, k=k, misses=misses
    )


def build_service(provider: str, workdir: str) -> RagService:
    settings = get_settings()
    store = VectorStore(os.path.join(workdir, "chroma"))
    if provider == "fake":
        embedder = FakeEmbedder()
    else:
        embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
    return RagService(
        embedder,
        store,
        FakeAnswerer(),  # not used: evaluation only retrieves
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_score=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--pdf-dir", required=True, help="folder with the PDFs named in the question set")
    parser.add_argument("--questions", required=True, help="JSON file with questions")
    parser.add_argument("--k", type=int, default=4, help="how many top results to check (default 4)")
    parser.add_argument("--provider", choices=["openai", "fake"], default="openai")
    args = parser.parse_args()

    items = load_items(args.questions)
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.pdf_dir}", file=sys.stderr)
        return 1
    missing = {i.filename for i in items} - {p.name for p in pdfs}
    if missing:
        print(f"Question set mentions PDFs that are not in the folder: {sorted(missing)}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as workdir:
        service = build_service(args.provider, workdir)
        for pdf in pdfs:
            try:
                service.ingest_pdf(pdf.name, pdf.read_bytes())
            except PdfError as exc:
                print(f"Skipping {pdf.name}: {exc}", file=sys.stderr)
        report = evaluate(service, items, args.k)

    print(f"Provider:          {args.provider}")
    print(f"Questions:         {report.total}")
    print(f"Hit@1:             {report.hit_at_1:.1%}")
    print(f"Hit@{report.k}:             {report.hit_at_k:.1%}")
    print(f"MRR:               {report.mrr:.3f}")
    if report.misses:
        print("\nQuestions where the right page was NOT in the top results:")
        for q in report.misses:
            print(f"  - {q}")
    if args.provider == "fake":
        print("\nNote: 'fake' is an offline word-match demo. Do not report these numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
