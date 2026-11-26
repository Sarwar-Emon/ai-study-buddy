"""Turn a PDF into small, page-aware text chunks.

Every chunk remembers which page it came from, so answers can cite the page.
"""

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PdfError(ValueError):
    """The file is not a readable PDF, or it has no text we can use."""


@dataclass(frozen=True)
class Chunk:
    page: int  # 1-based page number
    index: int  # position of the chunk inside the whole document
    text: str


def extract_pages(data: bytes) -> list[str]:
    """Return the text of each page (empty string for pages with no text)."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PdfError("Encrypted PDFs are not supported.")
        return [(page.extract_text() or "") for page in reader.pages]
    except PdfError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise PdfError("Could not read this file as a PDF.") from exc


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into pieces of about `size` characters.

    Cuts happen at spaces, and neighbouring pieces share `overlap` characters,
    so a sentence that sits on a boundary still appears whole in one piece.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    text = _normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        # step forward, keeping `overlap` characters, but always make progress
        next_start = max(end - overlap, start + 1)
        # begin the next piece at a word boundary
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return pieces


def chunk_pdf(data: bytes, size: int, overlap: int) -> list[Chunk]:
    """Read a PDF and return all of its chunks, in order."""
    pages = extract_pages(data)
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        for piece in split_text(page_text, size, overlap):
            chunks.append(Chunk(page=page_number, index=len(chunks), text=piece))
    if not chunks:
        raise PdfError(
            "No text found in this PDF. Scanned documents need OCR, which is not supported."
        )
    return chunks
