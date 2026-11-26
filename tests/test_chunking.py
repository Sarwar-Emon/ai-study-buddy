import pytest

from app.chunking import PdfError, chunk_pdf, split_text
from tests.conftest import make_pdf


def test_short_text_is_one_piece():
    assert split_text("hello   world", 100, 10) == ["hello world"]


def test_empty_text_gives_no_pieces():
    assert split_text("   \n  ", 100, 10) == []


def test_long_text_is_split_with_overlap_and_no_word_is_cut():
    words = [f"word{i}" for i in range(200)]
    pieces = split_text(" ".join(words), size=120, overlap=30)
    assert len(pieces) > 1
    assert all(len(p) <= 120 for p in pieces)
    # every word appears whole in at least one piece
    joined = " ".join(pieces)
    for w in words:
        assert w in joined.split()
    # neighbouring pieces share some text
    assert set(pieces[0].split()) & set(pieces[1].split())


def test_split_always_makes_progress_on_one_giant_word():
    pieces = split_text("x" * 5000, size=100, overlap=20)
    assert len(pieces) > 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_bad_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_text("some text here", size, overlap)


def test_chunks_remember_their_page():
    pdf = make_pdf(["alpha beta gamma", "delta epsilon zeta"])
    chunks = chunk_pdf(pdf, 1000, 100)
    assert [c.page for c in chunks] == [1, 2]
    assert [c.index for c in chunks] == [0, 1]
    assert "delta" in chunks[1].text


def test_not_a_pdf_raises_pdf_error():
    with pytest.raises(PdfError):
        chunk_pdf(b"this is not a pdf", 1000, 100)


def test_pdf_without_text_raises_pdf_error():
    blank = make_pdf([""])
    with pytest.raises(PdfError, match="No text"):
        chunk_pdf(blank, 1000, 100)
