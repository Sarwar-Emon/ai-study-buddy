import json

import pytest

from evaluation.run_eval import EvalItem, evaluate, load_items
from tests.conftest import PHISHING_PDF


def test_evaluate_scores_hits_and_misses(service, pdf_bytes):
    service.ingest_pdf("a.pdf", pdf_bytes)
    items = [
        EvalItem("Which network does Docker Compose use?", "a.pdf", 2),  # page 2 has the answer
        EvalItem("What features detect phishing URLs?", "a.pdf", 1),  # page 1 has the answer
        EvalItem("What features detect phishing URLs?", "a.pdf", 2),  # wrong page on purpose
    ]
    report = evaluate(service, items, k=2)
    assert report.total == 3
    assert report.hit_at_1 == pytest.approx(2 / 3)
    assert report.hit_at_k == pytest.approx(1.0)  # only 2 pages, so page 2 is always in the top 2
    assert 0 < report.mrr <= 1


def test_evaluate_reports_questions_that_miss(service, pdf_bytes):
    service.ingest_pdf("a.pdf", pdf_bytes)
    report = evaluate(service, [EvalItem("Docker Compose network", "a.pdf", 99)], k=2)
    assert report.hit_at_k == 0
    assert report.misses == ["Docker Compose network"]


def test_empty_question_set_is_an_error(service):
    with pytest.raises(ValueError):
        evaluate(service, [], k=4)


def test_load_items_reads_json(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"question": "q?", "filename": "a.pdf", "page": "3"}]))
    assert load_items(str(path)) == [EvalItem("q?", "a.pdf", 3)]
    assert len(PHISHING_PDF) == 2
