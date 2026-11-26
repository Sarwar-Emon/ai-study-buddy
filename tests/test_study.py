import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.progress import ProgressStore
from app.study import FakeStudyModel, NotFoundError, StudyError, StudyService
from app.verify import FakeJudge, VerifierError
from tests.conftest import make_pdf

PAGES = [
    "Phishing detection uses URL length, hostname length and the number of dots.\n"
    "Random forests handle these tabular features well.",
    "Docker Compose runs the frontend and backend on an internal network.\n"
    "Only the frontend port is published to the host.",
    "Rate limiting counts requests in a sliding window for each caller.\n"
    "Callers over the limit receive a 429 response.",
]


@pytest.fixture
def doc(service):
    record, _ = service.ingest_pdf("notes.pdf", make_pdf(PAGES))
    return record


@pytest.fixture
def make_study(service, tmp_path):
    def build(model=None, judge="fake") -> StudyService:
        return StudyService(
            service.store,
            model or FakeStudyModel(),
            FakeJudge() if judge == "fake" else judge,
            ProgressStore(str(tmp_path / "progress.db")),
        )

    return build


# ----- store and progress -----
def test_get_chunks_come_back_in_reading_order(service, doc):
    pages = [c.page for c in service.store.get_chunks(doc.doc_id)]
    assert pages == sorted(pages) and set(pages) == {1, 2, 3}
    assert service.store.get_chunks("missing") == []


def test_progress_ranks_weakest_pages_first(tmp_path):
    progress = ProgressStore(str(tmp_path / "p.db"))
    progress.record("d", "a.pdf", [(1, True), (1, True), (2, False), (2, True), (3, False)])
    stats = progress.page_stats()
    assert [(s.page, s.accuracy, s.weak) for s in stats] == [(3, 0.0, True), (2, 0.5, True), (1, 1.0, False)]
    progress.delete_document("d")
    assert progress.page_stats() == []


def test_progress_survives_reopening(tmp_path):
    ProgressStore(str(tmp_path / "p.db")).record("d", "a.pdf", [(1, False)])
    assert ProgressStore(str(tmp_path / "p.db")).page_stats()[0].attempts == 1


# ----- quiz -----
def test_quiz_questions_are_well_formed_and_verified(make_study, doc):
    questions, dropped = make_study().make_quiz(doc.doc_id, 3)
    assert dropped == 0 and len(questions) == 3
    for q in questions:
        assert len(q.options) == 4 and 0 <= q.answer_index < 4
        assert q.verified is True and q.filename == "notes.pdf" and 1 <= q.page <= 3


def test_correct_answer_survives_the_shuffle(make_study, doc):
    for q in make_study().make_quiz(doc.doc_id, 3)[0]:
        # the fake model's correct option is the page's own first sentence
        assert q.options[q.answer_index] in PAGES[q.page - 1].replace("\n", " ")


def test_page_range_is_respected(make_study, doc):
    questions, _ = make_study().make_quiz(doc.doc_id, 3, page_start=2, page_end=2)
    assert {q.page for q in questions} == {2}


def test_count_is_capped_by_available_passages(make_study, doc):
    assert len(make_study().make_quiz(doc.doc_id, 10)[0]) == 3


def test_unknown_document_is_not_found(make_study):
    with pytest.raises(NotFoundError):
        make_study().make_quiz("nope", 3)


class Scripted:
    """A model that returns whatever we give it."""

    def __init__(self, quiz=None, cards=None):
        self._quiz, self._cards = quiz or [], cards or []

    def quiz(self, passages):
        return self._quiz

    def flashcards(self, passages):
        return self._cards

    def explain(self, text):
        return "- ok"


GOOD = {
    "passage": 1,
    "question": "How do the frontend and backend run in Docker Compose?",
    "options": ["On an internal network", "On the public internet", "On a USB stick", "Inside a database"],
    "answer": 0,
    "explanation": "It says so.",
}
MADE_UP = {
    "passage": 1,
    "question": "Which transformer reached 99% accuracy on ImageNet?",
    "options": ["Vision models", "Random forests", "Kernel machines", "Nearest neighbours"],
    "answer": 0,
    "explanation": "Invented.",
}


def test_question_the_page_does_not_back_up_is_dropped(make_study, doc):
    study = make_study(Scripted(quiz=[GOOD, MADE_UP]))
    questions, dropped = study.make_quiz(doc.doc_id, 3, page_start=2, page_end=2)
    assert dropped == 1 and len(questions) == 1
    assert questions[0].question.startswith("How do the frontend")


def test_all_questions_unbacked_is_an_error(make_study, doc):
    with pytest.raises(StudyError):
        make_study(Scripted(quiz=[MADE_UP])).make_quiz(doc.doc_id, 3, page_start=2, page_end=2)


@pytest.mark.parametrize(
    "broken",
    [
        {**GOOD, "options": GOOD["options"][:3]},  # only 3 options
        {**GOOD, "answer": 9},  # answer index out of range
        {**GOOD, "answer": True},  # a bool is not an index
        {**GOOD, "passage": 2},  # points at a passage that was not sent (only 1 was)
        {**GOOD, "options": ["A", "A", "B", "C"]},  # duplicate options
        {**GOOD, "question": "  "},  # empty question
        {"nonsense": 1},
    ],
)
def test_malformed_model_output_is_rejected(make_study, doc, broken):
    with pytest.raises(StudyError):
        make_study(Scripted(quiz=[broken])).make_quiz(doc.doc_id, 3, page_start=2, page_end=2)


def test_without_a_checker_questions_are_kept_but_unverified(make_study, doc):
    study = make_study(Scripted(quiz=[GOOD, MADE_UP]), judge=None)
    questions, dropped = study.make_quiz(doc.doc_id, 3, page_start=2, page_end=2)
    assert dropped == 0 and [q.verified for q in questions] == [None, None]


def test_a_failing_checker_does_not_block_the_quiz(make_study, doc):
    class Broken:
        def judge(self, claims, hits):
            raise VerifierError("down")

    questions, _ = make_study(Scripted(quiz=[GOOD]), judge=Broken()).make_quiz(
        doc.doc_id, 3, page_start=2, page_end=2
    )
    assert questions[0].verified is None


# ----- flashcards -----
def test_flashcards_are_verified(make_study, doc):
    cards, dropped = make_study().make_flashcards(doc.doc_id, 3)
    assert dropped == 0 and len(cards) == 3 and all(c.verified for c in cards)


def test_flashcard_the_page_does_not_back_up_is_dropped(make_study, doc):
    good = {"passage": 1, "front": "Which ports does Compose publish?", "back": "Only the frontend port is published to the host."}
    bad = {"passage": 1, "front": "What accuracy did transformers reach?", "back": "Ninety nine percent on ImageNet."}
    cards, dropped = make_study(Scripted(cards=[good, bad])).make_flashcards(
        doc.doc_id, 3, page_start=2, page_end=2
    )
    assert dropped == 1 and cards[0].front.startswith("Which ports")


# ----- progress, gaps and review -----
def test_record_and_gaps(make_study, doc):
    study = make_study()
    study.record(doc.doc_id, [(2, False), (2, False), (1, True)])
    gaps = study.gaps()
    assert [(g.page, g.weak) for g in gaps] == [(2, True), (1, False)]


def test_record_rejects_a_page_that_does_not_exist(make_study, doc):
    with pytest.raises(StudyError):
        make_study().record(doc.doc_id, [(99, True)])


def test_gaps_ignore_deleted_documents(make_study, service, doc):
    study = make_study()
    study.record(doc.doc_id, [(1, False)])
    service.delete_document(doc.doc_id)
    assert study.gaps() == []


def test_review_summarises_one_page(make_study, doc):
    filename, summary = make_study().review(doc.doc_id, 2)
    assert filename == "notes.pdf" and "Docker Compose" in summary
    with pytest.raises(NotFoundError):
        make_study().review(doc.doc_id, 50)


# ----- API -----
@pytest.fixture
def api(service, settings, make_study):
    def build(model=None, **overrides):
        s = settings.model_copy(update=overrides) if overrides else settings
        return TestClient(create_app(service=service, settings=s, study=make_study(model)))

    return build


def test_api_quiz_flashcards_attempts_gaps_review_flow(api, doc):
    client = api()
    quiz = client.post("/api/v1/quiz", json={"doc_id": doc.doc_id, "count": 3}).json()
    assert len(quiz["questions"]) == 3 and quiz["dropped"] == 0
    assert set(quiz["questions"][0]) >= {"question", "options", "answer_index", "page", "verified"}

    cards = client.post("/api/v1/flashcards", json={"doc_id": doc.doc_id, "count": 2}).json()
    assert len(cards["cards"]) == 2

    sent = client.post(
        "/api/v1/attempts",
        json={"doc_id": doc.doc_id, "results": [{"page": 2, "correct": False}, {"page": 1, "correct": True}]},
    )
    assert sent.status_code == 204
    gaps = client.get("/api/v1/gaps").json()
    assert [(g["page"], g["weak"]) for g in gaps] == [(2, True), (1, False)]

    review = client.post("/api/v1/review", json={"doc_id": doc.doc_id, "page": 2}).json()
    assert review["page"] == 2 and review["summary"]

    assert client.delete(f"/api/v1/documents/{doc.doc_id}").status_code == 204
    assert client.get("/api/v1/gaps").json() == []


def test_api_validation_and_not_found(api, doc):
    client = api()
    assert client.post("/api/v1/quiz", json={"doc_id": "nope", "count": 3}).status_code == 404
    assert client.post("/api/v1/quiz", json={"doc_id": doc.doc_id, "count": 0}).status_code == 422
    assert client.post("/api/v1/quiz", json={"doc_id": doc.doc_id, "count": 11}).status_code == 422
    inverted = {"doc_id": doc.doc_id, "page_start": 3, "page_end": 1}
    assert client.post("/api/v1/flashcards", json=inverted).status_code == 422
    bad_page = {"doc_id": doc.doc_id, "results": [{"page": 99, "correct": True}]}
    assert client.post("/api/v1/attempts", json=bad_page).status_code == 422
    assert client.post("/api/v1/attempts", json={"doc_id": doc.doc_id, "results": []}).status_code == 422
    assert client.post("/api/v1/review", json={"doc_id": doc.doc_id, "page": 50}).status_code == 404


def test_api_bad_model_output_is_502(api, doc):
    client = api(Scripted(quiz=[{"nonsense": 1}]))
    response = client.post("/api/v1/quiz", json={"doc_id": doc.doc_id, "count": 2})
    assert response.status_code == 502 and "usable" in response.json()["detail"]


def test_api_provider_failure_is_502(api, doc):
    class Down(Scripted):
        def quiz(self, passages):
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

    response = api(Down()).post("/api/v1/quiz", json={"doc_id": doc.doc_id, "count": 2})
    assert response.status_code == 502 and "provider" in response.json()["detail"].lower()


def test_api_generation_is_rate_limited(api, doc):
    client = api()
    codes = [client.post("/api/v1/flashcards", json={"doc_id": doc.doc_id, "count": 1}).status_code for _ in range(12)]
    assert codes[:10] == [200] * 10 and codes[10:] == [429, 429]


def test_api_study_routes_need_the_key(api, doc):
    client = api(api_key="s3cret")
    assert client.post("/api/v1/quiz", json={"doc_id": doc.doc_id}).status_code == 401
    assert client.get("/api/v1/gaps").status_code == 401
    ok = client.get("/api/v1/gaps", headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


def test_study_service_is_built_on_first_use(tmp_path, pdf_bytes):
    """No injected study service: the app builds one (fake provider) when first needed."""
    settings = Settings(provider="fake", data_dir=str(tmp_path), _env_file=None)
    client = TestClient(create_app(settings=settings))
    doc_id = client.post("/api/v1/documents", files={"file": ("p.pdf", pdf_bytes, "application/pdf")}).json()["doc_id"]
    assert client.post("/api/v1/quiz", json={"doc_id": doc_id, "count": 2}).status_code == 200


# ----- speed knob: reasoning_effort is only sent when set -----
@pytest.mark.parametrize("effort, expected", [("low", {"reasoning_effort": "low"}), ("", {})])
def test_reasoning_effort_reaches_every_model_call(monkeypatch, effort, expected):
    from types import SimpleNamespace

    from app import llm, study, verify
    from app.store import Hit

    sent = []

    class FakeCompletions:
        def create(self, **kwargs):
            sent.append(kwargs)
            msg = SimpleNamespace(content='[{"id": 1, "supported": true}]')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class FakeClient:
        def __init__(self, api_key, base_url=None, max_retries=2):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    for module in (llm, verify, study):
        monkeypatch.setattr(module, "OpenAI", FakeClient)

    hit = Hit("d", "f.pdf", 1, "some passage text", 1.0)
    llm.OpenAIAnswerer("k", "m", "", effort).answer("q?", [hit])
    verify.LLMJudge("k", "m", "", effort).judge([("claim [1].", [1])], [hit])
    study.LLMStudyModel("k", "m", "", effort).explain("text")  # one _chat call is enough for the study model

    assert len(sent) == 3
    for call in sent:
        assert {k: v for k, v in call.items() if k == "reasoning_effort"} == expected


# ----- coming back short: ask again, read answers loosely, say how many -----
class ShortFirst:
    """Returns only one question the first time, then everything it is asked for."""

    def __init__(self):
        self.calls = 0

    def quiz(self, passages):
        self.calls += 1
        items = FakeStudyModel().quiz(passages)
        return items[:1] if self.calls == 1 else items

    def flashcards(self, passages):
        self.calls += 1
        items = FakeStudyModel().flashcards(passages)
        return items[:1] if self.calls == 1 else items

    def explain(self, text):
        return "- ok"


def test_quiz_asks_again_when_the_model_comes_back_short(make_study, doc):
    model = ShortFirst()
    questions, _ = make_study(model).make_quiz(doc.doc_id, 3)
    assert len(questions) == 3  # asked for 3, first answer had 1, the second round filled the gap
    assert model.calls == 2


def test_flashcards_ask_again_too(make_study, doc):
    cards, _ = make_study(ShortFirst()).make_flashcards(doc.doc_id, 3)
    assert len(cards) == 3


def test_it_gives_up_after_a_few_rounds_and_returns_what_it_has(make_study, doc):
    class OnlyOnce(Scripted):
        calls = 0

        def quiz(self, passages):
            OnlyOnce.calls += 1
            return FakeStudyModel().quiz(passages)[:1] if OnlyOnce.calls == 1 else []

    questions, _ = make_study(OnlyOnce()).make_quiz(doc.doc_id, 3)
    assert len(questions) == 1 and OnlyOnce.calls == 3  # tried 3 rounds, then returned what it had


def test_unusable_output_in_one_round_does_not_sink_the_quiz(make_study, doc):
    class GarbageThenGood:
        calls = 0

        def quiz(self, passages):
            GarbageThenGood.calls += 1
            if GarbageThenGood.calls == 1:
                raise StudyError("unreadable")
            return FakeStudyModel().quiz(passages)

        def explain(self, text):
            return ""

    questions, _ = make_study(GarbageThenGood()).make_quiz(doc.doc_id, 2)
    assert len(questions) == 2


@pytest.mark.parametrize("answer", ["B", "b", "1", 1, 1.0, "On an internal network"])
def test_answer_and_passage_numbers_are_read_loosely(make_study, doc, answer):
    item = {
        "passage": "1",  # text instead of a number
        "question": "How do the frontend and backend run in Docker Compose?",
        "options": ["On a USB stick", "On an internal network", "On a ferry", "Inside a database"],
        "answer": answer,  # every case here means option index 1
        "explanation": "",
    }
    questions, _ = make_study(Scripted(quiz=[item])).make_quiz(doc.doc_id, 1, page_start=2, page_end=2)
    assert questions[0].options[questions[0].answer_index] == "On an internal network"


def test_api_reports_how_many_were_asked_for(api, doc):
    client = api()
    body = client.post("/api/v1/quiz", json={"doc_id": doc.doc_id, "count": 10}).json()
    assert body["requested"] == 10 and len(body["questions"]) == 3  # the document only has 3 passages
    cards = client.post("/api/v1/flashcards", json={"doc_id": doc.doc_id, "count": 7}).json()
    assert cards["requested"] == 7
