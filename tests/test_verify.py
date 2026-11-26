from types import SimpleNamespace

import pytest

from app import verify
from app.main import build_service
from app.config import Settings
from app.rag import RagService
from app.store import Hit
from app.verify import FakeJudge, LLMJudge, VerifierError, split_claims, verify_answer

PASSAGE = "Docker Compose runs the frontend and backend on an internal network. Only the frontend port is published."
HITS = [Hit("d1", "notes.pdf", 2, PASSAGE, 0.9)]


# ----- splitting -----
def test_split_keeps_citations_and_decimals():
    text = "The runtime fell by 26.97% on the benchmark [1][2]. A second claim is here [2]."
    claims = split_claims(text)
    assert [c[1] for c in claims] == [[1, 2], [2]]
    assert claims[0][0].startswith("The runtime fell by 26.97%")


def test_split_handles_lists_new_lines_and_trailing_citations():
    text = "1. First claim goes here fully.\n- Second claim goes here fully. [1]\n\nShort."
    claims = split_claims(text)
    assert [c[0] for c in claims] == ["First claim goes here fully.", "Second claim goes here fully. [1]"]
    assert claims[1][1] == [1]  # the lone citation stayed with its sentence; "Short." is dropped


def test_split_caps_the_number_of_claims():
    text = " ".join(f"Claim number {i} says something useful." for i in range(40))
    assert len(split_claims(text)) == verify.MAX_CLAIMS


# ----- verifying -----
def test_supported_and_made_up_claims_are_told_apart():
    answer = (
        "Docker Compose runs the frontend and backend on an internal network [1]. "
        "The system reached 99% accuracy on ImageNet using transformers [1]."
    )
    result = verify_answer(FakeJudge(), answer, HITS)
    assert [c.supported for c in result.claims] == [True, False]
    assert result.trust_score == 0.5


def test_citation_to_a_missing_passage_is_unsupported_without_asking_the_judge():
    class Boom:
        def judge(self, claims, hits):
            raise AssertionError("must not judge a claim that cites a missing passage")

    result = verify_answer(Boom(), "Docker Compose runs the frontend on a network [7].", HITS)
    assert result.claims[0].supported is False
    assert result.trust_score == 0.0


def test_refusal_text_has_nothing_to_verify():
    assert verify_answer(FakeJudge(), "I could not find this in the uploaded documents.", HITS) is None


def test_wrong_number_of_verdicts_is_an_error():
    class Short:
        def judge(self, claims, hits):
            return []

    with pytest.raises(VerifierError):
        verify_answer(Short(), "Docker Compose runs the frontend on a network [1].", HITS)


# ----- the model-backed judge (client is mocked, no network) -----
def make_judge(monkeypatch, reply):
    class FakeCompletions:
        def create(self, **kwargs):
            if isinstance(reply, Exception):
                raise reply
            msg = SimpleNamespace(content=reply)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class FakeClient:
        def __init__(self, api_key, base_url=None, max_retries=2):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(verify, "OpenAI", FakeClient)
    return LLMJudge("key", "model-x")


CLAIMS = [("First claim [1].", [1]), ("Second claim [1].", [1])]


def test_llm_judge_reads_plain_and_fenced_json(monkeypatch):
    plain = make_judge(monkeypatch, '[{"id": 1, "supported": true}, {"id": 2, "supported": false}]')
    assert plain.judge(CLAIMS, HITS) == [True, False]
    fenced = make_judge(monkeypatch, '```json\n[{"id": 1, "supported": false}, {"id": 2, "supported": true}]\n```')
    assert fenced.judge(CLAIMS, HITS) == [False, True]


def test_llm_judge_treats_a_skipped_claim_as_unsupported(monkeypatch):
    judge = make_judge(monkeypatch, '[{"id": 1, "supported": true}]')
    assert judge.judge(CLAIMS, HITS) == [True, False]


@pytest.mark.parametrize("reply", ["no json here", '{"id": 1}', '[{"nope": 1}]'])
def test_llm_judge_bad_output_is_a_verifier_error(monkeypatch, reply):
    with pytest.raises(VerifierError):
        make_judge(monkeypatch, reply).judge(CLAIMS, HITS)


def test_llm_judge_provider_failure_is_a_verifier_error(monkeypatch):
    import httpx
    import openai

    error = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    with pytest.raises(VerifierError):
        make_judge(monkeypatch, error).judge(CLAIMS, HITS)


# ----- inside the pipeline -----
class Hallucinating:
    """Answerer that adds one sentence the document never said."""

    def answer(self, question, hits):
        return (
            "Docker Compose runs the frontend and backend on an internal network [1]. "
            "The system reached 99% accuracy on ImageNet using transformers [1]."
        )


def test_service_flags_the_made_up_sentence(service, pdf_bytes):
    service.ingest_pdf("a.pdf", pdf_bytes)
    service._answerer = Hallucinating()
    result = service.ask("Docker Compose runs the frontend and backend on an internal network")
    assert [c.supported for c in result.verification.claims] == [True, False]


def test_failed_trust_check_does_not_break_the_answer(service, pdf_bytes):
    class Broken:
        def judge(self, claims, hits):
            raise VerifierError("down")

    service.ingest_pdf("a.pdf", pdf_bytes)
    service._verifier = Broken()
    result = service.ask("Which network does Docker Compose use?")
    assert result.grounded is True and result.verification is None


def test_api_returns_the_trust_report(client, pdf_bytes):
    client.post("/api/v1/documents", files={"file": ("p.pdf", pdf_bytes, "application/pdf")})
    body = client.post("/api/v1/ask", json={"question": "Which network does Docker Compose use?"}).json()
    report = body["verification"]
    assert 0 <= report["trust_score"] <= 1
    assert report["claims"] and {"text", "cites", "supported"} <= set(report["claims"][0])


def test_refused_answer_has_no_trust_report(client, pdf_bytes):
    client.post("/api/v1/documents", files={"file": ("p.pdf", pdf_bytes, "application/pdf")})
    body = client.post("/api/v1/ask", json={"question": "quantum chromodynamics neutron stars"}).json()
    assert body["grounded"] is False and body["verification"] is None


def test_trust_check_can_be_turned_off(tmp_path):
    off = Settings(provider="fake", data_dir=str(tmp_path), verify_answers=False, _env_file=None)
    assert isinstance(build_service(off), RagService)
    assert build_service(off)._verifier is None
