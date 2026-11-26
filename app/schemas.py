"""Request and response shapes for the API."""

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=10)


class CitationOut(BaseModel):
    number: int
    filename: str
    page: int
    score: float
    snippet: str


class ClaimOut(BaseModel):
    text: str
    cites: list[int]
    supported: bool


class VerificationOut(BaseModel):
    trust_score: float  # share of claims backed by their cited passages (0 to 1)
    claims: list[ClaimOut]


class AskOut(BaseModel):
    answer: str
    grounded: bool
    citations: list[CitationOut]
    verification: VerificationOut | None = None


# ----- study tools -----
class GenerateIn(BaseModel):
    doc_id: str = Field(min_length=1)
    count: int = Field(default=5, ge=1, le=10)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)


class QuizQuestionOut(BaseModel):
    question: str
    options: list[str]
    answer_index: int
    explanation: str
    page: int
    filename: str
    verified: bool | None


class QuizOut(BaseModel):
    questions: list[QuizQuestionOut]
    requested: int  # how many were asked for; fewer may come back
    dropped: int  # questions removed because their page did not back them up


class FlashcardOut(BaseModel):
    front: str
    back: str
    page: int
    filename: str
    verified: bool | None


class FlashcardsOut(BaseModel):
    cards: list[FlashcardOut]
    requested: int
    dropped: int


class AttemptIn(BaseModel):
    page: int = Field(ge=1)
    correct: bool


class AttemptsIn(BaseModel):
    doc_id: str = Field(min_length=1)
    results: list[AttemptIn] = Field(min_length=1, max_length=50)


class GapOut(BaseModel):
    doc_id: str
    filename: str
    page: int
    attempts: int
    correct: int
    accuracy: float
    weak: bool


class ReviewIn(BaseModel):
    doc_id: str = Field(min_length=1)
    page: int = Field(ge=1)


class ReviewOut(BaseModel):
    filename: str
    page: int
    summary: str
