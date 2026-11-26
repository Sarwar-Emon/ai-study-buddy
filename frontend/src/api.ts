export interface DocumentInfo {
  doc_id: string;
  filename: string;
  pages: number;
  chunks: number;
}

export interface Citation {
  number: number;
  filename: string;
  page: number;
  score: number;
  snippet: string;
}

export interface Claim {
  text: string;
  cites: number[];
  supported: boolean;
}

export interface Verification {
  trust_score: number;
  claims: Claim[];
}

export interface AskResult {
  answer: string;
  grounded: boolean;
  citations: Citation[];
  verification: Verification | null;
}

/** Turn a failed response into a readable message. */
async function failure(res: Response): Promise<Error> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return new Error(body.detail);
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return new Error(body.detail[0].msg);
  } catch {
    /* response was not JSON */
  }
  return new Error(`Request failed (${res.status}).`);
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch("/api/v1/documents");
  if (!res.ok) throw await failure(res);
  return res.json();
}

export async function uploadDocument(file: File): Promise<{ doc: DocumentInfo; created: boolean }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/v1/documents", { method: "POST", body: form });
  if (!res.ok) throw await failure(res);
  return { doc: await res.json(), created: res.status === 201 };
}

export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetch(`/api/v1/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
  if (!res.ok) throw await failure(res);
}

export async function ask(question: string): Promise<AskResult> {
  const res = await fetch("/api/v1/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw await failure(res);
  return res.json();
}

// ----- study tools -----
export interface QuizQuestion {
  question: string;
  options: string[];
  answer_index: number;
  explanation: string;
  page: number;
  filename: string;
  verified: boolean | null;
}

export interface Flashcard {
  front: string;
  back: string;
  page: number;
  filename: string;
  verified: boolean | null;
}

export interface Gap {
  doc_id: string;
  filename: string;
  page: number;
  attempts: number;
  correct: number;
  accuracy: number;
  weak: boolean;
}

export interface StudyRange {
  docId: string;
  count: number;
  pageStart?: number;
  pageEnd?: number;
}

export interface AttemptResult {
  page: number;
  correct: boolean;
}

async function postJson(path: string, body: unknown): Promise<Response> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await failure(res);
  return res;
}

function rangeBody(r: StudyRange) {
  return { doc_id: r.docId, count: r.count, page_start: r.pageStart, page_end: r.pageEnd };
}

export async function makeQuiz(r: StudyRange): Promise<{ questions: QuizQuestion[]; requested: number; dropped: number }> {
  return (await postJson("/api/v1/quiz", rangeBody(r))).json();
}

export async function makeFlashcards(r: StudyRange): Promise<{ cards: Flashcard[]; requested: number; dropped: number }> {
  return (await postJson("/api/v1/flashcards", rangeBody(r))).json();
}

export async function recordAttempts(docId: string, results: AttemptResult[]): Promise<void> {
  await postJson("/api/v1/attempts", { doc_id: docId, results });
}

export async function listGaps(): Promise<Gap[]> {
  const res = await fetch("/api/v1/gaps");
  if (!res.ok) throw await failure(res);
  return res.json();
}

export async function reviewPage(docId: string, page: number): Promise<string> {
  const res = await postJson("/api/v1/review", { doc_id: docId, page });
  return (await res.json()).summary;
}
