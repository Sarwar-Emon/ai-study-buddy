# AI Study Buddy

An AI study assistant for your own course material. Upload lecture notes, slides or papers (as PDF), then:

- **Ask questions** and get answers that point to the exact pages they came from.
- **Take quizzes and exams** and **practice with flashcards**, all made from your documents.
- **See your weak spots**: the pages you keep getting wrong, with a plain-language recap of each one.

It is a full-stack app: a React + TypeScript interface, a FastAPI backend, a Chroma vector database (retrieval-augmented generation, RAG), and an LLM provider of your choice (Google Gemini's free tier, or OpenAI). It answers only from your documents. When they do not contain the answer, it says so instead of guessing. Every answer, quiz question and flashcard is **checked against the page it came from**, and anything the page does not back up is flagged or dropped.

> **Status:** working prototype for learning and portfolio use. It is not hardened for public internet deployment (see [Limitations](#limitations)).

## How it works

```
Browser (React + TypeScript)
   |
   v
Nginx  ----- adds the API key, forwards /api/ ----->  FastAPI backend
                                                        |
        Upload a PDF                                    |  Ask a question
        1. extract text page by page                    |  1. embed the question
        2. split into overlapping chunks (keeps page)   |  2. find the closest chunks (Chroma)
        3. embed each chunk                             |  3. if nothing is close enough: refuse
        4. store vectors + page numbers (Chroma)        |  4. otherwise send the chunks to the LLM
                                                        |  5. trust check: verify each claim vs its citations
                                                        |  6. return answer + citations + trust report
```

## Features

- **Quiz and exam mode.** Multiple-choice questions from a document or a page range. *Learn* mode shows the answer after each question; *Exam* mode hides everything until you submit.
- **Flashcards.** Flip a card, then mark "I knew it" or "Not yet".
- **Page strip.** Every document shows one small tile per page, colored by how well you know it (not practiced, needs another look, doing fine). In Weak spots the strip is large, and clicking any page gives a plain-language recap of it.
- **Weak spots.** Quiz and flashcard results are saved per page. The page list is sorted weakest first, and each page has an "Explain this page" button that gives a short, simple recap written from that page only.
- **Checked study material.** Each question and card is verified against its source page. Items the page does not support are removed, and the screen says how many were left out.
- **Page-level citations.** Every answer lists the files and pages it used, with the matching text.
- **Refuses when unsure.** If no passage is similar enough to the question (`MIN_SCORE`), the app says it could not find the answer and does not call the LLM.
- **Claim-level trust check.** After each answer, every sentence is checked against the pages it cites. Claims the sources do not back up are listed as "could not verify", and the answer gets a trust score (supported claims / all claims). A citation to a passage number that does not exist is caught without any model call.
- **Grounded prompt.** The model is told to use only the supplied passages, and to ignore instructions that appear inside documents (basic prompt-injection defense).
- **Duplicate-safe uploads.** Files are identified by a hash of their content, so uploading the same PDF twice does not create duplicates.
- **Safe API.** File type and size checks, a real-PDF check, API-key authentication, and a rate limit on questions.
- **Runs offline for testing.** A built-in fake embedder and answerer let the whole test suite run with no network and no API key.

## Tech stack

| Layer | Technology |
| --- | --- |
| Frontend | React, TypeScript, Vite |
| Web server / proxy | Nginx |
| Backend | Python, FastAPI, Uvicorn |
| Vector database | Chroma (stored on disk) |
| Progress tracking | SQLite (stored on disk, per page) |
| Models | Any OpenAI-compatible API: Google Gemini (free tier) or OpenAI. Set by `.env` |
| PDF reading | pypdf |
| Packaging | Docker, Docker Compose |
| Tests | pytest |

## Quick start (Docker)

You need Docker and an API key. The default setup uses Google Gemini's **free** tier, which needs no credit card: get a key at <https://aistudio.google.com/apikey>.

```bash
cp .env.example .env
# edit .env: paste your key after OPENAI_API_KEY= and set API_KEY to any long random text

docker compose up -d --build
```

Open <http://127.0.0.1:8080>, upload a PDF, and ask a question. Stop with `docker compose down`.

Only the frontend is published, and only on `127.0.0.1`. The backend is reachable just from inside the Compose network. Indexed documents live in a Docker volume, so they survive restarts.

## Local development

```bash
# backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # set OPENAI_API_KEY (+ base URL for Gemini)
uvicorn app.main:app --reload   # http://127.0.0.1:8000

# frontend (second terminal)
cd frontend
npm install
npm run dev                     # http://127.0.0.1:5173, proxies /api to the backend
```

To try the app with no API key at all, start the backend with `PROVIDER=fake`. Answers are then a simple word-matching demo, not real AI.

## API

All `/api/v1` routes need the header `X-API-Key` when `API_KEY` is set. Interactive docs are at `/docs`.

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/health` | Health check (no key needed) |
| `POST` | `/api/v1/documents` | Upload a PDF (`multipart/form-data`, field `file`). `201` if new, `200` if already indexed |
| `GET` | `/api/v1/documents` | List indexed documents |
| `DELETE` | `/api/v1/documents/{doc_id}` | Remove a document and its vectors |
| `POST` | `/api/v1/ask` | Ask a question: `{"question": "...", "top_k": 4}` |
| `POST` | `/api/v1/quiz` | Make a quiz: `{"doc_id": "...", "count": 5, "page_start": 1, "page_end": 9}` (pages optional, `count` 1 to 10) |
| `POST` | `/api/v1/flashcards` | Make flashcards, same body as `/quiz` |
| `POST` | `/api/v1/attempts` | Save results: `{"doc_id": "...", "results": [{"page": 3, "correct": false}]}` |
| `GET` | `/api/v1/gaps` | Pages you have practiced, weakest first (`weak` is true under 70% right) |
| `POST` | `/api/v1/review` | Plain-language recap of one page: `{"doc_id": "...", "page": 3}` |

Example answer:

```json
{
  "answer": "The frontend and backend run on an internal Docker network [1].",
  "grounded": true,
  "citations": [
    { "number": 1, "filename": "notes.pdf", "page": 2, "score": 0.61, "snippet": "Docker Compose runs ..." }
  ]
}
```

Errors use normal HTTP codes: `401` bad key, `413` file too large, `415` not a PDF, `422` bad input or a PDF with no text, `429` too many questions, `502` the AI provider failed, `503` the API key is missing.

## Configuration

Set these as environment variables or in `.env`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | (none) | Your provider key (Gemini or OpenAI) |
| `OPENAI_BASE_URL` | (empty = OpenAI) | Set to `https://generativelanguage.googleapis.com/v1beta/openai/` for Gemini |
| `API_KEY` | (none) | If set, clients must send it as `X-API-Key` |
| `CHAT_MODEL` | `gpt-4o-mini` | Model that writes answers (`.env.example` uses `gemini-3.8-flash`) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model that turns text into vectors (`.env.example` uses `gemini-embedding-2`; if that name is rejected, try `gemini-embedding-001`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Characters per chunk, and how much neighbours share |
| `TOP_K` | `4` | Passages sent to the model |
| `MIN_SCORE` | `0.2` | Minimum cosine similarity; below this the app refuses to answer |
| `MAX_UPLOAD_MB` | `20` | Largest accepted PDF |
| `REASONING_EFFORT` | (empty) | Speed knob for models that think before answering (for example Gemini 3: try `low` or `minimal`). Empty means it is not sent. Not every model accepts it |
| `VERIFY_ANSWERS` | `true` | Turn the trust check on or off. It costs one extra model call per answer (set `false` if you hit free-tier limits) |
| `PROVIDER` | `openai` | `fake` runs fully offline (tests and demos only) |

## Free tier notes and troubleshooting

- **Privacy:** on Google's free tier, the text you send may be used to improve Google products. Do not upload private or sensitive documents. Paid tiers (and OpenAI's API) have different terms.
- **Quizzes feel slow?** Making a quiz takes two model calls in a row (write, then check), so it takes longer than a chat answer. `docker compose logs backend` prints a line like `quiz: 5 passages, writing 6.2s, checking 3.1s` so you can see which step is slow. Fewer questions helps a little. If the writing step is slow, try `REASONING_EFFORT=low` in `.env`. If the checking step is slow and you can live without it, set `VERIFY_ANSWERS=false`.
- **Rate limits:** the free tier allows only a few requests per minute. The app retries automatically; a large PDF may take a while to index. If you still see `502`, wait a minute and retry.
- **Changed models or provider?** Vectors from different embedding models cannot be mixed. The app then returns a `503` explaining this. Reset with `docker compose down -v` (this deletes indexed documents) and upload again.
- **Something fails?** Run `docker compose logs backend` to see the real error (wrong key, wrong model name, quota).
- Free-tier model names change over time. Check Google AI Studio for current names if a model is rejected.

## How the study tools work

1. **Pick passages.** The backend picks up to `count` passages from the document (spread over the pages, optionally limited to a page range).
2. **Write questions.** One model call writes one question or card per passage, as JSON. Anything that is not the shape we asked for (wrong number of options, answer out of range, duplicate options, a passage that was not sent) is thrown away. Answers like `"B"` or `"1"` are accepted. If fewer than you asked for come back (the model skipped some, or the page did not back them up), the app asks again with other passages, up to 3 rounds in total. If it still falls short, the screen says "Made 3 of 5".
3. **Check them.** The same claim checker used for answers checks each question and its correct answer against its own passage. Unsupported items are dropped. If the checker is off (`VERIFY_ANSWERS=false`) or fails, items are kept and marked "not checked".
4. **Shuffle.** Models like to put the right answer first, so options are shuffled (the same way every time for the same question).
5. **Remember.** Results are saved per page in SQLite. "Weak spots" ranks pages by accuracy, and "Explain this page" recaps the page from its own text.

Quiz generation and the checker use the same `CHAT_MODEL`. Generation is rate limited to 10 requests a minute.

## About the trust check

LLMs sometimes add a sentence that sounds right but is not in the sources, and a citation next to it makes it look trustworthy. The check goes after exactly that:

1. Split the answer into claims (about one per sentence, at most 12).
2. A second model call judges each claim against only the passages it cites. Numbers, names and results must match.
3. Claims citing a passage that does not exist are marked unsupported with no model call.
4. The API returns `verification`: `trust_score` and the list of claims with `supported` true or false. The UI shows the score and lists the unverified claims.

Limits: the checker is also an LLM, so it can be wrong in both directions. It reads only the retrieved passages, so it cannot tell you the retrieval missed a better passage. If the check itself fails (provider error, bad output), the answer is still returned, with no trust report.

## Design decisions

- **Chunks never cross pages.** Each chunk belongs to one page, so a citation always names a real page.
- **Overlap.** Neighbouring chunks share 200 characters so a sentence on a boundary is not lost.
- **Refusal threshold instead of trusting the LLM.** Checking similarity before calling the model is cheaper and more reliable than hoping the model notices it has no evidence. `MIN_SCORE` should be tuned with the evaluation script below.
- **Temperature 0.** Answers should be repeatable.
- **Swappable parts.** The embedder, answerer, and store are separate classes behind small interfaces, which is what makes the offline tests possible.
- **Service built on first use.** `/health` works even if the API key is missing, and the other routes return a clear `503`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs offline. It covers chunking, retrieval and citations, refusing off-topic questions, upload validation, authentication, rate limiting, provider failures (`502`), the OpenAI wrappers (with the client mocked), the trust check (claim splitting, made-up sentences, missing citations, bad checker output), and the study tools (quiz and flashcard generation, dropping unsupported questions, malformed model output, progress and weak-spot ranking, the API).

## Measuring retrieval quality

Retrieval is the part of RAG that most often goes wrong, so it is measured separately from the LLM.

1. Put a few PDFs in a folder.
2. Write a question set like [`evaluation/questions.example.json`](evaluation/questions.example.json). Each question names the PDF and page that contains the answer.
3. Run:

```bash
python -m evaluation.run_eval --pdf-dir ./my_papers --questions evaluation/questions.json --k 4
```

It prints **Hit@1**, **Hit@k** (is the right page in the top k results?) and **MRR**, and lists the questions that missed. Use those numbers to tune `CHUNK_SIZE`, `TOP_K`, and `MIN_SCORE`.

## Limitations

- **Quiz quality depends on the model.** The page check catches invented facts, but a question can still be too easy, too hard, or oddly worded. Weak spots are tracked per page, not per concept.
- **Slides must be PDFs.** Export PowerPoint or Keynote to PDF first. Slides with mostly images give little to study.
- **Scanned PDFs** (images of text) are rejected; there is no OCR.
- **Tables, equations, and figures** are not understood; only extracted text is indexed.
- **One shared API key.** The Nginx proxy adds the same key for every browser user, so this is a single-user or private-network setup, not per-user login.
- **In-memory rate limiter.** It works for one server process. Several servers would need a shared store such as Redis.
- **Everyone sees every document.** There are no user accounts or per-user collections.
- **Answer quality depends on the LLM.** Citations show where an answer came from, but they do not prove the answer is correct. Check important answers against the source page.
- Public deployment would need real authentication, HTTPS, monitoring, and a security review.

## Live demo on Render (free)

The repo includes `Dockerfile.render` and `render.yaml`, which run the whole app as one container.

1. Push the repo to GitHub.
2. On render.com, choose **New → Blueprint**, pick this repo, and apply.
3. When asked for `OPENAI_API_KEY`, paste a Gemini key. Use a separate key for the demo so you can delete it later.
4. After the build finishes, Render gives you a URL like `https://ai-study-buddy.onrender.com`.

Demo mode (`DEMO_MODE=true`) protects your free quota:

- All visitors share a daily cap on requests that call the model (`DEMO_DAILY_REQUESTS`, default 150).
- Uploads are limited to `MAX_UPLOAD_MB` (5 on Render) and `DEMO_MAX_DOCS` documents (default 4).
- A sample study guide (`samples/study_sample.pdf`) loads on startup so visitors can try it at once.
- A banner tells visitors not to upload private files.

On the free plan the app sleeps when idle, so the first visit can take about a minute, and the disk is erased on each restart. Both are fine for a demo.
