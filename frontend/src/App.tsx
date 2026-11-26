import { DragEvent, FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  ask,
  deleteDocument,
  getConfig,
  listDocuments,
  listGaps,
  uploadDocument,
  type Citation,
  type DocumentInfo,
  type Gap,
  type PublicConfig,
  type Verification,
} from "./api";
import Flashcards from "./study/Flashcards";
import Gaps from "./study/Gaps";
import Logo from "./study/Logo";
import PageStrip from "./study/PageStrip";
import Quiz from "./study/Quiz";
import type { SetupValues } from "./study/Setup";

interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  grounded?: boolean;
  citations?: Citation[];
  verification?: Verification | null;
}

const MAX_QUESTION = 1000;

type Tab = "ask" | "quiz" | "cards" | "gaps";
const TABS: { id: Tab; label: string }[] = [
  { id: "ask", label: "Ask" },
  { id: "quiz", label: "Quiz" },
  { id: "cards", label: "Flashcards" },
  { id: "gaps", label: "Weak spots" },
];

function TrustReport({ report }: { report: Verification }) {
  const total = report.claims.length;
  const good = report.claims.filter((c) => c.supported).length;
  const level = good === total ? "high" : good / total >= 0.5 ? "mid" : "low";
  const unsupported = report.claims.filter((c) => !c.supported);
  return (
    <div className={`trust ${level}`}>
      <p>
        <strong>
          {good} of {total} {total > 1 ? "claims are" : "claim is"} backed by the cited pages.
        </strong>
      </p>
      {unsupported.length > 0 && (
        <>
          <p>Could not verify these. Check them in the source:</p>
          <ul>
            {unsupported.map((c, i) => (
              <li key={i}>{c.text}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong.";
}

export default function App() {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [gaps, setGaps] = useState<Gap[]>([]);
  const [config, setConfig] = useState<PublicConfig | null>(null);

  useEffect(() => {
    void getConfig().then(setConfig);
  }, []);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("ask");
  const [setup, setSetup] = useState<SetupValues>({ docId: "", count: 5, pageStart: "", pageEnd: "" });
  const [notice, setNotice] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const nextId = useRef(1);
  const endOfChat = useRef<HTMLDivElement>(null);

  const showError = useCallback((text: string) => setNotice({ kind: "error", text }), []);
  const refreshGaps = useCallback(() => {
    listGaps()
      .then(setGaps)
      .catch(() => setGaps([]));
  }, []);

  useEffect(() => {
    listDocuments()
      .then(setDocs)
      .catch((err) => showError(errorText(err)));
    refreshGaps();
  }, [showError, refreshGaps]);

  // keep the study document valid when documents are added or deleted
  useEffect(() => {
    if (docs.length > 0 && !docs.some((d) => d.doc_id === setup.docId)) {
      setSetup((s) => ({ ...s, docId: docs[0].doc_id, pageStart: "", pageEnd: "" }));
    }
  }, [docs, setup.docId]);

  useEffect(() => {
    endOfChat.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  const activeDoc = docs.find((d) => d.doc_id === setup.docId);

  function pickDoc(docId: string) {
    if (docId !== setup.docId) setSetup({ ...setup, docId, pageStart: "", pageEnd: "" });
  }

  async function onUpload(files: FileList | null) {
    if (!files || files.length === 0) return;
    setUploading(true);
    setNotice(null);
    try {
      for (const file of Array.from(files)) {
        const { doc, created } = await uploadDocument(file);
        setNotice({
          kind: "ok",
          text: created ? `Added ${doc.filename} (${doc.pages} pages).` : `${doc.filename} is already in your library.`,
        });
      }
      setDocs(await listDocuments());
    } catch (err) {
      showError(errorText(err));
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    void onUpload(event.dataTransfer.files);
  }

  async function onDelete(doc: DocumentInfo) {
    setNotice(null);
    setConfirming(null);
    try {
      await deleteDocument(doc.doc_id);
      setDocs((current) => current.filter((d) => d.doc_id !== doc.doc_id));
      refreshGaps();
    } catch (err) {
      showError(errorText(err));
    }
  }

  async function onAsk(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || asking) return;

    setMessages((m) => [...m, { id: nextId.current++, role: "user", text }]);
    setQuestion("");
    setAsking(true);
    setNotice(null);
    try {
      const result = await ask(text);
      setMessages((m) => [
        ...m,
        {
          id: nextId.current++,
          role: "assistant",
          text: result.answer,
          grounded: result.grounded,
          citations: result.citations,
          verification: result.verification,
        },
      ]);
    } catch (err) {
      showError(errorText(err));
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="shell">
      {config?.demo && (
        <p className="demo-note" role="note">
          Public demo. Please do not upload private files. It holds up to {config.max_documents} PDFs
          of {config.max_upload_mb} MB each, and uses a shared daily limit, so it can stop for the
          day. Idle copies may take a minute to wake up.
        </p>
      )}
      <header className="topbar">
        <div className="brand">
          <Logo />
          <span>Study Buddy</span>
        </div>
        <div className="tabs" role="tablist" aria-label="Study tools">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={tab === t.id}
              className={tab === t.id ? "on" : ""}
              onClick={() => setTab(t.id)}
            >
              <span>{t.label}</span>
            </button>
          ))}
        </div>
      </header>

      <div className="layout">
        <aside className="library" aria-labelledby="library-title">
          <h2 id="library-title">Library</h2>
          <label
            className={`drop ${uploading ? "busy" : ""} ${dragging ? "over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input
              ref={fileInput}
              type="file"
              accept="application/pdf,.pdf"
              multiple
              disabled={uploading}
              onChange={(e) => onUpload(e.target.files)}
            />
            <strong>{uploading ? "Reading your PDF…" : "Add PDFs"}</strong>
            <span>{uploading ? "This can take a minute for long files." : "Click here, or drop files on this box."}</span>
          </label>

          {docs.length === 0 ? (
            <p className="empty">Notes, slides exported as PDF, or papers. Everything you ask and study comes from these files.</p>
          ) : (
            <ul className="docs">
              {docs.map((doc) => (
                <li key={doc.doc_id} className={doc.doc_id === setup.docId ? "active" : ""}>
                  <button type="button" className="docpick" onClick={() => pickDoc(doc.doc_id)} aria-pressed={doc.doc_id === setup.docId}>
                    <strong title={doc.filename}>{doc.filename}</strong>
                    <span className="muted">{doc.pages} {doc.pages === 1 ? "page" : "pages"}</span>
                    <PageStrip pages={doc.pages} gaps={gaps.filter((g) => g.doc_id === doc.doc_id)} size="sm" />
                  </button>
                  {confirming === doc.doc_id ? (
                    <div className="confirm">
                      <span>Delete it and its results?</span>
                      <button type="button" className="danger" onClick={() => onDelete(doc)}>
                        Delete
                      </button>
                      <button type="button" onClick={() => setConfirming(null)}>
                        Keep
                      </button>
                    </div>
                  ) : (
                    <button type="button" className="quiet" onClick={() => setConfirming(doc.doc_id)} aria-label={`Delete ${doc.filename}`}>
                      Delete
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </aside>

        <main className="desk">
          {notice && (
            <div className={`notice ${notice.kind}`} role={notice.kind === "error" ? "alert" : "status"}>
              {notice.text}
            </div>
          )}

          {docs.length === 0 && (tab === "quiz" || tab === "cards") && <p className="empty">Add a PDF to the library first.</p>}

          <div hidden={tab !== "quiz" || docs.length === 0}>
            <Quiz doc={activeDoc} setup={setup} onSetup={setSetup} onError={showError} onRecorded={refreshGaps} />
          </div>
          <div hidden={tab !== "cards" || docs.length === 0}>
            <Flashcards doc={activeDoc} setup={setup} onSetup={setSetup} onError={showError} onRecorded={refreshGaps} />
          </div>
          <div hidden={tab !== "gaps"}>
            <Gaps docs={docs} gaps={gaps} onError={showError} />
          </div>

          <section className="ask" hidden={tab !== "ask"} aria-label="Ask a question">
            <div className="messages" aria-live="polite">
              {messages.length === 0 && (
                <p className="empty">
                  {docs.length === 0
                    ? "Add a PDF to the library, then ask it a question."
                    : "Ask anything about your documents. Each answer shows the pages it came from."}
                </p>
              )}
              {messages.map((m) => (
                <article key={m.id} className={`msg ${m.role} ${m.grounded === false ? "ungrounded" : ""}`}>
                  <p>{m.text}</p>
                  {m.verification && <TrustReport report={m.verification} />}
                  {m.citations && m.citations.length > 0 && (
                    <ol className="citations">
                      {m.citations.map((c) => (
                        <li key={c.number}>
                          <details>
                            <summary>
                              <span className="num">{c.number}</span>
                              <span className="pagechip">page {c.page}</span>
                              <span className="file">{c.filename}</span>
                            </summary>
                            <blockquote>{c.snippet}</blockquote>
                          </details>
                        </li>
                      ))}
                    </ol>
                  )}
                </article>
              ))}
              {asking && <p className="hint">Looking through your pages…</p>}
              <div ref={endOfChat} />
            </div>

            <form className="askbar" onSubmit={onAsk}>
              <input
                type="text"
                value={question}
                maxLength={MAX_QUESTION}
                placeholder="Ask a question about your PDFs"
                aria-label="Your question"
                onChange={(e) => setQuestion(e.target.value)}
              />
              <button type="submit" className="primary" disabled={asking || question.trim() === "" || docs.length === 0}>
                Ask
              </button>
            </form>
          </section>
        </main>
      </div>
    </div>
  );
}
