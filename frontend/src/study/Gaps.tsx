import { useState } from "react";
import { reviewPage, type DocumentInfo, type Gap } from "../api";
import PageStrip from "./PageStrip";

interface Props {
  docs: DocumentInfo[];
  gaps: Gap[];
  onError: (text: string) => void;
}

interface Recap {
  docId: string;
  page: number;
  text: string | null; // null while loading
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong.";
}

export default function Gaps({ docs, gaps, onError }: Props) {
  const [recap, setRecap] = useState<Recap | null>(null);

  async function pick(docId: string, page: number) {
    setRecap({ docId, page, text: null });
    try {
      const text = await reviewPage(docId, page);
      setRecap((current) => (current && current.docId === docId && current.page === page ? { docId, page, text } : current));
    } catch (err) {
      setRecap(null);
      onError(message(err));
    }
  }

  if (docs.length === 0) return <p className="empty">Add a PDF to the library, then take a quiz. Your weak pages will show up here.</p>;

  const weak = gaps.filter((g) => g.weak);
  const name = (docId: string) => docs.find((d) => d.doc_id === docId)?.filename ?? "";

  return (
    <div className="gaps">
      <p className="lead">
        {gaps.length === 0
          ? "Take a quiz or go through some flashcards and your results will color the pages below."
          : weak.length > 0
            ? `${weak.length} ${weak.length > 1 ? "pages need" : "page needs"} another look.`
            : "No weak pages right now. Nice work."}
      </p>
      <ul className="legend" aria-label="Color key">
        <li>
          <span className="tile none" /> Not practiced
        </li>
        <li>
          <span className="tile weak" /> Needs another look
        </li>
        <li>
          <span className="tile fine" /> Doing fine
        </li>
      </ul>

      {docs.map((doc) => (
        <section key={doc.doc_id} className="docgaps">
          <h3>{doc.filename}</h3>
          <PageStrip
            pages={doc.pages}
            gaps={gaps.filter((g) => g.doc_id === doc.doc_id)}
            size="lg"
            selected={recap?.docId === doc.doc_id ? recap.page : null}
            onPick={(page) => void pick(doc.doc_id, page)}
          />
          {recap?.docId === doc.doc_id && (
            <div className="recap" aria-live="polite">
              <p className="recap-title">
                Page {recap.page} in plain words
              </p>
              {recap.text === null ? <p className="hint">Reading the page…</p> : <p className="recap-text">{recap.text}</p>}
            </div>
          )}
        </section>
      ))}

      {weak.length > 0 && (
        <section className="worklist">
          <h3>Start here</h3>
          <ul>
            {weak.slice(0, 5).map((g) => (
              <li key={`${g.doc_id}:${g.page}`}>
                <button type="button" onClick={() => void pick(g.doc_id, g.page)}>
                  <span className="pagechip">page {g.page}</span>
                  <span className="grow">{name(g.doc_id)}</span>
                  <span className="muted">
                    {g.correct} of {g.attempts} right
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {weak.length > 5 && <p className="hint">Plus {weak.length - 5} more in red above.</p>}
        </section>
      )}
    </div>
  );
}
