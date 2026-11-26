import { useState } from "react";
import { makeFlashcards, recordAttempts, type DocumentInfo, type Flashcard } from "../api";
import Setup, { toRange, type SetupValues } from "./Setup";
import { useElapsed } from "./useElapsed";

interface Props {
  doc: DocumentInfo | undefined;
  setup: SetupValues;
  onSetup: (next: SetupValues) => void;
  onError: (text: string) => void;
  onRecorded: () => void;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong.";
}

export default function Flashcards({ doc, setup, onSetup, onError, onRecorded }: Props) {
  const [cards, setCards] = useState<Flashcard[]>([]);
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [ratings, setRatings] = useState<boolean[]>([]);
  const [loading, setLoading] = useState(false);
  const [dropped, setDropped] = useState(0);
  const [requested, setRequested] = useState(0);
  const [done, setDone] = useState(false);
  const elapsed = useElapsed(loading);

  async function generate() {
    setLoading(true);
    try {
      const result = await makeFlashcards(toRange(setup));
      setCards(result.cards);
      setDropped(result.dropped);
      setRequested(result.requested);
      setIndex(0);
      setFlipped(false);
      setRatings([]);
      setDone(false);
    } catch (err) {
      onError(message(err));
    } finally {
      setLoading(false);
    }
  }

  async function rate(knewIt: boolean) {
    const next = [...ratings, knewIt];
    setRatings(next);
    setFlipped(false);
    if (index + 1 < cards.length) {
      setIndex(index + 1);
      return;
    }
    setDone(true);
    try {
      await recordAttempts(
        setup.docId,
        cards.map((c, i) => ({ page: c.page, correct: next[i] })),
      );
      onRecorded();
    } catch (err) {
      onError(`Your ratings were not saved. ${message(err)}`);
    }
  }

  const card = cards[index];
  const known = ratings.filter(Boolean).length;

  return (
    <div className="study">
      <Setup doc={doc} values={setup} onChange={onSetup} countLabel="Cards" maxCount={10} />
      <div className="toolbar">
        <button type="button" className="primary" onClick={generate} disabled={loading || !setup.docId}>
          {loading ? `Writing and checking cards… ${elapsed}s` : cards.length ? "New cards" : "Make flashcards"}
        </button>
      </div>
      {cards.length > 0 && cards.length < requested && (
        <p className="hint" role="status">
          Made {cards.length} of {requested} cards.
          {dropped > 0 && ` ${dropped} ${dropped > 1 ? "were" : "was"} left out because the page did not back ${dropped > 1 ? "them" : "it"} up.`}{" "}
          Try again, or choose more pages.
        </p>
      )}

      {card && !done && (
        <div>
          <p className="progress" aria-live="polite">
            Card {index + 1} of {cards.length}
          </p>
          <button type="button" className="flash" onClick={() => setFlipped(!flipped)} aria-label={flipped ? "Show the question" : "Show the answer"}>
            <span key={String(flipped)} className="face">
              <span className="side">{flipped ? "Answer" : "Question"}</span>
              <span className="text">{flipped ? card.back : card.front}</span>
            </span>
          </button>
          <p className="sources center">
            <span className="pagechip">page {card.page}</span>
            {card.verified === true && <span className="note good">Checked against the page</span>}
            {card.verified === null && <span className="note">Not checked</span>}
          </p>
          {flipped ? (
            <div className="toolbar center">
              <button type="button" onClick={() => void rate(false)}>
                Not yet
              </button>
              <button type="button" className="primary" onClick={() => void rate(true)}>
                I knew it
              </button>
            </div>
          ) : (
            <p className="hint center">Click the card to see the answer.</p>
          )}
        </div>
      )}

      {done && (
        <div className="result" role="status">
          <p className="score">
            You knew {known} of {cards.length}.
          </p>
          <p>The ones you missed now show up in Weak spots.</p>
        </div>
      )}
    </div>
  );
}
