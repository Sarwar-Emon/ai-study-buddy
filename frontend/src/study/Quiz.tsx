import { useState } from "react";
import { makeQuiz, recordAttempts, type DocumentInfo, type QuizQuestion } from "../api";
import Setup, { toRange, type SetupValues } from "./Setup";
import { useElapsed } from "./useElapsed";

interface Props {
  doc: DocumentInfo | undefined;
  setup: SetupValues;
  onSetup: (next: SetupValues) => void;
  onError: (text: string) => void;
  onRecorded: () => void;
}

type Mode = "learn" | "exam";
const LETTERS = ["A", "B", "C", "D"];

function message(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong.";
}

export default function Quiz({ doc, setup, onSetup, onError, onRecorded }: Props) {
  const [mode, setMode] = useState<Mode>("learn");
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [finished, setFinished] = useState(false);
  const [loading, setLoading] = useState(false);
  const [dropped, setDropped] = useState(0);
  const [requested, setRequested] = useState(0);
  const [recorded, setRecorded] = useState(true);
  const elapsed = useElapsed(loading);

  async function generate() {
    setLoading(true);
    try {
      const quiz = await makeQuiz(toRange(setup));
      setQuestions(quiz.questions);
      setDropped(quiz.dropped);
      setRequested(quiz.requested);
      setAnswers(quiz.questions.map(() => null));
      setFinished(false);
      setRecorded(true);
    } catch (err) {
      onError(message(err));
    } finally {
      setLoading(false);
    }
  }

  async function finish(finalAnswers: (number | null)[]) {
    setFinished(true);
    const results = questions.map((q, i) => ({ page: q.page, correct: finalAnswers[i] === q.answer_index }));
    try {
      await recordAttempts(setup.docId, results);
      onRecorded();
    } catch (err) {
      setRecorded(false);
      onError(`Your score was not saved. ${message(err)}`);
    }
  }

  function choose(index: number, option: number) {
    if (finished || (mode === "learn" && answers[index] !== null)) return;
    const next = answers.map((a, i) => (i === index ? option : a));
    setAnswers(next);
    if (mode === "learn" && next.every((a) => a !== null)) void finish(next);
  }

  const score = questions.filter((q, i) => answers[i] === q.answer_index).length;
  const answeredCount = answers.filter((a) => a !== null).length;
  const answeredAll = answers.length > 0 && answeredCount === answers.length;
  // Learn mode shows the answer as soon as you pick. Exam mode waits until you submit.
  const reveal = (i: number) => (mode === "learn" ? answers[i] !== null : finished);

  return (
    <div className="study">
      <Setup doc={doc} values={setup} onChange={onSetup} countLabel="Questions" maxCount={10} />
      <div className="toolbar">
        <div className="seg" role="group" aria-label="Quiz mode">
          <button type="button" className={mode === "learn" ? "on" : ""} aria-pressed={mode === "learn"} onClick={() => setMode("learn")} disabled={loading}>
            Learn
          </button>
          <button type="button" className={mode === "exam" ? "on" : ""} aria-pressed={mode === "exam"} onClick={() => setMode("exam")} disabled={loading}>
            Exam
          </button>
        </div>
        <button type="button" className="primary" onClick={generate} disabled={loading || !setup.docId}>
          {loading ? `Writing and checking questions… ${elapsed}s` : questions.length ? "New quiz" : "Make a quiz"}
        </button>
      </div>
      <p className="hint">
        {mode === "learn"
          ? "Learn mode shows the right answer after each question."
          : "Exam mode keeps the answers hidden until you submit."}
      </p>

      {questions.length > 0 && questions.length < requested && (
        <p className="hint" role="status">
          Made {questions.length} of {requested} questions.
          {dropped > 0 && ` ${dropped} ${dropped > 1 ? "were" : "was"} left out because the page did not back ${dropped > 1 ? "them" : "it"} up.`}{" "}
          Try again, or choose more pages.
        </p>
      )}

      {questions.length > 0 && (
        <p className="progress" aria-live="polite">
          {finished ? "Finished" : `${answeredCount} of ${questions.length} answered`}
        </p>
      )}

      {questions.map((q, i) => (
        <article key={`${q.question}-${i}`} className="question">
          <h3>{q.question}</h3>
          <ul className="options">
            {q.options.map((option, o) => {
              const chosen = answers[i] === o;
              const shown = reveal(i);
              const state = shown ? (o === q.answer_index ? "right" : chosen ? "wrong" : "") : chosen ? "picked" : "";
              return (
                <li key={o}>
                  <button type="button" className={`option ${state}`} onClick={() => choose(i, o)} disabled={finished}>
                    <span className="letter" aria-hidden="true">
                      {LETTERS[o]}
                    </span>
                    <span>{option}</span>
                  </button>
                </li>
              );
            })}
          </ul>
          {reveal(i) && (
            <div className="feedback">
              <p>
                <strong>{answers[i] === q.answer_index ? "Correct." : "Not quite."}</strong> {q.explanation}
              </p>
              <p className="sources">
                <span className="pagechip">page {q.page}</span>
                {q.verified === true && <span className="note good">Checked against the page</span>}
                {q.verified === null && <span className="note">Not checked</span>}
              </p>
            </div>
          )}
        </article>
      ))}

      {mode === "exam" && questions.length > 0 && !finished && (
        <button type="button" className="primary" onClick={() => void finish(answers)} disabled={!answeredAll}>
          Submit exam
        </button>
      )}

      {finished && (
        <div className="result" role="status">
          <p className="score">
            You got {score} of {questions.length} right.
          </p>
          <p>{recorded ? "Your results are saved. Open Weak spots to see which pages to review." : "Your results were not saved."}</p>
        </div>
      )}
    </div>
  );
}
