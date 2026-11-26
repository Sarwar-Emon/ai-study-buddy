import type { DocumentInfo } from "../api";

export interface SetupValues {
  docId: string;
  count: number;
  pageStart: string;
  pageEnd: string;
}

export function toRange(v: SetupValues) {
  const num = (text: string) => (text.trim() === "" ? undefined : Number(text));
  return { docId: v.docId, count: v.count, pageStart: num(v.pageStart), pageEnd: num(v.pageEnd) };
}

interface Props {
  doc: DocumentInfo | undefined;
  values: SetupValues;
  onChange: (next: SetupValues) => void;
  countLabel: string;
  maxCount: number;
}

/** How many items, and an optional page range. The document is the one picked in the library. */
export default function Setup({ doc, values, onChange, countLabel, maxCount }: Props) {
  const set = (patch: Partial<SetupValues>) => onChange({ ...values, ...patch });
  return (
    <div className="setup">
      <p className="studying">
        From <strong title={doc?.filename}>{doc?.filename ?? "no document"}</strong>
      </p>
      <div className="fields">
        <label>
          {countLabel}
          <input
            type="number"
            min={1}
            max={maxCount}
            value={values.count}
            onChange={(e) => set({ count: Math.min(maxCount, Math.max(1, Number(e.target.value) || 1)) })}
          />
        </label>
        <label>
          First page
          <input
            type="number"
            min={1}
            max={doc?.pages}
            placeholder="1"
            value={values.pageStart}
            onChange={(e) => set({ pageStart: e.target.value })}
          />
        </label>
        <label>
          Last page
          <input
            type="number"
            min={1}
            max={doc?.pages}
            placeholder={doc ? String(doc.pages) : ""}
            value={values.pageEnd}
            onChange={(e) => set({ pageEnd: e.target.value })}
          />
        </label>
      </div>
    </div>
  );
}
