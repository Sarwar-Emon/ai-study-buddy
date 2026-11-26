import type { Gap } from "../api";

interface Props {
  pages: number;
  gaps: Gap[]; // results for this document only
  size: "sm" | "lg";
  selected?: number | null;
  onPick?: (page: number) => void;
}

type Level = "none" | "weak" | "fine";

function levelOf(gap: Gap | undefined): Level {
  if (!gap) return "none";
  return gap.weak ? "weak" : "fine";
}

/** One tile per page, coloured by how well you know it. The same strip is used small (library) and large (weak spots). */
export default function PageStrip({ pages, gaps, size, selected, onPick }: Props) {
  const byPage = new Map(gaps.map((g) => [g.page, g]));
  const tiles = Array.from({ length: pages }, (_, i) => i + 1);

  return (
    <div className={`strip ${size}`} role={onPick ? "group" : "img"} aria-label={`Pages of this document, ${pages} in total`}>
      {tiles.map((page) => {
        const gap = byPage.get(page);
        const level = levelOf(gap);
        const label = gap
          ? `Page ${page}, ${Math.round(gap.accuracy * 100)}% right`
          : `Page ${page}, not practiced yet`;
        const cls = `tile ${level} ${selected === page ? "picked" : ""}`;
        return onPick ? (
          <button key={page} type="button" className={cls} title={label} aria-label={label} aria-pressed={selected === page} onClick={() => onPick(page)}>
            {size === "lg" ? page : null}
          </button>
        ) : (
          <span key={page} className={cls} title={label} />
        );
      })}
    </div>
  );
}
