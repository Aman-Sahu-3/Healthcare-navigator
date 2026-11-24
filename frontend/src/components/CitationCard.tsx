import { forwardRef, useState } from "react";
import type { Citation } from "../api/types";

interface Props {
  citation: Citation;
  active: boolean;
}

export const CitationCard = forwardRef<HTMLLIElement, Props>(function CitationCard(
  { citation, active },
  ref,
) {
  const [expanded, setExpanded] = useState(false);
  const pct = Math.round(Math.max(0, Math.min(1, citation.similarity)) * 100);
  const long = citation.snippet.length > 160;

  return (
    <li
      ref={ref}
      id={`citation-${citation.id}`}
      tabIndex={-1}
      data-active={active || undefined}
      className={`rounded-xl border bg-white p-4 transition-colors focus:outline-none ${
        active ? "cite-flash border-brand-600" : "border-slate-200"
      }`}
    >
      <div className="flex items-start gap-3">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-sm font-semibold text-brand-700 ring-1 ring-brand-100">
          {citation.id}
        </span>
        <div className="min-w-0 flex-1">
          <p className="font-medium leading-snug text-slate-900">{citation.title}</p>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-xs text-slate-500">
            <span className="truncate" title={citation.source}>
              {citation.source}
            </span>
            <span aria-hidden="true">·</span>
            <span className="font-medium text-slate-700">Page {citation.page}</span>
          </p>
        </div>
      </div>

      <blockquote
        className={`mt-3 border-l-2 border-slate-200 pl-3 text-sm leading-relaxed text-slate-600 ${
          expanded ? "" : "line-clamp-3"
        }`}
      >
        {citation.snippet}
      </blockquote>
      {long && (
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="mt-1 text-xs font-medium text-brand-700 hover:underline"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}

      <div className="mt-3 flex items-center gap-2" title="Similarity between your question and this passage">
        <span className="text-xs text-slate-500">Match</span>
        <div
          className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100"
          role="meter"
          aria-label="Passage similarity"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
        >
          <div className="h-full rounded-full bg-brand-600" style={{ width: `${pct}%` }} />
        </div>
        <span className="w-9 text-right text-xs tabular-nums text-slate-600">{pct}%</span>
      </div>
    </li>
  );
});
