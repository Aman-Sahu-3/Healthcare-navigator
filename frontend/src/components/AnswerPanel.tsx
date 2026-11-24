import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import type { QueryResponse } from "../api/types";
import { answerAsText, type Inline, parseAnswer } from "../lib/answer";
import { CitationCard } from "./CitationCard";
import { ConfidenceBadge } from "./ConfidenceBadge";

export function AnswerPanel({ result }: { result: QueryResponse }) {
  const [activeId, setActiveId] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const cardRefs = useRef(new Map<number, HTMLLIElement>());
  const blocks = useMemo(() => parseAnswer(result.answer), [result.answer]);
  const byId = useMemo(
    () => new Map(result.citations.map((c) => [c.id, c])),
    [result.citations],
  );

  useEffect(() => setActiveId(null), [result]);

  const jumpTo = (id: number) => {
    setActiveId(null);
    requestAnimationFrame(() => {
      setActiveId(id);
      const el = cardRefs.current.get(id);
      el?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
      el?.focus({ preventScroll: true });
    });
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(answerAsText(result.answer, result.citations));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked — ignore */
    }
  };

  const renderInline = (parts: Inline[]): ReactNode[] =>
    parts.map((p, i) => {
      if (p.kind === "bold") return <strong key={i}>{p.text}</strong>;
      if (p.kind === "text") return <span key={i}>{p.text}</span>;
      const c = byId.get(p.id);
      if (!c) return <span key={i}>[{p.id}]</span>;
      return (
        <button
          key={i}
          type="button"
          onClick={() => jumpTo(p.id)}
          title={`${c.title}, page ${c.page}`}
          aria-label={`Citation ${p.id}: ${c.title}, page ${c.page}`}
          className="mx-0.5 inline-flex h-5 min-w-5 -translate-y-1 items-center justify-center rounded bg-brand-50 px-1 align-baseline text-[11px] font-semibold text-brand-700 ring-1 ring-brand-100 hover:bg-brand-100"
        >
          {p.id}
        </button>
      );
    });

  if (result.insufficient_evidence) {
    return (
      <section
        aria-label="Answer"
        className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
            No supported answer
          </h2>
          <ConfidenceBadge confidence={result.confidence} />
        </div>
        <p className="mt-3 text-slate-700">{result.answer}</p>
        <p className="mt-3 text-sm text-slate-500">
          The indexed guidelines don't contain enough evidence to answer this. Try rephrasing,
          or check whether a relevant guideline has been added.
        </p>
        <Meta result={result} />
      </section>
    );
  }

  return (
    <section aria-label="Answer" className="space-y-5">
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Answer</h2>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={copy}
              className="rounded-md px-2 py-1 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-700"
            >
              {copied ? "Copied" : "Copy"}
            </button>
            <ConfidenceBadge confidence={result.confidence} />
          </div>
        </div>

        <div className="mt-4 space-y-3 leading-relaxed text-slate-800">
          {blocks.map((b, i) =>
            b.kind === "paragraph" ? (
              <p key={i}>{renderInline(b.content)}</p>
            ) : b.ordered ? (
              <ol key={i} className="list-decimal space-y-1 pl-5">
                {b.items.map((item, j) => (
                  <li key={j}>{renderInline(item)}</li>
                ))}
              </ol>
            ) : (
              <ul key={i} className="list-disc space-y-1 pl-5">
                {b.items.map((item, j) => (
                  <li key={j}>{renderInline(item)}</li>
                ))}
              </ul>
            ),
          )}
        </div>
        <Meta result={result} />
      </div>

      {result.citations.length > 0 && (
        <div>
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
            Sources ({result.citations.length})
          </h3>
          <ol className="grid gap-3 md:grid-cols-2">
            {result.citations.map((c) => (
              <CitationCard
                key={c.id}
                citation={c}
                active={activeId === c.id}
                ref={(el) => {
                  if (el) cardRefs.current.set(c.id, el);
                  else cardRefs.current.delete(c.id);
                }}
              />
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

function Meta({ result }: { result: QueryResponse }) {
  return (
    <p className="mt-5 border-t border-slate-100 pt-3 text-xs text-slate-400">
      {result.retrieved_chunks} passages searched · {result.model} ·{" "}
      {(result.latency_ms / 1000).toFixed(1)} s
    </p>
  );
}
