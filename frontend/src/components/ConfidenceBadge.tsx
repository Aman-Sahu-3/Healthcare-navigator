import { useEffect, useId, useRef, useState } from "react";
import type { Confidence } from "../api/types";

const STYLES = {
  HIGH: {
    badge: "bg-emerald-50 text-emerald-800 ring-emerald-600/25",
    dot: "bg-emerald-500",
    label: "High confidence",
  },
  MEDIUM: {
    badge: "bg-amber-50 text-amber-800 ring-amber-600/30",
    dot: "bg-amber-500",
    label: "Medium confidence",
  },
  LOW: {
    badge: "bg-rose-50 text-rose-800 ring-rose-600/25",
    dot: "bg-rose-500",
    label: "Low confidence",
  },
} as const;

export function ConfidenceBadge({ confidence }: { confidence: Confidence }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const s = STYLES[confidence.level];

  // Close on Escape or a click outside the badge.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const onClick = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onClick);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={panelId}
        data-level={confidence.level}
        className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-medium ring-1 ring-inset ${s.badge} hover:brightness-95`}
      >
        <span className={`h-2 w-2 rounded-full ${s.dot}`} aria-hidden="true" />
        {s.label}
        <span className="font-normal opacity-70">{Math.round(confidence.score * 100)}%</span>
      </button>
      {open && (
        <div
          id={panelId}
          role="region"
          aria-label="Why this confidence level"
          className="absolute right-0 z-10 mt-2 w-72 max-w-[calc(100vw-2rem)] rounded-lg border border-slate-200 bg-white p-3 text-sm shadow-lg"
        >
          <p className="mb-1 font-medium text-slate-700">Why this level</p>
          <ul className="list-disc space-y-0.5 pl-4 text-slate-600">
            {confidence.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-slate-500">
            Based on how closely the cited passages match your question and how many
            independent passages support the answer.
          </p>
        </div>
      )}
    </div>
  );
}
