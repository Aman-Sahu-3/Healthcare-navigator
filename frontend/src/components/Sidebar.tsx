import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { HistoryEntry } from "../hooks/useHistory";

const LEVEL_DOT = { HIGH: "bg-emerald-500", MEDIUM: "bg-amber-500", LOW: "bg-rose-500" };

interface HistoryProps {
  entries: HistoryEntry[];
  selectedId: string | null;
  onSelect: (entry: HistoryEntry) => void;
  onClear: () => void;
}

export function HistoryList({ entries, selectedId, onSelect, onClear }: HistoryProps) {
  return (
    <section aria-labelledby="history-heading" className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-2 flex items-center justify-between">
        <h2 id="history-heading" className="text-sm font-semibold text-slate-700">
          Recent questions
        </h2>
        {entries.length > 0 && (
          <button
            type="button"
            onClick={onClear}
            className="text-xs text-slate-400 hover:text-slate-600"
          >
            Clear
          </button>
        )}
      </div>
      {entries.length === 0 ? (
        <p className="text-xs text-slate-400">Your questions will appear here on this device.</p>
      ) : (
        <ul className="-mx-2 max-h-80 space-y-0.5 overflow-y-auto">
          {entries.map((e) => (
            <li key={e.id}>
              <button
                type="button"
                onClick={() => onSelect(e)}
                aria-current={selectedId === e.id || undefined}
                className={`flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left text-sm hover:bg-slate-50 ${
                  selectedId === e.id ? "bg-brand-50 text-brand-800" : "text-slate-600"
                }`}
              >
                <span
                  className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${LEVEL_DOT[e.result.confidence.level]}`}
                  aria-hidden="true"
                />
                <span className="line-clamp-2">{e.result.question}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function SourcesList() {
  const sources = useQuery({ queryKey: ["sources"], queryFn: api.sources, staleTime: 60_000 });

  return (
    <section aria-labelledby="sources-heading" className="rounded-2xl border border-slate-200 bg-white p-4">
      <details>
        <summary className="cursor-pointer list-none">
          <span id="sources-heading" className="text-sm font-semibold text-slate-700">
            Indexed documents
          </span>
          <span className="ml-2 text-xs text-slate-400">
            {sources.data ? sources.data.count : sources.isError ? "unavailable" : "…"}
          </span>
        </summary>
        {sources.data && (
          <ul className="mt-3 max-h-72 space-y-2 overflow-y-auto pr-1">
            {sources.data.documents.map((d) => (
              <li key={d.source} className="text-xs">
                <p className="line-clamp-2 font-medium text-slate-700">{d.title}</p>
                <p className="text-slate-400">
                  {d.total_pages ?? "?"} pages · {d.chunks} passages
                </p>
              </li>
            ))}
          </ul>
        )}
      </details>
    </section>
  );
}
