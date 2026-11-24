import { describeError } from "../api/client";

export function ErrorBanner({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const { title, detail } = describeError(error);
  return (
    <div role="alert" className="rounded-2xl border border-rose-200 bg-rose-50 p-4">
      <p className="font-medium text-rose-900">{title}</p>
      <p className="mt-1 text-sm text-rose-800">{detail}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-md bg-white px-3 py-1 text-sm font-medium text-rose-800 ring-1 ring-rose-200 hover:bg-rose-100"
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function AnswerSkeleton() {
  return (
    <div
      aria-busy="true"
      aria-label="Loading answer"
      className="animate-pulse rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
    >
      <div className="flex justify-between">
        <div className="h-3 w-20 rounded bg-slate-200" />
        <div className="h-6 w-36 rounded-full bg-slate-200" />
      </div>
      <div className="mt-5 space-y-2.5">
        <div className="h-3 w-full rounded bg-slate-200" />
        <div className="h-3 w-11/12 rounded bg-slate-200" />
        <div className="h-3 w-4/5 rounded bg-slate-200" />
        <div className="h-3 w-2/3 rounded bg-slate-200" />
      </div>
      <p className="mt-5 text-xs text-slate-400">
        Searching the guidelines and checking every claim against its source…
      </p>
    </div>
  );
}

export function EmptyState() {
  return (
    <div className="rounded-2xl border border-dashed border-slate-300 bg-white/60 p-8 text-center">
      <p className="font-medium text-slate-700">Ask a question to get started</p>
      <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
        Every answer is grounded in the indexed guidelines. Click a numbered marker to see the
        exact passage and page it came from.
      </p>
    </div>
  );
}
