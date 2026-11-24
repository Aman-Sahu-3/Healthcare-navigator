import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";

const MAX_LEN = 1000;
const EXAMPLES = [
  "What is the first-line treatment for hypertension in adults?",
  "How soon must HIV post-exposure prophylaxis be started?",
  "What is the recommended naloxone dose for opioid overdose?",
  "What are the advantages of using clinical guidelines?",
];

interface Props {
  onSubmit: (question: string, topK: number) => void;
  loading: boolean;
  initialQuestion?: string;
}

export function QueryForm({ onSubmit, loading, initialQuestion = "" }: Props) {
  const [question, setQuestion] = useState(initialQuestion);
  const [topK, setTopK] = useState(5);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => setQuestion(initialQuestion), [initialQuestion]);

  const trimmed = question.trim();
  const valid = trimmed.length >= 3 && question.length <= MAX_LEN;

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (valid && !loading) onSubmit(trimmed, topK);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form onSubmit={submit} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <label htmlFor="question" className="sr-only">
        Clinical question
      </label>
      <textarea
        id="question"
        ref={ref}
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={onKeyDown}
        rows={3}
        maxLength={MAX_LEN + 50}
        placeholder="Ask a clinical question, e.g. What is the first-line treatment for hypertension?"
        className="w-full resize-y rounded-lg border-0 bg-transparent p-2 text-base text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-0"
        autoFocus
      />
      <div className="mt-2 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-3">
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <label className="flex items-center gap-1.5">
            Passages
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="rounded-md border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-700"
            >
              {[3, 5, 8, 10].map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <span className={question.length > MAX_LEN ? "text-rose-600" : ""}>
            {question.length}/{MAX_LEN}
          </span>
          <span className="hidden sm:inline">Enter to ask · Shift+Enter for new line</span>
        </div>
        <button
          type="submit"
          disabled={!valid || loading}
          className="inline-flex items-center gap-2 rounded-lg bg-brand-700 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-brand-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading && (
            <span
              className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
              aria-hidden="true"
            />
          )}
          {loading ? "Searching guidelines…" : "Ask"}
        </button>
      </div>

      {!question && (
        <div className="mt-3 flex flex-wrap gap-2" aria-label="Example questions">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => {
                setQuestion(ex);
                ref.current?.focus();
              }}
              className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-600 hover:border-brand-600 hover:text-brand-700"
            >
              {ex}
            </button>
          ))}
        </div>
      )}
    </form>
  );
}
