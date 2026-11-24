import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api/client";
import type { QueryResponse } from "./api/types";
import { AnswerPanel } from "./components/AnswerPanel";
import { Disclaimer } from "./components/Disclaimer";
import { Header } from "./components/Header";
import { QueryForm } from "./components/QueryForm";
import { HistoryList, SourcesList } from "./components/Sidebar";
import { AnswerSkeleton, EmptyState, ErrorBanner } from "./components/StatusViews";
import { type HistoryEntry, useHistory } from "./hooks/useHistory";

export default function App() {
  const history = useHistory();
  const [shown, setShown] = useState<{ id: string | null; result: QueryResponse } | null>(null);
  const [formQuestion, setFormQuestion] = useState("");

  const ask = useMutation({
    mutationFn: api.query,
    onSuccess: (result) => {
      const entry = history.add(result);
      setShown({ id: entry.id, result });
    },
  });

  const submit = (question: string, topK: number) => {
    setShown(null);
    ask.mutate({ question, top_k: topK });
  };

  const selectHistory = (entry: HistoryEntry) => {
    ask.reset();
    setShown({ id: entry.id, result: entry.result });
    setFormQuestion(entry.result.question);
  };

  return (
    <div className="min-h-screen">
      <Header />
      <Disclaimer />
      <main className="mx-auto grid max-w-6xl gap-6 px-4 py-6 lg:grid-cols-[1fr_280px]">
        <div className="min-w-0 space-y-5">
          <QueryForm onSubmit={submit} loading={ask.isPending} initialQuestion={formQuestion} />
          {ask.isPending && <AnswerSkeleton />}
          {ask.isError && (
            <ErrorBanner
              error={ask.error}
              onRetry={ask.variables ? () => ask.mutate(ask.variables) : undefined}
            />
          )}
          {!ask.isPending && shown && <AnswerPanel result={shown.result} />}
          {!ask.isPending && !ask.isError && !shown && <EmptyState />}
        </div>
        <aside className="space-y-4">
          <HistoryList
            entries={history.entries}
            selectedId={shown?.id ?? null}
            onSelect={selectHistory}
            onClear={history.clear}
          />
          <SourcesList />
        </aside>
      </main>
      <footer className="mx-auto max-w-6xl px-4 pb-8 text-center text-xs text-slate-400">
        Answers cite the source document and page. Verify before clinical use.
      </footer>
    </div>
  );
}
