import { useCallback, useState } from "react";
import type { QueryResponse } from "../api/types";

export interface HistoryEntry {
  id: string;
  askedAt: number;
  result: QueryResponse;
}

const KEY = "hkn.history.v1";
const MAX = 20;

// localStorage can be missing or throw (private mode, blocked storage) — the app must
// still work, so every access is wrapped and history simply becomes in-memory.
function load(): HistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    const data = raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
    return Array.isArray(data) ? data.filter((e) => e?.result?.question) : [];
  } catch {
    return [];
  }
}

function save(entries: HistoryEntry[]) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(entries));
  } catch {
    /* ignore quota / disabled storage */
  }
}

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>(load);

  const add = useCallback((result: QueryResponse): HistoryEntry => {
    const entry: HistoryEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      askedAt: Date.now(),
      result,
    };
    setEntries((prev) => {
      const next = [
        entry,
        ...prev.filter((e) => e.result.question !== result.question),
      ].slice(0, MAX);
      save(next);
      return next;
    });
    return entry;
  }, []);

  const clear = useCallback(() => {
    setEntries([]);
    save([]);
  }, []);

  return { entries, add, clear };
}
