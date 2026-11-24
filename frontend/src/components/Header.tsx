import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function Header() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
    retry: false,
  });

  let dot = "bg-slate-300";
  let label = "Checking server…";
  if (health.isError) {
    dot = "bg-rose-500";
    label = "Server unreachable";
  } else if (health.data) {
    const ok = health.data.status === "ok";
    dot = ok ? "bg-emerald-500" : "bg-amber-500";
    label = ok
      ? `Online · ${health.data.index?.documents ?? 0} documents`
      : `Limited: ${health.data.issues.join("; ")}`;
  }

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
        <div className="flex items-center gap-3">
          <img src="/favicon.svg" alt="" className="h-8 w-8 shrink-0" />
          <div>
            <h1 className="text-sm font-semibold leading-tight text-slate-900 sm:text-base">
              Healthcare Knowledge Navigator
            </h1>
            <p className="hidden text-xs text-slate-500 sm:block">
              Evidence-based answers from clinical guidelines
            </p>
          </div>
        </div>
        <div
          className="flex max-w-[50%] items-center gap-2 text-xs text-slate-600"
          role="status"
          title={label}
        >
          <span className={`h-2 w-2 shrink-0 rounded-full ${dot}`} aria-hidden="true" />
          <span className="hidden truncate sm:inline">{label}</span>
          <span className="sm:hidden">{health.isError ? "Offline" : "Online"}</span>
        </div>
      </div>
    </header>
  );
}
