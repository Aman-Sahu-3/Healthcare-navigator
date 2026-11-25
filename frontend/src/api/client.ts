import type {
  ErrorBody,
  HealthResponse,
  QueryRequest,
  QueryResponse,
  SourcesResponse,
} from "./types";

// Dev: "/api" (proxied by Vite). Production: the deployed API origin.
export const API_BASE = (import.meta.env.VITE_API_URL || "/api").replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public requestId?: string | null,
    public retryAfter?: number,
    public details?: { field: string; message: string }[],
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 60_000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      signal: controller.signal,
    });
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === "AbortError";
    throw new ApiError(
      0,
      aborted ? "client_timeout" : "network_error",
      aborted ? "The request took too long" : "Cannot reach the API server",
    );
  } finally {
    clearTimeout(timer);
  }

  if (res.ok) return (await res.json()) as T;

  let body: ErrorBody | null = null;
  try {
    body = (await res.json()) as ErrorBody;
  } catch {
    /* non-JSON error (e.g. proxy HTML page) */
  }
  const retry = Number(res.headers.get("Retry-After"));
  throw new ApiError(
    res.status,
    body?.error?.code ?? "http_error",
    body?.error?.message ?? `Request failed (${res.status})`,
    body?.request_id ?? res.headers.get("X-Request-ID"),
    Number.isFinite(retry) && retry > 0 ? retry : undefined,
    body?.error?.details,
  );
}

export const api = {
  // 90 s: on free hosting the first request may include a ~1 min server wake-up.
  query: (body: QueryRequest) =>
    request<QueryResponse>("/query", { method: "POST", body: JSON.stringify(body) }, 90_000),
  health: () => request<HealthResponse>("/health", undefined, 30_000),
  sources: () => request<SourcesResponse>("/sources", undefined, 15_000),
};

/** Human-friendly message for an error from the API. */
export function describeError(err: unknown): { title: string; detail: string } {
  if (!(err instanceof ApiError)) {
    return { title: "Something went wrong", detail: String(err) };
  }
  switch (err.code) {
    case "network_error":
      return {
        title: "Can't reach the server",
        detail: "Check that the API is running (uvicorn on port 8000) and try again.",
      };
    case "client_timeout":
    case "llm_timeout":
      return { title: "The answer took too long", detail: "Please try again in a moment." };
    case "rate_limited":
      return {
        title: "Too many questions",
        detail: `Please wait ${err.retryAfter ?? 60} seconds before asking again.`,
      };
    case "llm_rate_limited":
      return {
        title: "The language model is busy",
        detail: "The provider's rate limit or quota was reached. Try again shortly.",
      };
    case "llm_not_configured":
      return { title: "Answering is not configured", detail: err.message };
    case "llm_auth_error":
      return { title: "LLM provider rejected the API key", detail: "Check the server's API key." };
    case "validation_error":
      return {
        title: "Please check your question",
        detail: err.details?.map((d) => d.message).join(" · ") || err.message,
      };
    default:
      return {
        title: "Something went wrong",
        detail: err.requestId ? `${err.message} (ref ${err.requestId})` : err.message,
      };
  }
}
