import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { QueryResponse } from "./api/types";
import App from "./App";

const ANSWER: QueryResponse = {
  question: "What is first-line treatment for hypertension?",
  answer:
    "First-line options include:\n- Thiazide diuretics [1]\n- ACE inhibitors [2]\n\nMonotherapy is preferred initially [1].",
  citations: [
    {
      id: 1,
      source: "who_htn_2021.pdf",
      title: "WHO Hypertension Guideline",
      page: 12,
      snippet: "Thiazide-like diuretics are recommended as first-line agents.",
      similarity: 0.82,
    },
    {
      id: 2,
      source: "kenya_guidelines.pdf",
      title: "Kenya Clinical Guidelines",
      page: 40,
      snippet: "ACE inhibitors may be used as first-line therapy.",
      similarity: 0.71,
    },
  ],
  confidence: { level: "HIGH", score: 0.86, reasons: ["Best cited passage similarity 0.82"] },
  insufficient_evidence: false,
  retrieved_chunks: 5,
  model: "openai/gpt-oss-20b",
  latency_ms: 1240,
};

const REFUSAL: QueryResponse = {
  ...ANSWER,
  question: "What is the capital of France?",
  answer: "Insufficient evidence: the context does not cover this.",
  citations: [],
  confidence: { level: "LOW", score: 0, reasons: ["Model reported insufficient evidence"] },
  insufficient_evidence: true,
};

type Route = { status: number; body: unknown; headers?: Record<string, string> };

function mockApi(queryResponses: Route[]) {
  const queue = [...queryResponses];
  const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
    const json = (r: Route) =>
      new Response(JSON.stringify(r.body), { status: r.status, headers: r.headers });
    if (url.endsWith("/health"))
      return json({
        status: 200,
        body: {
          status: "ok",
          version: "0.1.0",
          environment: "test",
          llm_model: "m",
          llm_configured: true,
          index: { documents: 26, chunks: 900 },
          issues: [],
        },
      });
    if (url.endsWith("/sources"))
      return json({
        status: 200,
        body: {
          count: 1,
          documents: [{ source: "who.pdf", title: "WHO Guideline", total_pages: 80, chunks: 120 }],
        },
      });
    if (url.endsWith("/query")) return json(queue.shift() ?? { status: 500, body: {} });
    throw new Error(`unexpected ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderApp() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}

describe("App", () => {
  it("tells the user when a sleeping server is waking up", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      vi.stubGlobal(
        "fetch",
        vi.fn(() => new Promise<Response>(() => {})), // server not answering yet
      );
      renderApp();
      expect(screen.getByRole("status")).toHaveTextContent("Checking server");
      await vi.advanceTimersByTimeAsync(4500);
      await waitFor(() =>
        expect(screen.getByRole("status")).toHaveTextContent(/Waking up the server/),
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows server status and the disclaimer", async () => {
    mockApi([]);
    renderApp();
    expect(await screen.findByText(/Online · 26 documents/)).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent(/Not medical advice/);
  });

  it("asks a question and renders answer, badge and citation cards", async () => {
    const fetchMock = mockApi([{ status: 200, body: ANSWER }]);
    const user = userEvent.setup();
    renderApp();

    await user.type(screen.getByLabelText("Clinical question"), ANSWER.question);
    await user.click(screen.getByRole("button", { name: "Ask" }));

    const answer = await screen.findByRole("region", { name: "Answer" });
    expect(within(answer).getByText("Thiazide diuretics")).toBeInTheDocument();
    expect(within(answer).getByText("ACE inhibitors").closest("ul")).toBeInTheDocument();

    const badge = screen.getByRole("button", { name: /High confidence/ });
    expect(badge).toHaveAttribute("data-level", "HIGH");
    expect(badge).toHaveTextContent("86%");

    const card = document.getElementById("citation-2")!;
    expect(card).toHaveTextContent("Kenya Clinical Guidelines");
    expect(card).toHaveTextContent("Page 40");

    const body = JSON.parse(String(fetchMock.mock.calls.find((c) => c[0].endsWith("/query"))![1]!.body));
    expect(body).toEqual({ question: ANSWER.question, top_k: 5 });
  });

  it("highlights the citation card when a marker is clicked", async () => {
    mockApi([{ status: 200, body: ANSWER }]);
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText("Clinical question"), "first-line hypertension{Enter}");

    const marker = await screen.findByRole("button", {
      name: "Citation 2: Kenya Clinical Guidelines, page 40",
    });
    await user.click(marker);
    await waitFor(() =>
      expect(document.getElementById("citation-2")).toHaveAttribute("data-active", "true"),
    );
    expect(document.getElementById("citation-1")).not.toHaveAttribute("data-active");
  });

  it("explains confidence on click", async () => {
    mockApi([{ status: 200, body: ANSWER }]);
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText("Clinical question"), "hypertension{Enter}");
    await user.click(await screen.findByRole("button", { name: /High confidence/ }));
    expect(screen.getByRole("region", { name: "Why this confidence level" })).toHaveTextContent(
      "Best cited passage similarity 0.82",
    );
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("region", { name: "Why this confidence level" })).toBeNull();
  });

  it("renders a refusal distinctly with a LOW badge", async () => {
    mockApi([{ status: 200, body: REFUSAL }]);
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText("Clinical question"), "capital of France{Enter}");
    expect(await screen.findByText("No supported answer")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Low confidence/ })).toHaveAttribute(
      "data-level",
      "LOW",
    );
    expect(screen.queryByText(/Sources \(/)).not.toBeInTheDocument();
  });

  it("shows a friendly error and retries", async () => {
    mockApi([
      {
        status: 429,
        body: { error: { code: "rate_limited", message: "Too many" } },
        headers: { "Retry-After": "20" },
      },
      { status: 200, body: ANSWER },
    ]);
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText("Clinical question"), "hypertension{Enter}");

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Too many questions");
    expect(alert).toHaveTextContent("20 seconds");
    await user.click(within(alert).getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("region", { name: "Answer" })).toBeInTheDocument();
  });

  it("disables Ask for too-short questions and fills examples", async () => {
    mockApi([]);
    const user = userEvent.setup();
    renderApp();
    const ask = screen.getByRole("button", { name: "Ask" });
    expect(ask).toBeDisabled();
    await user.type(screen.getByLabelText("Clinical question"), "ab");
    expect(ask).toBeDisabled();
    await user.clear(screen.getByLabelText("Clinical question"));
    await user.click(screen.getByRole("button", { name: /naloxone dose/ }));
    expect(screen.getByLabelText("Clinical question")).toHaveValue(
      "What is the recommended naloxone dose for opioid overdose?",
    );
    expect(ask).toBeEnabled();
  });

  it("keeps history and restores an answer without a new request", async () => {
    const fetchMock = mockApi([
      { status: 200, body: ANSWER },
      { status: 200, body: REFUSAL },
    ]);
    const user = userEvent.setup();
    renderApp();
    const box = screen.getByLabelText("Clinical question");
    await user.type(box, "hypertension{Enter}");
    await screen.findByText("Thiazide diuretics");
    await user.clear(box);
    await user.type(box, "France{Enter}");
    await screen.findByText("No supported answer");

    const history = screen.getByRole("region", { name: "Recent questions" });
    const items = within(history).getAllByRole("button").filter((b) => b.textContent !== "Clear");
    expect(items.map((b) => b.textContent)).toEqual([REFUSAL.question, ANSWER.question]);

    const queryCalls = () => fetchMock.mock.calls.filter((c) => c[0].endsWith("/query")).length;
    expect(queryCalls()).toBe(2);
    await user.click(items[1]!);
    expect(await screen.findByText("Thiazide diuretics")).toBeInTheDocument();
    expect(queryCalls()).toBe(2);
    expect(JSON.parse(window.localStorage.getItem("hkn.history.v1")!)).toHaveLength(2);
  });

  it("works when localStorage throws", async () => {
    vi.spyOn(window.localStorage, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    mockApi([{ status: 200, body: ANSWER }]);
    const user = userEvent.setup();
    renderApp();
    await user.type(screen.getByLabelText("Clinical question"), "hypertension{Enter}");
    expect(await screen.findByText("Thiazide diuretics")).toBeInTheDocument();
  });
});
