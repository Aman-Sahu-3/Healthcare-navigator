import { describe, expect, it, vi } from "vitest";
import { ApiError, api, describeError } from "./client";

function mockFetch(status: number, body: unknown, headers: Record<string, string> = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status, headers })),
  );
}

describe("api client", () => {
  it("returns JSON on success and calls /api by default", async () => {
    mockFetch(200, { status: "ok" });
    await expect(api.health()).resolves.toEqual({ status: "ok" });
    expect(vi.mocked(fetch).mock.calls[0]?.[0]).toBe("/api/health");
  });

  it("parses the backend error format and Retry-After", async () => {
    mockFetch(
      429,
      { error: { code: "rate_limited", message: "Too many" }, request_id: "r1" },
      { "Retry-After": "12" },
    );
    const err = await api.query({ question: "abc" }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 429, code: "rate_limited", retryAfter: 12, requestId: "r1" });
    expect(describeError(err).detail).toContain("12 seconds");
  });

  it("maps network failures", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    const err = await api.health().catch((e: unknown) => e);
    expect(err).toMatchObject({ code: "network_error", status: 0 });
    expect(describeError(err).title).toMatch(/reach the server/);
  });

  it("handles non-JSON error pages", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("<html>502</html>", { status: 502 })));
    const err = await api.health().catch((e: unknown) => e);
    expect(err).toMatchObject({ status: 502, code: "http_error" });
  });

  it("joins validation details", () => {
    const err = new ApiError(422, "validation_error", "Invalid", null, undefined, [
      { field: "question", message: "too short" },
    ]);
    expect(describeError(err).detail).toBe("too short");
  });
});
