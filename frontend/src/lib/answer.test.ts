import { describe, expect, it } from "vitest";
import { answerAsText, parseAnswer, parseInline } from "./answer";

describe("parseInline", () => {
  it("splits text, bold and citation markers", () => {
    expect(parseInline("Use **thiazides** first [1][2].")).toEqual([
      { kind: "text", text: "Use " },
      { kind: "bold", text: "thiazides" },
      { kind: "text", text: " first " },
      { kind: "cite", id: 1 },
      { kind: "cite", id: 2 },
      { kind: "text", text: "." },
    ]);
  });

  it("treats html as plain text", () => {
    expect(parseInline("<img src=x onerror=alert(1)>")).toEqual([
      { kind: "text", text: "<img src=x onerror=alert(1)>" },
    ]);
  });
});

describe("parseAnswer", () => {
  it("builds paragraphs and lists", () => {
    const blocks = parseAnswer(
      "First-line options [1]:\n- Thiazides [1]\n- ACE inhibitors [2]\n\nReview yearly.\n1. Step one\n2. Step two",
    );
    expect(blocks.map((b) => b.kind)).toEqual(["paragraph", "list", "paragraph", "list"]);
    const list = blocks[1];
    expect(list?.kind === "list" && list.items.length).toBe(2);
    const ordered = blocks[3];
    expect(ordered?.kind === "list" && ordered.ordered).toBe(true);
  });

  it("joins wrapped lines into one paragraph", () => {
    const blocks = parseAnswer("Line one\nline two");
    expect(blocks).toHaveLength(1);
    expect(blocks[0]).toEqual({
      kind: "paragraph",
      content: [{ kind: "text", text: "Line one line two" }],
    });
  });
});

describe("answerAsText", () => {
  it("appends a source list", () => {
    const text = answerAsText("A [1].", [{ id: 1, title: "WHO HTN", page: 12, source: "who.pdf" }]);
    expect(text).toContain("Sources:\n[1] WHO HTN, p. 12 (who.pdf)");
  });
});
