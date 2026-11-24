/**
 * Turn the model's plain-text answer into blocks (paragraphs / lists) of inline segments
 * (text, **bold**, and [n] citation markers) so React can render them safely — no HTML
 * from the model is ever injected into the page.
 */

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "cite"; id: number };

export type Block =
  | { kind: "paragraph"; content: Inline[] }
  | { kind: "list"; ordered: boolean; items: Inline[][] };

const INLINE = /(\[\d+\]|\*\*[^*\n]+\*\*)/g;
const BULLET = /^\s*(?:[-*•])\s+(.*)$/;
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/;

export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  for (const part of text.split(INLINE)) {
    if (!part) continue;
    const cite = /^\[(\d+)\]$/.exec(part);
    if (cite) out.push({ kind: "cite", id: Number(cite[1]) });
    else if (part.startsWith("**") && part.endsWith("**") && part.length > 4)
      out.push({ kind: "bold", text: part.slice(2, -2) });
    else out.push({ kind: "text", text: part });
  }
  return out;
}

export function parseAnswer(answer: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushPara = () => {
    if (para.length) blocks.push({ kind: "paragraph", content: parseInline(para.join(" ")) });
    para = [];
  };
  const flushList = () => {
    if (list) {
      blocks.push({ kind: "list", ordered: list.ordered, items: list.items.map(parseInline) });
    }
    list = null;
  };

  for (const raw of answer.replace(/\r\n/g, "\n").split("\n")) {
    const line = raw.trim();
    const bullet = BULLET.exec(raw);
    const numbered = NUMBERED.exec(raw);
    if (bullet || numbered) {
      flushPara();
      const ordered = Boolean(numbered && !bullet);
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push(((bullet ?? numbered) as RegExpExecArray)[1] ?? "");
    } else if (!line) {
      flushPara();
      flushList();
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return blocks;
}

/** Plain text for copying: keeps [n] markers and appends the source list. */
export function answerAsText(
  answer: string,
  citations: { id: number; title: string; page: number; source: string }[],
): string {
  const refs = citations.map((c) => `[${c.id}] ${c.title}, p. ${c.page} (${c.source})`);
  return refs.length ? `${answer}\n\nSources:\n${refs.join("\n")}` : answer;
}
