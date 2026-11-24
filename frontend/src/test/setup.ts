import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";

/**
 * In-memory Web Storage for tests.
 *
 * Node 25+ ships its own experimental `localStorage` global which, without the
 * --localstorage-file flag, is undefined and hides jsdom's implementation. Installing
 * our own makes the tests behave the same on every Node version.
 */
export class MemoryStorage implements Storage {
  private data = new Map<string, string>();
  get length() {
    return this.data.size;
  }
  clear() {
    this.data.clear();
  }
  getItem(key: string) {
    return this.data.has(key) ? (this.data.get(key) as string) : null;
  }
  key(index: number) {
    return [...this.data.keys()][index] ?? null;
  }
  removeItem(key: string) {
    this.data.delete(key);
  }
  setItem(key: string, value: string) {
    this.data.set(key, String(value));
  }
}

beforeEach(() => {
  const storage = new MemoryStorage();
  for (const target of [window, globalThis]) {
    Object.defineProperty(target, "localStorage", {
      value: storage,
      configurable: true,
      writable: true,
    });
  }
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
