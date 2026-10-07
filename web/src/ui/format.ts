/** Make a token's whitespace visible: "·" for spaces, "↵" for newlines. */
export function showToken(text: string): string {
  return text.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "→");
}

export function pct(p: number): string {
  if (p >= 0.995) return "100%";
  if (p >= 0.1) return `${(p * 100).toFixed(0)}%`;
  if (p >= 0.001) return `${(p * 100).toFixed(1)}%`;
  return "<0.1%";
}

/** 16_234_567 -> "16.2M" */
export function compact(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}K`;
  return String(n);
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  text?: string,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
