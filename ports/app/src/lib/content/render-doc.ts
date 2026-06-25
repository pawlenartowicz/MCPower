// Shared documentation renderer: resolves leyline-style wiki links to the published
// docs site, renders trusted vault Markdown, and sanitizes the result. Also routes
// anchor clicks to the host's external opener so the app's own webview never navigates.
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { isTauri } from '@tauri-apps/api/core';

/** Base URL of the published documentation wiki — the single switch for doc routing. */
export const DOCS_BASE_URL = 'https://docs.mcpower.app';

/** The account-less bug-report form (mcpower.app/report). */
export const REPORT_BASE_URL = 'https://mcpower.app/report';

// Inlined by vite `define`; falls back to 'dev' in unit tests (no define).
const APP_VERSION = typeof __MCPOWER_VERSION__ !== 'undefined' ? __MCPOWER_VERSION__ : 'dev';

/**
 * URL to the report form, pre-tagged with this shell (port=app|wasm) and
 * version. `detail` (e.g. a crash dump) prefills the form's description; it is
 * truncated to keep the URL within browser length limits.
 */
export function reportUrl(detail?: string): string {
  const port = import.meta.env.VITE_TARGET === 'wasm' ? 'wasm' : 'app';
  const params = new URLSearchParams({ port, version: APP_VERSION });
  if (detail) params.set('detail', detail.slice(0, 1500));
  return `${REPORT_BASE_URL}?${params.toString()}`;
}

// Leyline wiki links: [[path]] or [[path|alias]]. The path is vault-relative
// (e.g. concepts/effect-sizes) and maps verbatim to {DOCS_BASE_URL}/{path-minus-.md}.
const WIKI_LINK = /\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]/g;

function resolveWikiLinks(md: string): string {
  return md.replace(WIKI_LINK, (_match, path: string, alias?: string) => {
    const slug = path.trim().replace(/\.md$/, '');
    const text = (alias ?? slug).trim();
    return `[${text}](${DOCS_BASE_URL}/${slug})`;
  });
}

// GitHub/Obsidian callout marker `> [!type] Title`. marked has no callout support,
// so the raw `[!type]` would render literally. Drop the marker and keep the title
// as a bold lead-in; the rest of the blockquote body is left untouched.
const CALLOUT_MARKER = /^(>[ \t]*)\[!\w+\][ \t]*(.*)$/gm;

function plainCallouts(md: string): string {
  return md.replace(CALLOUT_MARKER, (_m, quote: string, title: string) =>
    title.trim() ? `${quote}**${title.trim()}**` : quote.trimEnd(),
  );
}

/** Render vault Markdown to sanitized HTML, with wiki links pointed at the live docs. */
export function renderDoc(md: string): string {
  // Drop the YAML frontmatter first — marked has no frontmatter support and would
  // otherwise render the raw `---`/`title:`/`description:` block into the page.
  const html = marked.parse(resolveWikiLinks(plainCallouts(stripFrontmatter(md))), {
    async: false,
  }) as string;
  return DOMPurify.sanitize(html);
}

/** Open an external URL in the system browser — never navigate the app's own webview. */
export function openExternal(href: string): void {
  if (isTauri()) {
    void import('@tauri-apps/plugin-opener').then(({ openUrl }) => openUrl(href));
  } else {
    window.open(href, '_blank', 'noopener,noreferrer');
  }
}

// ---------------------------------------------------------------------------
// Tip backing: extract a popover-ready first paragraph (page or section) from a
// vault page, and resolve section anchors the way the published site does.
// ---------------------------------------------------------------------------

/**
 * GitHub-style heading slug exactly as goldmark v1.8.2's WithAutoHeadingID emits
 * it — the convention the docs publisher uses for in-page anchors. ASCII
 * alphanumerics are lowercased and kept; each space, '-' or '_' becomes a single
 * '-'; every other punctuation mark and any non-ASCII character is dropped. (The
 * publisher's in-document -1/-2 collision suffix is not replicated: backing
 * headings are unique on their own page.)
 */
export function slugify(text: string): string {
  let out = '';
  for (const ch of text.trim()) {
    if (ch.charCodeAt(0) > 0x7f) continue; // non-ASCII → dropped
    if (/[A-Za-z0-9]/.test(ch)) out += ch.toLowerCase();
    else if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '-' || ch === '_') out += '-';
    // all other ASCII punctuation → dropped
  }
  return out;
}

// ATX heading: 1–6 '#', whitespace, the text, optional trailing '#' run.
const HEADING = /^#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$/;

/** Heading text of a Markdown line, or null when the line is not an ATX heading. */
function headingText(line: string): string | null {
  const m = HEADING.exec(line);
  return m?.[1] ?? null;
}

/** Drop a leading `---`-fenced YAML frontmatter block, if the page has one. */
function stripFrontmatter(md: string): string {
  const m = /^---[ \t]*\r?\n[\s\S]*?\r?\n---[ \t]*(?:\r?\n|$)/.exec(md);
  return m ? md.slice(m[0].length) : md;
}

/** True iff some heading in `md` slugifies to `anchor` — the broken-anchor detector. */
export function hasHeading(md: string, anchor: string): boolean {
  return stripFrontmatter(md)
    .split(/\r?\n/) // CRLF-safe: Windows git autocrlf checks docs out with \r\n
    .some((line) => {
      const t = headingText(line);
      return t !== null && slugify(t) === anchor;
    });
}

/**
 * First self-contained paragraph of a vault page (or of one section), returned as
 * Markdown for renderDoc. With no anchor: skips a leading H1 and returns the
 * page's first paragraph. With an anchor: returns the first paragraph beneath the
 * heading whose slug matches; if no heading matches, falls back to the page's
 * first paragraph so a stale anchor still yields a useful popover (the fallback is
 * itself non-empty, so use hasHeading — not emptiness — to detect a broken anchor).
 */
export function firstParagraph(md: string, anchor?: string): string {
  const lines = stripFrontmatter(md).split(/\r?\n/); // CRLF-safe, mirrors hasHeading
  let i = 0;
  if (anchor) {
    const idx = lines.findIndex((line) => {
      const t = headingText(line);
      return t !== null && slugify(t) === anchor;
    });
    if (idx === -1) return firstParagraph(md); // stale anchor → page intro
    i = idx + 1;
  } else {
    while (i < lines.length && lines[i]!.trim() === '') i++;
    if (/^#[ \t]+/.test(lines[i] ?? '')) i++; // skip a leading H1
  }
  while (i < lines.length && lines[i]!.trim() === '') i++;
  const para: string[] = [];
  while (i < lines.length && lines[i]!.trim() !== '') para.push(lines[i++]!);
  return para.join('\n').trim();
}
