/** Rewrite a page's screenshot placeholder embed(s) to the canonical
 *  theme-swapping pair
 *    `![[assets/<id>-setup.png|600|theme-light]]`
 *    `![[assets/<id>-setup-dark.png|600|theme-dark]]`
 *  Leyline shows the `theme-light` image only in light mode and `theme-dark`
 *  only in dark, swapping as the reader toggles (notes/2-features/markdown.md;
 *  parser at leyline internal/web/render/imagesize.go — size + variant tokens
 *  combine, order-independent). Matches any embed whose target ends in
 *  `<id>-setup*.png` — legacy `<family>/<id>-setup-1.png`, the canonical light
 *  `assets/<id>-setup.png`, and the dark `assets/<id>-setup-dark.png` alike —
 *  so re-running collapses an existing pair back to itself. If none is present,
 *  insert the pair on its own lines right after the first H1. */
export function reconcileEmbed(markdown, id) {
  const pair =
    `![[assets/${id}-setup.png|600|theme-light]]\n` +
    `![[assets/${id}-setup-dark.png|600|theme-dark]]`;
  // Embed targeting this id's setup image (light or dark variant), any path
  // prefix, optional -N / -dark suffix, any |options, plus its trailing newline
  // so blanking a sibling embed removes the whole line (keeps the pair idempotent).
  const re = new RegExp(
    String.raw`!\[\[[^\]\n]*${id}-setup(?:-\d+|-dark)?\.png(?:\|[^\]\n]*)?\]\]\n?`,
    'g',
  );
  if (re.test(markdown)) {
    let first = true;
    const text = markdown.replace(re, () => (first ? ((first = false), pair + '\n') : ''))
      // collapse any blank line left where a duplicate placeholder was blanked
      .replace(/\n{3,}/g, '\n\n');
    return { text, changed: text !== markdown };
  }
  const lines = markdown.split('\n');
  const h1 = lines.findIndex((l) => /^#\s/.test(l));
  const at = h1 === -1 ? 0 : h1 + 1;
  lines.splice(at, 0, '', pair);
  return { text: lines.join('\n'), changed: true };
}
