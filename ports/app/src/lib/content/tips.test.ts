import { describe, it, expect } from 'vitest';
import { TIP_SOURCES } from './tips';
import { firstParagraph, hasHeading } from './render-doc';

// Every info-icon tip must back onto a real popover paragraph and a working
// see-more URL. The hasHeading assertion (c) is the load-bearing guard: a later
// doc edit that renames an anchored heading would leave firstParagraph's
// no-heading fallback returning a non-empty (but wrong) paragraph, which (a)
// alone cannot catch.
describe('TIP_SOURCES', () => {
  for (const [key, src] of Object.entries(TIP_SOURCES)) {
    describe(key, () => {
      it('(a) has a non-empty backing first paragraph', () => {
        expect(firstParagraph(src.md, src.anchor).length).toBeGreaterThan(0);
      });

      // InfoIcon builds the see-more href from `doc`; its test pins that builder.
      it('(b) names a vault-relative, extensionless doc page', () => {
        expect(src.doc).not.toMatch(/^\//); // no leading slash
        expect(src.doc).not.toMatch(/\.md$/); // extensionless (leyline pretty URL)
      });

      const anchor = src.anchor;
      if (anchor) {
        it('(c) anchor resolves to a real heading', () => {
          expect(hasHeading(src.md, anchor)).toBe(true);
        });
      }
    });
  }
});
