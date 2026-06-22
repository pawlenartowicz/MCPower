import { describe, it, expect } from 'vitest';
import { renderDoc, DOCS_BASE_URL, slugify, hasHeading, firstParagraph } from './render-doc';

describe('renderDoc', () => {
  it('rewrites aliased wiki links to external docs URLs', () => {
    const html = renderDoc('See [[concepts/effect-sizes|effect sizes]].');
    expect(html).toContain(`href="${DOCS_BASE_URL}/concepts/effect-sizes"`);
    expect(html).toContain('>effect sizes<');
  });

  it('rewrites bare wiki links and strips the .md slug', () => {
    const html = renderDoc('Open [[tutorial-app/regression.md]].');
    expect(html).toContain(`href="${DOCS_BASE_URL}/tutorial-app/regression"`);
    expect(html).toContain('>tutorial-app/regression<'); // link text drops the .md too
  });

  it('renders ordinary markdown', () => {
    const html = renderDoc('# Title\n\n- a\n- b');
    expect(html).toContain('<h1>Title</h1>');
    expect(html).toContain('<li>a</li>');
  });

  it('sanitizes dangerous HTML', () => {
    const html = renderDoc('hi <img src=x onerror=alert(1)> <script>alert(1)</script>');
    expect(html).not.toContain('onerror');
    expect(html).not.toContain('<script>');
  });
});

describe('slugify (goldmark WithAutoHeadingID parity)', () => {
  it('lowercases and turns spaces into hyphens', () => {
    expect(slugify('The ICC')).toBe('the-icc');
    expect(slugify('Choosing which tests to target')).toBe('choosing-which-tests-to-target');
  });

  it('matches the real backing anchors', () => {
    expect(slugify('Baseline probability')).toBe('baseline-probability');
    expect(slugify('Factors and the reference level')).toBe('factors-and-the-reference-level');
    expect(slugify('Matched predictors lock to the data')).toBe('matched-predictors-lock-to-the-data');
  });

  it('drops punctuation and "&" (does not hyphenate them) and treats _ like a separator', () => {
    // goldmark drops non-alphanumerics other than space/-/_; no separator collapsing.
    expect(slugify('Tom & Jerry, Inc.')).toBe('tom--jerry-inc');
    expect(slugify('a_b-c')).toBe('a-b-c');
    expect(slugify('Run, and clear')).toBe('run-and-clear');
  });

  it('drops non-ASCII characters', () => {
    expect(slugify('Naïve café')).toBe('nave-caf');
  });
});

describe('hasHeading', () => {
  const md = '# Page\n\nIntro.\n\n## The ICC\n\nbody\n\n## Run, and clear\n\nmore';
  it('is true when a heading slugifies to the anchor', () => {
    expect(hasHeading(md, 'the-icc')).toBe(true);
    expect(hasHeading(md, 'run-and-clear')).toBe(true); // un-numbered/punctuated slug
  });
  it('is false when no heading matches', () => {
    expect(hasHeading(md, 'nope')).toBe(false);
  });
  it('matches headings in CRLF-checked-out docs (Windows git autocrlf)', () => {
    expect(hasHeading(md.replace(/\n/g, '\r\n'), 'the-icc')).toBe(true);
  });
});

describe('firstParagraph', () => {
  it('with no anchor strips the H1 and returns the first paragraph', () => {
    expect(firstParagraph('# Title\n\nFirst para here.\n\n## Section\n\nSecond.')).toBe(
      'First para here.',
    );
  });

  it('with an anchor returns the matching section’s first paragraph', () => {
    const md = '# T\n\nIntro.\n\n## The ICC\n\nICC para.\n\nmore';
    expect(firstParagraph(md, 'the-icc')).toBe('ICC para.');
  });

  it('matches an un-numbered, punctuated heading slug', () => {
    const md = '# T\n\nIntro.\n\n## Run, and clear\n\nRun para.';
    expect(firstParagraph(md, 'run-and-clear')).toBe('Run para.');
  });

  it('falls back to the page first paragraph when the anchor is missing', () => {
    const md = '# T\n\nIntro.\n\n## S\n\nSec.';
    expect(firstParagraph(md, 'no-such-anchor')).toBe('Intro.');
  });

  it('returns empty string for an H1-only page', () => {
    expect(firstParagraph('# Only a title')).toBe('');
  });

  it('strips a leading frontmatter block', () => {
    expect(firstParagraph('---\ntitle: x\n---\n\n# T\n\nBody para.')).toBe('Body para.');
  });

  it('resolves anchors in CRLF-checked-out docs (Windows git autocrlf)', () => {
    const md = '# T\n\nIntro.\n\n## The ICC\n\nICC para.\n\nmore'.replace(/\n/g, '\r\n');
    expect(firstParagraph(md, 'the-icc')).toBe('ICC para.');
  });
});
