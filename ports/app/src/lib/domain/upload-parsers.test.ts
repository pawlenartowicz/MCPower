// Vitest tests for upload-parsers.ts — the multi-format ingestion layer.
// Covers CSV/TSV dialect autodetect + override, XLSX sheet listing/selection,
// SPSS .sav parsing (values + variable labels), preview slicing, and error mapping.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseFile } from './upload-parsers';

function csvFile(text: string, name = 'data.csv'): File {
    return new File([text], name, { type: 'text/csv' });
}

// Binary fixtures live next to this test; vitest runs with cwd = ports/app.
function fixtureFile(name: string, mime: string): File {
    const bytes = readFileSync(resolve('src/lib/domain/__fixtures__', name));
    return new File([bytes], name, { type: mime });
}

// ---------------------------------------------------------------------------
// CSV / TSV — PapaParse
// ---------------------------------------------------------------------------

describe('parseFile — CSV/TSV dialect', () => {
    it('parses a comma CSV and reports the detected separator', async () => {
        const out = await parseFile(csvFile('a,b,c\n1,2,3\n4,5,6'));
        expect(out.format).toBe('csv');
        expect(out.parsed.header).toEqual(['a', 'b', 'c']);
        expect(out.parsed.columns[0]).toEqual(['1', '4']);
        expect(out.parsed.nRows).toBe(2);
        expect(out.dialect?.separator).toBe(',');
    });

    it('autodetects a semicolon separator', async () => {
        const out = await parseFile(csvFile('a;b;c\n1;2;3'));
        expect(out.dialect?.separator).toBe(';');
        expect(out.parsed.header).toEqual(['a', 'b', 'c']);
    });

    it('autodetects a tab separator (.tsv)', async () => {
        const out = await parseFile(csvFile('a\tb\tc\n1\t2\t3', 'data.tsv'));
        expect(out.dialect?.separator).toBe('\t');
        expect(out.parsed.header).toEqual(['a', 'b', 'c']);
    });

    it('autodetects a pipe separator', async () => {
        const out = await parseFile(csvFile('a|b|c\n1|2|3'));
        expect(out.dialect?.separator).toBe('|');
    });

    it('handles quoted fields with an embedded separator', async () => {
        const out = await parseFile(csvFile('"x","y"\n"a,b","c"'));
        expect(out.parsed.header).toEqual(['x', 'y']);
        expect(out.parsed.columns[0]).toEqual(['a,b']);
    });

    // Migrated from the deleted parseCsvText quirk tests.
    it('does not emit a phantom trailing field for a row ending in a quoted field', async () => {
        const out = await parseFile(csvFile('"a","b","c"\n1,2,3'));
        expect(out.parsed.header).toEqual(['a', 'b', 'c']);
    });

    it('parses an mtcars-style header without a duplicate empty column', async () => {
        const out = await parseFile(csvFile('"","mpg","carb"\n"Mazda RX4",21,4'));
        expect(out.parsed.header).toEqual(['', 'mpg', 'carb']);
        expect(out.parsed.columns).toHaveLength(3);
        expect(out.parsed.columns[0]).toEqual(['Mazda RX4']);
    });

    it('honours a single-quote override', async () => {
        const out = await parseFile(csvFile("a,b\n'x,y',z"), { dialect: { separator: ',', quote: "'" } });
        expect(out.parsed.columns[0]).toEqual(['x,y']);
        expect(out.dialect?.quote).toBe("'");
    });

    it('pads short rows to the header width (lenient, no throw)', async () => {
        const out = await parseFile(csvFile('a,b,c\n1,2\n3,4,5'));
        expect(out.parsed.columns[0]).toEqual(['1', '3']);
        expect(out.parsed.columns[2]).toEqual(['', '5']);
    });

    it('rejects a file with an unterminated quote instead of silently mangling it', async () => {
        await expect(parseFile(csvFile('a,b\n"x,1'))).rejects.toThrow(/Couldn't parse this file/);
    });

    it('slices to previewRows', async () => {
        const text = 'a,b\n' + Array.from({ length: 20 }, (_, i) => `${i},${i}`).join('\n');
        const out = await parseFile(csvFile(text), { previewRows: 3 });
        expect(out.parsed.nRows).toBe(3);
    });
});

// ---------------------------------------------------------------------------
// XLSX — read-excel-file
// ---------------------------------------------------------------------------

describe('parseFile — XLSX', () => {
    const XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';

    it('lists sheets and reads the first sheet by default', async () => {
        const out = await parseFile(fixtureFile('two_sheet.xlsx', XLSX_MIME));
        expect(out.format).toBe('xlsx');
        expect(out.sheetNames).toEqual(['responses', 'summary']);
        expect(out.parsed.header).toEqual(['id', 'score', 'cohort', 'completed', 'signup']);
        expect(out.parsed.columns[0]).toEqual(['1', '2', '3', '4', '5']);
    });

    it('stringifies numeric, boolean, and Date cells', async () => {
        const out = await parseFile(fixtureFile('two_sheet.xlsx', XLSX_MIME));
        expect(out.parsed.columns[1]![0]).toBe('78.2'); // score (float)
        expect(out.parsed.columns[3]![0]).toBe('true'); // completed (boolean)
        // signup Date → ISO string (not the tz-shifted "Mon Jan 15 2024…" of String(date))
        expect(out.parsed.columns[4]![0]).toMatch(/^2024-01-15/);
    });

    it('selects a sheet by name (positional readSheet argument)', async () => {
        const out = await parseFile(fixtureFile('two_sheet.xlsx', XLSX_MIME), { sheet: 'summary' });
        // Proves the sheet arg is honoured positionally — 'summary' is sheet 2, not the default.
        expect(out.parsed.header).toEqual(['cohort', 'mean_score']);
    });
});

// ---------------------------------------------------------------------------
// SPSS .sav — sav-reader
// ---------------------------------------------------------------------------

describe('parseFile — .sav', () => {
    const SAV_MIME = 'application/x-spss-sav';

    it('parses values and exposes variable labels', async () => {
        const out = await parseFile(fixtureFile('sample.sav', SAV_MIME));
        expect(out.format).toBe('sav');
        expect(out.parsed.header).toEqual(['q1_satisfaction', 'age_intake', 'site_id']);
        expect(out.parsed.columns[0]).toEqual(['4', '5', '3', '4', '2']);
        expect(out.parsed.nRows).toBe(5);
        expect(out.variableLabels).toMatchObject({
            q1_satisfaction: 'Overall satisfaction (1-5 Likert)',
            age_intake: 'Age in years at intake',
            site_id: 'Site identifier',
        });
    });

    it('rejects a ZLIB-compressed .sav with an actionable message', async () => {
        await expect(parseFile(fixtureFile('compressed.sav', SAV_MIME))).rejects.toThrow(
            /ZLIB compression/i,
        );
    });
});

// ---------------------------------------------------------------------------
// Unsupported types
// ---------------------------------------------------------------------------

describe('parseFile — unsupported', () => {
    it('rejects an unknown extension', async () => {
        await expect(parseFile(new File(['{}'], 'data.json'))).rejects.toThrow(/Unsupported file type/);
    });
});
