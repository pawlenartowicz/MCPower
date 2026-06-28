<script lang="ts">
  // Citation footer at the foot of the results scroll area (shown under every run tab).
  // This is the only in-app citation surface that reaches the WASM/web shell — the
  // browser build has no native Help menu (menu.ts attachMenuRouter is a no-op there).
  // The DOI + BibTeX mirror CITATION.cff and the port READMEs — change together.
  import { openExternal } from '$lib/content/render-doc';

  const DOI = '10.5281/zenodo.16502734';
  const DOI_URL = `https://doi.org/${DOI}`;
  const BIBTEX = `@software{mcpower2025,
  author    = {Lenartowicz, Pawe{\\l}},
  title     = {{MCPower}: Monte Carlo Power Analysis for Complex Statistical Models},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {${DOI}},
  url       = {${DOI_URL}}
}`;

  let copied = $state(false);
  async function copyBibtex() {
    await navigator.clipboard.writeText(BIBTEX);
    copied = true;
    setTimeout(() => (copied = false), 1500);
  }
</script>

<div class="mt-6 border-t border-border pt-3 text-xs text-muted-foreground">
  Using MCPower in research? Please cite — Lenartowicz (2025).
  <button class="underline underline-offset-2 hover:text-foreground" onclick={() => openExternal(DOI_URL)}>
    doi:{DOI}
  </button>
  ·
  <button class="underline underline-offset-2 hover:text-foreground" onclick={copyBibtex}>
    {copied ? 'Copied ✓' : 'Copy BibTeX'}
  </button>
</div>
