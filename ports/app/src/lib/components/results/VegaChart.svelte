<script lang="ts">
  // Renders a theme-naked Vega-Lite spec (JSON string) from the engine via
  // vega-embed, applying a host theme overlay derived from live CSS vars.
  // Re-embeds on spec change and on theme change; vega-embed is lazy-imported
  // so the ~1 MB Vega bundle loads only when a chart first appears.
  import { onThemeChange } from '$lib/charts/theme';
  import { embedThemed } from '$lib/charts/embed';

  interface Props {
    spec: string;
    testid?: string;
    /** `target_{idx}` → effect-name map; relabels generic plot tokens to real
     *  names on the axes/legend. Omit to leave the engine's tokens as-is. */
    labelMap?: Record<string, string>;
  }
  const { spec, testid = 'vega-chart', labelMap }: Props = $props();

  let host = $state<HTMLDivElement>();
  let view: { finalize: () => void } | null = null;
  let seq = 0; // guards against out-of-order async re-renders

  async function renderChart() {
    if (!host) return;
    const mySeq = ++seq;
    let result: { view: { finalize: () => void }; finalize: () => void };
    try {
      result = await embedThemed(host, spec, labelMap);
    } catch (err) {
      // Malformed/regressed engine spec: surface it instead of leaving a silently
      // empty chart. Still renders nothing (does not throw), so the UI is intact.
      console.warn(`[VegaChart:${testid}] failed to render spec`, err);
      return;
    }
    if (mySeq !== seq || !host) {
      result.view.finalize();
      return; // a newer render superseded this one
    }
    view?.finalize();
    view = result.view;
  }

  $effect(() => {
    void spec; // track: re-render when the spec string changes
    void labelMap; // track: re-render when the label map changes
    void renderChart();
    const off = onThemeChange(() => void renderChart());
    return () => {
      off();
      view?.finalize();
      view = null;
    };
  });
</script>

<div bind:this={host} class="w-full" data-testid={testid}></div>
