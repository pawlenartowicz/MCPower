<script lang="ts">
  // Test-only stub that replaces VegaChart in unit tests. vega-embed needs a
  // real DOM/SVG surface that jsdom does not provide, so component tests that
  // only care about surrounding markup (tables, selectors) mock the chart child
  // to this no-op. Renders a minimal div with data-testid so testid-based
  // queries still work.
  interface Props {
    spec?: string;
    testid?: string;
    labelMap?: Record<string, string>;
  }
  const { testid = 'vega-chart' }: Props = $props();
</script>

<div data-testid={testid}></div>
