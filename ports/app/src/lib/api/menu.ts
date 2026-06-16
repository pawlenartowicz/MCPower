// Menu event router: maps Tauri native-menu event IDs to store actions and async run commands.
import { familyStore } from '$lib/stores/family.svelte';
import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
import { uiStore } from '$lib/stores/ui.svelte';
import { runStore } from '$lib/stores/run.svelte';
import { familyConfigToAppSpec } from '$lib/domain/app-spec-adapter';
import { openExternal, DOCS_BASE_URL } from '$lib/content/render-doc';
import { SIMULATION } from '$lib/configs/app-config';

type UnlistenFn = () => void;

export function routeMenuEvent(id: string): void {
  switch (id) {
    case 'view.settings':        uiStore.settingsOpen = true; return;
    case 'view.history':         uiStore.historyOpen = true; return;
    case 'help.documentation':   openExternal(`${DOCS_BASE_URL}/tutorial-app/index`); return;
    case 'help.acknowledgments': uiStore.acknowledgmentsOpen = true; return;
    case 'edit.reset_family':    uiStore.resetConfirmOpen = true; return;
    case 'file.new':             familyStore.resetActive(); return;
    case 'view.toggle_config':   sharedPrefs.activePane = 'config'; return;
    case 'view.toggle_results':  sharedPrefs.activePane = 'results'; return;
    case 'run.find_power':       void startFindPowerFromShortcut(); return;
    case 'run.find_n':           void startFindSampleSizeFromShortcut(); return;
    case 'run.cancel':           void runStore.cancel(); return;
    case 'run.rerun':            void runStore.replayLast(); return;
    case 'file.export_results':  void exportResults(); return;
    default: console.log(`menu: ${id} (no-op)`);
  }
}

async function startFindPowerFromShortcut() {
  if (familyStore.active !== 'regression') return;
  const cfg = familyStore.byFamily.regression;
  const { spec, errors } = familyConfigToAppSpec(
    'regression', cfg, familyStore.regressionOutcome,
  );
  if (!spec) { console.warn(errors[0] ?? 'cannot run yet'); return; }
  await runStore.startFindPower(spec, cfg.findPower.n);
}

async function startFindSampleSizeFromShortcut() {
  if (familyStore.active !== 'regression') return;
  const cfg = familyStore.byFamily.regression;
  const { spec, errors } = familyConfigToAppSpec(
    'regression', cfg, familyStore.regressionOutcome,
  );
  if (!spec) { console.warn(errors[0] ?? 'cannot run yet'); return; }
  const ssb = cfg.findSampleSize;
  const by =
    ssb.by === 'auto'
      ? { Auto: { count: SIMULATION.cluster_auto_count } }
      : { Fixed: ssb.by };
  await runStore.startFindSampleSize(
    spec,
    [ssb.from, ssb.to],
    { Grid: { by, mode: 'Linear' } },
  );
}

async function exportResults() {
  const active = runStore.runTabs.find((t) => t.id === runStore.activeTabId);
  if (!active) {
    console.warn('export: no active tab');
    return;
  }
  const filename = `${active.label.replace(/\s+/g, '-').toLowerCase()}.json`;
  const payload = {
    label: active.label,
    kind: active.kind,
    spec: active.spec,
    sample_size: active.sample_size,
    bounds: active.bounds,
    method: active.method,
    effect_names: active.effect_names,
    result: active.result,
  };
  if (import.meta.env.VITE_TARGET === 'wasm') {
    const { downloadJson } = await import('./menu-wasm');
    downloadJson(filename, payload);
    return;
  }
  try {
    const { save } = await import('@tauri-apps/plugin-dialog');
    const { writeTextFile } = await import('@tauri-apps/plugin-fs');
    const path = await save({
      filters: [{ name: 'JSON', extensions: ['json'] }],
      defaultPath: filename,
    });
    if (!path) return; // user cancelled
    await writeTextFile(path, JSON.stringify(payload, null, 2));
  } catch (err) {
    console.error('export failed:', err);
  }
}

export async function attachMenuRouter(): Promise<UnlistenFn> {
  if (import.meta.env.VITE_TARGET === 'wasm') return () => {};
  const { listen } = await import('@tauri-apps/api/event');
  return listen<string>('menu', ({ payload }) => routeMenuEvent(payload));
}
