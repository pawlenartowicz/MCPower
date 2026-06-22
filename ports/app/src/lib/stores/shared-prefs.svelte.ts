// Shared-prefs store: user preferences (theme, font size, pane layout) persisted via the settings layer with debounced saves.
import { loadSettings, saveSettings, type SharedPrefsSnapshot } from '$lib/persistence/settings';

export type Theme = 'light' | 'dark' | 'system';

function createSharedPrefs() {
  let theme = $state<Theme>('system');
  let fontSize = $state<number>(17);
  let splitterFraction = $state<number>(0.33);
  let modelExpanded = $state<boolean>(true);
  let runExpanded = $state<boolean>(false);
  let correlationsExpanded = $state<boolean>(false);
  let uploadExpanded = $state<boolean>(false);
  let activePane = $state<'config' | 'results'>('config');
  let scenariosEnabled = $state<boolean>(false);
  let scriptLanguage = $state<'python' | 'r'>('python');
  // null = use all cores (Tauri default); positive integer = fixed pool size.
  // Tauri-only: the WASM shell ignores this field (single-core workers by design).
  let nThreads = $state<number | null>(null);
  let ready = $state<boolean>(false);
  let saveTimer: ReturnType<typeof setTimeout> | null = null;

  function snapshot(): SharedPrefsSnapshot {
    return {
      theme,
      fontSize,
      splitterFraction,
      modelExpanded,
      runExpanded,
      correlationsExpanded,
      uploadExpanded,
      activePane,
      scenariosEnabled,
      scriptLanguage,
      nThreads,
    };
  }

  function scheduleSave() {
    if (!ready) return; // don't save during initial hydration
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      void saveSettings(snapshot());
    }, 200);
  }

  void loadSettings()
    .then((snap) => {
      // 'wild' was the dark-pink variant name; renamed to 'dark'. Migrate old snapshots.
      if (snap.theme !== undefined) theme = (snap.theme as string) === 'wild' ? 'dark' : snap.theme;
      if (snap.fontSize !== undefined) fontSize = snap.fontSize;
      if (snap.splitterFraction !== undefined) splitterFraction = snap.splitterFraction;
      if (snap.modelExpanded !== undefined) modelExpanded = snap.modelExpanded;
      if (snap.runExpanded !== undefined) runExpanded = snap.runExpanded;
      if (snap.correlationsExpanded !== undefined) correlationsExpanded = snap.correlationsExpanded;
      if (snap.uploadExpanded !== undefined) uploadExpanded = snap.uploadExpanded;
      if (snap.activePane !== undefined) activePane = snap.activePane;
      if (snap.scenariosEnabled !== undefined) scenariosEnabled = snap.scenariosEnabled;
      if (snap.scriptLanguage !== undefined) scriptLanguage = snap.scriptLanguage;
      if (snap.nThreads !== undefined) nThreads = snap.nThreads;
    })
    .finally(() => {
      ready = true;
    });

  return {
    get theme() {
      return theme;
    },
    set theme(v: Theme) {
      theme = v;
      scheduleSave();
    },
    get fontSize() {
      return fontSize;
    },
    set fontSize(v: number) {
      fontSize = v;
      scheduleSave();
    },
    get splitterFraction() {
      return splitterFraction;
    },
    set splitterFraction(v: number) {
      splitterFraction = v;
      scheduleSave();
    },
    get modelExpanded() {
      return modelExpanded;
    },
    set modelExpanded(v: boolean) {
      modelExpanded = v;
      scheduleSave();
    },
    get runExpanded() {
      return runExpanded;
    },
    set runExpanded(v: boolean) {
      runExpanded = v;
      scheduleSave();
    },
    get correlationsExpanded() {
      return correlationsExpanded;
    },
    set correlationsExpanded(v: boolean) {
      correlationsExpanded = v;
      scheduleSave();
    },
    get uploadExpanded() {
      return uploadExpanded;
    },
    set uploadExpanded(v: boolean) {
      uploadExpanded = v;
      scheduleSave();
    },
    get activePane() {
      return activePane;
    },
    set activePane(v: 'config' | 'results') {
      activePane = v;
      scheduleSave();
    },
    get scenariosEnabled() {
      return scenariosEnabled;
    },
    set scenariosEnabled(v: boolean) {
      scenariosEnabled = v;
      scheduleSave();
    },
    get scriptLanguage() {
      return scriptLanguage;
    },
    set scriptLanguage(v: 'python' | 'r') {
      scriptLanguage = v;
      scheduleSave();
    },
    get nThreads() {
      return nThreads;
    },
    set nThreads(v: number | null) {
      nThreads = v;
      scheduleSave();
    },
    get ready() {
      return ready;
    },
  };
}

export const sharedPrefs = createSharedPrefs();
