/** The prefs snapshot seeded into kv['settings.json'].snapshot.
 *  `theme` ('light'|'dark') drives the app's ThemeProvider (toggles the `.dark`
 *  class on <html>, independent of the OS colorScheme). Every config section is
 *  expanded so one element-screenshot of ConfigPanel shows the full
 *  Upload/Model/Correlations/Run form. */
export function captureSnapshot(theme = 'light') {
  return {
    theme,
    fontSize: 17,            // app default (shared-prefs.svelte.ts)
    splitterFraction: 0.5,   // wide config pane so interaction-effect rows (long
                             // `a:b:c` names + INTERACTION badge + stepper + presets)
                             // fit without horizontal clipping
    modelExpanded: true,
    runExpanded: true,
    correlationsExpanded: true,
    uploadExpanded: true,
    activePane: 'config',
    scenariosEnabled: false,
    scriptLanguage: 'python',
    nThreads: null,
  };
}
