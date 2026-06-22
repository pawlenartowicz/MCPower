// UI store: transient overlay/modal open-state flags (settings, history, acknowledgments, reset-confirm).
function createUiStore() {
  let settingsOpen = $state(false);
  let historyOpen = $state(false);
  let acknowledgmentsOpen = $state(false);
  let resetConfirmOpen = $state(false);

  return {
    get settingsOpen() {
      return settingsOpen;
    },
    set settingsOpen(v: boolean) {
      settingsOpen = v;
    },
    get historyOpen() {
      return historyOpen;
    },
    set historyOpen(v: boolean) {
      historyOpen = v;
    },
    get acknowledgmentsOpen() {
      return acknowledgmentsOpen;
    },
    set acknowledgmentsOpen(v: boolean) {
      acknowledgmentsOpen = v;
    },
    get resetConfirmOpen() {
      return resetConfirmOpen;
    },
    set resetConfirmOpen(v: boolean) {
      resetConfirmOpen = v;
    },
  };
}

export const uiStore = createUiStore();
