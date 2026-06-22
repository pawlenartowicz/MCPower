// Demo-mode store: UI state (run/hint state, result tabs) for the guided demo flow.
import type { DemoPowerResult } from '$lib/domain/result';

export type RunState = 'idle' | 'running' | 'done' | 'not-ready';
export type HintState = 'neutral' | 'attention' | 'hidden';
export type SubView = 'bars' | 'curve' | 'table' | 'script';

export interface RunTab {
  id: string;
  label: string;
  kind: 'find-power' | 'find-sample-size';
  subView: SubView;
  result: DemoPowerResult;
}

let tabCounter = 0;
function nextTabId() {
  tabCounter += 1;
  return `tab-${tabCounter}`;
}

function createDemoStore() {
  let runState = $state<RunState>('idle');
  let hintState = $state<HintState>('neutral');
  const progress = $state<{ completed: number; total: number }>({ completed: 0, total: 0 });
  const runTabs = $state<RunTab[]>([]);
  let activeTabId = $state<string | null>(null);

  function pushTab(tab: Omit<RunTab, 'id'>) {
    const t: RunTab = { ...tab, id: nextTabId() };
    runTabs.push(t);
    activeTabId = t.id;
    return t;
  }

  function removeTab(id: string) {
    const idx = runTabs.findIndex((t) => t.id === id);
    if (idx === -1) return;
    runTabs.splice(idx, 1);
    if (activeTabId === id) activeTabId = runTabs[runTabs.length - 1]?.id ?? null;
  }

  function clearTabs() {
    runTabs.length = 0;
    activeTabId = null;
  }

  return {
    get runState() {
      return runState;
    },
    set runState(v: RunState) {
      runState = v;
    },
    get hintState() {
      return hintState;
    },
    set hintState(v: HintState) {
      hintState = v;
    },
    get progress() {
      return progress;
    },
    get runTabs() {
      return runTabs;
    },
    get activeTabId() {
      return activeTabId;
    },
    set activeTabId(v: string | null) {
      activeTabId = v;
    },
    pushTab,
    removeTab,
    clearTabs,
  };
}

export const demoStore = createDemoStore();
