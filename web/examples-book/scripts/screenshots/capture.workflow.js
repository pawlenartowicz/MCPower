export const meta = {
  name: 'examples-book-screenshots',
  description: 'Author per-example app configs, validate via the shipped adapter gate, fix, capture ConfigPanel screenshots, reconcile embeds, review.',
  phases: [
    { title: 'Author configs' },
    { title: 'Validate' },
    { title: 'Fix loop' },
    { title: 'Capture' },
    { title: 'Embed + review' },
  ],
};

const ROOT = 'mcpower/web/examples-book';
const SCREENS = `${ROOT}/scripts/screenshots`;

// Phase 1 — author one config per id (fan-out, zero write contention: per-id files).
phase('Author configs');
// The workflow sandbox has no filesystem access, so it cannot run discoverIds()
// itself; the id list is injected. Prefer args.ids (discovered at launch); fall
// back to this snapshot of `discoverIds(chunks)` if args plumbing drops the value.
const ALL_IDS = [
  'anova-01', 'anova-02', 'anova-03', 'anova-04', 'anova-05', 'anova-06',
  'glm-01', 'glm-02', 'glm-03', 'glm-04', 'glm-05', 'glm-06', 'glm-07', 'glm-08', 'glm-09',
  'glmm-01', 'glmm-02', 'glmm-03', 'glmm-04',
  'lmm-01', 'lmm-02', 'lmm-03', 'lmm-04', 'lmm-05', 'lmm-06', 'lmm-07', 'lmm-09',
  'ols-01', 'ols-02', 'ols-03', 'ols-04', 'ols-05', 'ols-06', 'ols-07', 'ols-08', 'ols-09',
  'ols-10', 'ols-11', 'ols-12', 'ols-13', 'ols-14', 'ols-15', 'ols-16', 'ols-18',
];
const ids = args?.ids?.length ? args.ids : ALL_IDS;
const AUTHOR = (id) => `You are writing ONE file and executing NOTHING.
Read ${ROOT}/${id.split('-')[0]}/${id}.md and ${ROOT}/chunks/${id}.py, and the FamilyConfig
interface + per-family default factory in mcpower/ports/app/src/lib/domain/family.ts.
Write ${SCREENS}/${id}.config.json as {"outcomeKind": "...","config": <FamilyConfig>}:
clone the family default config, then set formula/variables/effects/findPower.n (and
cluster for mixed) from the chunk. outcomeKind: 'binary' for glm/glmm else 'continuous'.
CRITICAL — effect names must be the app's CANONICAL effect-row names, because the harness
re-enters effect values through inputs keyed data-testid="effect-<name>". The chunk's
set_effects(...) ALREADY uses these canonical names — copy them verbatim into config.effects.
For reference, the derivation (effect-names.ts expandMainEffect/expandInteraction) is:
  - continuous / binary predictor "x"      -> effect name "x"
  - synthetic factor "g" non-ref level     -> "g[2]", "g[3]", ... (integer labels, level 1 = reference)
  - continuous interaction "a:b"           -> effect name "a:b"
  - factor-involved interaction            -> "g[2]:h[2]" etc. (no bare "g:h" after expansion)
Mirror the two committed fixtures ${SCREENS}/ols-01.config.json
and ${SCREENS}/glmm-01.config.json for the wrapper shape and field defaults
(seed 2137, OLS 1600 / mixed 800 sims, alpha 0.05, targetPower 80, bounds 30->200).
Return the path you wrote. Do not run any command.`;
await parallel(ids.map((id) => () => agent(AUTHOR(id), { label: `author:${id}`, phase: 'Author configs' })));

// Phase 2 — central validation (single agent runs the shipped-adapter gate once).
phase('Validate');
const validateOut = await agent(
  `Run: node ${SCREENS}/capture-screens.mjs --mode validate
Then read ${SCREENS}/validation.json and return JSON {"failed":[ids with ok=false]}.`,
  { label: 'validate', phase: 'Validate', schema: { type: 'object', properties: { failed: { type: 'array', items: { type: 'string' } } }, required: ['failed'] } },
);
let failed = validateOut?.failed ?? [];
if (failed.length > ids.length * 0.5) {
  log(`HALT: ${failed.length}/${ids.length} configs invalid on first pass — systemic, not per-id. Surfacing instead of fanning out.`);
  return { halted: true, failed };
}

// Phase 3 — bounded fix loop (script-owned, <=2 rounds).
phase('Fix loop');
for (let round = 1; round <= 2 && failed.length; round++) {
  await parallel(failed.map((id) => () => agent(
    `Config ${SCREENS}/${id}.config.json failed the familyConfigToAppSpec gate. Read its
errors in ${SCREENS}/validation.json, re-read the page + chunk + family.ts, and rewrite
the config so it builds a valid AppSpec. Write the file only; run nothing.`,
    { label: `fix:${id} (r${round})`, phase: 'Fix loop' },
  )));
  const re = await agent(
    `Run: node ${SCREENS}/capture-screens.mjs --mode validate --id ${failed.join(' --id ')}
Read ${SCREENS}/validation.json; return {"failed":[still-failing ids]}.`,
    { label: `revalidate r${round}`, phase: 'Fix loop', schema: { type: 'object', properties: { failed: { type: 'array', items: { type: 'string' } } }, required: ['failed'] } },
  );
  failed = re?.failed ?? [];
}

// Phase 4 — central capture (single agent, one browser, serial).
phase('Capture');
const green = ids.filter((id) => !failed.includes(id));
await agent(
  `Run: node ${SCREENS}/capture-screens.mjs --mode capture --id ${green.join(' --id ')}
Confirm ${ROOT}/assets/<id>-setup.png exists for each. Return the list of PNGs written.`,
  { label: 'capture', phase: 'Capture' },
);

// Phase 5 — embed reconciliation + per-family review (fan-out per family).
phase('Embed + review');
const families = [...new Set(green.map((id) => id.split('-')[0]))];
await parallel(families.map((fam) => () => agent(
  `For book family "${fam}": for each green id, import { reconcileEmbed } from
${SCREENS}/lib/embed.mjs (via a one-off node -e), apply it to ${ROOT}/${fam}/<id>.md and
write the result back. Then verify every embed resolves to an existing
${ROOT}/assets/<id>-setup.png and the config matches the page's stated analysis.
Report any mismatch. Write files only.`,
  { label: `embed+review:${fam}`, phase: 'Embed + review' },
)));

return { halted: false, failed, green };
