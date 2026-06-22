export const meta = {
  name: 'generate-examples-book',
  description: 'Produce the ~40-example recognition-indexed MCPower examples book',
  phases: [
    { title: 'Discover' }, { title: 'Pool review' }, { title: 'Per-family review' },
    { title: 'Gap review' }, { title: 'Map refs' }, { title: 'Author' },
    { title: 'Execute' }, { title: 'Fix loop' }, { title: 'Assemble' },
    { title: 'Family review' },
  ],
}

// Absolute paths so spawned agents are cwd-independent (the Workflow runtime's
// base dir differs from the agents' Bash/Write/Read cwd).
const REPO = '/home/plenartowicz/Projekty/MCPower-Project'
const BOOK = `${REPO}/mcpower/web/examples-book`

const CONVENTIONS = `
AUTHORING CONVENTIONS (load-bearing, follow verbatim):
- Only public class is MCPower (R: MCPower$new(...)). Never LinearRegression / PowerAnalysis.
- Realistic, domain-specific variable names — never x1/y/group. Rotate three domains across
  the book: clinical/health, ecology/biology, social science (NO education/psych framing like
  test_score/study_hours; educational ATTAINMENT — graduate, years_education — is social science).
  Vary names within a domain: not every continuous predictor is "dose".
- Every page carries a "Same design, other fields" sub-list in ## Variations: the IDENTICAL model
  structure in the OTHER TWO domains, as prose-only formulas (cross-domain twins).
- Factor dummies are integer-indexed (treatment[2], habitat[3]) because every chunk uses simulated
  data, not uploads. The data-value convention (cyl[6], origin[Japan]) is for UPLOAD examples only —
  there are none here; an effect name with a data-value label would fail validation.
- '*' expands to a + b + a:b; never also write a*b. Use ':' for interaction-only.
- No kernel internals: say "native engine"; never mcpower[JIT] or "pure Python".
- Effect benchmarks: continuous 0.10/0.25/0.40; binary-or-factor 0.20/0.50/0.80.
- Sim defaults: OLS 1600 sims; mixed 800; alpha 0.05; target power 80%; seed 2137.
- Wikilinks stay inside the book vault: [[family/id|alias]]. Never link to docs/ or source.
- Effect sizes are FABRICATED-PLAUSIBLE on the benchmark scale, never a study's raw numbers.
- Page order (see scripts/PAGE_ANATOMY.md): title+formula -> ## Variations (with the twins block)
  -> ## Not this setup? -> ## If you'd rather have… -> ## Copy-paste setup (Python then R, at the
  BOTTOM) -> screenshot placeholder.
`

const EXPRESSIBILITY = `
ENGINE EXPRESSIBILITY (verify against the LIVE validation source — the charter's
"single-grouping" line is STALE; trust the code, not the docs).
All paths below are under /home/plenartowicz/Projekty/MCPower-Project/mcpower/ :
- crates/engine-spec-builder/src/formula.rs       (formula parse surface)
- crates/engine-spec-builder/src/validate.rs       (factor/binary/effect bounds)
- crates/engine-spec-builder/src/error.rs           (SpecError variants)
- crates/engine-contract/src/validate.rs            (invariants 18-21: cluster/slope/grouping)
- crates/engine-contract/src/error.rs               (ContractError variants)
Known bounds to honour: estimators OLS / GLM(binary, logit) / MLE(needs exactly one
cluster); factors 2-20 levels; no term removal ('-'), no functions (log(x), I(x^2));
correlations continuous-only and PSD; posthoc OLS-only. Mixed models are the trap:
a random slope needs a random intercept (tau^2 > 0) AND its predictor present as a
direct fixed effect; <= 1 nested grouping level; crossed grouping needs FixedClusters.
A formula only survives if it round-trips through this validation. When in doubt, DROP it
-- Phase 7 execution is the hard backstop and unexpressible formulas fail there.
`

const POOL_SCHEMA = { type: 'object', required: ['candidates'], properties: { candidates: { type: 'array', items: {
  type: 'object', required: ['family', 'title', 'formula', 'recognition_hook'], properties: {
    family: { type: 'string', enum: ['anova', 'ols', 'glm', 'lmm', 'glmm'] },
    title: { type: 'string' }, formula: { type: 'string' }, recognition_hook: { type: 'string' } } } } } }

const SELECTION_SCHEMA = { type: 'object', required: ['selected'], properties: { selected: { type: 'array', items: {
  type: 'object', required: ['id', 'family', 'title', 'formula', 'estimator', 'expressible'], properties: {
    id: { type: 'string' }, family: { type: 'string', enum: ['anova', 'ols', 'glm', 'lmm', 'glmm'] },
    title: { type: 'string' }, formula: { type: 'string' }, estimator: { type: 'string', enum: ['ols', 'glm', 'mle'] },
    expressible: { type: 'boolean' }, note: { type: 'string' } } } } } }

const REFMAP_SCHEMA = { type: 'object', required: ['id', 'not_this_setup', 'if_youd_rather'], properties: {
  id: { type: 'string' },
  not_this_setup: { type: 'array', items: { type: 'string' } },
  if_youd_rather: { type: 'array', items: { type: 'object', required: ['id', 'why'],
    properties: { id: { type: 'string' }, why: { type: 'string' } } } } } }

const AUTHORED_SCHEMA = { type: 'object', required: ['id', 'page_path', 'chunk_py', 'chunk_r'], properties: {
  id: { type: 'string' }, page_path: { type: 'string' }, chunk_py: { type: 'string' }, chunk_r: { type: 'string' } } }

const EXEC_SCHEMA = { type: 'object', required: ['results'], properties: { results: { type: 'object',
  additionalProperties: { type: 'object', required: ['ok'], properties: { ok: { type: 'boolean' }, err: { type: ['string', 'null'] } } } } } }

const FAMILY_REVIEW_SCHEMA = { type: 'object', required: ['family', 'refs_ok', 'snippets_ok', 'issues'], properties: {
  family: { type: 'string' }, refs_ok: { type: 'boolean' }, snippets_ok: { type: 'boolean' },
  issues: { type: 'array', items: { type: 'string' } } } }

phase('Discover')
// Candidate pool of 3-4x target across analysis TYPES; domain is rotated at authoring time
// (clinical/ecology/social), so discover by analysis type and let the rotation be assigned later.
const TARGETS = { anova: 5, ols: 15, glm: 8, lmm: 6, glmm: 6 }  // targets, not quotas (spec 2.2)
const pools = await parallel(Object.entries(TARGETS).map(([fam, n]) => () =>
  agent(`Web-search the power analyses students and early researchers actually run that map to
the ${fam.toUpperCase()} family. Build a candidate pool of ~${n * 4} distinct ANALYSIS TYPES.
Recognition hooks come from real studies; effect sizes are fabricated-plausible. The book rotates
three domains (clinical/health, ecology/biology, social science) across the final set — keep the
candidates analysis-type-focused; domain assignment happens at authoring.
${CONVENTIONS}`, { label: `discover:${fam}`, phase: 'Discover', schema: POOL_SCHEMA })))
const pool = pools.filter(Boolean).flatMap(p => p.candidates)
log(`pool: ${pool.length} candidates`)

phase('Pool review')
// Cull dedupes AND culls formulas the engine can't express, checked against live source.
const culled = await agent(`Here is the full candidate pool as JSON:
${JSON.stringify(pool)}
Dedupe toward analysis-TYPE coverage and CULL every formula the engine cannot express.
Read the validation source to decide expressibility. Assign each survivor a stable kebab id
(<family>-NN). ${EXPRESSIBILITY}`, { label: 'cull-pool', phase: 'Pool review', schema: SELECTION_SCHEMA })

phase('Per-family review')
const selectionResult = await agent(`From this expressibility-checked set:
${JSON.stringify(culled.selected)}
Select the final ~40 across families. Targets (rebalance as the pool dictates):
ANOVA 5, OLS 15, GLM 8, LMM 6, GLMM 6. Recognition is the spine; feature coverage is a
checklist satisfied across the set; rotate the three domains (clinical/health, ecology/biology,
social science) roughly evenly across the selection — each domain should appear in every family
where it fits naturally.`,
  { label: 'per-family-select', phase: 'Per-family review', schema: SELECTION_SCHEMA })

phase('Gap review')
const finalSel = await agent(`Final gap pass on this selection:
${JSON.stringify(selectionResult.selected)}
Available pool to backfill from: ${JSON.stringify(culled.selected)}
Add or swap to cover any missing analysis type or MCPower feature. Keep ~40 total, all
expressible. Return the final selected list.`,
  { label: 'gap-review', phase: 'Gap review', schema: SELECTION_SCHEMA })
const selected = finalSel.selected.filter(s => s.expressible)
log(`selected: ${selected.length}`)

const authored = await pipeline(
  selected,
  (s) => agent(`Map cross-links for example "${s.id}" (${s.title}, ${s.formula}).
Full selected set: ${JSON.stringify(selected.map(x => ({ id: x.id, family: x.family, title: x.title, formula: x.formula })))}
Pick the nearest OTHER ids for "Not this setup?" and a few "If you'd rather have…" alternatives.`,
    { label: `refs:${s.id}`, phase: 'Map refs', schema: REFMAP_SCHEMA })
    .then(refs => ({ s, refs })),

  ({ s, refs }) => agent(`Author the example page and code chunks for "${s.id}".
Read ${BOOK}/scripts/PAGE_ANATOMY.md and follow it exactly. Model: ${s.formula} (estimator ${s.estimator}).
Cross-links to embed: ${JSON.stringify(refs)}.
WRITE three files, do NOT execute anything:
  1. ${BOOK}/${s.family}/${s.id}.md  -- full page with EMPTY chunk:py / chunk:r marker fences.
  2. ${BOOK}/chunks/${s.id}.py        -- self-contained runnable Python (imports + build chain +
     one find_power/find_sample_size call; no print, no plot).
  3. ${BOOK}/chunks/${s.id}.R         -- the R mirror.
The chunk files are the single source for the page's fenced blocks (injected later).
${CONVENTIONS}`,
    { label: `author:${s.id}`, phase: 'Author', schema: AUTHORED_SCHEMA }),
)
const ids = authored.filter(Boolean).map(a => a.id)
log(`authored: ${ids.length} pages`)

// A storm of rate-limit/API failures kills author agents BEFORE execution, leaving
// `ids` empty -- the >50% execute-time guard can't see that (0/max(1,0)=0) and the
// run would report a false green over an empty book. Catch the systemic authoring
// wipeout here and halt honestly instead.
const authorFailed = selected.filter(s => !ids.includes(s.id)).map(s => s.id)
if (ids.length === 0 || authorFailed.length > selected.length * 0.5) {
  log(`HALT: ${authorFailed.length}/${selected.length} authoring agents produced no page (likely systemic/rate-limit). Not executing.`)
  return { selected: selected.map(s => s.id), green: false, residual_failures: authorFailed, counts: countByFamily(selected), halted: true, stage: 'author' }
}

// Python-only validation gate (the R port's chunks are still authored and shown
// on the pages, but executing them needs a cold Rust rebuild that risks OOM, so
// validation is Python-only — see ../scripts/validate.py).
async function runExecutors(onlyIds) {
  const arg = onlyIds && onlyIds.length ? ' ' + onlyIds.join(' ') : ''
  const py = await agent(`Run the Python validator ONCE. Steps, exactly:
  1. Activate the workspace .venv: source ${REPO}/.venv/bin/activate
  2. In the SAME shell command, run: python ${BOOK}/scripts/validate.py${arg}
     Allow a long timeout (~600s) -- ~40 chunks run full simulation counts
     sequentially; do NOT kill it as a hang.
  3. Read ${BOOK}/results.json and return its parsed contents as {results: {...}}.
Do NOT run maturin or any build; do NOT parallelise.`,
    { label: 'exec:py', phase: 'Execute', schema: EXEC_SCHEMA })
  const scope = onlyIds && onlyIds.length ? onlyIds : ids
  const merged = {}
  for (const id of scope) {
    merged[id] = { ok: py.results?.[id]?.ok ?? false, err: py.results?.[id]?.err || null }
  }
  return merged
}

phase('Execute')
let results = await runExecutors(null)
let failed = Object.entries(results).filter(([, v]) => !v.ok).map(([id]) => id)
const failRate = failed.length / Math.max(1, ids.length)
if (failRate > 0.5) {
  log(`HALT: ${failed.length}/${ids.length} failed (>50%) -- likely systemic/environmental, not per-id.`)
  return { selected: selected.map(s => s.id), green: false, residual_failures: failed, counts: countByFamily(selected), halted: true }
}

phase('Fix loop')
const MAX_ROUNDS = 2
for (let round = 1; round <= MAX_ROUNDS && failed.length; round++) {
  log(`fix round ${round}: ${failed.length} failing -> ${failed.join(', ')}`)
  await parallel(failed.map(id => () => {
    const s = selected.find(x => x.id === id)
    return agent(`Chunk "${id}" failed validation: ${JSON.stringify(results[id].err)}.
Model: ${s.formula} (estimator ${s.estimator}). Rewrite ${BOOK}/chunks/${id}.py and
${BOOK}/chunks/${id}.R so they run without error. Fix the CODE only; do NOT execute anything.
If the formula is fundamentally inexpressible, simplify it to the nearest expressible model and
note the change at the top of the chunk as a comment. ${EXPRESSIBILITY}`,
      { label: `fix:${id}`, phase: 'Fix loop' })
  }))
  // A subset executor run only reports the ids it ran, so overwriting `results`
  // wholesale would drop every already-passing id back to failed. Merge in
  // place, then recompute `failed` over the full persistent map.
  Object.assign(results, await runExecutors(failed))
  failed = ids.filter(id => !results[id].ok)
}

phase('Assemble')
// Inject validated chunks into page fences, then populate family example-indexes.
await agent(`Run: python ${BOOK}/scripts/inject_chunks.py
This fills every page's chunk fences from the validated chunk files. Then, for each family
(anova, ols, glm, lmm, glmm), replace the <!-- examples-index --> ... <!-- /examples-index -->
block in ${BOOK}/<family>/index.md with a bullet list of that family's examples as wikilinks
[[<family>/<id>|<title>]], using this selection: ${JSON.stringify(selected.map(s => ({ id: s.id, family: s.family, title: s.title })))}.
Do not touch any other content.`, { label: 'assemble', phase: 'Assemble' })

phase('Family review')
const reviews = await parallel(['anova', 'ols', 'glm', 'lmm', 'glmm'].map(fam => () =>
  agent(`Review the ${fam} family of the examples book at ${BOOK}/${fam}/.
Confirm: every wikilink resolves to a real page in the book; each page follows PAGE_ANATOMY.md;
each page's chunk fences are non-empty. Read ${BOOK}/results.json
and confirm this family's ids are ok:true. Report issues.`,
    { label: `review:${fam}`, phase: 'Family review', schema: FAMILY_REVIEW_SCHEMA })))

function countByFamily(items) {
  const c = {}
  for (const s of items) c[s.family] = (c[s.family] || 0) + 1
  return c
}

return {
  selected: selected.map(s => s.id),
  green: failed.length === 0,
  residual_failures: failed,
  counts: countByFamily(selected),
  reviews: reviews.filter(Boolean),
}
