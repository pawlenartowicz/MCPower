// Family tutorial markdown, single-sourced from the documentation vault via the $docs
// alias. Vite embeds these at build; dev hot-reloads on edit. No copy, no generated HTML.
import type { Entrypoint } from '$lib/domain/family';
import anova from '$docs/tutorial-app/anova.md?raw';
import regression from '$docs/tutorial-app/regression.md?raw';
import mixed from '$docs/tutorial-app/mixed-models.md?raw';

export const TUTORIAL: Record<Entrypoint, string> = { anova, regression, mixed };
