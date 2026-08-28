// Wording for the sample-size unit — the single spelling source for the unit
// chips, the echo line under the from/to/step row, the required-N table column,
// the sample-size curve's x-axis title and the status-bar run summary. Every
// surface that names the unit reads it from here so the app never says
// "participants" in one place and "subject clusters" in another.
//
// The unit is the divisor between total observations N and the quantity the
// user plans in: with a fixed cluster size they plan primary clusters, with a
// fixed cluster count they plan observations per cluster. The engine searches
// and reports N either way; the labels below are the host layer over it.

/** Wire values of the engine's sample-size unit (`SampleSizeUnit`). */
export type SampleSizeUnit = 'observations' | 'clusters' | 'cluster_size';

/** Which cluster knob the user fixed (`ClusterConfig.dimKind`). */
export type ClusterDimKind = 'n_clusters' | 'cluster_size';

/** A run of label text; `mono` marks the cluster name, rendered in the mono
 *  face the Clusters card uses for it. */
export interface LabelPart {
  text: string;
  mono?: boolean;
}

/** Flatten label parts to plain text (chip `aria`/tests, status bar). */
export function partsText(parts: LabelPart[]): string {
  return parts.map((p) => p.text).join('');
}

/**
 * The one derived unit a cluster dimension can express. A fixed cluster size
 * makes the number of clusters the free quantity; a fixed cluster count makes
 * the observations per cluster the free one.
 */
export function derivedUnit(dimKind: ClusterDimKind): SampleSizeUnit {
  return dimKind === 'cluster_size' ? 'clusters' : 'cluster_size';
}

/** Chip label: a generic noun, never the cluster name — the chips sit in a
 *  narrow row and the name has room in the echo line under them. The derived
 *  nouns borrow the Clusters card's own vocabulary for the free quantity. */
export function unitChipParts(unit: SampleSizeUnit): LabelPart[] {
  if (unit === 'clusters') return [{ text: 'cluster number' }];
  if (unit === 'cluster_size') return [{ text: 'cluster size' }];
  return [{ text: 'observations' }];
}

/**
 * A sample-size bound given in `unit`, expressed in total observations.
 * `fixed` is the Clusters card value the unit is measured against
 * (`clusterSize` under a fixed size, `nClusters` under a fixed count); it is
 * unused for `observations`, where the bound is already a total.
 */
export function boundToObservations(unit: SampleSizeUnit, bound: number, fixed: number): number {
  return unit === 'observations' ? bound : bound * fixed;
}

/** The unit as a plain noun phrase: "observations", "subject clusters",
 *  "observations per subject". Used for the table header, the axis title and
 *  the status-bar summary. */
export function unitNoun(unit: SampleSizeUnit, clusterName: string): string {
  if (unit === 'clusters') return `${clusterName} clusters`;
  if (unit === 'cluster_size') return `observations per ${clusterName}`;
  return 'observations';
}

/** Required-N table column header for the unit column (shown before the N column). */
export function unitColumnHeader(unit: SampleSizeUnit, clusterName: string): string {
  return `Required ${unitNoun(unit, clusterName)}`;
}

/** x-axis title for the sample-size curve. The engine emits no title once the
 *  result carries a divisor, so the host must set one on every layer. */
export function unitAxisTitle(unit: SampleSizeUnit, clusterName: string): string {
  const noun = unitNoun(unit, clusterName);
  return noun.charAt(0).toUpperCase() + noun.slice(1);
}

function num(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}

/**
 * The from/to range restated in the other unit, from the Clusters card's own
 * fixed value (`clusterSize` under a fixed size, `nClusters` under a fixed
 * count). `from`/`to` are in `unit`; `fixed` is that card value. Returns an
 * empty array when there is nothing to restate (no usable fixed value).
 */
export function unitEchoParts(
  unit: SampleSizeUnit,
  dimKind: ClusterDimKind,
  clusterName: string,
  from: number,
  to: number,
  fixed: number,
): LabelPart[] {
  if (!Number.isFinite(fixed) || fixed < 1 || !Number.isFinite(from) || !Number.isFinite(to)) {
    return [];
  }
  const name: LabelPart = { text: clusterName, mono: true };
  // The multiplying branches are exact, the dividing ones are rounded — hence
  // "=" above and "≈" below.
  const obs = (b: number) => num(boundToObservations(unit, b, fixed));
  if (unit === 'clusters') {
    // Bounds are clusters → observations, and the fixed size per cluster.
    return [{ text: `= ${obs(from)} to ${obs(to)} observations, ${num(fixed)} per ` }, name];
  }
  if (unit === 'cluster_size') {
    // Bounds are observations per cluster → observations, over the fixed count.
    return [
      { text: `= ${obs(from)} to ${obs(to)} observations, ${num(fixed)} ` },
      name,
      { text: ' clusters' },
    ];
  }
  // Bounds are observations → the derived quantity the fixed knob leaves free.
  if (dimKind === 'cluster_size') {
    return [
      { text: `≈ ${num(from / fixed)} to ${num(to / fixed)} ` },
      name,
      { text: ` clusters, ${num(fixed)} observations each` },
    ];
  }
  return [
    { text: `≈ ${num(from / fixed)} to ${num(to / fixed)} observations per ` },
    name,
    { text: `, ${num(fixed)} ` },
    name,
    { text: ' clusters' },
  ];
}
