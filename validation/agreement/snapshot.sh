#!/usr/bin/env bash
# snapshot.sh <name> — freeze the outputs the paper reads into
# validation/snapshots/<name>/, plus a MANIFEST.md (date, commit, dirty flag,
# per-file sha256 and the JSON meta.packages / lang_version).
#
# Copied: every results/*_r.json here and the <stem>_points.csv /
# <stem>_summary.csv beside it; validation/data/{df_calibration,l3_maxima,
# data_generation_recovery}.csv; benchmarks/results/{py,py_1t,r,r_1t,jl}.json
# when present. Smoke outputs (agreement_smoke*, lme_ex2_smoke*) never match
# *_r.json, and plots are not copied.
#
# Refuses — non-zero exit, nothing written — when the target exists, when the
# files carrying meta.packages.mcpower disagree on the version, or when a JSON's
# meta.n_sims_scale is not 1 (a smoke or scaled run; absent counts as not 1),
# or when a results/*_r.json lacks a <stem>_points.csv or <stem>_summary.csv
# that is newer than the JSON.
# Every check runs before anything is created; the snapshot is built in a temp
# dir and moved into place last.
set -euo pipefail

name=${1:-}
[[ $name =~ ^[A-Za-z0-9._-]+$ && $name != . && $name != .. ]] \
  || { echo "usage: $0 <name>   (letters, digits, . _ -)" >&2; exit 1; }

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
validation=$(dirname "$here")
repo=$(dirname "$validation")
target=$validation/snapshots/$name

die() { echo "snapshot.sh: $*" >&2; exit 1; }

meta() {  # meta <json> n_sims_scale|mcpower|packages|lang_version — "" when absent
  python3 - "$1" "$2" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1])).get("meta", {})
pk = m.get("packages")
v = {"n_sims_scale": m.get("n_sims_scale"),
     "mcpower": (pk or {}).get("mcpower"),
     "packages": json.dumps(pk, separators=(",", ":")) if pk is not None else None,
     "lang_version": m.get("lang_version")}[sys.argv[2]]
print("" if v is None else v)
EOF
}

[[ -e $target ]] && die "$target already exists; a new snapshot needs a new name"

files=()
shopt -s nullglob
for json in "$here"/results/*_r.json; do
  files+=("$json")
  for csv in "${json%.json}_points.csv" "${json%.json}_summary.csv"; do
    [[ -f $csv ]] || die "${csv#"$repo"/}: missing (rerun the agreement script that wrote ${json#"$repo"/})"
    [[ $csv -nt $json ]] || die "${csv#"$repo"/}: not newer than ${json#"$repo"/} (stale CSV)"
    files+=("$csv")
  done
done
for f in df_calibration.csv l3_maxima.csv data_generation_recovery.csv; do
  [[ -f $validation/data/$f ]] || die "missing $validation/data/$f"
  files+=("$validation/data/$f")
done
for f in py py_1t r r_1t jl; do
  [[ -f $repo/benchmarks/results/$f.json ]] && files+=("$repo/benchmarks/results/$f.json")
done

versions=()
for f in "${files[@]}"; do
  [[ $f == *.json ]] || continue
  scale=$(meta "$f" n_sims_scale)
  [[ $scale == 1 || $scale == 1.0 ]] \
    || die "${f#"$repo"/}: meta.n_sims_scale is '${scale}', not 1 (smoke or scaled run)"
  v=$(meta "$f" mcpower)
  [[ -n $v ]] && versions+=("$v ${f#"$repo"/}")
done
if (( ${#versions[@]} )) && [[ $(printf '%s\n' "${versions[@]}" | cut -d' ' -f1 | sort -u | wc -l) -gt 1 ]]; then
  die "mixed meta.packages.mcpower versions:"$'\n'"$(printf '  %s\n' "${versions[@]}")"
fi

# Read before the temp dir exists, so the dirty flag never sees the snapshot itself.
commit=$(git -C "$repo" rev-parse HEAD)
dirty=no
[[ -n $(git -C "$repo" status --porcelain) ]] && dirty=yes

mkdir -p "$validation/snapshots"
tmp=$(mktemp -d "$validation/snapshots/.$name.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
chmod 755 "$tmp"

{
  echo "# Snapshot $name"
  echo
  echo "- Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Commit: $commit"
  echo "- Uncommitted changes (git status --porcelain non-empty): $dirty"
  echo
  echo "| File | Source | sha256 | meta.packages | lang_version |"
  echo "|---|---|---|---|---|"
  for f in "${files[@]}"; do
    base=$(basename "$f")
    cp -p "$f" "$tmp/$base"
    sha=$(sha256sum "$tmp/$base" | cut -d' ' -f1)
    pk=""; lv=""
    if [[ $base == *.json ]]; then
      pk=$(meta "$tmp/$base" packages); lv=$(meta "$tmp/$base" lang_version)
    fi
    echo "| $base | ${f#"$repo"/} | $sha | ${pk:+\`$pk\`} | $lv |"
  done
} > "$tmp/MANIFEST.md"

mv -T "$tmp" "$target"
echo "Wrote ${target#"$repo"/} (${#files[@]} files + MANIFEST.md)"
