"""Theme helpers and spec-emit wrappers for the Vega-Lite specs emitted by the engine.

The engine emits theme-naked specs (data + encoding only); themes live in
``mcpower/configs/plot-themes.json``, are embedded into the compiled extension
at build time, and exposed via ``_engine.list_plot_themes`` / ``_engine.plot_theme``.
``_apply_theme`` grafts a chosen theme onto a spec's ``config`` block.

The engine exposes plot-set functions that return ordered ``(block_key, spec_json)``
pairs. This module builds the neutral envelope from result dicts, calls those
functions, applies host-side post-emit rewrites (relabelling, CI styling, correction
axis note), and routes blocks to files or a stacked HTML page.

Pure stdlib — no plotting dependency.
"""

from __future__ import annotations

import json
import os
import re
import sys
import webbrowser
from typing import Any, Dict, List, Optional, Tuple


def _next_free_path(path: str) -> str:
    """Return `path` if free, else `<stem>_2.<ext>`, `_3`, … so successive
    plots never clobber an earlier save."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{stem}_{i}{ext}"):
        i += 1
    return f"{stem}_{i}{ext}"


def _is_headless() -> bool:
    """True when no display is available (CI / SSH / headless)."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _open_or_report(path: str) -> str:
    """Open `path` in the OS default viewer unless headless; return a message."""
    if _is_headless():
        return f"Wrote {path} (no display detected — open it manually)."
    try:
        webbrowser.open(f"file://{os.path.abspath(path)}")
    except Exception:
        return f"Wrote {path} (could not auto-open — open it manually)."
    return f"Wrote {path} and opened it in your browser."


def available_themes() -> List[str]:
    """Names of the embedded plot themes, in declaration order."""
    from .. import _engine

    return _engine.list_plot_themes()


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (mutating ``base``).

    Nested dicts merge key-by-key so ``axis.*`` sub-keys combine rather than
    the whole ``axis`` block being clobbered; non-dict values overwrite.
    """
    for key, value in overlay.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(existing, value)
        else:
            base[key] = value
    return base


_PLOT_FORMATS = ("png", "svg", "pdf", "html")


def _render(spec: Dict[str, Any], path: str, *, scale: float = 2.0, ppi: Optional[float] = None) -> None:
    """Render a parsed Vega-Lite ``spec`` dict to ``path`` (format by suffix).

    PNG/PDF write bytes; SVG writes text. Requires the optional ``vl-convert``
    renderer (``pip install mcpower[plot]``); raises ImportError with that hint if
    absent. Raises ValueError for an unsupported suffix.
    """
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if suffix not in _PLOT_FORMATS or suffix == "html":
        raise ValueError(
            f"unsupported plot format '.{suffix}'; use one of: {', '.join(f for f in _PLOT_FORMATS if f != 'html')}"
        )
    try:
        import vl_convert as vlc
    except ImportError as e:  # no silent downgrade — there is no spec-file output
        raise ImportError(
            "saving plots needs the renderer: pip install mcpower[plot]"
        ) from e

    if suffix == "png":
        kwargs: Dict[str, Any] = {"scale": scale}
        if ppi is not None:
            kwargs["ppi"] = ppi
        data = vlc.vegalite_to_png(spec, **kwargs)
        with open(path, "wb") as f:
            f.write(data)
    elif suffix == "svg":
        with open(path, "w") as f:
            f.write(vlc.vegalite_to_svg(spec))
    elif suffix == "pdf":
        data = vlc.vegalite_to_pdf(spec)
        with open(path, "wb") as f:
            f.write(data)


def _apply_theme(spec_json: str, theme: str) -> str:
    """Merge the named theme into ``spec["config"]`` and return the new JSON.

    ``spec_json`` is a theme-naked Vega-Lite spec (as emitted by the engine);
    ``theme`` is one of :func:`available_themes`. Raises ``ValueError`` (from
    the engine) for an unknown theme name.
    """
    spec = json.loads(spec_json)
    from .. import _engine

    theme_config = json.loads(_engine.plot_theme(theme))
    config = spec.setdefault("config", {})
    _deep_merge(config, theme_config)
    _style_ci_marks(spec, config.get("axis", {}).get("titleColor", CI_DEFAULT_COLOR))
    return json.dumps(spec)


CI_DEFAULT_COLOR = "#333333"  # readable on the default white plot background


def _style_ci_marks(spec: Dict[str, Any], color: str) -> None:
    """In place: make errorbar CIs legible. The engine emits errorbar marks with
    no colour, so they inherit the bar colour and vanish. Vega-Lite forbids
    ``color`` in ``config.errorbar``, so the contrasting colour is set on the mark
    itself — host-applied colour/size, part of the host theme overlay. End ticks
    are enabled so interval bounds are visible. Single-series error bars (no
    ``color`` encoding) get a foreground whisker; grouped/multi-series error bars
    keep their per-series colour and just gain ticks."""
    _walk_ci(spec, color)


def _walk_ci(node: Any, color: str) -> None:
    if isinstance(node, list):
        for child in node:
            _walk_ci(child, color)
        return
    if not isinstance(node, dict):
        return
    mark = node.get("mark")
    mark_type = mark if isinstance(mark, str) else (mark.get("type") if isinstance(mark, dict) else None)
    if mark_type == "errorbar":
        md = {"type": "errorbar"} if isinstance(mark, str) else dict(mark)
        has_color_enc = isinstance(node.get("encoding"), dict) and node["encoding"].get("color") is not None
        if has_color_enc:
            md["ticks"] = True
        else:
            md["ticks"] = {"color": color}
            md["rule"] = {"color": color, "strokeWidth": 1.5}
        node["mark"] = md
    for value in node.values():
        _walk_ci(value, color)


def _relabel_targets(spec: Dict[str, Any], label_map: Dict[str, str]) -> None:
    """In place: rewrite every ``target`` data-field value via ``label_map``
    (``target_{idx}`` → effect name).

    The engine emits generic ``target_{idx}`` tokens because effect labels are
    host-owned; this overlays the host's real names — the same post-emit,
    host-applied pattern as :func:`_apply_theme`. Walks the whole spec (nested
    layers, ``vconcat``, ``facet``, marker data) so bar, curve, joint and
    composite specs are all covered. No-op for an empty map. Vega-Lite keys the
    axis/legend off the field value, so relabelling the data relabels the chart
    without touching the engine's encoding."""
    if not label_map:
        return
    _walk_relabel(spec, label_map)


def _walk_relabel(node: Any, label_map: Dict[str, str]) -> None:
    if isinstance(node, list):
        for child in node:
            _walk_relabel(child, label_map)
        return
    if not isinstance(node, dict):
        return
    data = node.get("data")
    if isinstance(data, dict) and isinstance(data.get("values"), list):
        for row in data["values"]:
            if isinstance(row, dict):
                token = row.get("target")
                if isinstance(token, str) and token in label_map:
                    row["target"] = label_map[token]
    for value in node.values():
        _walk_relabel(value, label_map)


def _rewrite_correction_axis_title(spec: Dict[str, Any], correction_name: str) -> None:
    """In place: rewrite every encoding axis title equal to exactly ``"Power"``
    to ``"Power (<Correction>-corrected)"`` where the correction name has its
    first letter capitalised.

    Joint-curve titles (``"P(detect >= k)"``, ``"P(exactly k)"``) are not
    touched — only the literal ``"Power"`` axis title is rewritten.
    """
    cap = correction_name[0].upper() + correction_name[1:] if correction_name else correction_name
    new_title = f"Power ({cap}-corrected)"
    _walk_correction_title(spec, new_title)


def _walk_correction_title(node: Any, new_title: str) -> None:
    if isinstance(node, list):
        for child in node:
            _walk_correction_title(child, new_title)
        return
    if not isinstance(node, dict):
        return
    encoding = node.get("encoding")
    if isinstance(encoding, dict):
        for _field_key, enc in encoding.items():
            if isinstance(enc, dict):
                # Title may be directly on the field encoding dict
                if enc.get("title") == "Power":
                    enc["title"] = new_title
                # Or nested in an axis sub-key
                axis = enc.get("axis")
                if isinstance(axis, dict) and axis.get("title") == "Power":
                    axis["title"] = new_title
    for key, value in node.items():
        if key == "encoding":
            continue  # already rewritten above — don't re-visit
        _walk_correction_title(value, new_title)


# ── Envelope building ────────────────────────────────────────────────────────

def _build_envelope(result: Dict[str, Any], meta: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """Build the neutral plot envelope for the new engine plot-set functions.

    ``result`` is the raw engine result dict (single-scenario or multi-scenario
    envelope from PowerResult / SampleSizeResult). ``meta`` provides correction
    info. ``kind`` is ``"find_power"`` or ``"find_sample_size"``.

    Neutral key selection:
      - ``power`` ← corrected if correction is active, else uncorrected.
      - ``ci``    ← corrected if correction is active, else uncorrected.
      - ``histogram`` ← ``success_count_histogram_corrected`` always (for
        sample-size only; power results omit it).
    """
    corr = bool(meta.get("correction") and meta["correction"] != "none")
    pkey = "power_corrected" if corr else "power_uncorrected"
    ckey = "ci_corrected" if corr else "ci_uncorrected"

    from .tables import _scenarios
    scen_list = _scenarios(result)

    scenarios_out = []
    for name, inner in scen_list:
        entry: Dict[str, Any] = {
            "label": name,
            "sample_sizes": inner["sample_sizes"],
            "target_indices": inner["target_indices"],
            # Contrast identities so the bridge can emit target_{p}_vs_{n}
            # tokens for the entries appended past the marginals.
            "contrast_pairs": inner.get("contrast_pairs") or [],
            "power": inner[pkey],
            "ci": inner[ckey],
        }
        if kind == "find_power":
            # The overall/omnibus test is a first-class result: it draws one more
            # bar (last) on the power-at-N chart, matching the table. Absent when
            # the family suppressed the overall test (mixed/GLMM) → no bar.
            entry["overall_power"] = inner.get("overall_significant_rate")
            entry["overall_ci"] = inner.get("overall_significant_ci")
        if kind == "find_sample_size":
            entry["histogram"] = inner.get("success_count_histogram_corrected", [])
        scenarios_out.append(entry)

    return {"scenarios": scenarios_out}


def _plot_blocks(
    result: Dict[str, Any],
    meta: Dict[str, Any],
    kind: str,
    *,
    label_map: Dict[str, str],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return the ordered ``(block_key, spec_dict)`` list for ``result``.

    Builds the neutral envelope, calls the engine plot-set function with
    ``show_ci=True`` and the run's target power line, then applies
    ``_relabel_targets``, ``_style_ci_marks``, and (when correction is active)
    ``_rewrite_correction_axis_title`` to every block spec.

    The target power comes from ``meta["target_power"]`` (present in both
    find_power and find_sample_size meta dicts).
    """
    from .. import _engine

    envelope = _build_envelope(result, meta, kind)
    target_power: Optional[float] = meta.get("target_power")

    if kind == "find_power":
        raw_blocks = _engine.power_plot_set_json(
            envelope, show_ci=True, target_power_line=target_power
        )
    else:
        raw_blocks = _engine.sample_size_plot_set_json(
            envelope, show_ci=True, target_power_line=target_power
        )

    corr = bool(meta.get("correction") and meta["correction"] != "none")
    correction_name: Optional[str] = meta.get("correction") if corr else None
    ci_color = CI_DEFAULT_COLOR

    blocks: List[Tuple[str, Dict[str, Any]]] = []
    for key, spec_json in raw_blocks:
        spec = json.loads(spec_json)
        _relabel_targets(spec, label_map)
        # Default CI colour applied here; _apply_theme (themed paths) will re-run
        # _style_ci_marks with the theme's titleColor, intentionally overriding this.
        _style_ci_marks(spec, ci_color)
        if correction_name:
            _rewrite_correction_axis_title(spec, correction_name)
        blocks.append((key, spec))
    return blocks


# ── Path derivation ──────────────────────────────────────────────────────────

_NONALNUM_RE = re.compile(r"[^a-z0-9]+")


def _sanitize_label(label: str) -> str:
    """Sanitize a scenario label to a safe filename fragment.

    Lowercase; every run of non-alphanumeric characters becomes a single ``_``.
    Trailing ``_`` is preserved (matches spec: the suffix itself is the sanitized
    label, so callers strip nothing extra)."""
    return _NONALNUM_RE.sub("_", label.lower())


def _derive_block_path(stem: str, ext: str, block_key: str) -> str:
    """Derive the output path for a single block, given the user's base ``stem``
    (everything before ``.``) and extension ``ext`` (including ``.``).

    Block routing:
      - ``power`` / ``curve``        → ``<stem><ext>``   (unchanged)
      - ``scenario:<label>``         → ``<stem>_<sanitized><ext>``
      - ``overlay``                  → ``<stem>_overlay<ext>``
      - ``at_least_k``               → ``<stem>_at_least_k<ext>``
      - ``exactly_k``                → ``<stem>_exactly_k<ext>``
    """
    if block_key in ("power", "curve"):
        return f"{stem}{ext}"
    if block_key.startswith("scenario:"):
        label = block_key[len("scenario:"):]
        return f"{stem}_{_sanitize_label(label)}{ext}"
    return f"{stem}_{block_key}{ext}"


def _unique_block_paths(
    user_path: str, blocks: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any], str]]:
    """Pair each ``(block_key, spec)`` with its output path.

    Deduplicates in-call collisions (two scenario labels that sanitize
    identically) by appending ``_2``, ``_3``, … to the later collision using the
    same ``_next_free_path`` numbering pattern but against the in-call set (not
    the filesystem).
    """
    stem, ext = os.path.splitext(user_path)
    seen: Dict[str, int] = {}
    result = []
    for key, spec in blocks:
        base = _derive_block_path(stem, ext, key)
        if base not in seen:
            seen[base] = 1
            path = base
        else:
            seen[base] += 1
            n = seen[base]
            base_stem, base_ext = os.path.splitext(base)
            path = f"{base_stem}_{n}{base_ext}"
        result.append((key, spec, path))
    return result


# ── HTML rendering ───────────────────────────────────────────────────────────

def _write_stacked_html(
    blocks: List[Tuple[str, Dict[str, Any]]], path: str, *, theme: Optional[str]
) -> None:
    """Write a single self-contained HTML file with all block specs stacked via
    the engine's CDN HTML template. ``theme`` is applied to every spec before
    embedding; ``None`` → theme-naked (no config block added).

    ``</`` inside spec JSON is escaped as ``<\\/`` before insertion to prevent
    script tag termination.
    """
    from .. import _engine

    template = _engine.plot_html_template()

    themed_specs = []
    for _key, spec in blocks:
        spec_json = json.dumps(spec)
        if theme is not None:
            spec_json = _apply_theme(spec_json, theme)
        themed_specs.append(json.loads(spec_json))

    specs_json = json.dumps(themed_specs).replace("</", "<\\/")
    html = template.replace("{{SPECS}}", specs_json)

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Public save / view ───────────────────────────────────────────────────────

def save_result_plot(
    result: Dict[str, Any],
    meta: Dict[str, Any],
    kind: str,
    path: str,
    *,
    theme: str = "light-print",
    scale: float = 2.0,
    ppi: Optional[float] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    """Save the plot(s) for ``result`` to ``path``.

    HTML suffix → one stacked file; other suffixes → one file per block with
    derived names (see ``_unique_block_paths``). Default theme is ``"light-print"``;
    pass ``theme=None`` for theme-naked output.
    """
    if label_map is None:
        label_map = {}
    blocks = _plot_blocks(result, meta, kind, label_map=label_map)
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if suffix not in _PLOT_FORMATS:
        raise ValueError(
            f"unsupported plot format '.{suffix}'; use one of: {', '.join(_PLOT_FORMATS)}"
        )
    if suffix == "html":
        _write_stacked_html(blocks, path, theme=theme)
        return

    # Non-HTML: one file per block.
    try:
        import vl_convert as vlc  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "saving plots needs the renderer: pip install mcpower[plot]"
        ) from e

    for key, spec, block_path in _unique_block_paths(path, blocks):
        if theme is not None:
            spec = json.loads(_apply_theme(json.dumps(spec), theme))
        _render(spec, block_path, scale=scale, ppi=ppi)


def view_result_plot(
    result: Dict[str, Any],
    meta: Dict[str, Any],
    kind: str,
    *,
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    """Write a stacked light-print-themed HTML in cwd (uniquely named) and open it."""
    if label_map is None:
        label_map = {}
    blocks = _plot_blocks(result, meta, kind, label_map=label_map)
    basename = "find_power.html" if kind == "find_power" else "find_sample_size.html"
    out = _next_free_path(basename)
    _write_stacked_html(blocks, out, theme="light-print")
    msg = _open_or_report(out)
    print(msg)


def mimebundle_spec(
    result: Dict[str, Any],
    meta: Dict[str, Any],
    kind: str,
    *,
    label_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Return a single light-print-themed spec dict for Jupyter rich repr.

    Block selection:
      - ``find_power``                    → ``power``
      - ``find_sample_size``, 1 scenario  → ``curve``
      - ``find_sample_size``, ≥2 scenarios → ``overlay``
    """
    if label_map is None:
        label_map = {}
    blocks = _plot_blocks(result, meta, kind, label_map=label_map)
    block_dict = {k: s for k, s in blocks}

    if kind == "find_power":
        spec = block_dict.get("power", next(iter(block_dict.values())))
    else:
        from .tables import _scenarios
        n_scenarios = len(_scenarios(result))
        if n_scenarios >= 2:
            spec = block_dict.get("overlay", block_dict.get("curve", next(iter(block_dict.values()))))
        else:
            spec = block_dict.get("curve", next(iter(block_dict.values())))

    spec = json.loads(_apply_theme(json.dumps(spec), "light-print"))
    return spec
