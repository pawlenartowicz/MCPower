#!/usr/bin/env python3
"""Regenerate the mcpower.app social-share cards under web/site/.

Brand cards built from one engine run:

  * web/site/og-card.png  1200x630 (1.91:1) — the wide link-preview card wired
    into every page's og:image/twitter:image. Content stays inside the 16:9
    safe box, so X / Bluesky / LinkedIn / Facebook and any 16:9-or-wider crop
    lose nothing.
  * web/site/og-card-square.png  1080x1080 — square card, curves full-bleed
    under a centred wordmark/tagline, for square/chat-app contexts. Hosted
    alongside the wide card; not referenced by meta tags.

Three stages, all in this file:

  1. Engine run  -> a genuine power-by-N curve (not decoration). The exact
     scenario/seed/grid below is the cards' data of record; rerunning it
     reproduces the same curve.
  2. HTML emit   -> a self-contained build artifact per variant (temp dir,
     never committed): the cream MCPower wordmark reused verbatim from
     web/site/mcpower-logo.svg, the card tagline, a generated curve SVG, and a
     scrim that dims the chart behind the text. Site fonts (Fraunces + JetBrains
     Mono) are base64-embedded so the rasterizer needs no network/font install.
  3. Rasterize   -> headless Chromium via the app's @playwright/test toolchain
     (the only browser the repo already vendors), screenshotting the card node
     at deviceScaleFactor 1 for an exact-size PNG.

Run:  .venv/bin/python mcpower/scripts/gen-og-card.py   (from the workspace root)
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Resolved relative to this file so the script runs from any cwd.
SCRIPTS_DIR = Path(__file__).resolve().parent
MCPOWER = SCRIPTS_DIR.parent                       # the publishable repo root
SITE = MCPOWER / "web" / "site"
PORTS_APP = MCPOWER / "ports" / "app"               # vendors @playwright/test

# One colour per predictor, biomarker (the lead) first in rose.
COLORS = ["#ff5e8a", "#c9a6e8", "#f1a8bf", "#93b1e6", "#7ec8bf", "#8e8295"]
BG = "#110b1c"
CREAM = "#ede5f0"
ROSE = "#ff5e8a"
MUTED = "#8e8295"

# Per-variant geometry. The plot data rectangle is (px0,px1) x (py_top,py_bot)
# in canvas px; labels="edge" hangs the per-line names off the right of the
# plot (wide card), "none" drops all chart text so the curves read as a pure
# backdrop (square card). The wide card's px1 is pulled in far enough that the
# longest end label ends well left of the 16:9 side cut (~x 1160 of 1200).
WIDE = {
    "name": "wide",
    "W": 1200, "H": 630,
    "px0": 446, "px1": 1024,
    "py_top": 88, "py_bot": 544,
    "labels": "edge",
    "dim": False,
    "out": SITE / "og-card.png",
}
SQUARE = {
    "name": "square",
    "W": 1080, "H": 1080,
    "px0": 36, "px1": 1044,
    "py_top": 64, "py_bot": 1016,
    "labels": "none",
    "dim": True,
    "out": SITE / "og-card-square.png",
}


def run_engine():
    """Return (sample_sizes, predictors, power[N][k], ci[N][k]=(lo,hi))."""
    import mcpower

    m = mcpower.MCPower("outcome = biomarker + age + dose + bmi + sleep + stress")
    m.set_effects("biomarker=0.40, age=0.22, dose=0.18, bmi=0.14, sleep=0.11, stress=0.08")
    m.set_seed(2137)
    m.set_simulations(1600)
    res = m.find_sample_size(
        from_size=30, to_size=230, by=25, target_power=0.8,
        verbose=False, progress_callback=False,
    )
    return (
        list(res["sample_sizes"]),
        list(m.predictor_vars_order),
        res["power_uncorrected"],
        res["ci_uncorrected"],
    )


def build_curve_svg(g, sizes, predictors, power, ci) -> str:
    px0, px1, py_top, py_bot = g["px0"], g["px1"], g["py_top"], g["py_bot"]
    show_text = g["labels"] != "none"

    def sx(n):
        return px0 + (n - 30) / (230 - 30) * (px1 - px0)

    def sy(p):
        return py_bot - p * (py_bot - py_top)

    n_pts, n_k = len(sizes), len(predictors)
    parts = [f'<svg viewBox="0 0 {g["W"]} {g["H"]}" xmlns="http://www.w3.org/2000/svg" class="chart">']

    # Horizontal gridlines at 0.2..1.0; the 0.8 line is the dashed target ref.
    for p in (0.2, 0.4, 0.6, 0.8, 1.0):
        y = sy(p)
        if p == 0.8:
            parts.append(
                f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" '
                f'stroke="{ROSE}" stroke-width="1.5" stroke-dasharray="7 6" opacity="0.55"/>'
            )
        else:
            parts.append(
                f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" '
                f'stroke="{CREAM}" stroke-width="1" opacity="0.10"/>'
            )

    # Axis lines (left + bottom) — only on the labelled (wide) card.
    if show_text:
        parts.append(
            f'<line x1="{px0}" y1="{py_top}" x2="{px0}" y2="{py_bot}" stroke="{CREAM}" stroke-width="1.2" opacity="0.28"/>'
            f'<line x1="{px0}" y1="{py_bot}" x2="{px1}" y2="{py_bot}" stroke="{CREAM}" stroke-width="1.2" opacity="0.28"/>'
        )

    # 95% CI bands (drawn under the lines): lo left->right, hi right->left.
    for k in range(n_k):
        lo = " ".join(f"{sx(sizes[i]):.1f},{sy(ci[i][k][0]):.1f}" for i in range(n_pts))
        hi = " ".join(f"{sx(sizes[i]):.1f},{sy(ci[i][k][1]):.1f}" for i in range(n_pts - 1, -1, -1))
        parts.append(f'<polygon points="{lo} {hi}" fill="{COLORS[k]}" opacity="0.13"/>')

    # Power lines; biomarker (k=0) leads, thicker.
    for k in range(n_k):
        pts = " ".join(f"{sx(sizes[i]):.1f},{sy(power[i][k]):.1f}" for i in range(n_pts))
        width = 4.5 if k == 0 else 2.6
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{COLORS[k]}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    if show_text:
        # Per-line end labels, de-collided so the low cluster stays legible.
        ends = sorted(
            ({"name": predictors[k], "color": COLORS[k], "y": sy(power[-1][k])} for k in range(n_k)),
            key=lambda d: d["y"],
        )
        for i in range(1, len(ends)):
            if ends[i]["y"] - ends[i - 1]["y"] < 23.0:
                ends[i]["y"] = ends[i - 1]["y"] + 23.0
        lx = px1 + 12
        for e in ends:
            parts.append(
                f'<text x="{lx}" y="{e["y"]:.1f}" class="endlabel" fill="{e["color"]}" '
                f'dominant-baseline="middle">{e["name"]}</text>'
            )

        for p in (0.2, 0.4, 0.6, 0.8, 1.0):
            parts.append(
                f'<text x="{px0 - 9}" y="{sy(p) + 4:.1f}" class="tick" fill="{MUTED}" '
                f'text-anchor="end">{p:.1f}</text>'
            )
        for n in (30, 80, 130, 180, 230):
            parts.append(
                f'<text x="{sx(n):.1f}" y="{py_bot + 24:.1f}" class="tick" fill="{MUTED}" '
                f'text-anchor="middle">{n}</text>'
            )
        parts.append(
            f'<text x="{(px0 + px1) / 2:.1f}" y="{py_bot + 50:.1f}" class="axistitle" '
            f'fill="{MUTED}" text-anchor="middle">sample size (N)</text>'
        )
        # Anchored in the clear band above the dashed line (biomarker is already
        # past 0.8 here, the rest still climbing), away from curves and labels.
        parts.append(
            f'<text x="700" y="{sy(0.8) - 10:.1f}" class="tick" fill="{ROSE}" '
            f'opacity="0.9">target power 0.8</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def font_face(family: str, style: str, weight: str, woff2: Path) -> str:
    b64 = base64.b64encode(woff2.read_bytes()).decode("ascii")
    return (
        "@font-face{"
        f"font-family:'{family}';font-style:{style};font-weight:{weight};"
        f"src:url(data:font/woff2;base64,{b64}) format('woff2');"
        "}"
    )


def _fonts_css() -> str:
    fonts = SITE / "fonts"
    return "".join([
        font_face("Fraunces", "normal", "400 600", fonts / "fraunces-latin.woff2"),
        font_face("Fraunces", "italic", "400 500", fonts / "fraunces-italic-latin.woff2"),
        font_face("JetBrains Mono", "normal", "400 600", fonts / "jetbrains-mono-latin.woff2"),
    ])


def _wordmark() -> str:
    # Reuse the wordmark verbatim: take the logo's <svg> and tint it cream.
    logo = (SITE / "mcpower-logo.svg").read_text()
    svg = re.search(r"<svg\b.*?</svg>", logo, re.DOTALL).group(0)
    svg = re.sub(r'fill="currentColor"', f'fill="{CREAM}"', svg, count=1)
    return svg.replace("<svg", '<svg class="wordmark"', 1)


TAGLINE = ('Power analysis by simulation. <i>Any design</i> from t-test to '
           'mixed models, in your browser, on your desktop, or in Python and R.')

# Shared CSS for the curve text on both cards.
CHART_CSS = (
    ".chart{position:absolute;inset:0}"
    ".endlabel{font-family:'JetBrains Mono',monospace;font-size:17px;font-weight:500}"
    ".tick{font-family:'JetBrains Mono',monospace;font-size:14px}"
    ".axistitle{font-family:'JetBrains Mono',monospace;font-size:15px;letter-spacing:.04em}"
)


def build_html(g, curve_svg: str) -> str:
    faces = _fonts_css()
    wordmark = _wordmark()
    frame = (f"#card{{position:relative;width:{g['W']}px;height:{g['H']}px;"
             f"background:{BG};overflow:hidden;font-family:'JetBrains Mono',monospace}}")

    if g["name"] == "wide":
        # Left text column; a left->right scrim keeps the wordmark/tagline crisp
        # over the chart. Both column and chart sit inside the 16:9 safe box.
        layout = f"""
.scrim{{position:absolute;inset:0;background:linear-gradient(90deg,
  {BG} 0%, {BG} 27%, rgba(17,11,28,.86) 40%, rgba(17,11,28,.40) 56%, rgba(17,11,28,.06) 100%)}}
.left{{position:absolute;left:70px;top:0;bottom:0;width:412px;
  display:flex;flex-direction:column;justify-content:center}}
.wordmark{{width:352px;height:auto;display:block;
  filter:drop-shadow(0 0 26px rgba(255,94,138,.22))}}
.rule{{width:104px;height:4px;border-radius:2px;background:{ROSE};margin:30px 0 26px}}
.tagline{{font-family:'Fraunces',serif;font-weight:500;font-size:27px;line-height:1.4;
  letter-spacing:.01em;color:{CREAM};max-width:418px}}
.tagline i{{font-style:italic;color:{ROSE}}}
.url{{position:absolute;left:70px;bottom:54px;font-family:'JetBrains Mono',monospace;
  font-size:19px;font-weight:500;letter-spacing:.02em;color:{CREAM};opacity:.82}}"""
        body = f"""
  {curve_svg}
  <div class="scrim"></div>
  <div class="left">
    {wordmark}
    <div class="rule"></div>
    <p class="tagline">{TAGLINE}</p>
  </div>
  <div class="url">mcpower.app</div>"""
    else:
        # Square: curves full-bleed as a backdrop. A vertical scrim darkest
        # through the middle keeps the centred wordmark/tagline legible.
        layout = f"""
.scrim{{position:absolute;inset:0;background:linear-gradient(180deg,
  rgba(17,11,28,.42) 0%, rgba(17,11,28,.82) 34%, rgba(17,11,28,.85) 60%, rgba(17,11,28,.55) 100%)}}
.center{{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;padding:0 90px}}
.wordmark{{width:600px;height:auto;display:block;
  filter:drop-shadow(0 0 34px rgba(255,94,138,.30))}}
.rule{{width:128px;height:5px;border-radius:3px;background:{ROSE};margin:40px 0 34px}}
.tagline{{font-family:'Fraunces',serif;font-weight:500;font-size:33px;line-height:1.42;
  letter-spacing:.01em;color:{CREAM};max-width:660px}}
.tagline i{{font-style:italic;color:{ROSE}}}
.url{{position:absolute;left:0;right:0;bottom:56px;font-family:'JetBrains Mono',monospace;
  font-size:23px;font-weight:500;letter-spacing:.02em;color:{CREAM};opacity:.82;text-align:center}}"""
        body = f"""
  {curve_svg}
  <div class="scrim"></div>
  <div class="center">
    {wordmark}
    <div class="rule"></div>
    <p class="tagline">{TAGLINE}</p>
  </div>
  <div class="url">mcpower.app</div>"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>
{faces}
*{{margin:0;padding:0;box-sizing:border-box}}
{frame}
{CHART_CSS}
{layout}
</style></head><body>
<div id="card">{body}
</div>
</body></html>"""


RASTERIZER = r"""
const { chromium } = require('@playwright/test');
(async () => {
  const [htmlPath, outPath, w, h] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: +w, height: +h },
    deviceScaleFactor: 1,
  });
  await page.goto('file://' + htmlPath);
  await page.evaluate(() => document.fonts.ready);
  const el = await page.$('#card');
  await el.screenshot({ path: outPath });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def render(g, sizes, predictors, power, ci) -> None:
    html = build_html(g, build_curve_svg(g, sizes, predictors, power, ci))
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "card.html"
        js_path = Path(tmp) / "rasterize.cjs"
        html_path.write_text(html)
        js_path.write_text(RASTERIZER)
        # NODE_PATH points require() at the app's vendored playwright; module
        # resolution keys off the script's dir, not cwd, so cwd alone wouldn't
        # find it.
        env = {**os.environ, "NODE_PATH": str(PORTS_APP / "node_modules")}
        proc = subprocess.run(
            ["node", str(js_path), str(html_path), str(g["out"]), str(g["W"]), str(g["H"])],
            cwd=PORTS_APP, capture_output=True, text=True, env=env,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout + proc.stderr)
            raise SystemExit(proc.returncode)

    # Re-encode losslessly: Pillow's PNG encoder packs the flat background and
    # gradient tighter than Chromium's, for headroom under the OG size budget.
    from PIL import Image

    Image.open(g["out"]).save(g["out"], optimize=True)
    kb = g["out"].stat().st_size / 1024
    print(f"      wrote {g['out'].relative_to(MCPOWER)} ({g['W']}x{g['H']}, {kb:.0f} KB)")


def main() -> int:
    print("[1/2] running engine for the power curve…")
    sizes, predictors, power, ci = run_engine()
    print(f"      N grid {sizes}; predictors {predictors}")

    print("[2/2] rendering cards via headless Chromium (@playwright/test)…")
    for g in (WIDE, SQUARE):
        render(g, sizes, predictors, power, ci)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
