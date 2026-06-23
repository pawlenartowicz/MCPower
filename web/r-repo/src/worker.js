// Cloudflare Worker for r.mcpower.app — serves the R port's prebuilt binaries
// out of GitHub Release assets, reconstructing the CRAN directory tree from the
// flat _R<m>-suffixed asset names. Stateless; edge-cached by request path.
//
// Release resolution: list-and-filter (per_page=100), pick the highest-semver
// r-* tag. (KV is optional hardening — Appendix A of the impl plan.)
// Linux /src/contrib negotiation is added in Phase 3.

const API = "https://api.github.com";

function ghHeaders(env) {
  const h = { "Accept": "application/vnd.github+json", "User-Agent": "mcpower-r-repo" };
  if (env.GH_TOKEN) h["Authorization"] = `Bearer ${env.GH_TOKEN}`;
  return h;
}

// Highest-semver r-X.Y.Z release (non-latest, so /releases/latest is unusable).
async function resolveRelease(env, ctx) {
  const cache = caches.default;
  const key = new Request("https://r.mcpower.app/__release.json");
  let hit = await cache.match(key);
  if (hit) return hit.json();

  const res = await fetch(`${API}/repos/${env.GH_REPO}/releases?per_page=100`, { headers: ghHeaders(env) });
  if (!res.ok) throw new Error(`release list ${res.status}`);
  const rels = await res.json();
  const cmp = (a, b) => a.map(Number).reduce((acc, n, i) => acc || n - Number(b[i]), 0);
  const r = rels
    .filter((x) => /^r-\d+\.\d+\.\d+$/.test(x.tag_name) && !x.draft)
    .map((x) => ({ rel: x, v: x.tag_name.slice(2).split(".") }))
    .sort((a, b) => cmp(b.v, a.v))[0];
  if (!r) throw new Error("no r-* release");

  const out = {
    tag: r.rel.tag_name,
    version: r.rel.tag_name.slice(2),
    // asset name -> browser_download_url
    assets: Object.fromEntries(r.rel.assets.map((a) => [a.name, a.browser_download_url])),
  };
  const body = JSON.stringify(out);
  ctx.waitUntil(cache.put(key, new Response(body, { headers: { "Cache-Control": "max-age=180" } })));
  return out;
}

// Built minors come from the _R<m> binary asset suffixes present on the release.
function builtMinors(assets, plat) {
  const ext = plat === "win" ? "zip" : plat === "mac" ? "tgz" : "tar.gz";
  const re = new RegExp(`_R(\\d+\\.\\d+)\\.${ext.replace(".", "\\.")}$`);
  const set = new Set();
  for (const name of Object.keys(assets)) { const m = name.match(re); if (m) set.add(m[1]); }
  return set;
}

// Compare "X.Y" R-minor strings numerically (so 4.10 > 4.9, which lexical sort breaks).
const cmpMinor = (a, b) => {
  const [a1, a2] = a.split(".").map(Number);
  const [b1, b2] = b.split(".").map(Number);
  return a1 - b1 || a2 - b2;
};

const CT = {
  zip: "application/zip",
  tgz: "application/x-gzip",
  "tar.gz": "application/x-gzip",
  gz: "application/x-gzip",
  PACKAGES: "text/plain; charset=utf-8",
};

// Follow GitHub's 302 to objects.githubusercontent.com server-side; stream body.
// Never cache the signed redirect URL — cache by the r.mcpower.app request path.
async function streamAsset(url, contentType) {
  if (!url) return new Response("not found", { status: 404 }); // gated minor present but this asset missing (partial upload)
  const r = await fetch(url, { redirect: "follow" });
  if (!r.ok) return new Response("not found", { status: 404 });
  return new Response(r.body, { status: 200, headers: { "Content-Type": contentType, "Cache-Control": "public, max-age=300" } });
}

const EMPTY_PACKAGES = () =>
  new Response("", { status: 200, headers: { "Content-Type": CT.PACKAGES, "Cache-Control": "public, max-age=120" } });

export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    const path = url.pathname;
    const cache = caches.default;

    // Win/Mac routes are path-keyed (minor is in the path). Serve cache early.
    const cached = await cache.match(req);
    if (cached) return cached;

    let rel;
    try { rel = await resolveRelease(env, ctx); }
    catch (e) { return new Response(`resolve error: ${e.message}`, { status: 502 }); }
    const V = rel.version;

    // --- Manual download: stable, version-less URLs for the manual-download docs
    //     page (docs.mcpower.app/tutorial-r/manual-download). Each maps to one
    //     release asset; the URL keeps the archive extension so a
    //     `install.packages(url, repos=NULL)` recognises the type. This is how
    //     oldrel Linux + any off-window R reaches a prebuilt binary (the one-liner
    //     repo only auto-serves the release Linux build — see /src/contrib below).
    const dl = path.match(/^\/download\/(linux|windows|macos)\/mcpower-R(\d+\.\d+)\.(?:tar\.gz|zip|tgz)$/);
    if (dl) {
      const plat = dl[1], m = dl[2];
      const asset = plat === "linux" ? `mcpower_${V}_linux_x86_64_R${m}.tar.gz`
                  : plat === "windows" ? `mcpower_${V}_R${m}.zip`
                  : `mcpower_${V}_R${m}.tgz`;
      const ct = plat === "windows" ? CT.zip : CT["tar.gz"];
      const resp = await streamAsset(rel.assets[asset], ct);
      if (resp.status === 200) ctx.waitUntil(cache.put(req, resp.clone()));
      return resp;
    }

    // --- Windows: /bin/windows/contrib/<m>/(PACKAGES[.gz]|PACKAGES.rds|mcpower_V.zip)
    // --- macOS:   /bin/macosx/big-sur-arm64/contrib/<m>/(...)  V.tgz
    const win = path.match(/^\/bin\/windows\/contrib\/(\d+\.\d+)\/(.+)$/);
    const mac = path.match(/^\/bin\/macosx\/big-sur-arm64\/contrib\/(\d+\.\d+)\/(.+)$/);
    const route = win ? { plat: "win", ext: "zip" } : mac ? { plat: "mac", ext: "tgz" } : null;

    if (route) {
      const m = (win || mac)[1];
      const file = (win || mac)[2];
      const minors = builtMinors(rel.assets, route.plat);
      const inWindow = minors.has(m);

      let resp;
      if (file === "PACKAGES.rds") {
        resp = new Response("not found", { status: 404 });           // we never emit .rds
      } else if (file === "PACKAGES.gz") {
        resp = inWindow
          ? await streamAsset(rel.assets[`PACKAGES.${route.plat}-${m}.gz`], CT.gz)
          : new Response("not found", { status: 404 });               // → client falls back to plain PACKAGES
      } else if (file === "PACKAGES") {
        resp = inWindow
          ? await streamAsset(rel.assets[`PACKAGES.${route.plat}-${m}`], CT.PACKAGES)
          : EMPTY_PACKAGES();                                         // off-window minor → clean "not available"
      } else if (file === `mcpower_${V}.${route.ext}`) {
        resp = inWindow
          ? await streamAsset(rel.assets[`mcpower_${V}_R${m}.${route.ext}`], CT[route.ext])
          : new Response("not found", { status: 404 });
      } else {
        resp = new Response("not found", { status: 404 });
      }
      if (resp.status === 200) ctx.waitUntil(cache.put(req, resp.clone()));
      return resp;
    }

    // --- Linux /src/contrib/* — the one-liner repo path. R sends no minor/platform
    //     on install (bare `libcurl/X` UA, Phase 0 §4) and the source path has no
    //     minor, so there is NO per-minor negotiation: serve the RELEASE = highest
    //     built Linux minor, unconditionally. oldrel Linux uses /download/ instead.
    //     R requests PACKAGES.rds first; 404-ing it makes R fall back to .gz
    //     (verified Phase 0 §5).
    if (path.startsWith("/src/contrib/")) {
      const minors = [...builtMinors(rel.assets, "linux")].sort((a, b) => cmpMinor(b, a));
      const m = minors[0];   // highest built Linux minor = release
      const file = path.slice("/src/contrib/".length);

      let resp;
      if (!m) {
        resp = file === "PACKAGES" ? EMPTY_PACKAGES() : new Response("not found", { status: 404 });
      } else if (file === "PACKAGES.rds") {
        resp = new Response("not found", { status: 404 });            // → R falls back to .gz
      } else if (file === "PACKAGES.gz") {
        resp = await streamAsset(rel.assets[`PACKAGES.src-${m}.gz`], CT.gz);
      } else if (file === "PACKAGES") {
        resp = await streamAsset(rel.assets[`PACKAGES.src-${m}`], CT.PACKAGES);
      } else if (file === `mcpower_${V}.tar.gz`) {
        resp = await streamAsset(rel.assets[`mcpower_${V}_linux_x86_64_R${m}.tar.gz`], CT["tar.gz"]);
      } else {
        resp = new Response("not found", { status: 404 });
      }
      if (resp.status === 200) ctx.waitUntil(cache.put(req, resp.clone()));
      return resp;
    }

    return new Response("mcpower R repository", { status: 200, headers: { "Content-Type": "text/plain" } });
  },
};
