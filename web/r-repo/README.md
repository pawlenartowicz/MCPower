# r.mcpower.app — Cloudflare Worker (R binary repository)

Serves the R port's prebuilt binaries from GitHub Release assets. Decoupled from
the site Pages pipeline — deployed by its own `wrangler deploy`.

## One-time setup (operator)
1. GitHub PAT — fine-grained, **read-only**, public repos only (no write scopes).
   `npx wrangler secret put GH_TOKEN`   (paste the PAT)
2. Deploy:  `npx wrangler deploy`
3. Custom domain: Cloudflare dashboard → Workers → mcpower-r-repo → Settings →
   Domains & Routes → Add custom domain → `r.mcpower.app`.

## Verify
install.packages("mcpower", repos = "https://r.mcpower.app")   # Win / macOS / Linux (release R)

The one-liner auto-serves the *release* R binary per platform. oldrel + off-window R
use the manual links (docs.mcpower.app/tutorial-r/manual-download), which resolve to
stable routes:  GET /download/{linux,windows,macos}/mcpower-R<minor>.{tar.gz,zip,tgz}
