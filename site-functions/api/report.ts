// Cloudflare Pages Function — POST /api/report
//
// The account-less intake path for the bug-report form (web/site/report). It:
//   1. validates + size-caps the JSON body,
//   2. verifies the Turnstile token server-side,
//   3. opens a public GitHub issue (same field layout as
//      .github/ISSUE_TEMPLATE/bug_report.yml — keep the two in step),
//   4. best-effort emails the maintainer the reporter's contact (never written
//      into the public issue), and
//   5. returns the created issue URL.
//
// Secrets live only here (Pages env): GITHUB_TOKEN, TURNSTILE_SECRET, and the
// optional REPORT_EMAIL send binding. This file holds no secret values.
//
// PREREQUISITES (maintainer-provisioned, outside the repo):
//   - GitHub: fine-grained PAT (issues:write on the repo) →
//     `wrangler pages secret put GITHUB_TOKEN --project-name=mcpower-site`
//   - Turnstile: widget secret →
//     `wrangler pages secret put TURNSTILE_SECRET --project-name=mcpower-site`
//     (and swap the TEST sitekey in web/site/report/index.html for the real one)
//   - Email (optional): a Cloudflare Email Sending binding named REPORT_EMAIL,
//     plus a verified sender; replace MAINTAINER_EMAIL / SENDER_EMAIL below.

const REPO = 'pawlenartowicz/mcpower';
const GITHUB_API = `https://api.github.com/repos/${REPO}/issues`;
const TURNSTILE_VERIFY = 'https://challenges.cloudflare.com/turnstile/v0/siteverify';

// REPLACE with the verified maintainer recipient + sender on mcpower.app.
const MAINTAINER_EMAIL = 'bugs@mcpower.app';
const SENDER_EMAIL = 'noreply@mcpower.app';

const MAX_BODY_BYTES = 32 * 1024;
const PORTS = ['Python', 'R', 'Desktop app', 'Web app'];
const LIMITS = { title: 200, description: 8000, version: 60, os: 120, email: 254 };

/** Cloudflare Email Sending binding (only the `send` we use). */
interface SendEmail {
  send(message: unknown): Promise<void>;
}

export interface Env {
  GITHUB_TOKEN: string;
  TURNSTILE_SECRET: string;
  REPORT_EMAIL?: SendEmail;
}

interface ReportFields {
  port: string;
  version: string;
  os: string;
  title: string;
  description: string;
  email: string;
  token: string;
}

function json(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** Validate + normalize the request body, or return an error string. */
function parseFields(raw: unknown): { fields: ReportFields } | { error: string } {
  if (typeof raw !== 'object' || raw === null) return { error: 'Malformed request body.' };
  const r = raw as Record<string, unknown>;
  const str = (v: unknown): string => (typeof v === 'string' ? v.trim() : '');

  const fields: ReportFields = {
    port: str(r.port),
    version: str(r.version),
    os: str(r.os),
    title: str(r.title),
    description: str(r.description),
    email: str(r.email),
    // Turnstile injects this field name into the form.
    token: str(r['cf-turnstile-response']),
  };

  // Version + OS are optional (web-app reporters often don't know a version).
  if (!PORTS.includes(fields.port)) return { error: 'Please choose a valid port.' };
  if (!fields.title) return { error: 'Title is required.' };
  if (!fields.description) return { error: 'A description is required.' };
  if (!fields.token) return { error: 'Bot check missing — please retry.' };

  if (fields.title.length > LIMITS.title) return { error: 'Title is too long.' };
  if (fields.description.length > LIMITS.description) return { error: 'Description is too long.' };
  if (fields.version.length > LIMITS.version) return { error: 'Version is too long.' };
  if (fields.os.length > LIMITS.os) return { error: 'Operating system is too long.' };
  if (fields.email && fields.email.length > LIMITS.email) return { error: 'Email is too long.' };

  return { fields };
}

/** Verify the Turnstile token against Cloudflare's siteverify endpoint. */
async function verifyTurnstile(token: string, secret: string, ip: string | null): Promise<boolean> {
  const form = new FormData();
  form.append('secret', secret);
  form.append('response', token);
  if (ip) form.append('remoteip', ip);
  const res = await fetch(TURNSTILE_VERIFY, { method: 'POST', body: form });
  if (!res.ok) return false;
  const data = (await res.json()) as { success?: boolean };
  return data.success === true;
}

/** Render the issue body — mirrors the .github bug_report.yml field layout. */
function issueBody(f: ReportFields): string {
  return [
    `**Port:** ${f.port}`,
    `**Version:** ${f.version || '—'}`,
    // Web-app reporters describe a browser, native ports an OS (see web/site/report).
    `**${f.port === 'Web app' ? 'Browser' : 'OS'}:** ${f.os || '—'}`,
    '',
    '### What happened',
    '',
    f.description,
    '',
    '---',
    '*Filed via the [web form](https://mcpower.app/report).*',
  ].join('\n');
}

/** Create the GitHub issue; returns its html_url + number, or null on failure. */
async function createIssue(
  f: ReportFields,
  token: string,
): Promise<{ url: string; number: number } | null> {
  const res = await fetch(GITHUB_API, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'Content-Type': 'application/json',
      'User-Agent': 'mcpower-report-form',
    },
    body: JSON.stringify({ title: f.title, body: issueBody(f), labels: ['bug'] }),
  });
  if (!res.ok) return null;
  const data = (await res.json()) as { html_url?: string; number?: number };
  if (!data.html_url || typeof data.number !== 'number') return null;
  return { url: data.html_url, number: data.number };
}

/** Minimal RFC 822 message — avoids a MIME dependency for a plaintext notice. */
export function buildRawEmail(from: string, to: string, subject: string, text: string): string {
  return [
    `From: ${from}`,
    `To: ${to}`,
    `Subject: ${subject}`,
    'MIME-Version: 1.0',
    'Content-Type: text/plain; charset=utf-8',
    '',
    text,
  ].join('\r\n');
}

/**
 * Best-effort private notice to the maintainer linking the reporter's contact
 * to the issue number. The reporter email is NEVER written into the public
 * issue — it travels only here. A failure is swallowed: the issue already exists.
 */
async function notifyMaintainer(env: Env, email: string, issueNumber: number): Promise<void> {
  if (!env.REPORT_EMAIL || !email) return;
  try {
    const { EmailMessage } = (await import('cloudflare:email')) as {
      EmailMessage: new (from: string, to: string, raw: string) => unknown;
    };
    const subject = `New MCPower report with contact (issue #${issueNumber})`;
    const text = `Reporter left contact ${email} on issue #${issueNumber}:\nhttps://github.com/${REPO}/issues/${issueNumber}`;
    const raw = buildRawEmail(SENDER_EMAIL, MAINTAINER_EMAIL, subject, text);
    await env.REPORT_EMAIL.send(new EmailMessage(SENDER_EMAIL, MAINTAINER_EMAIL, raw));
  } catch (err) {
    console.error('maintainer email failed (issue already filed):', err);
  }
}

export async function handleReport(request: Request, env: Env): Promise<Response> {
  // Content-Length is a cheap early-out, but it can be absent or wrong — also
  // cap the actual bytes read before parsing so a missing header can't bypass it.
  const len = Number(request.headers.get('content-length') ?? '0');
  if (len > MAX_BODY_BYTES) return json({ error: 'Report is too large.' }, 400);

  let raw: unknown;
  try {
    const text = await request.text();
    if (text.length > MAX_BODY_BYTES) return json({ error: 'Report is too large.' }, 400);
    raw = JSON.parse(text);
  } catch {
    return json({ error: 'Malformed request body.' }, 400);
  }

  const parsed = parseFields(raw);
  if ('error' in parsed) return json({ error: parsed.error }, 400);
  const { fields } = parsed;

  const ok = await verifyTurnstile(
    fields.token,
    env.TURNSTILE_SECRET,
    request.headers.get('CF-Connecting-IP'),
  );
  if (!ok) return json({ error: 'Bot check failed — please retry.' }, 403);

  const issue = await createIssue(fields, env.GITHUB_TOKEN);
  if (!issue) return json({ error: 'Could not create the issue upstream. Please try again later.' }, 502);

  await notifyMaintainer(env, fields.email, issue.number);

  return json({ url: issue.url, number: issue.number }, 200);
}

// Cloudflare Pages Functions entry point for POST /api/report.
export const onRequestPost = (context: { request: Request; env: Env }): Promise<Response> =>
  handleReport(context.request, context.env);
