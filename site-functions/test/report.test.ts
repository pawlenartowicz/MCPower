import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildRawEmail, handleReport, type Env } from '../api/report';

// notifyMaintainer dynamically imports cloudflare:email; provide a stand-in so
// the email-attempt test can run under node. Unused when REPORT_EMAIL is absent.
vi.mock('cloudflare:email', () => ({
  EmailMessage: class {
    constructor(
      public from: string,
      public to: string,
      public raw: string,
    ) {}
  },
}));

const VALID = {
  port: 'Python',
  version: '1.0.0',
  os: 'Ubuntu 24.04',
  title: 'find_power crashes',
  description: 'It threw an error on a valid model.',
  email: '',
  'cf-turnstile-response': 'tok',
};

function makeRequest(body: unknown, headers: Record<string, string> = {}): Request {
  return new Request('https://mcpower.app/api/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: typeof body === 'string' ? body : JSON.stringify(body),
  });
}

interface Routes {
  turnstileSuccess?: boolean;
  turnstileOk?: boolean; // HTTP-level ok
  githubOk?: boolean;
}

function installFetch(routes: Routes): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('siteverify')) {
      return new Response(JSON.stringify({ success: routes.turnstileSuccess ?? true }), {
        status: routes.turnstileOk === false ? 500 : 200,
      });
    }
    if (url.includes('api.github.com')) {
      if (routes.githubOk === false) return new Response('nope', { status: 422 });
      return new Response(
        JSON.stringify({ html_url: 'https://github.com/pawlenartowicz/mcpower/issues/42', number: 42 }),
        { status: 201 },
      );
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

const env: Env = { GITHUB_TOKEN: 'gh', TURNSTILE_SECRET: 'ts' };

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('handleReport', () => {
  it('creates an issue and returns its url on the happy path', async () => {
    const fetchMock = installFetch({});
    const res = await handleReport(makeRequest(VALID), env);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      url: 'https://github.com/pawlenartowicz/mcpower/issues/42',
      number: 42,
    });
    // Turnstile verified before the issue was created.
    const urls = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urls.some((u) => u.includes('siteverify'))).toBe(true);
    expect(urls.some((u) => u.includes('api.github.com'))).toBe(true);
  });

  it('rejects a missing required field with 400 and never calls upstream', async () => {
    const fetchMock = installFetch({});
    const { title, ...noTitle } = VALID;
    const res = await handleReport(makeRequest(noTitle), env);
    expect(res.status).toBe(400);
    expect((await res.json()).error).toMatch(/title/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('accepts a report with version and OS omitted (both optional)', async () => {
    installFetch({});
    const { version, os, ...noMeta } = VALID;
    const res = await handleReport(makeRequest(noMeta), env);
    expect(res.status).toBe(200);
  });

  it('labels the env line "Browser" for the web app in the issue body', async () => {
    const fetchMock = installFetch({});
    await handleReport(makeRequest({ ...VALID, port: 'Web app', os: 'Safari 17' }), env);
    const ghCall = fetchMock.mock.calls.find((c) => String(c[0]).includes('api.github.com'));
    const body = JSON.parse((ghCall![1] as RequestInit).body as string);
    expect(body.body).toContain('**Browser:** Safari 17');
    expect(body.body).not.toContain('**OS:**');
  });

  it('rejects an unknown port with 400', async () => {
    installFetch({});
    const res = await handleReport(makeRequest({ ...VALID, port: 'Fortran' }), env);
    expect(res.status).toBe(400);
  });

  it('returns 403 when Turnstile verification fails, without creating an issue', async () => {
    const fetchMock = installFetch({ turnstileSuccess: false });
    const res = await handleReport(makeRequest(VALID), env);
    expect(res.status).toBe(403);
    const urls = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urls.some((u) => u.includes('api.github.com'))).toBe(false);
  });

  it('returns 502 when the GitHub API fails', async () => {
    installFetch({ githubOk: false });
    const res = await handleReport(makeRequest(VALID), env);
    expect(res.status).toBe(502);
  });

  it('rejects oversized bodies by content-length with 400', async () => {
    const fetchMock = installFetch({});
    const res = await handleReport(makeRequest(VALID, { 'content-length': String(64 * 1024) }), env);
    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects malformed JSON with 400', async () => {
    installFetch({});
    const res = await handleReport(makeRequest('{not json'), env);
    expect(res.status).toBe(400);
  });

  it('emails the maintainer when a contact is supplied, and still succeeds', async () => {
    installFetch({});
    const send = vi.fn(async () => {});
    const res = await handleReport(
      makeRequest({ ...VALID, email: 'reporter@example.com' }),
      { ...env, REPORT_EMAIL: { send } },
    );
    expect(res.status).toBe(200);
    expect(send).toHaveBeenCalledOnce();
  });

  it('does not fail the request when the email send throws', async () => {
    installFetch({});
    const send = vi.fn(async () => {
      throw new Error('email down');
    });
    const res = await handleReport(
      makeRequest({ ...VALID, email: 'reporter@example.com' }),
      { ...env, REPORT_EMAIL: { send } },
    );
    expect(res.status).toBe(200);
  });
});

describe('buildRawEmail', () => {
  it('renders RFC822 headers and keeps the reporter contact out of the issue body', () => {
    const raw = buildRawEmail('a@x', 'b@y', 'subj', 'reporter@example.com on #42');
    expect(raw).toContain('From: a@x');
    expect(raw).toContain('To: b@y');
    expect(raw).toContain('Subject: subj');
    expect(raw).toContain('reporter@example.com');
  });
});
