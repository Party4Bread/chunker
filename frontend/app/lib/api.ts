import type { BatchUploadOut, InferOut, Project, RecordOut, RecordPatch, RecordSummary, TranslateSourceOut } from "./types";

const API_BASE = "/api";

/**
 * Error carrying the HTTP status so callers can branch on it — notably 429
 * (rate limited) from the source-MT endpoint, which surfaces a Retry-After.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly retryAfter: number | null;

  constructor(status: number, statusText: string, body: string, retryAfter: number | null) {
    super(`${status} ${statusText}: ${body}`);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

async function raiseApiError(res: Response): Promise<never> {
  const body = await res.text();
  const raw = res.headers.get("Retry-After");
  const seconds = raw != null && raw.trim() !== "" ? Number(raw) : NaN;
  throw new ApiError(res.status, res.statusText, body, Number.isFinite(seconds) ? seconds : null);
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) await raiseApiError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  listProjects: () => jsonFetch<Project[]>("/projects"),
  createProject: (name: string, model_name?: string) =>
    jsonFetch<Project>("/projects", {
      method: "POST",
      body: JSON.stringify({ name, model_name }),
    }),
  deleteProject: (slug: string) => jsonFetch<void>(`/projects/${slug}`, { method: "DELETE" }),

  listRecords: (slug: string) => jsonFetch<RecordSummary[]>(`/projects/${slug}/records`),
  getRecord: (slug: string, id: number) => jsonFetch<RecordOut>(`/projects/${slug}/records/${id}`),
  patchRecord: (slug: string, id: number, body: RecordPatch) =>
    jsonFetch<RecordOut>(`/projects/${slug}/records/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  deleteRecord: (slug: string, id: number) => jsonFetch<void>(`/projects/${slug}/records/${id}`, { method: "DELETE" }),

  reinfer: (
    slug: string,
    id: number,
    opts: { persist?: boolean; start_src_index?: number; start_tgt_index?: number } = {},
  ) =>
    jsonFetch<InferOut>(`/projects/${slug}/records/${id}/reinfer`, {
      method: "POST",
      body: JSON.stringify({
        persist: opts.persist ?? true,
        start_src_index: opts.start_src_index ?? 0,
        start_tgt_index: opts.start_tgt_index ?? 0,
      }),
    }),
  translateSource: (
    slug: string,
    id: number,
    opts: { texts?: string[]; targetLanguage?: string } = {},
  ) =>
    jsonFetch<TranslateSourceOut>(`/projects/${slug}/records/${id}/translate-source`, {
      method: "POST",
      body: JSON.stringify({
        target_language: opts.targetLanguage?.trim() || null,
        texts: opts.texts ?? null,
      }),
    }),

  uploadRecord: async (
    slug: string,
    src: File,
    tgt: File,
    opts: { title?: string; runModel?: boolean } = {},
  ): Promise<RecordOut> => {
    const fd = new FormData();
    fd.append("src_file", src);
    fd.append("tgt_file", tgt);
    if (opts.title) fd.append("title", opts.title);
    fd.append("run_model", opts.runModel === false ? "false" : "true");
    const res = await fetch(`${API_BASE}/projects/${slug}/records/upload`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) await raiseApiError(res);
    return (await res.json()) as RecordOut;
  },

  uploadBatchRecords: async (
    slug: string,
    srcFiles: File[],
    tgtFiles: File[],
    opts: { runModel?: boolean } = {},
  ): Promise<BatchUploadOut> => {
    const fd = new FormData();
    for (const file of srcFiles) fd.append("src_files", file);
    for (const file of tgtFiles) fd.append("tgt_files", file);
    fd.append("run_model", opts.runModel === false ? "false" : "true");
    const res = await fetch(`${API_BASE}/projects/${slug}/records/upload/batch`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) await raiseApiError(res);
    return (await res.json()) as BatchUploadOut;
  },

  exportUrl: (slug: string, include: "reviewed" | "all" = "reviewed") =>
    `${API_BASE}/projects/${slug}/export.jsonl?include=${include}`,
};
