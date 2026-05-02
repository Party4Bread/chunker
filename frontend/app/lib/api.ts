import type { InferOut, Project, RecordOut, RecordPatch, RecordSummary } from "./types";

const API_BASE = "/api";

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
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

  reinfer: (slug: string, id: number, persist = true) =>
    jsonFetch<InferOut>(`/projects/${slug}/records/${id}/reinfer`, {
      method: "POST",
      body: JSON.stringify({ persist }),
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
    if (!res.ok) {
      throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
    }
    return (await res.json()) as RecordOut;
  },

  exportUrl: (slug: string, include: "reviewed" | "all" = "reviewed") =>
    `${API_BASE}/projects/${slug}/export.jsonl?include=${include}`,
};
