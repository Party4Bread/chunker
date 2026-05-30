import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate, useParams } from "react-router";
import { Toolbar } from "~/components/Toolbar";
import { api } from "~/lib/api";

export default function UploadPair() {
  const { slug = "" } = useParams();
  const navigate = useNavigate();
  const [src, setSrc] = useState<File | null>(null);
  const [tgt, setTgt] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [runModel, setRunModel] = useState(true);
  const [cleanHtml, setCleanHtml] = useState(true);

  const mut = useMutation({
    mutationFn: async () => {
      if (!src || !tgt) throw new Error("pick both files");
      return api.uploadRecord(slug, src, tgt, { title: title.trim() || undefined, runModel, cleanHtml });
    },
    onSuccess: (rec) => navigate(`/projects/${slug}/records/${rec.id}`),
  });

  return (
    <div className="min-h-screen">
      <Toolbar
        crumbs={[{ to: "/", label: "Projects" }, { to: `/projects/${slug}`, label: slug }, { label: "Upload" }]}
      />
      <main className="mx-auto max-w-2xl space-y-4 p-4">
        <form
          className="space-y-4 rounded-lg bg-white p-4 ring-1 ring-neutral-200"
          onSubmit={(e) => {
            e.preventDefault();
            mut.mutate();
          }}
        >
          <label className="block">
            <span className="eyebrow text-neutral-700">label (optional)</span>
            <input
              type="text"
              className="mt-1 w-full rounded-md border border-neutral-200 px-3 py-2 text-sm"
              value={title}
              placeholder="e.g. blog-post-2024-04"
              onChange={(e) => setTitle(e.target.value)}
            />
          </label>
          <label className="block">
            <span className="eyebrow text-neutral-700">source language</span>
            <input
              type="file"
              accept=".txt,text/plain"
              className="mt-1 block w-full text-sm"
              onChange={(e) => setSrc(e.target.files?.[0] ?? null)}
            />
            <span className="mt-1 block text-2xs text-neutral-500">{src ? src.name : "plain text, UTF-8"}</span>
          </label>
          <label className="block">
            <span className="eyebrow text-neutral-700">target language</span>
            <input
              type="file"
              accept=".txt,text/plain"
              className="mt-1 block w-full text-sm"
              onChange={(e) => setTgt(e.target.files?.[0] ?? null)}
            />
            <span className="mt-1 block text-2xs text-neutral-500">{tgt ? tgt.name : "plain text, UTF-8"}</span>
          </label>
          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              className="mt-1"
              checked={cleanHtml}
              onChange={(e) => setCleanHtml(e.target.checked)}
            />
            <span>
              <span className="font-medium text-ink">clean HTML before chunking</span>
              <span className="block text-xs text-neutral-500">
                removes tags, scripts, styles, comments, and decodes HTML entities while keeping readable line breaks.
              </span>
            </span>
          </label>
          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              className="mt-1"
              checked={runModel}
              onChange={(e) => setRunModel(e.target.checked)}
            />
            <span>
              <span className="font-medium text-ink">seed alignments with the model</span>
              <span className="block text-xs text-neutral-500">
                uncheck to start from a blank record and label by hand.
              </span>
            </span>
          </label>
          <div className="flex items-center gap-2">
            <button type="submit" className="btn-primary" disabled={mut.isPending || !src || !tgt}>
              {mut.isPending ? "uploading…" : "Upload"}
            </button>
            {mut.isError && (
              <span className="text-sm text-red-600">{(mut.error as Error).message}</span>
            )}
          </div>
        </form>
        <p className="text-xs text-neutral-500">
          Each file is split into chunks with{" "}
          <a
            href="https://github.com/segment-any-text/wtpsplit"
            className="underline decoration-neutral-300 hover:decoration-ink"
            target="_blank"
            rel="noreferrer"
          >
            wtpsplit
          </a>
          ; with the model on, your local <span className="font-mono">vllm serve</span> is asked to propose
          which source chunks align with which target chunks. You can edit everything afterwards.
        </p>
      </main>
    </div>
  );
}
