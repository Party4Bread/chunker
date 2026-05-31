import { useMutation } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { Toolbar } from "~/components/Toolbar";
import { api } from "~/lib/api";

export default function UploadPair() {
  const { slug = "" } = useParams();
  const navigate = useNavigate();
  const [src, setSrc] = useState<File | null>(null);
  const [tgt, setTgt] = useState<File | null>(null);
  const [srcFiles, setSrcFiles] = useState<File[]>([]);
  const [tgtFiles, setTgtFiles] = useState<File[]>([]);
  const [title, setTitle] = useState("");
  const [runModel, setRunModel] = useState(true);
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [batchResult, setBatchResult] = useState<{ created: number; firstId: number | null; errors: string[] } | null>(null);

  const mut = useMutation({
    mutationFn: async () => {
      if (!src || !tgt) throw new Error("pick both files");
      return api.uploadRecord(slug, src, tgt, { title: title.trim() || undefined, runModel });
    },
    onSuccess: (rec) => navigate(`/projects/${slug}/records/${rec.id}`),
  });

  const batchMut = useMutation({
    mutationFn: () => {
      if (srcFiles.length === 0 || tgtFiles.length === 0) throw new Error("pick source and target files");
      return api.uploadBatchRecords(slug, srcFiles, tgtFiles, { runModel });
    },
    onSuccess: (out) => {
      const firstId = out.records[0]?.id ?? null;
      setBatchResult({
        created: out.records.length,
        firstId,
        errors: out.errors.map((err) => {
          const srcName = err.src_file ? `source ${err.src_file}` : "";
          const tgtName = err.tgt_file ? `target ${err.tgt_file}` : "";
          const files = [srcName, tgtName].filter(Boolean).join(" / ");
          return files ? `${files}: ${err.detail}` : err.detail;
        }),
      });
      if (out.records.length > 0 && out.errors.length === 0) navigate(`/projects/${slug}/records/${out.records[0].id}`);
    },
  });

  const sortedSrcNames = useMemo(() => srcFiles.map((f) => f.webkitRelativePath || f.name).sort(), [srcFiles]);
  const sortedTgtNames = useMemo(() => tgtFiles.map((f) => f.webkitRelativePath || f.name).sort(), [tgtFiles]);
  const pending = mut.isPending || batchMut.isPending;

  return (
    <div className="min-h-screen">
      <Toolbar
        crumbs={[{ to: "/", label: "Projects" }, { to: `/projects/${slug}`, label: slug }, { label: "Upload" }]}
      />
      <main className="mx-auto max-w-2xl space-y-4 p-4">
        <form
          className="space-y-4 rounded-lg bg-surface p-4 ring-1 ring-neutral-200"
          onSubmit={(e) => {
            e.preventDefault();
            if (mode === "single") mut.mutate();
            else batchMut.mutate();
          }}
        >
          <div className="inline-flex rounded-md border border-neutral-200 p-0.5">
            <button
              type="button"
              className={`rounded px-3 py-1.5 text-xs font-medium ${mode === "single" ? "bg-ink text-brand-fg" : "text-neutral-600 hover-fade"}`}
              onClick={() => setMode("single")}
            >
              pair
            </button>
            <button
              type="button"
              className={`rounded px-3 py-1.5 text-xs font-medium ${mode === "batch" ? "bg-ink text-brand-fg" : "text-neutral-600 hover-fade"}`}
              onClick={() => setMode("batch")}
            >
              batch
            </button>
          </div>

          {mode === "single" ? (
            <>
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
            </>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2">
              <FileSetPicker
                label="source chapters"
                files={sortedSrcNames}
                onChange={setSrcFiles}
              />
              <FileSetPicker
                label="target chapters"
                files={sortedTgtNames}
                onChange={setTgtFiles}
              />
            </div>
          )}
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
            <button
              type="submit"
              className="btn-primary"
              disabled={
                pending ||
                (mode === "single" ? !src || !tgt : srcFiles.length === 0 || tgtFiles.length === 0)
              }
            >
              {pending ? "uploading…" : mode === "single" ? "Upload" : "Upload batch"}
            </button>
            {(mut.isError || batchMut.isError) && (
              <span className="text-sm text-red-600">{((mut.error || batchMut.error) as Error).message}</span>
            )}
          </div>
          {batchResult && mode === "batch" && (
            <div className="rounded-md border border-neutral-200 bg-neutral-50 px-3 py-2 text-sm">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-medium text-ink">{batchResult.created} records created</span>
                {batchResult.firstId && (
                  <button
                    type="button"
                    className="btn !min-h-[32px] !px-2 text-xs"
                    onClick={() => navigate(`/projects/${slug}/records/${batchResult.firstId}`)}
                  >
                    open first
                  </button>
                )}
              </div>
              {batchResult.errors.length > 0 && (
                <ul className="mt-2 space-y-1 text-xs text-red-600">
                  {batchResult.errors.map((err) => (
                    <li key={err}>{err}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
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

function FileSetPicker({
  label,
  files,
  onChange,
}: {
  label: string;
  files: string[];
  onChange: (files: File[]) => void;
}) {
  return (
    <label className="block">
      <span className="eyebrow text-neutral-700">{label}</span>
      <input
        type="file"
        accept=".txt,text/plain"
        multiple
        {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
        className="mt-1 block w-full text-sm"
        onChange={(e) => onChange(Array.from(e.target.files ?? []))}
      />
      <span className="mt-1 block text-2xs text-neutral-500">
        {files.length === 0 ? "select multiple UTF-8 text files" : `${files.length} files selected`}
      </span>
      {files.length > 0 && (
        <ol className="mt-2 max-h-40 space-y-1 overflow-auto rounded border border-neutral-200 bg-neutral-50 p-2">
          {files.slice(0, 20).map((name) => (
            <li key={name} className="truncate font-mono text-2xs text-neutral-600" title={name}>
              {name}
            </li>
          ))}
          {files.length > 20 && (
            <li className="font-mono text-2xs text-neutral-500">+{files.length - 20} more</li>
          )}
        </ol>
      )}
    </label>
  );
}
