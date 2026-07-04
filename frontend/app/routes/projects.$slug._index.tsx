import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router";
import { ConfirmDialog } from "~/components/ConfirmDialog";
import { Toolbar } from "~/components/Toolbar";
import { VisuallyHidden } from "~/components/VisuallyHidden";
import { api } from "~/lib/api";

export default function ProjectIndex() {
  const { slug = "" } = useParams();
  const qc = useQueryClient();
  const records = useQuery({
    queryKey: ["records", slug],
    queryFn: () => api.listRecords(slug),
    enabled: !!slug,
  });

  const deleteMut = useMutation({
    mutationFn: (id: number) => api.deleteRecord(slug, id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["records", slug] }),
  });
  const [pendingDelete, setPendingDelete] = useState<number | null>(null);

  const counts = useMemo(() => {
    const rows = records.data ?? [];
    const reviewed = rows.filter((r) => r.status === "reviewed").length;
    return { total: rows.length, reviewed, draft: rows.length - reviewed };
  }, [records.data]);

  return (
    <div className="min-h-screen">
      <VisuallyHidden as="h1">Records in {slug}</VisuallyHidden>
      <Toolbar
        crumbs={[{ to: "/", label: "Projects" }, { label: slug }]}
        right={
          <>
            <Link to={`/projects/${slug}/upload`} className="btn-primary">
              + Upload pair
            </Link>
            <a
              href={api.exportUrl(slug, "reviewed")}
              className="btn"
              title="download reviewed records as JSONL (training-set shape)"
            >
              ↓ Export reviewed
            </a>
            <a
              href={api.exportUrl(slug, "all")}
              className="btn !min-h-[36px] !px-2 text-xs"
              title="download every record as JSONL"
            >
              ↓ All
            </a>
          </>
        }
      />
      <main className="mx-auto max-w-5xl space-y-3 p-4">
        {counts.total > 0 && (
          <p className="text-xs text-neutral-500">
            <span className="font-mono text-ink">{counts.reviewed}</span> reviewed ·{" "}
            <span className="font-mono text-ink">{counts.draft}</span> still draft
            {counts.draft > 0 && counts.total > 0 && (
              <span className="ml-2 text-neutral-400">
                ({Math.round((counts.reviewed / counts.total) * 100)}% done)
              </span>
            )}
          </p>
        )}
        {records.isLoading && <p className="text-sm text-neutral-500">loading…</p>}
        {records.isError && (
          <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            couldn't load records: {(records.error as Error).message}
          </p>
        )}
        <ul className="flex flex-col gap-2">
          {(records.data ?? []).map((r) => (
            <li key={r.id} className="chunk-card flex flex-wrap items-center justify-between gap-3">
              <div className="min-w-0">
                <Link
                  to={`/projects/${slug}/records/${r.id}`}
                  className="block truncate text-sm font-semibold text-ink hover:underline"
                >
                  <span className="font-mono text-xs text-neutral-400">#{r.id}</span> {r.title || "untitled"}
                </Link>
                <p className="mt-1 text-xs text-neutral-500">
                  <span className="font-mono text-ink">{r.n_src_chunks}</span> source ·{" "}
                  <span className="font-mono text-ink">{r.n_tgt_chunks}</span> target ·{" "}
                  <span className="font-mono text-ink">{r.n_pairs}</span> alignments
                </p>
              </div>
              <div className="flex items-center gap-2">
                <StatusChip status={r.status} />
                <button
                  type="button"
                  className="btn-danger !min-h-[36px] !min-w-[36px] !px-2 text-xs"
                  onClick={() => setPendingDelete(r.id)}
                  aria-label={`delete record ${r.id}`}
                  title="delete record"
                >
                  ×
                </button>
              </div>
            </li>
          ))}
          {records.data && records.data.length === 0 && (
            <li className="rounded-md border border-dashed border-neutral-300 bg-surface p-8 text-center">
              <p className="text-sm font-medium text-ink">No records yet</p>
              <p className="mt-1 text-xs text-neutral-500">
                Upload a source / target text pair to get started.
              </p>
              <Link to={`/projects/${slug}/upload`} className="btn-primary mt-4 inline-flex">
                + Upload pair
              </Link>
            </li>
          )}
        </ul>
      </main>
      <ConfirmDialog
        open={pendingDelete != null}
        title={pendingDelete != null ? `Delete record #${pendingDelete}?` : ""}
        description="This can't be undone."
        confirmLabel="Delete"
        destructive
        onConfirm={() => {
          if (pendingDelete != null) deleteMut.mutate(pendingDelete);
          setPendingDelete(null);
        }}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}

function StatusChip({ status }: { status: "draft" | "reviewed" }) {
  if (status === "reviewed") {
    return (
      <span className="chip bg-aligned/10 text-ink ring-1 ring-aligned/30">
        <span className="h-1.5 w-1.5 rounded-full bg-aligned" />
        reviewed
      </span>
    );
  }
  return (
    <span className="chip bg-srcOnly/10 text-ink ring-1 ring-srcOnly/30">
      <span className="h-1.5 w-1.5 rounded-full bg-srcOnly" />
      draft
    </span>
  );
}
