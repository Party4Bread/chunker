import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";
import { ConfirmDialog } from "~/components/ConfirmDialog";
import { Toolbar } from "~/components/Toolbar";
import { VisuallyHidden } from "~/components/VisuallyHidden";
import { api } from "~/lib/api";
import type { Project } from "~/lib/types";

export default function ProjectsIndex() {
  const qc = useQueryClient();
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const [name, setName] = useState("");

  const createMut = useMutation({
    mutationFn: () => api.createProject(name.trim()),
    onSuccess: () => {
      setName("");
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  const deleteMut = useMutation({
    mutationFn: (slug: string) => api.deleteProject(slug),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["projects"] }),
  });
  const [pendingDelete, setPendingDelete] = useState<Project | null>(null);

  return (
    <div className="min-h-screen">
      <VisuallyHidden as="h1">Projects</VisuallyHidden>
      <Toolbar crumbs={[{ label: "Projects" }]} />
      <main className="mx-auto max-w-4xl space-y-6 p-4">
        <section className="rounded-lg bg-white p-4 ring-1 ring-neutral-200">
          <h2 className="mb-1 text-base font-semibold text-ink">Start a new project</h2>
          <p className="mb-3 text-xs text-neutral-500">
            Each project gets its own database. Records you upload here stay isolated.
          </p>
          <form
            className="flex flex-wrap items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (name.trim()) createMut.mutate();
            }}
          >
            <input
              type="text"
              placeholder="project name (e.g. ko-en-novels)"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="flex-1 rounded-md border border-neutral-200 px-3 py-2 text-sm"
            />
            <button type="submit" className="btn-primary" disabled={!name.trim() || createMut.isPending}>
              {createMut.isPending ? "creating…" : "Create project"}
            </button>
          </form>
        </section>

        <section>
          <h2 className="mb-3 text-base font-semibold text-ink">Your projects</h2>
          {projects.isLoading && <p className="text-sm text-neutral-500">loading…</p>}
          {projects.isError && (
            <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              couldn't load projects: {(projects.error as Error).message}
            </p>
          )}
          <ul className="grid gap-3 sm:grid-cols-2">
            {(projects.data ?? []).map((p) => (
              <li key={p.slug} className="chunk-card flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <Link
                    to={`/projects/${p.slug}`}
                    className="block truncate text-base font-semibold text-ink hover:underline"
                  >
                    {p.name}
                  </Link>
                  <p className="mt-1 text-xs text-neutral-500">
                    {p.record_count === 0
                      ? "no records yet"
                      : `${p.record_count} record${p.record_count === 1 ? "" : "s"}`}
                    <span className="mx-1.5 text-neutral-300">·</span>
                    <span className="font-mono">{p.model_name}</span>
                  </p>
                </div>
                <button
                  type="button"
                  className="btn-danger !min-h-[36px] !min-w-[36px] !px-2 text-xs"
                  onClick={() => setPendingDelete(p)}
                  aria-label={`delete ${p.name}`}
                  title="delete project"
                >
                  ×
                </button>
              </li>
            ))}
            {projects.data && projects.data.length === 0 && (
              <li className="col-span-full rounded-md border border-dashed border-neutral-300 bg-white p-8 text-center">
                <p className="text-sm font-medium text-ink">No projects yet</p>
                <p className="mt-1 text-xs text-neutral-500">
                  Create one above to start uploading source / target text pairs.
                </p>
              </li>
            )}
          </ul>
        </section>
      </main>
      <ConfirmDialog
        open={!!pendingDelete}
        title={pendingDelete ? `Delete "${pendingDelete.name}"?` : ""}
        description="This removes the project's database file. You can't undo it."
        confirmLabel="Delete"
        destructive
        onConfirm={() => {
          if (pendingDelete) deleteMut.mutate(pendingDelete.slug);
          setPendingDelete(null);
        }}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}
