import type { Pair } from "~/lib/alignment";

interface ModelDiffBannerProps {
  modelPairs: Pair[];
  draftPairs: Pair[];
  parseError?: boolean;
  pending?: boolean;
  onApply?: () => void;
  onDiscard?: () => void;
  applying?: boolean;
  canApplyWhenSame?: boolean;
}

function key(p: Pair): string {
  return `${p[0]}-${p[1]}`;
}

function diff(model: Pair[], draft: Pair[]) {
  const modelSet = new Set(model.map(key));
  const draftSet = new Set(draft.map(key));
  return {
    kept: model.filter((p) => draftSet.has(key(p))).length,
    removed: model.filter((p) => !draftSet.has(key(p))).length,
    added: draft.filter((p) => !modelSet.has(key(p))).length,
  };
}

export function ModelDiffBanner({
  modelPairs,
  draftPairs,
  parseError,
  pending,
  onApply,
  onDiscard,
  applying,
  canApplyWhenSame,
}: ModelDiffBannerProps) {
  if (parseError) {
    return (
      <div className="flex items-start justify-between gap-3 rounded-md border border-srcOnly/30 bg-srcOnly/[0.06] px-3 py-2 text-sm">
        <span className="flex items-start gap-2">
          <span className="mt-1.5 inline-block h-2 w-2 rounded-full bg-srcOnly" />
          <span>
            <span className="font-medium text-ink">Model output did not parse.</span>{" "}
            <span className="text-neutral-600">Re-infer to try again, or place boundaries by hand.</span>
          </span>
        </span>
        {pending && onDiscard && (
          <button type="button" onClick={onDiscard} className="btn !min-h-[32px] !px-2 text-xs">
            Discard
          </button>
        )}
      </div>
    );
  }

  if (modelPairs.length === 0 && draftPairs.length === 0) {
    return (
      <p className="text-xs text-neutral-500">
        No model proposal yet. Press <span className="font-mono">Re-infer</span> to ask the model.
      </p>
    );
  }

  const d = diff(modelPairs, draftPairs);
  const totalChanges = d.removed + d.added;
  const canApply = totalChanges > 0 || !!canApplyWhenSame;

  if (pending) {
    return (
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-brand/30 bg-brand-subtle px-3 py-2 text-sm">
        <span className="flex items-center gap-3">
          <span className="font-medium text-ink">Model proposal ready.</span>
          {totalChanges === 0 ? (
            <span className="text-neutral-600">
              {canApplyWhenSame ? "Pair boundaries match; chunks may still change." : "Matches your current draft exactly."}
            </span>
          ) : (
            <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-neutral-600">
              {d.kept > 0 && <span><span className="font-mono text-ink">{d.kept}</span> would stay</span>}
              {d.removed > 0 && (
                <span className="flex items-center gap-1">
                  <span className="inline-block h-1 w-1 rounded-full bg-red-500" />
                  <span className="font-mono text-ink">{d.removed}</span> of yours dropped
                </span>
              )}
              {d.added > 0 && (
                <span className="flex items-center gap-1">
                  <span className="inline-block h-1 w-1 rounded-full bg-aligned" />
                  <span className="font-mono text-ink">{d.added}</span> would be added
                </span>
              )}
            </span>
          )}
        </span>
        <span className="flex items-center gap-2">
          <button type="button" onClick={onDiscard} className="btn !min-h-[32px] !px-2.5 text-xs">
            Discard
          </button>
          <button
            type="button"
            onClick={onApply}
            disabled={applying || !canApply}
            className="btn-primary !min-h-[32px] !px-3 text-xs"
          >
            {applying
              ? "Applying..."
              : totalChanges === 0
                ? "Apply rechunk"
                : `Apply ${totalChanges} change${totalChanges === 1 ? "" : "s"}`}
          </button>
        </span>
      </div>
    );
  }

  if (d.kept === modelPairs.length && d.removed === 0 && d.added === 0) {
    return (
      <p className="flex items-center gap-2 text-xs text-neutral-500">
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-aligned" />
        Matches the model's last proposal exactly ({modelPairs.length} alignment
        {modelPairs.length === 1 ? "" : "s"}).
      </p>
    );
  }

  return (
    <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-neutral-600">
      <span className="text-neutral-500">vs. last model run:</span>
      {d.kept > 0 && <span><span className="font-mono text-ink">{d.kept}</span> kept</span>}
      {d.removed > 0 && (
        <span className="flex items-center gap-1">
          <span className="inline-block h-1 w-1 rounded-full bg-red-500" />
          <span className="font-mono text-ink">{d.removed}</span> removed
        </span>
      )}
      {d.added > 0 && (
        <span className="flex items-center gap-1">
          <span className="inline-block h-1 w-1 rounded-full bg-aligned" />
          <span className="font-mono text-ink">{d.added}</span> added
        </span>
      )}
    </p>
  );
}
