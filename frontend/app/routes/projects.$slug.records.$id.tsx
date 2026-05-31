import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { AlignmentEditor, type AlignmentEditorActions } from "~/components/AlignmentEditor";
import { ConfirmDialog } from "~/components/ConfirmDialog";
import { HelpOverlay } from "~/components/HelpOverlay";
import { MobileActionBar } from "~/components/MobileActionBar";
import { MobileSegmentView } from "~/components/MobileSegmentView";
import { ModelDiffBanner } from "~/components/ModelDiffBanner";
import { Toolbar } from "~/components/Toolbar";
import { VisuallyHidden } from "~/components/VisuallyHidden";
import {
  type AlignmentState,
  type Pair,
  type Selection,
  type Side,
  addChunkAfter,
  bumpBoundary,
  buildSegments,
  clampPairs,
  clampSelection,
  deleteChunk,
  editChunkText,
  insertBoundaryAfterChunk,
  isOnlyTextEdit,
  mergeWithNext,
  moveChunkToNextSegment,
  moveChunkToPrevSegment,
  moveSelection,
  removeBoundary,
  splitChunk,
  switchSide,
} from "~/lib/alignment";
import { api } from "~/lib/api";
import { useHistory } from "~/lib/useHistory";
import { useHotkeys, modLabel } from "~/lib/useHotkeys";
import type { RecordOut, RecordSummary, TranslateSourceOut } from "~/lib/types";

const EMPTY_STATE: AlignmentState = { srcChunks: [], tgtChunks: [], pairs: [] };

function sameAlignmentState(a: AlignmentState, b: AlignmentState): boolean {
  return (
    a.srcChunks.length === b.srcChunks.length &&
    a.tgtChunks.length === b.tgtChunks.length &&
    a.pairs.length === b.pairs.length &&
    a.srcChunks.every((chunk, i) => chunk === b.srcChunks[i]) &&
    a.tgtChunks.every((chunk, i) => chunk === b.tgtChunks[i]) &&
    a.pairs.every((pair, i) => pair[0] === b.pairs[i]?.[0] && pair[1] === b.pairs[i]?.[1])
  );
}

export default function RecordEditor() {
  const { slug = "", id = "0" } = useParams();
  const recordId = Number(id);
  const qc = useQueryClient();
  const navigate = useNavigate();

  const recordQ = useQuery({
    queryKey: ["record", slug, recordId],
    queryFn: () => api.getRecord(slug, recordId),
    enabled: !!slug && recordId > 0,
    staleTime: 60_000,
    refetchOnMount: false,
  });

  const recordsQ = useQuery({
    queryKey: ["records", slug],
    queryFn: () => api.listRecords(slug),
    enabled: !!slug,
  });

  // Server-loaded record snapshot (separate from the editable state).
  const [meta, setMeta] = useState<RecordOut | null>(null);
  const history = useHistory<AlignmentState>(EMPTY_STATE);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [caret, setCaret] = useState(0);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [segmentIdx, setSegmentIdx] = useState(0);
  const [showTranslations, setShowTranslations] = useState(false);
  const [sourceTranslation, setSourceTranslation] = useState<TranslateSourceOut | null>(null);
  const dirtyVersionRef = useRef(0);

  // Reset history + selection whenever the server gives us a different record.
  useEffect(() => {
    if (!recordQ.data) return;
    setMeta(recordQ.data);
    history.reset({
      srcChunks: recordQ.data.src_chunks,
      tgtChunks: recordQ.data.tgt_chunks,
      pairs: recordQ.data.gt_pairs as Pair[],
    });
    setSelection({ side: "src", index: 0 });
    setEditingKey(null);
    setDirty(false);
    setSegmentIdx(0);
    setShowTranslations(false);
    setSourceTranslation(null);
  }, [recordQ.data?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const state = history.state;
  const segments = useMemo(() => buildSegments(state), [state]);
  const validPairs = useMemo(
    () => clampPairs(state.pairs, state.srcChunks.length, state.tgtChunks.length),
    [state.pairs, state.srcChunks.length, state.tgtChunks.length],
  );

  // Keep state in a ref so mutation callbacks can read the latest snapshot
  // without being re-created on every state change. This makes `actions` stable.
  const stateRef = useRef(state);
  stateRef.current = state;
  const metaRef = useRef(meta);
  metaRef.current = meta;

  // Keep selection valid as the underlying chunks change.
  useEffect(() => {
    setSelection((s) => clampSelection(s, state));
  }, [state.srcChunks.length, state.tgtChunks.length]);

  // Keep segmentIdx valid when segment count changes.
  useEffect(() => {
    setSegmentIdx((idx) => Math.max(0, Math.min(idx, segments.length - 1)));
  }, [segments.length]);

  // Sync segmentIdx <-> selection: when selection moves, jump mobile view to that segment.
  useEffect(() => {
    if (!selection) return;
    const found = segments.findIndex((seg) => {
      const range = selection.side === "src" ? seg.src_range : seg.tgt_range;
      return selection.index >= range[0] && selection.index < range[1];
    });
    if (found >= 0) setSegmentIdx(found);
  }, [selection, segments]);

  // When mobile user changes segment via swipe, point selection at the first chunk of that segment.
  const onSegmentChange = useCallback(
    (idx: number) => {
      setSegmentIdx(idx);
      const seg = segments[idx];
      if (!seg) return;
      if (seg.src.length > 0) setSelection({ side: "src", index: seg.src_range[0] });
      else if (seg.tgt.length > 0) setSelection({ side: "tgt", index: seg.tgt_range[0] });
    },
    [segments],
  );

  // Centralised mutation: apply pure function, decide commit-vs-live, mark dirty.
  // Reads state via ref so the callback identity is stable across renders —
  // every chunk card therefore sees the same `actions` object and doesn't
  // re-render on every keystroke.
  const historySet = history.set;
  const markDirty = useCallback(() => {
    dirtyVersionRef.current += 1;
    setDirty(true);
  }, []);
  const applyMutation = useCallback(
    (next: AlignmentState | null) => {
      if (next === null) return;
      const commit = !isOnlyTextEdit(stateRef.current, next);
      historySet(next, { commit });
      markDirty();
    },
    [historySet, markDirty],
  );

  const actions = useMemo<AlignmentEditorActions>(
    () => ({
      editChunkText: (side, i, text) =>
        applyMutation(editChunkText(stateRef.current, side, i, text)),
      splitChunk: (side, i, c) => applyMutation(splitChunk(stateRef.current, side, i, c)),
      mergeWithNext: (side, i) => applyMutation(mergeWithNext(stateRef.current, side, i)),
      deleteChunk: (side, i) => applyMutation(deleteChunk(stateRef.current, side, i)),
      addChunkAfter: (side, i) => {
        const { state: next, newIndex } = addChunkAfter(stateRef.current, side, i);
        applyMutation(next);
        setSelection({ side, index: newIndex });
      },
      insertBoundaryAfter: (side, i) =>
        applyMutation(insertBoundaryAfterChunk(stateRef.current, side, i)),
      bumpBoundary: (pair, side, delta) =>
        applyMutation(bumpBoundary(stateRef.current, pair, side, delta)),
      removeBoundary: (pair) => applyMutation(removeBoundary(stateRef.current, pair)),
      moveChunkToNextSegment: (side, i) =>
        applyMutation(moveChunkToNextSegment(stateRef.current, side, i)),
      moveChunkToPrevSegment: (side, i) =>
        applyMutation(moveChunkToPrevSegment(stateRef.current, side, i)),
    }),
    [applyMutation],
  );

  // ── Server mutations ────────────────────────────────────────────
  type SaveSnapshot = {
    version: number;
    state: AlignmentState;
    title: string | null;
    notes: string | null;
  };

  const toAlignmentState = useCallback(
    (record: RecordOut): AlignmentState => ({
      srcChunks: record.src_chunks,
      tgtChunks: record.tgt_chunks,
      pairs: record.gt_pairs as Pair[],
    }),
    [],
  );

  const saveMut = useMutation({
    mutationFn: (snapshot: SaveSnapshot) =>
      api.patchRecord(slug, recordId, {
        src_chunks: snapshot.state.srcChunks,
        tgt_chunks: snapshot.state.tgtChunks,
        gt_pairs: snapshot.state.pairs,
        notes: snapshot.notes,
        title: snapshot.title,
      }),
    onSuccess: (data, snapshot) => {
      qc.setQueryData(["record", slug, recordId], data);
      qc.invalidateQueries({ queryKey: ["records", slug] });
      if (dirtyVersionRef.current === snapshot.version) {
        const savedState = toAlignmentState(data);
        setMeta(data);
        if (!sameAlignmentState(snapshot.state, savedState)) {
          historySet(savedState, { commit: false });
        }
        setDirty(false);
      }
    },
  });

  const saveNow = useCallback(() => {
    const currentMeta = metaRef.current;
    if (!currentMeta || saveMut.isPending) return;
    saveMut.mutate({
      version: dirtyVersionRef.current,
      state: stateRef.current,
      title: currentMeta.title,
      notes: currentMeta.notes,
    });
  }, [saveMut.isPending, saveMut.mutate]);

  const reviewMut = useMutation({
    mutationFn: (status: "draft" | "reviewed") => api.patchRecord(slug, recordId, { status }),
    onSuccess: (data) => {
      setMeta(data);
      qc.setQueryData(["record", slug, recordId], data);
      qc.invalidateQueries({ queryKey: ["records", slug] });
    },
  });

  // Pending model proposal — re-infer fetches into here without touching the draft.
  // The labeler explicitly applies or discards it.
  const [proposal, setProposal] = useState<{ pairs: Pair[]; response: string; parseError: boolean } | null>(null);

  const reinferMut = useMutation({
    mutationFn: () => api.reinfer(slug, recordId, { persist: false }),
    onSuccess: (out) => {
      setProposal({
        pairs: out.pairs as Pair[],
        response: out.response,
        parseError: out.parse_error,
      });
    },
  });

  const reinferBelowMut = useMutation({
    mutationFn: () => {
      const seg = segments[segmentIdx];
      if (!seg) throw new Error("no segment selected");
      return api.reinfer(slug, recordId, {
        persist: false,
        start_src_index: seg.src_range[1],
        start_tgt_index: seg.tgt_range[1],
      });
    },
    onSuccess: (out) => {
      setProposal({
        pairs: out.pairs as Pair[],
        response: out.response,
        parseError: out.parse_error,
      });
    },
  });

  const translateMut = useMutation({
    mutationFn: () => api.translateSource(slug, recordId),
    onSuccess: (out) => {
      setSourceTranslation(out);
      setShowTranslations(true);
    },
  });

  const applyProposal = useCallback(() => {
    if (!proposal) return;
    history.set(
      { srcChunks: state.srcChunks, tgtChunks: state.tgtChunks, pairs: proposal.pairs },
      { commit: true },
    );
    setMeta((m) => (m ? { ...m, model_pairs: proposal.pairs as unknown as RecordOut["model_pairs"], model_response: proposal.response } : m));
    setProposal(null);
    markDirty();
  }, [proposal, history, state.srcChunks, state.tgtChunks, markDirty]);

  const discardProposal = useCallback(() => setProposal(null), []);

  // Discard a stale proposal whenever the user mutates chunks or boundaries themselves.
  useEffect(() => {
    if (proposal) setProposal(null);
    // Only react to structural shape changes from the user, not the proposal itself.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.srcChunks.length, state.tgtChunks.length, history.pastSize, history.futureSize]);

  // ── Record queue ────────────────────────────────────────────────
  const records: RecordSummary[] = recordsQ.data ?? [];
  const ordered = useMemo(() => [...records].sort((a, b) => a.id - b.id), [records]);
  const currentIdx = ordered.findIndex((r) => r.id === recordId);

  const findNextDraft = useCallback((): RecordSummary | null => {
    const after = ordered.slice(currentIdx + 1).find((r) => r.status === "draft");
    if (after) return after;
    const before = ordered.slice(0, currentIdx).find((r) => r.status === "draft");
    return before ?? null;
  }, [ordered, currentIdx]);

  const dirtyRef = useRef(false);
  const [pendingNav, setPendingNav] = useState<number | null>(null);
  const goToRecord = useCallback(
    (id: number | undefined) => {
      if (id == null) return;
      if (dirtyRef.current) {
        setPendingNav(id);
        return;
      }
      navigate(`/projects/${slug}/records/${id}`);
    },
    [navigate, slug],
  );

  const markReviewedAndAdvance = useCallback(() => {
    reviewMut.mutate("reviewed", {
      onSuccess: () => {
        const next = findNextDraft();
        if (next) goToRecord(next.id);
      },
    });
  }, [reviewMut, findNextDraft, goToRecord]);

  const goNext = useCallback(() => goToRecord(ordered[currentIdx + 1]?.id), [ordered, currentIdx, goToRecord]);
  const goPrev = useCallback(() => goToRecord(ordered[currentIdx - 1]?.id), [ordered, currentIdx, goToRecord]);

  // Track dirty in a ref so navigation handlers can read the latest value.
  useEffect(() => {
    dirtyRef.current = dirty;
  }, [dirty]);

  // Warn before browser reload / tab-close when dirty.
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty]);

  // Debounced autosave: 2s after the last edit, fire a PATCH.
  // No-op if the save is already in flight or there's nothing dirty.
  useEffect(() => {
    if (!dirty) return;
    const t = window.setTimeout(() => {
      if (saveMut.isPending) return;
      saveNow();
    }, 2000);
    return () => window.clearTimeout(t);
    // Re-fire when state.pairs / chunks shapes change (a structural commit).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dirty, state.pairs, state.srcChunks, state.tgtChunks, meta?.notes, meta?.title, saveNow, saveMut.isPending]);

  // ── Hotkeys ─────────────────────────────────────────────────────
  const selRef = useRef(selection);
  selRef.current = selection;
  // stateRef is declared earlier (used by `actions`) — no need to redeclare.

  const splitSelected = useCallback(() => {
    const s = selRef.current;
    if (!s) return;
    actions.splitChunk(s.side, s.index, caret);
  }, [actions, caret]);

  const mergeSelected = useCallback(() => {
    const s = selRef.current;
    if (!s) return;
    actions.mergeWithNext(s.side, s.index);
  }, [actions]);

  const deleteSelected = useCallback(() => {
    const s = selRef.current;
    if (!s) return;
    actions.deleteChunk(s.side, s.index);
  }, [actions]);

  const insertBoundarySelected = useCallback(() => {
    const s = selRef.current;
    if (!s) return;
    actions.insertBoundaryAfter(s.side, s.index);
  }, [actions]);

  const addChunkAfterSelected = useCallback(() => {
    const s = selRef.current;
    if (!s) return;
    actions.addChunkAfter(s.side, s.index);
  }, [actions]);

  const move = useCallback(
    (dir: "next" | "prev") => setSelection((sel) => moveSelection(sel, stateRef.current, dir)),
    [],
  );
  const switchSel = useCallback(
    (side: Side) => setSelection((sel) => switchSide(sel, side, stateRef.current)),
    [],
  );

  useHotkeys({
    "j": () => move("next"),
    "k": () => move("prev"),
    "h": () => switchSel("src"),
    "l": () => switchSel("tgt"),
    "s": splitSelected,
    "m": mergeSelected,
    "d": deleteSelected,
    "b": insertBoundarySelected,
    "o": addChunkAfterSelected,
    "Enter": () => {
      const s = selRef.current;
      if (s) setEditingKey(`${s.side}:${s.index}`);
    },
    "Escape": () => {
      if (helpOpen) {
        setHelpOpen(false);
      } else if (editingKey) {
        setEditingKey(null);
        (document.activeElement as HTMLElement | null)?.blur();
      }
    },
    "?": () => setHelpOpen((o) => !o),
    "shift+?": () => setHelpOpen((o) => !o),
    "mod+s": () => {
      if (dirty && !saveMut.isPending) saveNow();
    },
    "mod+z": () => history.undo(),
    "mod+shift+z": () => history.redo(),
    "mod+Enter": markReviewedAndAdvance,
    "g n": goNext,
    "g p": goPrev,
  });

  // ── Render ──────────────────────────────────────────────────────
  if (recordQ.isLoading || !meta) {
    return (
      <div className="min-h-screen">
        <Toolbar
          crumbs={[
            { to: "/", label: "Projects" },
            { to: `/projects/${slug}`, label: slug },
            { label: `#${recordId}` },
          ]}
        />
        <main className="p-4 text-sm text-neutral-500">loading…</main>
      </div>
    );
  }

  const errorBanner =
    saveMut.isError ||
    reinferMut.isError ||
    reinferBelowMut.isError ||
    reviewMut.isError ||
    translateMut.isError;
  const queuePosition = currentIdx >= 0 ? `${currentIdx + 1} / ${ordered.length}` : `· / ${ordered.length}`;
  const canReinferBelow = segmentIdx >= 0 && segmentIdx < segments.length - 1;

  return (
    <div className="editor-page min-h-screen lg:pb-8">
      <VisuallyHidden as="h1">{meta.title || `Record #${recordId}`}</VisuallyHidden>
      <Toolbar
        crumbs={[
          { to: "/", label: "Projects" },
          { to: `/projects/${slug}`, label: slug },
          { label: `#${recordId} · ${queuePosition}` },
        ]}
        right={
          <>
            {/* Desktop-only toolbar actions; mobile gets these via the bottom action bar */}
            <button
              type="button"
              className="btn hidden lg:inline-flex !min-h-[36px] !min-w-[36px] !px-2 text-xs"
              onClick={goPrev}
              title="previous record (g p)"
              disabled={currentIdx <= 0}
            >
              ◀
            </button>
            <button
              type="button"
              className="btn hidden lg:inline-flex !min-h-[36px] !min-w-[36px] !px-2 text-xs"
              onClick={goNext}
              title="next record (g n)"
              disabled={currentIdx < 0 || currentIdx >= ordered.length - 1}
            >
              ▶
            </button>
            <button
              type="button"
              className="btn hidden lg:inline-flex"
              onClick={() => reinferMut.mutate()}
              disabled={reinferMut.isPending}
            >
              {reinferMut.isPending ? "running model…" : "↻ Re-infer"}
            </button>
            <button
              type="button"
              className="btn hidden lg:inline-flex"
              onClick={() => reinferBelowMut.mutate()}
              disabled={reinferBelowMut.isPending || !canReinferBelow}
              title="re-pair segments below current segment"
            >
              {reinferBelowMut.isPending ? "running model…" : "↻ Re-pair below"}
            </button>
            <button
              type="button"
              className="btn hidden lg:inline-flex"
              onClick={() => history.undo()}
              disabled={!history.canUndo}
              title={`undo (${modLabel()}+z)`}
            >
              undo
            </button>
            <button
              type="button"
              className="btn-primary hidden lg:inline-flex"
              disabled={!dirty || saveMut.isPending}
              onClick={saveNow}
              title={`save (${modLabel()}+s)`}
              aria-live="polite"
            >
              {saveMut.isPending ? "Saving…" : dirty ? "Unsaved" : "Saved"}
            </button>
            <button
              type="button"
              className={`btn hidden lg:inline-flex ${
                meta.status === "reviewed" ? "border-aligned/40 bg-aligned/10 text-neutral-900" : ""
              }`}
              onClick={markReviewedAndAdvance}
              disabled={reviewMut.isPending}
              title={`mark reviewed and jump to next draft (${modLabel()}+enter)`}
            >
              {meta.status === "reviewed" ? "✓ Reviewed, continue" : "Mark reviewed, continue"}
            </button>
            <button
              type="button"
              className="btn !min-h-[36px] !min-w-[36px] !px-2 text-xs"
              onClick={() => setHelpOpen(true)}
              title="keyboard shortcuts (?)"
            >
              ?
            </button>
          </>
        }
      />
      <main className="mx-auto max-w-6xl space-y-3 p-3 sm:p-4">
        {errorBanner && (
          <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            <span className="font-medium">Couldn't reach the server.</span>{" "}
            {(saveMut.error as Error)?.message ||
              (reinferMut.error as Error)?.message ||
              (reinferBelowMut.error as Error)?.message ||
              (reviewMut.error as Error)?.message ||
              (translateMut.error as Error)?.message}
          </div>
        )}

        <input
          className="w-full rounded-md border border-neutral-200 bg-surface px-3 py-2 text-base font-semibold text-ink focus-visible:border-ink focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ink/30 focus:outline-none"
          value={meta.title ?? ""}
          placeholder="untitled"
          onChange={(e) => {
            setMeta({ ...meta, title: e.target.value });
            markDirty();
          }}
        />

        <ModelDiffBanner
          modelPairs={(proposal ? proposal.pairs : (meta.model_pairs as Pair[])) ?? []}
          draftPairs={state.pairs}
          parseError={proposal?.parseError}
          pending={!!proposal}
          onApply={applyProposal}
          onDiscard={discardProposal}
        />

        <div className="flex flex-wrap items-center gap-2 rounded-md border border-neutral-200 bg-surface px-3 py-2">
          <div className="flex rounded-md border border-neutral-200 p-0.5">
            <button
              type="button"
              className={`rounded px-3 py-1.5 text-xs font-medium ${!showTranslations ? "bg-ink text-brand-fg" : "text-neutral-600 hover-fade"}`}
              onClick={() => setShowTranslations(false)}
            >
              original
            </button>
            <button
              type="button"
              className={`rounded px-3 py-1.5 text-xs font-medium ${showTranslations ? "bg-ink text-brand-fg" : "text-neutral-600 hover-fade"}`}
              onClick={() => {
                if (!sourceTranslation) translateMut.mutate();
                else setShowTranslations(true);
              }}
              disabled={translateMut.isPending}
            >
              {translateMut.isPending ? "translating…" : "source MT"}
            </button>
          </div>
          {showTranslations && sourceTranslation?.parse_error && (
            <span className="text-xs text-red-600">Translation loaded, but the model response needed fallback parsing.</span>
          )}
        </div>

        <div className="hidden lg:block">
          <AlignmentEditor
            state={state}
            segments={segments}
            validPairs={validPairs}
            selection={selection}
            onSelect={setSelection}
            caret={caret}
            onCaretChange={setCaret}
            actions={actions}
            editingKey={editingKey}
            onRequestEdit={setEditingKey}
            sourceTranslations={showTranslations ? sourceTranslation?.translations ?? null : null}
          />
        </div>
        <div className="lg:hidden">
          <MobileSegmentView
            state={state}
            segments={segments}
            validPairs={validPairs}
            segmentIdx={segmentIdx}
            onSegmentChange={onSegmentChange}
            selection={selection}
            onSelect={setSelection}
            caret={caret}
            onCaretChange={setCaret}
            actions={actions}
            editingKey={editingKey}
            onRequestEdit={setEditingKey}
            sourceTranslations={showTranslations ? sourceTranslation?.translations ?? null : null}
          />
        </div>

        {meta.model_response && (
          <details className="rounded-lg border border-neutral-200 bg-surface p-3">
            <summary className="cursor-pointer text-xs font-medium text-neutral-600 hover:text-ink">
              raw model output
            </summary>
            <pre className="mt-2 overflow-x-auto whitespace-pre-wrap rounded bg-neutral-50 p-2 font-mono text-xs leading-relaxed text-neutral-700">
              {meta.model_response}
            </pre>
          </details>
        )}

        <label className="block">
          <span className="eyebrow text-neutral-700">notes</span>
          <textarea
            className="mt-1 w-full rounded-md border border-neutral-200 bg-surface px-3 py-2 text-sm focus-visible:border-ink focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ink/30 focus:outline-none"
            rows={2}
            value={meta.notes ?? ""}
            placeholder="anything worth remembering about this record…"
            onChange={(e) => {
              setMeta({ ...meta, notes: e.target.value });
              markDirty();
            }}
          />
        </label>
      </main>
      <MobileActionBar
        segmentIdx={segmentIdx}
        segmentTotal={segments.length}
        onSegmentPrev={() => onSegmentChange(Math.max(0, segmentIdx - 1))}
        onSegmentNext={() => onSegmentChange(Math.min(segments.length - 1, segmentIdx + 1))}
        onUndo={() => history.undo()}
        canUndo={history.canUndo}
        onSave={saveNow}
        saving={saveMut.isPending}
        dirty={dirty}
        onReviewAndNext={markReviewedAndAdvance}
        reviewing={reviewMut.isPending}
        reviewed={meta.status === "reviewed"}
        recordIdx={currentIdx}
        recordTotal={ordered.length}
        onRecordPrev={goPrev}
        onRecordNext={goNext}
      />
      <HelpOverlay open={helpOpen} onClose={() => setHelpOpen(false)} />
      <ConfirmDialog
        open={pendingNav != null}
        title="You have unsaved changes"
        description="Leave this record without saving? Your edits since the last save will be lost."
        confirmLabel="Leave without saving"
        cancelLabel="Stay"
        destructive
        onConfirm={() => {
          if (pendingNav != null) navigate(`/projects/${slug}/records/${pendingNav}`);
          setPendingNav(null);
        }}
        onCancel={() => setPendingNav(null)}
      />
    </div>
  );
}
