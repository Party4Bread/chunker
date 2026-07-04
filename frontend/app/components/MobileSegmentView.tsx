import { useRef } from "react";
import {
  type AlignmentState,
  type Pair,
  type Selection,
  type Side,
  formatChunkRange,
} from "~/lib/alignment";
import type { ChunkedSegment } from "~/lib/types";
import type { AlignmentEditorActions } from "./AlignmentEditor";
import { ChunkCard } from "./ChunkCard";

interface MobileSegmentViewProps {
  state: AlignmentState;
  segments: ChunkedSegment[];
  validPairs: Pair[];
  segmentIdx: number;
  onSegmentChange: (idx: number) => void;
  selection: Selection | null;
  onSelect: (sel: Selection) => void;
  caret: number;
  onCaretChange: (caret: number) => void;
  actions: AlignmentEditorActions;
  editingKey?: string | null;
  onRequestEdit?: (key: string | null) => void;
  sourceTranslations?: string[] | null;
}

const TYPE_LABEL: Record<ChunkedSegment["type"], string> = {
  aligned: "aligned",
  src_only_unaligned: "source only",
  tgt_only_unaligned: "target only",
  empty: "empty",
};

const TYPE_TONE: Record<ChunkedSegment["type"], string> = {
  aligned: "tone-aligned",
  src_only_unaligned: "tone-src-only",
  tgt_only_unaligned: "tone-tgt-only",
  empty: "tone-empty",
};

const TYPE_LABEL_TONE: Record<ChunkedSegment["type"], string> = {
  aligned: "tone-text-aligned",
  src_only_unaligned: "tone-text-src",
  tgt_only_unaligned: "tone-text-tgt",
  empty: "text-neutral-500",
};

const TYPE_DOT: Record<ChunkedSegment["type"], string> = {
  aligned: "bg-aligned",
  src_only_unaligned: "bg-srcOnly",
  tgt_only_unaligned: "bg-tgtOnly",
  empty: "bg-neutral-400",
};

function chunkKey(side: Side, absIdx: number): string {
  return `${side}:${absIdx}`;
}

export function MobileSegmentView({
  state,
  segments,
  validPairs,
  segmentIdx,
  onSegmentChange,
  selection,
  onSelect,
  caret,
  onCaretChange,
  actions,
  editingKey,
  onRequestEdit,
  sourceTranslations,
}: MobileSegmentViewProps) {
  const total = segments.length;
  const safeIdx = Math.min(Math.max(0, segmentIdx), Math.max(0, total - 1));
  const seg = segments[safeIdx];
  const trailingBoundary = safeIdx < total - 1 ? validPairs[safeIdx] : null;
  const selectedKey = selection ? chunkKey(selection.side, selection.index) : null;
  const translations = sourceTranslations
    ? seg?.src.map((_, i) => sourceTranslations[seg.src_range[0] + i] ?? "")
    : null;

  // Swipe handling.
  // Ignore swipes that start inside any interactive surface — textareas, inputs,
  // buttons. Otherwise selecting Korean text by drag flips the segment under the
  // labeler's finger.
  const swipeStart = useRef<{ x: number; y: number } | null>(null);
  const swipeFromInteractive = (target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) return false;
    return !!target.closest("textarea, input, button, [role='button'], a");
  };
  const onTouchStart = (e: React.TouchEvent) => {
    if (swipeFromInteractive(e.target)) {
      swipeStart.current = null;
      return;
    }
    const t = e.touches[0];
    swipeStart.current = { x: t.clientX, y: t.clientY };
  };
  const onTouchEnd = (e: React.TouchEvent) => {
    const start = swipeStart.current;
    swipeStart.current = null;
    if (!start) return;
    // If the user just selected text (drag-to-select), do not interpret
    // the same gesture as a segment swipe.
    if (typeof window !== "undefined" && (window.getSelection()?.toString().length ?? 0) > 0) {
      return;
    }
    const t = e.changedTouches[0];
    const dx = t.clientX - start.x;
    const dy = t.clientY - start.y;
    // Stricter thresholds so accidental scrolls don't flip segments.
    if (Math.abs(dx) < 80 || Math.abs(dy) > 40 || Math.abs(dx) < Math.abs(dy) * 2) return;
    if (dx < 0 && safeIdx < total - 1) onSegmentChange(safeIdx + 1);
    if (dx > 0 && safeIdx > 0) onSegmentChange(safeIdx - 1);
  };

  if (!seg) {
    return <div className="rounded-md border border-dashed border-neutral-300 p-6 text-center text-sm text-neutral-500">no segments</div>;
  }

  return (
    <section className="space-y-3" onTouchStart={onTouchStart} onTouchEnd={onTouchEnd}>
      <SegmentCounter
        idx={safeIdx}
        total={total}
        onPrev={() => safeIdx > 0 && onSegmentChange(safeIdx - 1)}
        onNext={() => safeIdx < total - 1 && onSegmentChange(safeIdx + 1)}
      />

      <article className={`rounded-lg p-3 ${TYPE_TONE[seg.type]}`}>
        <header className="mb-3 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 text-xs font-medium text-neutral-700">
          <span className="flex items-center gap-2">
            <span className={`h-1.5 w-1.5 rounded-full ${TYPE_DOT[seg.type]}`} />
            <span className={`font-semibold ${TYPE_LABEL_TONE[seg.type]}`}>{TYPE_LABEL[seg.type]}</span>
          </span>
          <span className="text-2xs text-neutral-500">
            source <span className="font-mono text-ink">{formatChunkRange(seg.src_range, state.srcChunks.length)}</span>
            <span className="ml-1 font-mono text-neutral-400">{seg.src.reduce((n, c) => n + c.length, 0)}ch</span>
            <span className="mx-1.5 text-neutral-300">·</span>
            target <span className="font-mono text-ink">{formatChunkRange(seg.tgt_range, state.tgtChunks.length)}</span>
            <span className="ml-1 font-mono text-neutral-400">{seg.tgt.reduce((n, c) => n + c.length, 0)}ch</span>
          </span>
        </header>

        <SideHeader side="src" />
        <ChunkColumn
          accent="src"
          chunks={seg.src}
          baseAbsIndex={seg.src_range[0]}
          translations={translations ?? undefined}
          hasPrevSegment={safeIdx > 0}
          hasNextSegment={safeIdx < total - 1}
          actions={actions}
          selectedKey={selectedKey}
          onSelect={onSelect}
          caret={caret}
          onCaretChange={onCaretChange}
          editingKey={editingKey ?? null}
          onRequestEdit={onRequestEdit}
        />

        <SideHeader side="tgt" extraTopMargin />
        <ChunkColumn
          accent="tgt"
          chunks={seg.tgt}
          baseAbsIndex={seg.tgt_range[0]}
          hasPrevSegment={safeIdx > 0}
          hasNextSegment={safeIdx < total - 1}
          actions={actions}
          selectedKey={selectedKey}
          onSelect={onSelect}
          caret={caret}
          onCaretChange={onCaretChange}
          editingKey={editingKey ?? null}
          onRequestEdit={onRequestEdit}
        />
      </article>

      {trailingBoundary && (
        <BoundaryDivider
          boundary={trailingBoundary}
          onRemove={() => actions.removeBoundary(trailingBoundary)}
          onBump={(side, delta) => actions.bumpBoundary(trailingBoundary, side, delta)}
        />
      )}
    </section>
  );
}

function SegmentCounter({ idx, total, onPrev, onNext }: { idx: number; total: number; onPrev: () => void; onNext: () => void }) {
  return (
    <div className="flex items-center justify-between rounded-md bg-surface px-3 py-2 ring-1 ring-neutral-200">
      <button
        type="button"
        className="btn !min-h-[40px] !min-w-[40px] !px-2 text-sm"
        onClick={onPrev}
        disabled={idx <= 0}
        aria-label="previous segment"
      >
        ◀
      </button>
      <span className="font-mono text-sm text-neutral-700">
        segment <span className="font-semibold text-neutral-900">{idx + 1}</span>
        <span className="text-neutral-400"> / {total}</span>
      </span>
      <button
        type="button"
        className="btn !min-h-[40px] !min-w-[40px] !px-2 text-sm"
        onClick={onNext}
        disabled={idx >= total - 1}
        aria-label="next segment"
      >
        ▶
      </button>
    </div>
  );
}

function SideHeader({ side, extraTopMargin = false }: { side: Side; extraTopMargin?: boolean }) {
  const dot = side === "src" ? "bg-srcOnly" : "bg-tgtOnly";
  const label = side === "src" ? "source" : "target";
  return (
    <div className={`mb-1 flex items-center gap-1.5 eyebrow ${extraTopMargin ? "mt-3" : ""}`}>
      <span className={`h-1 w-1 rounded-full ${dot}`} />
      {label}
    </div>
  );
}

interface ChunkColumnProps {
  accent: Side;
  chunks: string[];
  baseAbsIndex: number;
  hasPrevSegment: boolean;
  hasNextSegment: boolean;
  actions: AlignmentEditorActions;
  selectedKey: string | null;
  onSelect: (sel: Selection) => void;
  caret: number;
  onCaretChange: (caret: number) => void;
  editingKey: string | null;
  onRequestEdit?: (key: string | null) => void;
  /** Per-chunk source translations (source column only). When set, each source
   *  chunk's MT is stacked directly beneath it so the pairing stays obvious. */
  translations?: string[];
}

function ChunkColumn(props: ChunkColumnProps) {
  const { accent, chunks, baseAbsIndex, hasPrevSegment, hasNextSegment, actions, selectedKey, onSelect, caret, onCaretChange, editingKey, onRequestEdit, translations } = props;
  if (chunks.length === 0) {
    return (
      <p className="px-2 py-2 text-xs italic text-neutral-400">
        — no {accent === "src" ? "source" : "target"} match
      </p>
    );
  }
  return (
    <ol className="flex flex-col">
      {chunks.map((text, i) => {
        const absIdx = baseAbsIndex + i;
        const key = chunkKey(accent, absIdx);
        const isFirstInSeg = i === 0;
        const isLastInSeg = i === chunks.length - 1;
        return (
          <li key={key}>
            <ChunkCard
              displayIndex={absIdx + 1}
              text={text}
              accent={accent}
              variant="mobile"
              selected={selectedKey === key}
              editing={editingKey === key}
              onSelect={() => onSelect({ side: accent, index: absIdx })}
              onCaretChange={onCaretChange}
              onEdit={(t) => actions.editChunkText(accent, absIdx, t)}
              onSplit={(c) => actions.splitChunk(accent, absIdx, c)}
              onMergeNext={i < chunks.length - 1 ? () => actions.mergeWithNext(accent, absIdx) : undefined}
              onMoveToPrevSegment={
                isFirstInSeg && hasPrevSegment
                  ? () => actions.moveChunkToPrevSegment(accent, absIdx)
                  : undefined
              }
              onMoveToNextSegment={
                isLastInSeg && hasNextSegment
                  ? () => actions.moveChunkToNextSegment(accent, absIdx)
                  : undefined
              }
              onDelete={() => actions.deleteChunk(accent, absIdx)}
              onRequestEdit={(editing) => onRequestEdit?.(editing ? key : null)}
              caretFromState={selectedKey === key ? caret : null}
            />
            {translations && <MobileMTCard text={translations[i] ?? ""} displayIndex={absIdx + 1} />}
            {i < chunks.length - 1 && (
              <SplitSegmentBtn
                accent={accent}
                onClick={() => actions.insertBoundaryAfter(accent, absIdx)}
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}

// MT stacked directly under its source chunk on mobile, inset to read as a
// child of the chunk above it. Empty = not fetched yet (muted placeholder).
function MobileMTCard({ text, displayIndex }: { text: string; displayIndex: number }) {
  const pending = !text;
  return (
    <div className="ml-3 mt-1 rounded-md border border-neutral-200 bg-brand-subtle/60 px-3 py-2">
      <div className="mb-1 flex items-baseline gap-1.5">
        <span className="font-mono text-2xs font-semibold text-neutral-500">MT [|{displayIndex}|]</span>
        {!pending && <span className="font-mono text-2xs text-neutral-400">{text.length}ch</span>}
      </div>
      <p
        className={`whitespace-pre-wrap font-serif text-base leading-relaxed ${
          pending ? "italic text-neutral-400" : "text-ink"
        }`}
      >
        {text || "translating…"}
      </p>
    </div>
  );
}

function SplitSegmentBtn({ accent, onClick }: { accent: Side; onClick: () => void }) {
  const tone = accent === "src" ? "text-srcOnly bg-srcOnly/5 active:bg-srcOnly/15" : "text-tgtOnly bg-tgtOnly/5 active:bg-tgtOnly/15";
  return (
    <button
      type="button"
      onClick={onClick}
      className={`my-1 flex h-7 w-full items-center justify-center rounded text-2xs font-medium opacity-70 ${tone}`}
      aria-label="split segment here"
    >
      ↕ break segment here
    </button>
  );
}

interface BoundaryDividerProps {
  boundary: Pair;
  onRemove: () => void;
  onBump: (side: Side, delta: number) => void;
}

function BoundaryDivider({ boundary, onRemove, onBump }: BoundaryDividerProps) {
  return (
    <div className="rounded-lg border border-neutral-300 bg-surface p-3 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <span className="eyebrow">boundary</span>
        <button
          type="button"
          onClick={onRemove}
          className="rounded px-2 py-1 text-xs text-red-600 hover:bg-red-50 active:bg-red-100"
          aria-label="remove boundary"
        >
          merge ×
        </button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <BumpRow label="src" value={boundary[0]} onBump={(d) => onBump("src", d)} accent="srcOnly" />
        <BumpRow label="tgt" value={boundary[1]} onBump={(d) => onBump("tgt", d)} accent="tgtOnly" />
      </div>
    </div>
  );
}

function BumpRow({
  label,
  value,
  onBump,
  accent,
}: {
  label: string;
  value: number;
  onBump: (delta: number) => void;
  accent: "srcOnly" | "tgtOnly";
}) {
  const dot = accent === "srcOnly" ? "bg-srcOnly" : "bg-tgtOnly";
  return (
    <div className="flex items-center justify-between rounded-md border border-neutral-200 px-2 py-1.5">
      <div className="flex items-center gap-1.5">
        <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
        <span className="text-xs text-neutral-500">{label}</span>
      </div>
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => onBump(-1)}
          className="inline-flex h-9 w-9 items-center justify-center rounded text-neutral-600 hover-fade"
          aria-label={`decrease ${label}`}
        >
          ◀
        </button>
        <span className="min-w-[1.5rem] text-center font-mono text-sm font-semibold text-neutral-900">{value}</span>
        <button
          type="button"
          onClick={() => onBump(1)}
          className="inline-flex h-9 w-9 items-center justify-center rounded text-neutral-600 hover-fade"
          aria-label={`increase ${label}`}
        >
          ▶
        </button>
      </div>
    </div>
  );
}
