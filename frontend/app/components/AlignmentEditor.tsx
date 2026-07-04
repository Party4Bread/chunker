import {
  type AlignmentState,
  type Pair,
  type Selection,
  type Side,
  formatChunkRange,
} from "~/lib/alignment";
import type { ChunkedSegment } from "~/lib/types";
import { ChunkCard } from "./ChunkCard";

export type { AlignmentState } from "~/lib/alignment";

export interface AlignmentEditorActions {
  editChunkText: (side: Side, absIndex: number, text: string) => void;
  splitChunk: (side: Side, absIndex: number, caret: number) => void;
  mergeWithNext: (side: Side, absIndex: number) => void;
  deleteChunk: (side: Side, absIndex: number) => void;
  addChunkAfter: (side: Side, absIndex: number) => void;
  insertBoundaryAfter: (side: Side, absIndex: number) => void;
  bumpBoundary: (pair: Pair, side: Side, delta: number) => void;
  removeBoundary: (pair: Pair) => void;
  moveChunkToNextSegment: (side: Side, absIndex: number) => void;
  moveChunkToPrevSegment: (side: Side, absIndex: number) => void;
}

interface AlignmentEditorProps {
  state: AlignmentState;
  segments: ChunkedSegment[];
  validPairs: Pair[];
  selection: Selection | null;
  onSelect: (sel: Selection) => void;
  caret: number;
  onCaretChange: (caret: number) => void;
  actions: AlignmentEditorActions;
  /** Set when the user clicks a chunk's textarea — focuses it for typing. */
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

// Tone classes live in styles.css and lift their alphas in dark mode.
// Aligned is intentionally the quietest (it's "done"); the unaligned variants
// carry more chroma because they're the action items. Empty is a dashed
// border on no fill so it reads as absence rather than as a category.
const TYPE_TONE: Record<ChunkedSegment["type"], string> = {
  aligned: "tone-aligned",
  src_only_unaligned: "tone-src-only",
  tgt_only_unaligned: "tone-tgt-only",
  empty: "tone-empty",
};

const TYPE_DOT: Record<ChunkedSegment["type"], string> = {
  aligned: "bg-aligned",
  src_only_unaligned: "bg-srcOnly",
  tgt_only_unaligned: "bg-tgtOnly",
  empty: "bg-neutral-400",
};

const TYPE_LABEL_TONE: Record<ChunkedSegment["type"], string> = {
  aligned: "tone-text-aligned",
  src_only_unaligned: "tone-text-src",
  tgt_only_unaligned: "tone-text-tgt",
  empty: "text-neutral-500",
};

function chunkKey(side: Side, absIndex: number): string {
  return `${side}:${absIndex}`;
}

export function AlignmentEditor({
  state,
  segments,
  validPairs,
  selection,
  onSelect,
  caret,
  onCaretChange,
  actions,
  editingKey,
  onRequestEdit,
  sourceTranslations,
}: AlignmentEditorProps) {
  const selectedKey = selection ? chunkKey(selection.side, selection.index) : null;

  return (
    <section className="space-y-2">
      <header className="flex items-center justify-between text-xs text-neutral-600">
        <span>
          <span className="font-mono text-neutral-900">{state.srcChunks.length}</span> source ·{" "}
          <span className="font-mono text-neutral-900">{state.tgtChunks.length}</span> target ·{" "}
          <span className="font-mono text-neutral-900">{validPairs.length}</span> boundaries ·{" "}
          <span className="font-mono text-neutral-900">{segments.length}</span> segments
        </span>
        <div className="flex gap-1">
          <button
            type="button"
            className="btn !min-h-[32px] !px-2 text-xs"
            onClick={() => actions.addChunkAfter("src", state.srcChunks.length - 1)}
          >
            + source chunk
          </button>
          <button
            type="button"
            className="btn !min-h-[32px] !px-2 text-xs"
            onClick={() => actions.addChunkAfter("tgt", state.tgtChunks.length - 1)}
          >
            + target chunk
          </button>
        </div>
      </header>

      <ol className="flex flex-col">
        {segments.map((seg, segIdx) => {
          const isLast = segIdx === segments.length - 1;
          const boundary = !isLast ? validPairs[segIdx] : null;
          return (
            <li key={`seg-${segIdx}-${seg.src_range[0]}-${seg.tgt_range[0]}`}>
              <SegmentBlock
                seg={seg}
                segIdx={segIdx}
                nSrc={state.srcChunks.length}
                nTgt={state.tgtChunks.length}
                hasPrevSegment={segIdx > 0}
                hasNextSegment={segIdx < segments.length - 1}
                actions={actions}
                selectedKey={selectedKey}
                onSelect={onSelect}
                caret={caret}
                onCaretChange={onCaretChange}
                editingKey={editingKey ?? null}
                onRequestEdit={onRequestEdit}
                sourceTranslations={sourceTranslations}
              />
              {boundary && (
                <BoundaryDivider
                  boundary={boundary}
                  onRemove={() => actions.removeBoundary(boundary)}
                  onBump={(side, delta) => actions.bumpBoundary(boundary, side, delta)}
                />
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

interface SegmentBlockProps {
  seg: ChunkedSegment;
  segIdx: number;
  nSrc: number;
  nTgt: number;
  hasPrevSegment: boolean;
  hasNextSegment: boolean;
  actions: AlignmentEditorActions;
  selectedKey: string | null;
  onSelect: (sel: Selection) => void;
  caret: number;
  onCaretChange: (caret: number) => void;
  editingKey: string | null;
  onRequestEdit?: (key: string | null) => void;
  sourceTranslations?: string[] | null;
}

function SegmentBlock({
  seg,
  segIdx,
  nSrc,
  nTgt,
  hasPrevSegment,
  hasNextSegment,
  actions,
  selectedKey,
  onSelect,
  caret,
  onCaretChange,
  editingKey,
  onRequestEdit,
  sourceTranslations,
}: SegmentBlockProps) {
  const srcChars = seg.src.reduce((n, c) => n + c.length, 0);
  const tgtChars = seg.tgt.reduce((n, c) => n + c.length, 0);
  const translations = sourceTranslations
    ? seg.src.map((_, i) => sourceTranslations[seg.src_range[0] + i] ?? "")
    : null;
  return (
    <article className={`rounded-lg p-3 ${TYPE_TONE[seg.type]}`}>
      <header className="mb-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-xs font-medium text-neutral-700">
        <span className="flex items-center gap-2">
          <span className={`h-1.5 w-1.5 rounded-full ${TYPE_DOT[seg.type]}`} />
          segment {segIdx + 1}{" "}
          <span className="text-neutral-400">·</span>{" "}
          <span className={`font-semibold ${TYPE_LABEL_TONE[seg.type]}`}>{TYPE_LABEL[seg.type]}</span>
        </span>
        <span className="text-neutral-500">
          source <span className="font-mono text-ink">{formatChunkRange(seg.src_range, nSrc)}</span>
          <span className="ml-1 font-mono text-neutral-400">{srcChars}ch</span>
          <span className="mx-1.5 text-neutral-300">·</span>
          target <span className="font-mono text-ink">{formatChunkRange(seg.tgt_range, nTgt)}</span>
          <span className="ml-1 font-mono text-neutral-400">{tgtChars}ch</span>
        </span>
      </header>

      {/* When MT is shown, each source chunk carries its translation inline
          (see ChunkList) so the two stay row-aligned no matter how tall either
          gets — hence a 2-column layout (source+MT | target) rather than 3
          independent columns that drift apart down a long segment. */}
      <div className={`grid gap-3 ${translations ? "lg:grid-cols-[minmax(0,1.85fr)_minmax(0,1fr)]" : "lg:grid-cols-2"}`}>
        <ChunkList
          accent="src"
          chunks={seg.src}
          baseAbsIndex={seg.src_range[0]}
          translations={translations ?? undefined}
          hasPrevSegment={hasPrevSegment}
          hasNextSegment={hasNextSegment}
          actions={actions}
          selectedKey={selectedKey}
          onSelect={onSelect}
          caret={caret}
          onCaretChange={onCaretChange}
          editingKey={editingKey}
          onRequestEdit={onRequestEdit}
        />
        <ChunkList
          accent="tgt"
          chunks={seg.tgt}
          baseAbsIndex={seg.tgt_range[0]}
          hasPrevSegment={hasPrevSegment}
          hasNextSegment={hasNextSegment}
          actions={actions}
          selectedKey={selectedKey}
          onSelect={onSelect}
          caret={caret}
          onCaretChange={onCaretChange}
          editingKey={editingKey}
          onRequestEdit={onRequestEdit}
        />
      </div>
    </article>
  );
}

// MT card rendered beside a source chunk (same grid row → always aligned).
// An empty translation means "not fetched yet" — shown as a muted placeholder
// rather than an error, since on-demand translation fills it in shortly.
function MTCard({ text, displayIndex }: { text: string; displayIndex: number }) {
  const pending = !text;
  return (
    <div className="rounded-md border border-neutral-200 bg-brand-subtle/60 px-3 py-2">
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

interface ChunkListProps {
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
  /** Per-chunk source translations (source column only). When set, each chunk
   *  renders its MT card in the same row so the two stay vertically aligned. */
  translations?: string[];
}

function ChunkList(props: ChunkListProps) {
  const { accent, chunks, baseAbsIndex, hasPrevSegment, hasNextSegment, actions, selectedKey, onSelect, caret, onCaretChange, editingKey, onRequestEdit, translations } = props;
  if (chunks.length === 0) {
    return (
      <p className="flex h-full items-center justify-center px-3 py-2 text-xs italic text-neutral-400">
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
        const card = (
          <ChunkCard
            displayIndex={absIdx + 1}
            text={text}
            accent={accent}
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
        );
        return (
          <li key={key}>
            {translations ? (
              // Source card and its MT share one grid row, top-aligned, so they
              // never drift apart even when either wraps to several lines.
              <div className="grid items-start gap-2 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.85fr)]">
                {card}
                <MTCard text={translations[i] ?? ""} displayIndex={absIdx + 1} />
              </div>
            ) : (
              card
            )}
            {i < chunks.length - 1 && (
              <SplitSegmentAffordance
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

function SplitSegmentAffordance({ accent, onClick }: { accent: Side; onClick: () => void }) {
  const tone =
    accent === "src"
      ? "text-srcOnly bg-srcOnly/[0.04] hover:bg-srcOnly/15"
      : "text-tgtOnly bg-tgtOnly/[0.04] hover:bg-tgtOnly/15";
  return (
    <button
      type="button"
      onClick={onClick}
      className={`my-0.5 flex h-5 w-full items-center justify-center rounded text-2xs font-medium opacity-30 transition hover:opacity-100 focus:opacity-100 ${tone}`}
      aria-label="split segment here"
      title="break the segment after this chunk"
    >
      ↕ break here
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
    <div className="my-2 flex items-center gap-2 px-1">
      <span className="h-px flex-1 bg-neutral-300" />
      <div className="flex items-center gap-1 rounded-full border border-neutral-300 bg-surface px-2 py-0.5 text-2xs shadow-sm">
        <span className="font-medium text-neutral-500">boundary</span>
        <BumpGroup label="src" value={boundary[0]} onBump={(d) => onBump("src", d)} accent="srcOnly" />
        <BumpGroup label="tgt" value={boundary[1]} onBump={(d) => onBump("tgt", d)} accent="tgtOnly" />
        <button
          type="button"
          onClick={onRemove}
          className="ml-1 rounded px-1.5 py-0.5 text-red-600 hover:bg-red-50"
          title="remove this boundary (merges adjacent segments)"
          aria-label="remove boundary"
        >
          merge ×
        </button>
      </div>
      <span className="h-px flex-1 bg-neutral-300" />
    </div>
  );
}

function BumpGroup({
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
    <span className="flex items-center gap-0.5">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      <span className="text-neutral-500">{label}</span>
      <button
        type="button"
        onClick={() => onBump(-1)}
        className="rounded px-1 text-neutral-500 hover:text-ink hover-fade"
        title={`move ${label} boundary one chunk earlier`}
        aria-label={`decrease ${label} boundary`}
      >
        ◀
      </button>
      <span className="min-w-[1.25rem] text-center font-mono text-neutral-900">{value}</span>
      <button
        type="button"
        onClick={() => onBump(1)}
        className="rounded px-1 text-neutral-500 hover:text-ink hover-fade"
        title={`move ${label} boundary one chunk later`}
        aria-label={`increase ${label} boundary`}
      >
        ▶
      </button>
    </span>
  );
}
