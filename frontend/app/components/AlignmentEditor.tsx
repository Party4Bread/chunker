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
}

const TYPE_LABEL: Record<ChunkedSegment["type"], string> = {
  aligned: "aligned",
  src_only_unaligned: "source only",
  tgt_only_unaligned: "target only",
  empty: "empty",
};

const TYPE_TONE: Record<ChunkedSegment["type"], string> = {
  aligned: "bg-aligned/[0.04] ring-aligned/30",
  src_only_unaligned: "bg-srcOnly/[0.06] ring-srcOnly/30",
  tgt_only_unaligned: "bg-tgtOnly/[0.05] ring-tgtOnly/30",
  empty: "bg-neutral-100 ring-neutral-300",
};

const TYPE_DOT: Record<ChunkedSegment["type"], string> = {
  aligned: "bg-aligned",
  src_only_unaligned: "bg-srcOnly",
  tgt_only_unaligned: "bg-tgtOnly",
  empty: "bg-neutral-400",
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
                actions={actions}
                selectedKey={selectedKey}
                onSelect={onSelect}
                caret={caret}
                onCaretChange={onCaretChange}
                editingKey={editingKey ?? null}
                onRequestEdit={onRequestEdit}
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
  actions: AlignmentEditorActions;
  selectedKey: string | null;
  onSelect: (sel: Selection) => void;
  caret: number;
  onCaretChange: (caret: number) => void;
  editingKey: string | null;
  onRequestEdit?: (key: string | null) => void;
}

function SegmentBlock({
  seg,
  segIdx,
  nSrc,
  nTgt,
  actions,
  selectedKey,
  onSelect,
  caret,
  onCaretChange,
  editingKey,
  onRequestEdit,
}: SegmentBlockProps) {
  const srcChars = seg.src.reduce((n, c) => n + c.length, 0);
  const tgtChars = seg.tgt.reduce((n, c) => n + c.length, 0);
  return (
    <article className={`rounded-lg ring-1 p-3 ${TYPE_TONE[seg.type]}`}>
      <header className="mb-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-xs font-medium text-neutral-700">
        <span className="flex items-center gap-2">
          <span className={`h-1.5 w-1.5 rounded-full ${TYPE_DOT[seg.type]}`} />
          segment {segIdx + 1} <span className="text-neutral-400">·</span> {TYPE_LABEL[seg.type]}
        </span>
        <span className="text-neutral-500">
          source <span className="font-mono text-ink">{formatChunkRange(seg.src_range, nSrc)}</span>
          <span className="ml-1 font-mono text-neutral-400">{srcChars}ch</span>
          <span className="mx-1.5 text-neutral-300">·</span>
          target <span className="font-mono text-ink">{formatChunkRange(seg.tgt_range, nTgt)}</span>
          <span className="ml-1 font-mono text-neutral-400">{tgtChars}ch</span>
        </span>
      </header>

      <div className="grid gap-3 lg:grid-cols-2">
        <ChunkList
          accent="src"
          chunks={seg.src}
          baseAbsIndex={seg.src_range[0]}
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

interface ChunkListProps {
  accent: Side;
  chunks: string[];
  baseAbsIndex: number;
  actions: AlignmentEditorActions;
  selectedKey: string | null;
  onSelect: (sel: Selection) => void;
  caret: number;
  onCaretChange: (caret: number) => void;
  editingKey: string | null;
  onRequestEdit?: (key: string | null) => void;
}

function ChunkList(props: ChunkListProps) {
  const { accent, chunks, baseAbsIndex, actions, selectedKey, onSelect, caret, onCaretChange, editingKey, onRequestEdit } = props;
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
        return (
          <li key={key}>
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
              onDelete={() => actions.deleteChunk(accent, absIdx)}
              onRequestEdit={(editing) => onRequestEdit?.(editing ? key : null)}
              caretFromState={selectedKey === key ? caret : null}
            />
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
      <div className="flex items-center gap-1 rounded-full border border-neutral-300 bg-white px-2 py-0.5 text-2xs shadow-sm">
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
        className="rounded px-1 text-neutral-500 hover:bg-neutral-100 hover:text-neutral-900"
        title={`move ${label} boundary one chunk earlier`}
        aria-label={`decrease ${label} boundary`}
      >
        ◀
      </button>
      <span className="min-w-[1.25rem] text-center font-mono text-neutral-900">{value}</span>
      <button
        type="button"
        onClick={() => onBump(1)}
        className="rounded px-1 text-neutral-500 hover:bg-neutral-100 hover:text-neutral-900"
        title={`move ${label} boundary one chunk later`}
        aria-label={`increase ${label} boundary`}
      >
        ▶
      </button>
    </span>
  );
}
