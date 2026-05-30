import { useEffect, useLayoutEffect, useRef } from "react";
import { DEFAULT_LONG_CHUNK_LIMIT, DEFAULT_SHORT_CHUNK_LIMIT, getChunkHealth, type Side } from "~/lib/alignment";

interface ChunkCardProps {
  displayIndex: number;
  text: string;
  accent: Side;
  /** Whether this card is the editor's current selection (controls the visual ring). */
  selected: boolean;
  /** Whether the textarea should be auto-focused for editing. */
  editing: boolean;
  variant?: "desktop" | "mobile";
  onSelect: () => void;
  onCaretChange: (caret: number) => void;
  onEdit: (text: string) => void;
  onSplit: (caret: number) => void;
  onMergePrevious?: () => void;
  onMergeNext?: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  onPullFromNext?: () => void;
  onPushToNext?: () => void;
  onRechunkBelow?: () => void;
  /** Set when the chunk is first-in-segment and a previous segment exists. */
  onMoveToPrevSegment?: () => void;
  /** Set when the chunk is last-in-segment and a next segment exists. */
  onMoveToNextSegment?: () => void;
  onDelete: () => void;
  onRequestEdit: (editing: boolean) => void;
  caretFromState: number | null;
}

/**
 * One source / target chunk in the alignment editor.
 *
 * - The wrapper is a plain `<div>` (NOT a `role="button"`); the `<textarea>` inside
 *   is the only focusable element, so the focus ring lands on the actual editor.
 * - Selection (the `ring-2` highlight) is React-driven, independent of DOM focus,
 *   so the labeler's keyboard cursor can be visible even when they're "between"
 *   text-edit sessions.
 */
export function ChunkCard({
  displayIndex,
  text,
  accent,
  selected,
  editing,
  variant = "desktop",
  onSelect,
  onCaretChange,
  onEdit,
  onSplit,
  onMergePrevious,
  onMergeNext,
  onMoveUp,
  onMoveDown,
  onPullFromNext,
  onPushToNext,
  onRechunkBelow,
  onMoveToPrevSegment,
  onMoveToNextSegment,
  onDelete,
  onRequestEdit,
  caretFromState,
}: ChunkCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const gutter = accent === "src" ? "gutter-src" : "gutter-tgt";
  const health = getChunkHealth(text);
  const ring = selected ? "ring-2 ring-offset-1 ring-ink" : "";
  const warningTone = health.isEmpty
    ? "border-amber-300 bg-amber-50/60"
    : health.isLong
      ? "border-orange-300 bg-orange-50/50"
      : health.isShort
        ? "border-sky-300 bg-sky-50/60"
        : "border-neutral-200 bg-white";
  const iconSize = variant === "mobile" ? "h-9 w-9 text-base" : "h-7 w-7 text-sm";
  const wideIconSize = variant === "mobile" ? "h-9 min-w-[48px] px-1.5 text-xs" : "h-7 min-w-[42px] px-1.5 text-2xs";

  useEffect(() => {
    if (selected && ref.current) {
      ref.current.scrollIntoView({ block: "nearest" });
    }
  }, [selected]);

  useEffect(() => {
    if (editing && taRef.current) {
      taRef.current.focus();
      const c = caretFromState ?? text.length;
      taRef.current.setSelectionRange(c, c);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editing]);

  // Auto-grow the textarea to fit content so the chunk card never scrolls.
  // Uses scrollHeight measurement, which works correctly for CJK and any line length.
  useLayoutEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight}px`;
  }, [text]);

  return (
    <div
      ref={ref}
      className={`rounded-md border pl-3 pr-2.5 py-2 transition ${gutter} hover:border-neutral-300 ${warningTone} ${ring}`}
    >
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="flex flex-wrap items-baseline gap-1.5">
          <span className="font-mono text-2xs font-semibold text-neutral-500">Pair {displayIndex}</span>
          <span className="font-mono text-2xs text-neutral-400">[|{displayIndex}|]</span>
          <span
            className="font-mono text-2xs text-neutral-400"
            title={`${text.length} characters`}
          >
            {text.length}ch
          </span>
          {health.isEmpty && (
            <span className="rounded border border-amber-300 bg-amber-100 px-1 text-2xs font-medium text-amber-800">
              empty
            </span>
          )}
          {health.isLong && (
            <span
              className="rounded border border-orange-300 bg-orange-100 px-1 text-2xs font-medium text-orange-800"
              title={`More than ${DEFAULT_LONG_CHUNK_LIMIT} characters`}
            >
              long
            </span>
          )}
          {health.isShort && (
            <span
              className="rounded border border-sky-300 bg-sky-100 px-1 text-2xs font-medium text-sky-800"
              title={`Less than ${DEFAULT_SHORT_CHUNK_LIMIT} characters`}
            >
              short
            </span>
          )}
        </span>
        <div className="flex flex-wrap gap-0.5">
          <IconBtn
            label="split chunk at caret"
            sizeClass={iconSize}
            onClick={(e) => {
              e.stopPropagation();
              onSplit(taRef.current?.selectionStart ?? caretFromState ?? Math.floor(text.length / 2));
            }}
          >
            ✂
          </IconBtn>
          {onMergePrevious && (
            <IconBtn
              label="merge with previous chunk"
              sizeClass={iconSize}
              onClick={(e) => {
                e.stopPropagation();
                onMergePrevious();
              }}
            >
              P
            </IconBtn>
          )}
          {onMergeNext && (
            <IconBtn
              label="merge with next chunk"
              sizeClass={iconSize}
              onClick={(e) => {
                e.stopPropagation();
                onMergeNext();
              }}
            >
              ⇩
            </IconBtn>
          )}
          {onMoveUp && (
            <IconBtn
              label="move chunk up"
              sizeClass={iconSize}
              onClick={(e) => {
                e.stopPropagation();
                onMoveUp();
              }}
            >
              Up
            </IconBtn>
          )}
          {onMoveDown && (
            <IconBtn
              label="move chunk down"
              sizeClass={iconSize}
              onClick={(e) => {
                e.stopPropagation();
                onMoveDown();
              }}
            >
              Dn
            </IconBtn>
          )}
          {onPullFromNext && (
            <IconBtn
              label="pull first sentence from next chunk"
              sizeClass={wideIconSize}
              onClick={(e) => {
                e.stopPropagation();
                onPullFromNext();
              }}
            >
              Pull
            </IconBtn>
          )}
          {onPushToNext && (
            <IconBtn
              label="push last sentence to next chunk"
              sizeClass={wideIconSize}
              onClick={(e) => {
                e.stopPropagation();
                onPushToNext();
              }}
            >
              Push
            </IconBtn>
          )}
          {onRechunkBelow && (
            <IconBtn
              label="re-chunk below this pair"
              sizeClass={wideIconSize}
              onClick={(e) => {
                e.stopPropagation();
                onRechunkBelow();
              }}
            >
              Re-chunk
            </IconBtn>
          )}
          {onMoveToPrevSegment && (
            <IconBtn
              label="move chunk into previous segment"
              sizeClass={iconSize}
              onClick={(e) => {
                e.stopPropagation();
                onMoveToPrevSegment();
              }}
            >
              ⇈
            </IconBtn>
          )}
          {onMoveToNextSegment && (
            <IconBtn
              label="move chunk into next segment"
              sizeClass={iconSize}
              onClick={(e) => {
                e.stopPropagation();
                onMoveToNextSegment();
              }}
            >
              ⇊
            </IconBtn>
          )}
          <IconBtn
            label="delete chunk"
            sizeClass={iconSize}
            variant="danger"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          >
            ×
          </IconBtn>
        </div>
      </div>
      <textarea
        ref={taRef}
        aria-label={`${accent === "src" ? "source" : "target"} chunk ${displayIndex}`}
        className="w-full resize-none overflow-hidden rounded border border-transparent bg-transparent px-1 py-0.5 -mx-1 font-serif text-base leading-relaxed text-ink focus-visible:border-neutral-300 focus-visible:bg-neutral-50 focus:outline-none"
        rows={1}
        value={text}
        onChange={(e) => {
          onEdit(e.target.value);
          onCaretChange(e.target.selectionStart ?? 0);
        }}
        onSelect={(e) => onCaretChange((e.target as HTMLTextAreaElement).selectionStart ?? 0)}
        onFocus={() => {
          onSelect();
          onRequestEdit(true);
        }}
        onBlur={() => onRequestEdit(false)}
      />
    </div>
  );
}

function IconBtn({
  children,
  onClick,
  label,
  variant = "default",
  sizeClass,
}: {
  children: React.ReactNode;
  onClick: (e: React.MouseEvent) => void;
  label: string;
  variant?: "default" | "danger";
  sizeClass: string;
}) {
  const tone =
    variant === "danger"
      ? "text-red-600 hover:bg-red-50 active:bg-red-100"
      : "text-neutral-500 hover:bg-neutral-100 hover:text-ink active:bg-neutral-200";
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      className={`inline-flex items-center justify-center rounded transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink ${sizeClass} ${tone}`}
    >
      {children}
    </button>
  );
}
