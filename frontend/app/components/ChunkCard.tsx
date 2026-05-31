import { useCallback, useEffect, useLayoutEffect, useRef } from "react";
import type { Side } from "~/lib/alignment";

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
  onMergeNext?: () => void;
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
  onMergeNext,
  onMoveToPrevSegment,
  onMoveToNextSegment,
  onDelete,
  onRequestEdit,
  caretFromState,
}: ChunkCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const gutter = accent === "src" ? "gutter-src" : "gutter-tgt";
  const ring = selected ? "ring-2 ring-offset-1 ring-ink" : "";
  const iconSize = variant === "mobile" ? "h-9 w-9 text-base" : "h-7 w-7 text-sm";
  const resizeTextarea = useCallback(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight}px`;
  }, []);

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
      resizeTextarea();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editing]);

  // Auto-grow the textarea to fit content so the chunk card never scrolls.
  // Re-measure on width changes too: wrapping changes scrollHeight even when
  // text is unchanged, which made editors clip or become hard to focus after
  // resizing across layouts.
  useLayoutEffect(() => {
    resizeTextarea();
  }, [resizeTextarea, text]);

  useEffect(() => {
    const ta = taRef.current;
    if (!ta || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => resizeTextarea());
    ro.observe(ta);
    return () => ro.disconnect();
  }, [resizeTextarea]);

  return (
    <div
      ref={ref}
      className={`rounded-md border border-neutral-200 bg-surface pl-3 pr-2.5 py-2 transition ${gutter} hover:border-neutral-300 ${ring}`}
    >
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="flex items-baseline gap-1.5">
          <span className="font-mono text-2xs font-semibold text-neutral-500">[|{displayIndex}|]</span>
          <span
            className="font-mono text-2xs text-neutral-400"
            title={`${text.length} characters`}
          >
            {text.length}ch
          </span>
        </span>
        <div className="flex flex-wrap gap-0.5">
          <IconBtn
            label="split chunk at caret"
            sizeClass={iconSize}
            onClick={(e) => {
              e.stopPropagation();
              onSplit(caretFromState ?? 0);
            }}
          >
            ✂
          </IconBtn>
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
        className="min-h-[2.25rem] w-full resize-none overflow-hidden rounded border border-transparent bg-transparent px-1 py-0.5 -mx-1 font-serif text-base leading-relaxed text-ink focus-visible:border-neutral-300 focus-visible:bg-neutral-100 focus:outline-none"
        rows={1}
        value={text}
        onChange={(e) => {
          e.currentTarget.style.height = "auto";
          e.currentTarget.style.height = `${e.currentTarget.scrollHeight}px`;
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
      : "text-neutral-500 hover:text-ink hover-fade";
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
