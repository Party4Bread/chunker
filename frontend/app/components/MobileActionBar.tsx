interface MobileActionBarProps {
  segmentIdx: number;
  segmentTotal: number;
  onSegmentPrev: () => void;
  onSegmentNext: () => void;
  onUndo: () => void;
  canUndo: boolean;
  onSave: () => void;
  saving: boolean;
  dirty: boolean;
  onReviewAndNext: () => void;
  reviewing: boolean;
  reviewed: boolean;
  recordIdx: number;
  recordTotal: number;
  onRecordPrev: () => void;
  onRecordNext: () => void;
}

export function MobileActionBar({
  segmentIdx,
  segmentTotal,
  onSegmentPrev,
  onSegmentNext,
  onUndo,
  canUndo,
  onSave,
  saving,
  dirty,
  onReviewAndNext,
  reviewing,
  reviewed,
  recordIdx,
  recordTotal,
  onRecordPrev,
  onRecordNext,
}: MobileActionBarProps) {
  return (
    <nav
      aria-label="record actions"
      className="fixed inset-x-0 bottom-0 z-30 border-t border-neutral-200 bg-surface/95 backdrop-blur lg:hidden"
      style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}
    >
      <div className="flex items-center gap-1.5 px-2 pt-2 text-xs">
        <Btn label="prev segment" onClick={onSegmentPrev} disabled={segmentIdx <= 0}>
          ◀ seg
        </Btn>
        <span className="flex-1 text-center font-mono text-neutral-700">
          <span className="font-semibold text-neutral-900">{segmentIdx + 1}</span>
          <span className="text-neutral-400"> / {segmentTotal}</span>
        </span>
        <Btn label="next segment" onClick={onSegmentNext} disabled={segmentIdx >= segmentTotal - 1}>
          seg ▶
        </Btn>
        <Btn label="undo" onClick={onUndo} disabled={!canUndo}>
          ↺
        </Btn>
      </div>
      <div className="flex items-center gap-1.5 px-2 py-2">
        <Btn label="previous record" onClick={onRecordPrev} disabled={recordIdx <= 0} compact>
          ⟨ rec
        </Btn>
        <button
          type="button"
          onClick={onSave}
          disabled={!dirty || saving}
          aria-live="polite"
          className="flex-1 rounded-md bg-ink px-3 py-2.5 text-sm font-medium text-brand-fg shadow-sm disabled:opacity-50"
        >
          {saving ? "Saving…" : dirty ? "Save" : "Saved"}
        </button>
        <button
          type="button"
          onClick={onReviewAndNext}
          disabled={reviewing}
          className={`flex-1 rounded-md px-3 py-2.5 text-sm font-medium shadow-sm ${
            reviewed
              ? "bg-aligned/10 text-ink ring-1 ring-aligned/40"
              : "bg-aligned text-brand-fg"
          }`}
        >
          {reviewing ? "…" : reviewed ? "✓ Continue" : "Reviewed, continue"}
        </button>
        <Btn label="next record" onClick={onRecordNext} disabled={recordIdx < 0 || recordIdx >= recordTotal - 1} compact>
          rec ⟩
        </Btn>
      </div>
    </nav>
  );
}

function Btn({
  children,
  onClick,
  disabled,
  label,
  compact,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  label: string;
  compact?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className={`hover-fade inline-flex min-h-[40px] items-center justify-center rounded-md border border-neutral-300 bg-surface text-neutral-700 shadow-sm transition disabled:border-neutral-200 disabled:bg-neutral-100 disabled:text-neutral-400 ${
        compact ? "min-w-[52px] px-2 text-xs" : "min-w-[44px] px-2.5"
      }`}
    >
      {children}
    </button>
  );
}
