import { useEffect } from "react";
import { useFocusTrap } from "~/lib/useFocusTrap";

interface DialogProps {
  open: boolean;
  onClose: () => void;
  ariaLabel: string;
  children: React.ReactNode;
  /** Tailwind classes for the dialog surface (sizing, padding). */
  surfaceClassName?: string;
}

/**
 * Accessible modal dialog. Handles aria-modal, focus trap, focus restoration,
 * Escape, and click-outside dismissal. Locks body scroll while open.
 */
export function Dialog({
  open,
  onClose,
  ariaLabel,
  children,
  surfaceClassName = "max-h-[80vh] w-full max-w-xl overflow-auto rounded-lg border border-neutral-200 bg-surface p-5 shadow-lg",
}: DialogProps) {
  const containerRef = useFocusTrap<HTMLDivElement>(open, onClose);

  useEffect(() => {
    if (!open) return;
    const original = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = original;
    };
  }, [open]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={ariaLabel}
      ref={containerRef}
      tabIndex={-1}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-neutral-900/40 p-4 outline-none"
    >
      <div
        onMouseDown={(e) => e.stopPropagation()}
        className={surfaceClassName}
      >
        {children}
      </div>
    </div>
  );
}
