import { Dialog } from "./Dialog";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  description?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <Dialog
      open={open}
      onClose={onCancel}
      ariaLabel={title}
      surfaceClassName="w-full max-w-md rounded-lg border border-neutral-200 bg-surface p-5 shadow-lg"
    >
      <h2 className="text-base font-semibold text-ink">{title}</h2>
      {description && (
        <p className="mt-2 text-sm text-neutral-600">{description}</p>
      )}
      <div className="mt-5 flex justify-end gap-2">
        <button type="button" onClick={onCancel} className="btn !min-h-[36px] !px-3 text-sm">
          {cancelLabel}
        </button>
        <button
          type="button"
          onClick={onConfirm}
          className={`!min-h-[36px] !px-3 text-sm ${
            destructive
              ? "btn border-red-300 bg-red-600 text-white hover:bg-red-700"
              : "btn-primary"
          }`}
          autoFocus
        >
          {confirmLabel}
        </button>
      </div>
    </Dialog>
  );
}
