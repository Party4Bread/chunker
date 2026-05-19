import { Dialog } from "./Dialog";
import { modLabel } from "~/lib/useHotkeys";

interface HelpOverlayProps {
  open: boolean;
  onClose: () => void;
}

const SECTIONS: { title: string; rows: [string, string][] }[] = [
  {
    title: "Navigate",
    rows: [
      ["j / k", "next / previous chunk on current side"],
      ["h / l", "switch focus to source / target"],
      ["esc", "leave text editing, back to navigation"],
      ["enter", "edit selected chunk's text"],
    ],
  },
  {
    title: "Edit chunks",
    rows: [
      ["s", "split selected chunk at caret (or in the middle)"],
      ["m", "merge selected chunk with the next one on its side"],
      ["d", "delete selected chunk"],
      ["o", "add a new empty chunk after the selected one"],
    ],
  },
  {
    title: "Boundaries",
    rows: [
      ["b", "insert boundary after the selected chunk on its side"],
      ["click ◀ ▶", "shift the boundary on a divider one chunk earlier / later"],
      ["click merge ×", "remove a boundary, joining the two neighbouring segments"],
      ["click ⇈ / ⇊", "move the chunk into the previous / next segment"],
    ],
  },
  {
    title: "Record actions",
    rows: [
      [`${modLabel()}+s`, "save (also autosaves 2s after the last edit)"],
      [`${modLabel()}+z`, "undo last structural change"],
      [`${modLabel()}+shift+z`, "redo"],
      [`${modLabel()}+enter`, "mark reviewed and jump to next draft"],
      ["g n", "next record (any status)"],
      ["g p", "previous record"],
      ["?", "open / close this help"],
    ],
  },
];

export function HelpOverlay({ open, onClose }: HelpOverlayProps) {
  return (
    <Dialog open={open} onClose={onClose} ariaLabel="Keyboard shortcuts">
      <header className="mb-4 flex items-center justify-between">
        <h2 className="text-base font-semibold text-ink">Keyboard shortcuts</h2>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-neutral-500 hover:bg-neutral-100 hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
          aria-label="Close keyboard shortcuts"
        >
          ×
        </button>
      </header>
      <div className="grid gap-5 sm:grid-cols-2">
        {SECTIONS.map((sec) => (
          <section key={sec.title}>
            <h3 className="eyebrow mb-1.5">{sec.title}</h3>
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
              {sec.rows.map(([keys, desc]) => (
                <div key={keys} className="contents">
                  <dt className="font-mono text-ink">{keys}</dt>
                  <dd className="text-neutral-600">{desc}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
      <p className="mt-4 text-2xs text-neutral-500">
        letter shortcuts only fire when the cursor isn't in a text field. press{" "}
        <span className="font-mono">esc</span> to leave a textarea.
      </p>
    </Dialog>
  );
}
