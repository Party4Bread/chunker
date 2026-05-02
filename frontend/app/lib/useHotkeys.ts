import { useEffect, useRef } from "react";

export type Bindings = Record<string, (e: KeyboardEvent) => void>;

const MOD = typeof navigator !== "undefined" && /Mac|iPad|iPhone/.test(navigator.platform) ? "meta" : "ctrl";

const CHORD_TIMEOUT_MS = 1500;

function formatKey(e: KeyboardEvent): string {
  const parts: string[] = [];
  if ((MOD === "meta" && e.metaKey) || (MOD === "ctrl" && e.ctrlKey)) parts.push("mod");
  if (e.shiftKey) parts.push("shift");
  if (e.altKey) parts.push("alt");
  let key = e.key;
  if (key.length === 1) key = key.toLowerCase();
  parts.push(key);
  return parts.join("+");
}

function isInTextField(target: EventTarget | null): boolean {
  if (!target || !(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === "TEXTAREA") return true;
  if (tag === "INPUT") {
    const type = (target as HTMLInputElement).type;
    return type !== "checkbox" && type !== "radio" && type !== "button";
  }
  return false;
}

/**
 * Global keyboard layer.
 *
 * - "mod" maps to Cmd on macOS, Ctrl elsewhere.
 * - Plain letter shortcuts (e.g. "j") are suppressed while focus is in a text field.
 * - Modifier shortcuts (e.g. "mod+s") fire everywhere.
 * - "Escape" always fires (used to leave text fields back to navigation mode).
 * - Chords use a space: "g n" means press `g`, then `n` within 1.5 s.
 *   The leader key (e.g. `g`) is swallowed without invoking anything else.
 */
export function useHotkeys(bindings: Bindings, opts: { enabled?: boolean } = {}) {
  const ref = useRef(bindings);
  ref.current = bindings;
  const enabled = opts.enabled !== false;
  const pendingChord = useRef<{ leader: string; timer: number } | null>(null);

  useEffect(() => {
    if (!enabled) return;
    const clearChord = () => {
      if (pendingChord.current) {
        window.clearTimeout(pendingChord.current.timer);
        pendingChord.current = null;
      }
    };

    const handler = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      const key = formatKey(e);
      const inField = isInTextField(e.target);
      const isModifier = key.startsWith("mod+");
      const isEscape = e.key === "Escape";
      if (inField && !isModifier && !isEscape) return;

      // Continuation of a chord?
      if (pendingChord.current) {
        const chordKey = `${pendingChord.current.leader} ${key}`;
        const fn = ref.current[chordKey];
        clearChord();
        if (fn) {
          e.preventDefault();
          fn(e);
          return;
        }
        // Fall through: maybe this key starts a new chord or is a plain binding.
      }

      // Direct binding match?
      const direct = ref.current[key];
      if (direct) {
        e.preventDefault();
        direct(e);
        return;
      }

      // Is this key a leader for any chord binding?
      const leaderPrefix = `${key} `;
      const isLeader = Object.keys(ref.current).some((k) => k.startsWith(leaderPrefix));
      if (isLeader) {
        e.preventDefault();
        pendingChord.current = {
          leader: key,
          timer: window.setTimeout(clearChord, CHORD_TIMEOUT_MS),
        };
      }
    };
    window.addEventListener("keydown", handler);
    return () => {
      window.removeEventListener("keydown", handler);
      clearChord();
    };
  }, [enabled]);
}

export function modLabel(): string {
  return MOD === "meta" ? "⌘" : "Ctrl";
}
