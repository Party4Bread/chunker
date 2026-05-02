import { useCallback, useRef, useState } from "react";

interface HistoryState<T> {
  past: T[];
  present: T;
  future: T[];
}

const LIMIT = 50;

export interface UseHistoryReturn<T> {
  state: T;
  reset: (next: T) => void;
  set: (next: T, opts?: { commit?: boolean }) => void;
  undo: () => boolean;
  redo: () => boolean;
  canUndo: boolean;
  canRedo: boolean;
  pastSize: number;
  futureSize: number;
}

/**
 * Linear undo / redo history.
 *
 * `set(next)` defaults to a structural commit (pushes a snapshot to past).
 * `set(next, { commit: false })` updates `present` only — used for live edits
 * (e.g. typing in a textarea) that you don't want to spam the undo stack.
 * `reset(next)` replaces the entire history (use after server reload).
 */
export function useHistory<T>(initial: T): UseHistoryReturn<T> {
  const [hist, setHist] = useState<HistoryState<T>>({ past: [], present: initial, future: [] });
  const histRef = useRef(hist);
  histRef.current = hist;

  const reset = useCallback((next: T) => {
    const fresh = { past: [], present: next, future: [] };
    histRef.current = fresh;
    setHist(fresh);
  }, []);

  const set = useCallback((next: T, opts: { commit?: boolean } = {}) => {
    const commit = opts.commit !== false;
    setHist((h) => {
      if (!commit) return { ...h, present: next };
      const past = [...h.past, h.present];
      if (past.length > LIMIT) past.shift();
      return { past, present: next, future: [] };
    });
  }, []);

  const undo = useCallback((): boolean => {
    const h = histRef.current;
    if (h.past.length === 0) return false;
    const past = [...h.past];
    const previous = past.pop()!;
    const next = { past, present: previous, future: [h.present, ...h.future] };
    histRef.current = next;
    setHist(next);
    return true;
  }, []);

  const redo = useCallback((): boolean => {
    const h = histRef.current;
    if (h.future.length === 0) return false;
    const [first, ...rest] = h.future;
    const next = { past: [...h.past, h.present], present: first, future: rest };
    histRef.current = next;
    setHist(next);
    return true;
  }, []);

  return {
    state: hist.present,
    reset,
    set,
    undo,
    redo,
    canUndo: hist.past.length > 0,
    canRedo: hist.future.length > 0,
    pastSize: hist.past.length,
    futureSize: hist.future.length,
  };
}
