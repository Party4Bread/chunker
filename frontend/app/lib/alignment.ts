import type { ChunkedSegment } from "./types";

export type Pair = [number, number];
export type Side = "src" | "tgt";
export type Selection = { side: Side; index: number };

export interface AlignmentState {
  srcChunks: string[];
  tgtChunks: string[];
  pairs: Pair[];
}

export function sortedPairs(pairs: Pair[]): Pair[] {
  const seen = new Set<string>();
  const out: Pair[] = [];
  for (const p of pairs) {
    const key = `${p[0]}-${p[1]}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push([p[0], p[1]]);
  }
  out.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return out;
}

export function clampPairs(pairs: Pair[], nSrc: number, nTgt: number): Pair[] {
  return sortedPairs(
    pairs.filter((p) => {
      if (p[0] < 1 || p[0] > nSrc) return false;
      if (p[1] < 1 || p[1] > nTgt) return false;
      // [nSrc, nTgt] coincides with the implicit end boundary appended in
      // buildSegments; keeping it would spawn a ghost empty trailing segment.
      // A boundary at [k, nTgt] (k<nSrc) — or [nSrc, k] (k<nTgt) — is valid:
      // it puts all opposite-side content into the segment before the break.
      if (p[0] === nSrc && p[1] === nTgt) return false;
      return true;
    }),
  );
}

export function buildSegments(state: AlignmentState): ChunkedSegment[] {
  const sorted = clampPairs(state.pairs, state.srcChunks.length, state.tgtChunks.length);
  // Drop pairs whose tgt index goes backwards after the (src, tgt)-sort. Otherwise stale or
  // hand-edited data with non-monotonic boundaries would render as backwards/duplicated tgt slices.
  const valid: Pair[] = [];
  let lastTgt = 0;
  for (const p of sorted) {
    if (p[1] < lastTgt) continue;
    valid.push(p);
    lastTgt = p[1];
  }
  const out: ChunkedSegment[] = [];
  let prevSrc = 0;
  let prevTgt = 0;
  for (const [srcEnd, tgtEnd] of [...valid, [state.srcChunks.length, state.tgtChunks.length] as Pair]) {
    const srcDelta = srcEnd - prevSrc;
    const tgtDelta = tgtEnd - prevTgt;
    let type: ChunkedSegment["type"] = "empty";
    if (srcDelta > 0 && tgtDelta > 0) type = "aligned";
    else if (srcDelta > 0) type = "src_only_unaligned";
    else if (tgtDelta > 0) type = "tgt_only_unaligned";
    out.push({
      type,
      src_range: [prevSrc, srcEnd],
      tgt_range: [prevTgt, tgtEnd],
      src: state.srcChunks.slice(prevSrc, srcEnd),
      tgt: state.tgtChunks.slice(prevTgt, tgtEnd),
    });
    prevSrc = srcEnd;
    prevTgt = tgtEnd;
  }
  return out;
}

/** Returns true if only chunk text content changed (no structural shape change). */
export function isOnlyTextEdit(prev: AlignmentState, next: AlignmentState): boolean {
  if (prev.srcChunks.length !== next.srcChunks.length) return false;
  if (prev.tgtChunks.length !== next.tgtChunks.length) return false;
  if (prev.pairs.length !== next.pairs.length) return false;
  for (let i = 0; i < prev.pairs.length; i++) {
    if (prev.pairs[i][0] !== next.pairs[i][0]) return false;
    if (prev.pairs[i][1] !== next.pairs[i][1]) return false;
  }
  return true;
}

function chunksAndPairsFor(
  state: AlignmentState,
  side: Side,
  nextChunks: string[],
  shift: (p: Pair) => Pair,
): AlignmentState {
  const shifted = state.pairs.map(shift);
  return {
    srcChunks: side === "src" ? nextChunks : state.srcChunks,
    tgtChunks: side === "tgt" ? nextChunks : state.tgtChunks,
    pairs: clampPairs(
      shifted,
      side === "src" ? nextChunks.length : state.srcChunks.length,
      side === "tgt" ? nextChunks.length : state.tgtChunks.length,
    ),
  };
}

export function editChunkText(state: AlignmentState, side: Side, i: number, text: string): AlignmentState {
  const arr = side === "src" ? state.srcChunks : state.tgtChunks;
  if (i < 0 || i >= arr.length) return state;
  const next = [...arr];
  next[i] = text;
  return chunksAndPairsFor(state, side, next, (p) => p);
}

export function splitChunk(state: AlignmentState, side: Side, i: number, caret: number): AlignmentState {
  const arr = side === "src" ? state.srcChunks : state.tgtChunks;
  if (i < 0 || i >= arr.length) return state;
  const text = arr[i];
  const left = text.slice(0, caret).trim();
  const right = text.slice(caret).trim();
  const pieces = caret <= 0 || caret >= text.length || !left || !right ? [text, ""] : [left, right];
  const next = [...arr];
  next.splice(i, 1, ...pieces);
  const idx = side === "src" ? 0 : 1;
  return chunksAndPairsFor(state, side, next, (p) => {
    if (p[idx] >= i + 1) {
      const out: Pair = [p[0], p[1]];
      out[idx] = p[idx] + 1;
      return out;
    }
    return p;
  });
}

export function mergeWithNext(state: AlignmentState, side: Side, i: number): AlignmentState {
  const arr = side === "src" ? state.srcChunks : state.tgtChunks;
  if (i < 0 || i >= arr.length - 1) return state;
  const next = [...arr];
  next.splice(i, 2, `${arr[i]} ${arr[i + 1]}`.replace(/\s+/g, " ").trim());
  const idx = side === "src" ? 0 : 1;
  return chunksAndPairsFor(state, side, next, (p) => {
    // The boundary that sat between chunks i and i+1 is dissolved by the merge.
    // Setting p[idx]=0 makes clampPairs drop it, instead of repositioning it to
    // p[idx]=i (which would silently introduce a brand-new boundary in front of
    // the merged chunk and spawn a ghost empty segment).
    if (p[idx] === i + 1) {
      const out: Pair = [p[0], p[1]];
      out[idx] = 0;
      return out;
    }
    if (p[idx] > i + 1) {
      const out: Pair = [p[0], p[1]];
      out[idx] = p[idx] - 1;
      return out;
    }
    return p;
  });
}

export function deleteChunk(state: AlignmentState, side: Side, i: number): AlignmentState {
  const arr = side === "src" ? state.srcChunks : state.tgtChunks;
  if (i < 0 || i >= arr.length) return state;
  const next = arr.filter((_, k) => k !== i);
  const idx = side === "src" ? 0 : 1;
  return chunksAndPairsFor(state, side, next, (p) => {
    if (p[idx] > i) {
      const out: Pair = [p[0], p[1]];
      out[idx] = Math.max(1, p[idx] - 1);
      return out;
    }
    return p;
  });
}

export function addChunkAfter(state: AlignmentState, side: Side, i: number): { state: AlignmentState; newIndex: number } {
  const arr = side === "src" ? state.srcChunks : state.tgtChunks;
  const insertAt = i < 0 ? arr.length : i + 1;
  const next = [...arr];
  next.splice(insertAt, 0, "");
  const idx = side === "src" ? 0 : 1;
  return {
    state: chunksAndPairsFor(state, side, next, (p) => {
      if (p[idx] >= insertAt) {
        const out: Pair = [p[0], p[1]];
        out[idx] = p[idx] + 1;
        return out;
      }
      return p;
    }),
    newIndex: insertAt,
  };
}

/**
 * Insert a boundary right after the given absolute chunk index on `side`.
 * The opposite side's boundary is taken from the current segment's end on that side
 * (so the new boundary doesn't shift the opposite-side content out of place).
 */
export function insertBoundaryAfterChunk(
  state: AlignmentState,
  side: Side,
  absChunkIndex: number,
): AlignmentState | null {
  const segments = buildSegments(state);
  const containingSeg = segments.find((seg) => {
    const range = side === "src" ? seg.src_range : seg.tgt_range;
    return absChunkIndex >= range[0] && absChunkIndex < range[1];
  });
  if (!containingSeg) return null;
  const newSideEnd = absChunkIndex + 1;
  const oppositeEnd = side === "src" ? containingSeg.tgt_range[1] : containingSeg.src_range[1];
  const newPair: Pair = side === "src" ? [newSideEnd, oppositeEnd] : [oppositeEnd, newSideEnd];
  if (newPair[0] < 1 || newPair[0] > state.srcChunks.length) return null;
  if (newPair[1] < 1 || newPair[1] > state.tgtChunks.length) return null;
  if (newPair[0] === state.srcChunks.length && newPair[1] === state.tgtChunks.length) return null;
  if (state.pairs.some((p) => p[0] === newPair[0] && p[1] === newPair[1])) return null;
  return { ...state, pairs: clampPairs([...state.pairs, newPair], state.srcChunks.length, state.tgtChunks.length) };
}

export function removeBoundary(state: AlignmentState, pair: Pair): AlignmentState {
  return {
    ...state,
    pairs: clampPairs(
      state.pairs.filter((p) => !(p[0] === pair[0] && p[1] === pair[1])),
      state.srcChunks.length,
      state.tgtChunks.length,
    ),
  };
}

export function bumpBoundary(
  state: AlignmentState,
  pair: Pair,
  side: Side,
  delta: number,
): AlignmentState | null {
  const idx = side === "src" ? 0 : 1;
  const others = state.pairs.filter((p) => !(p[0] === pair[0] && p[1] === pair[1]));
  const limit = side === "src" ? state.srcChunks.length : state.tgtChunks.length;
  const lower = Math.max(0, ...others.filter((p) => (side === "src" ? p[0] < pair[0] : p[1] < pair[1])).map((p) => p[idx]));
  const upper = Math.min(
    limit,
    ...others.filter((p) => (side === "src" ? p[0] > pair[0] : p[1] > pair[1])).map((p) => p[idx]),
  );
  const next = pair[idx] + delta;
  if (next < Math.max(1, lower) || next > upper) return null;
  const updated: Pair = [pair[0], pair[1]];
  updated[idx] = next;
  if (updated[0] === state.srcChunks.length && updated[1] === state.tgtChunks.length) return null;
  if (others.some((p) => p[0] === updated[0] && p[1] === updated[1])) return null;
  return {
    ...state,
    pairs: clampPairs([...others, updated], state.srcChunks.length, state.tgtChunks.length),
  };
}

/**
 * Move the last chunk of `side` in the segment containing `absIndex` into the
 * next segment. Implemented as a one-step bump of that segment's trailing
 * boundary on `side`. Returns null if `absIndex` isn't the last chunk of its
 * segment on `side`, or if there's no next segment, or if the boundary can't
 * move (e.g. would collide with the leading boundary on that side).
 */
export function moveChunkToNextSegment(
  state: AlignmentState,
  side: Side,
  absIndex: number,
): AlignmentState | null {
  const segments = buildSegments(state);
  const segIdx = segments.findIndex((seg) => {
    const range = side === "src" ? seg.src_range : seg.tgt_range;
    return absIndex >= range[0] && absIndex < range[1];
  });
  if (segIdx < 0 || segIdx >= segments.length - 1) return null;
  const seg = segments[segIdx];
  const range = side === "src" ? seg.src_range : seg.tgt_range;
  if (absIndex !== range[1] - 1) return null;
  const trailing: Pair = [seg.src_range[1], seg.tgt_range[1]];
  return bumpBoundary(state, trailing, side, -1);
}

/**
 * Move the first chunk of `side` in the segment containing `absIndex` into
 * the previous segment. Implemented as a one-step bump of that segment's
 * leading boundary on `side`. Returns null if `absIndex` isn't the first chunk
 * of its segment on `side`, or if there's no previous segment, or if the
 * boundary can't move.
 */
export function moveChunkToPrevSegment(
  state: AlignmentState,
  side: Side,
  absIndex: number,
): AlignmentState | null {
  const segments = buildSegments(state);
  const segIdx = segments.findIndex((seg) => {
    const range = side === "src" ? seg.src_range : seg.tgt_range;
    return absIndex >= range[0] && absIndex < range[1];
  });
  if (segIdx <= 0) return null;
  const seg = segments[segIdx];
  const range = side === "src" ? seg.src_range : seg.tgt_range;
  if (absIndex !== range[0]) return null;
  const leading: Pair = [seg.src_range[0], seg.tgt_range[0]];
  return bumpBoundary(state, leading, side, 1);
}

// ── selection navigation ───────────────────────────────────────────

export function clampSelection(sel: Selection | null, state: AlignmentState): Selection | null {
  if (!sel) return null;
  const max = sel.side === "src" ? state.srcChunks.length : state.tgtChunks.length;
  if (max === 0) return null;
  return { side: sel.side, index: Math.min(Math.max(0, sel.index), max - 1) };
}

export function moveSelection(
  sel: Selection | null,
  state: AlignmentState,
  dir: "next" | "prev",
): Selection | null {
  if (!sel) {
    if (state.srcChunks.length > 0) return { side: "src", index: 0 };
    if (state.tgtChunks.length > 0) return { side: "tgt", index: 0 };
    return null;
  }
  const arr = sel.side === "src" ? state.srcChunks : state.tgtChunks;
  if (arr.length === 0) return null;
  const next = sel.index + (dir === "next" ? 1 : -1);
  if (next < 0) return { side: sel.side, index: 0 };
  if (next >= arr.length) return { side: sel.side, index: arr.length - 1 };
  return { side: sel.side, index: next };
}

export function switchSide(sel: Selection | null, side: Side, state: AlignmentState): Selection | null {
  const arr = side === "src" ? state.srcChunks : state.tgtChunks;
  if (arr.length === 0) return sel;
  const targetIndex = sel ? Math.min(sel.index, arr.length - 1) : 0;
  return { side, index: targetIndex };
}

/** Human-readable chunk range. "1–2 of 14" / "1 of 14" / "none". */
export function formatChunkRange(range: [number, number], total: number): string {
  const [start, end] = range;
  const span = end - start;
  if (span <= 0) return "none";
  if (span === 1) return `${start + 1} of ${total}`;
  return `${start + 1}–${end} of ${total}`;
}
