import {
  applyAlignmentPlaceholders,
  type AlignmentState,
  getAlignmentSuggestions,
  pullSentenceFromNext,
  pushSentenceToNext,
} from "./alignment";

function expectEqual(actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

const base: AlignmentState = {
  srcChunks: ["First sentence. Second sentence.", "Third sentence. Fourth sentence."],
  tgtChunks: [],
  pairs: [],
};

expectEqual(pushSentenceToNext(base, "src", 0).srcChunks, [
  "First sentence.",
  "Second sentence. Third sentence. Fourth sentence.",
]);

expectEqual(pullSentenceFromNext(base, "src", 0).srcChunks, [
  "First sentence. Second sentence. Third sentence.",
  "Fourth sentence.",
]);

expectEqual(pushSentenceToNext({ ...base, srcChunks: ["Only one sentence.", ""] }, "src", 0).srcChunks, [
  "",
  "Only one sentence.",
]);

expectEqual(pullSentenceFromNext({ ...base, srcChunks: ["Existing.", ""] }, "src", 0).srcChunks, [
  "Existing.",
  "",
]);

const missingTarget: AlignmentState = { srcChunks: ["a", "b", "c"], tgtChunks: ["x", "y"], pairs: [] };
expectEqual(getAlignmentSuggestions(missingTarget).filter((s) => s.type === "missing_target"), [
  { type: "missing_target", pairNumber: 3, srcChars: 1 },
]);
expectEqual(applyAlignmentPlaceholders(missingTarget).tgtChunks, ["x", "y", ""]);

const missingSource: AlignmentState = { srcChunks: ["a", "b"], tgtChunks: ["x", "y", "z"], pairs: [] };
expectEqual(getAlignmentSuggestions(missingSource).filter((s) => s.type === "missing_source"), [
  { type: "missing_source", pairNumber: 3, tgtChars: 1 },
]);
expectEqual(applyAlignmentPlaceholders(missingSource).srcChunks, ["a", "b", ""]);

const ratioOutlier: AlignmentState = { srcChunks: ["a".repeat(300)], tgtChunks: ["b"], pairs: [] };
expectEqual(getAlignmentSuggestions(ratioOutlier).filter((s) => s.type === "length_ratio_outlier"), [
  { type: "length_ratio_outlier", pairNumber: 1, srcChars: 300, tgtChars: 1, ratio: 300 },
]);

export {};
