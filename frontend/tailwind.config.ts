import type { Config } from "tailwindcss";

/**
 * Bitext-labeling identity:
 * - One brand axis: cool deep ink, low chroma. Carries chrome (toolbar, primary buttons, focus).
 * - Tinted neutral ramp pulled toward the brand hue (no pure grays).
 * - src/tgt accents (amber / blue) are STRUCTURAL — every chunk card has its side's accent on the left edge.
 * - Aligned / src-only / tgt-only segments pick up the same accent family at different chroma.
 * - All colors authored in OKLCH; chroma reduced near the lightness extremes per shared design laws.
 */
export default {
  content: ["./app/**/*.{ts,tsx}"],
  theme: {
    fontFamily: {
      sans: [
        '"IBM Plex Sans"',
        "system-ui",
        "-apple-system",
        "Segoe UI",
        "Roboto",
        "sans-serif",
      ],
      mono: [
        '"IBM Plex Mono"',
        "ui-monospace",
        "SFMono-Regular",
        "Menlo",
        "Consolas",
        "monospace",
      ],
      serif: [
        '"IBM Plex Serif"',
        '"Noto Serif KR"',
        "Georgia",
        "Cambria",
        '"Times New Roman"',
        "serif",
      ],
    },
    fontSize: {
      // 1.25 modular scale.
      "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      xs: ["0.75rem", { lineHeight: "1.125rem" }],
      sm: ["0.9375rem", { lineHeight: "1.4rem" }],
      base: ["1.125rem", { lineHeight: "1.625rem" }],
      lg: ["1.375rem", { lineHeight: "1.875rem" }],
      xl: ["1.75rem", { lineHeight: "2.25rem" }],
      "2xl": ["2.1875rem", { lineHeight: "2.625rem" }],
    },
    extend: {
      colors: {
        // Tinted neutral ramp, slight cool cast (hue 230) pulled toward brand.
        // Chroma trails off near the lightness extremes.
        neutral: {
          50: "oklch(98.5% 0.003 230 / <alpha-value>)",
          100: "oklch(96% 0.005 230 / <alpha-value>)",
          200: "oklch(92% 0.007 230 / <alpha-value>)",
          300: "oklch(86% 0.009 230 / <alpha-value>)",
          400: "oklch(70% 0.012 230 / <alpha-value>)",
          500: "oklch(58% 0.014 230 / <alpha-value>)",
          600: "oklch(48% 0.018 230 / <alpha-value>)",
          700: "oklch(38% 0.020 230 / <alpha-value>)",
          800: "oklch(28% 0.022 230 / <alpha-value>)",
          900: "oklch(22% 0.022 230 / <alpha-value>)",
        },
        ink: "oklch(22% 0.022 230 / <alpha-value>)",
        brand: {
          DEFAULT: "oklch(45% 0.10 230 / <alpha-value>)",
          fg: "oklch(98.5% 0.005 230 / <alpha-value>)",
          subtle: "oklch(94% 0.018 230 / <alpha-value>)",
        },
        // Source / target structural accents.
        srcOnly: "oklch(68% 0.15 60 / <alpha-value>)",
        tgtOnly: "oklch(62% 0.15 240 / <alpha-value>)",
        // Segment-type tints.
        aligned: "oklch(52% 0.13 155 / <alpha-value>)",
        empty: "oklch(70% 0.005 230 / <alpha-value>)",
      },
      minHeight: { touch: "44px" },
      minWidth: { touch: "44px" },
      borderRadius: {
        DEFAULT: "0.375rem",
      },
      boxShadow: {
        // Tinted shadows; never pure black.
        sm: "0 1px 2px 0 oklch(20% 0.02 230 / 0.04), 0 1px 1px 0 oklch(20% 0.02 230 / 0.03)",
        DEFAULT: "0 2px 4px -1px oklch(20% 0.02 230 / 0.06), 0 1px 2px 0 oklch(20% 0.02 230 / 0.04)",
        lg: "0 8px 24px -4px oklch(20% 0.02 230 / 0.10), 0 4px 8px -2px oklch(20% 0.02 230 / 0.06)",
      },
    },
  },
  plugins: [],
} satisfies Config;
