import type { Config } from "tailwindcss";

/**
 * Bitext-labeling identity:
 * - One brand axis: cool deep ink, low chroma. Carries chrome (toolbar, primary buttons, focus).
 * - Tinted neutral ramp pulled toward the brand hue (no pure grays).
 * - src/tgt accents (amber / blue) are STRUCTURAL — every chunk card has its side's accent on the left edge.
 * - Aligned / src-only / tgt-only segments pick up the same accent family at different chroma.
 * - All colors authored in OKLCH; chroma reduced near the lightness extremes per shared design laws.
 *
 * Each color references a CSS variable holding an "L C H" triplet so that
 * styles.css can swap the entire palette for dark mode without touching any
 * component className. <alpha-value> is Tailwind's opacity hook.
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
        neutral: {
          50: "oklch(var(--neutral-50) / <alpha-value>)",
          100: "oklch(var(--neutral-100) / <alpha-value>)",
          200: "oklch(var(--neutral-200) / <alpha-value>)",
          300: "oklch(var(--neutral-300) / <alpha-value>)",
          400: "oklch(var(--neutral-400) / <alpha-value>)",
          500: "oklch(var(--neutral-500) / <alpha-value>)",
          600: "oklch(var(--neutral-600) / <alpha-value>)",
          700: "oklch(var(--neutral-700) / <alpha-value>)",
          800: "oklch(var(--neutral-800) / <alpha-value>)",
          900: "oklch(var(--neutral-900) / <alpha-value>)",
        },
        // Tailwind's default red palette is RGB. We re-author the shades we
        // actually use in OKLCH so they participate in the theme swap.
        red: {
          50: "oklch(var(--red-50) / <alpha-value>)",
          100: "oklch(var(--red-100) / <alpha-value>)",
          200: "oklch(var(--red-200) / <alpha-value>)",
          300: "oklch(var(--red-300) / <alpha-value>)",
          500: "oklch(var(--red-500) / <alpha-value>)",
          600: "oklch(var(--red-600) / <alpha-value>)",
          700: "oklch(var(--red-700) / <alpha-value>)",
        },
        ink: "oklch(var(--ink) / <alpha-value>)",
        brand: {
          DEFAULT: "oklch(var(--brand) / <alpha-value>)",
          fg: "oklch(var(--brand-fg) / <alpha-value>)",
          subtle: "oklch(var(--brand-subtle) / <alpha-value>)",
        },
        // "surface" replaces bare bg-white — it tracks the theme.
        surface: "oklch(var(--surface) / <alpha-value>)",
        // Source / target structural accents.
        srcOnly: "oklch(var(--src-only) / <alpha-value>)",
        tgtOnly: "oklch(var(--tgt-only) / <alpha-value>)",
        // Segment-type tints.
        aligned: "oklch(var(--aligned) / <alpha-value>)",
        empty: "oklch(var(--empty) / <alpha-value>)",
      },
      minHeight: { touch: "44px" },
      minWidth: { touch: "44px" },
      borderRadius: {
        DEFAULT: "0.375rem",
      },
      boxShadow: {
        // Whole shadow stack lives in CSS variables so the dark theme can swap
        // to stronger near-black drops (lifted opacities) without us having to
        // re-derive every alpha here — see --shadow-* in styles.css.
        sm: "var(--shadow-sm)",
        DEFAULT: "var(--shadow-md)",
        lg: "var(--shadow-lg)",
      },
    },
  },
  plugins: [],
} satisfies Config;
