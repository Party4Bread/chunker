import { Link } from "react-router";

interface Crumb {
  to?: string;
  label: string;
}

export function Toolbar({ crumbs, right }: { crumbs: Crumb[]; right?: React.ReactNode }) {
  return (
    <header className="sticky top-0 z-10 flex flex-wrap items-center gap-3 border-b border-neutral-200 bg-neutral-50/90 px-3 py-2 backdrop-blur sm:px-4 sm:py-3">
      <Link to="/" className="group flex items-center gap-2 rounded px-1 -mx-1 py-0.5 hover:bg-neutral-100">
        <BrandMark />
        <span className="hidden text-sm font-semibold text-ink sm:inline">chunker</span>
      </Link>
      <span className="text-neutral-300" aria-hidden="true">·</span>
      <nav aria-label="breadcrumb" className="flex flex-1 flex-wrap items-center gap-x-1.5 gap-y-0.5 text-xs text-neutral-600 sm:text-sm">
        {crumbs.map((c, i) => (
          <span key={i} className="flex items-center gap-1.5">
            {i > 0 && <span className="text-neutral-300" aria-hidden="true">/</span>}
            {c.to ? (
              <Link to={c.to} className="rounded px-1 py-0.5 hover:bg-neutral-100 hover:text-ink">
                {c.label}
              </Link>
            ) : (
              <span className="font-semibold text-ink">{c.label}</span>
            )}
          </span>
        ))}
      </nav>
      {right && <div className="flex flex-wrap items-center gap-1.5">{right}</div>}
    </header>
  );
}

function BrandMark() {
  // Two interlocked squares = source/target alignment. Brand color carries the join.
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 22 22"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      className="shrink-0"
    >
      <rect x="2" y="2" width="11" height="11" rx="2" stroke="var(--color-src)" strokeWidth="1.5" />
      <rect x="9" y="9" width="11" height="11" rx="2" stroke="var(--color-tgt)" strokeWidth="1.5" />
      <rect x="9" y="9" width="4" height="4" fill="var(--color-brand)" />
    </svg>
  );
}
