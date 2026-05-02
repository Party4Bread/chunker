/**
 * Visually hide content while keeping it available to screen readers.
 * Uses the recognised "sr-only" technique from inclusive-design literature.
 */
export function VisuallyHidden({
  as: Tag = "span",
  children,
}: {
  as?: keyof React.JSX.IntrinsicElements;
  children: React.ReactNode;
}) {
  const Component = Tag as React.ElementType;
  return (
    <Component
      style={{
        position: "absolute",
        width: "1px",
        height: "1px",
        padding: 0,
        margin: "-1px",
        overflow: "hidden",
        clip: "rect(0,0,0,0)",
        whiteSpace: "nowrap",
        border: 0,
      }}
    >
      {children}
    </Component>
  );
}
