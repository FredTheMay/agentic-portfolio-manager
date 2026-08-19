// Inline SVG charts. No charting library on purpose: the whole set is a few
// hundred lines, it ships nothing to the client beyond the bundle it is
// already loading, and it keeps the app deployable behind a strict CSP.
//
// Every chart takes decimal STRINGS and parses for display only. Nothing here
// feeds a number back into the system.

interface Series {
  label: string;
  values: number[];
  className: string;
}

function path(values: number[], width: number, height: number, min: number, max: number): string {
  const span = max - min || 1;
  return values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width;
      const y = height - ((value - min) / span) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

/** Multi-series line chart with a gridded plot area. */
export function LineChart({
  series,
  height = 240,
  labels = [],
  format = (v: number) => v.toFixed(0),
}: {
  series: Series[];
  height?: number;
  labels?: string[];
  format?: (value: number) => string;
}) {
  const width = 900;
  const all = series.flatMap((s) => s.values);
  if (all.length < 2) return <p className="empty">Not enough data to plot.</p>;

  const min = Math.min(...all);
  const max = Math.max(...all);
  const ticks = 4;

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img"
           aria-label={series.map((s) => s.label).join(" versus ")}>
        {Array.from({ length: ticks + 1 }, (_, i) => {
          const y = (i / ticks) * height;
          return <line key={i} className="grid" x1={0} y1={y} x2={width} y2={y} />;
        })}
        {series.map((s) => (
          <path key={s.label} className={`line ${s.className}`} d={path(s.values, width, height, min, max)} />
        ))}
      </svg>
      <div className="axis">
        <span>{format(max)}</span>
        <span>{format(min)}</span>
      </div>
      <figcaption>
        {series.map((s) => (
          <span key={s.label} className="legend">
            <i className={`key ${s.className}`} />
            {s.label}
          </span>
        ))}
        {labels.length > 0 && (
          <span className="range">
            {labels[0]} → {labels[labels.length - 1]}
          </span>
        )}
      </figcaption>
    </figure>
  );
}

/** Scatter with a highlighted point — used for the efficient frontier. */
export function ScatterChart({
  points,
  selected,
  xLabel,
  yLabel,
}: {
  points: { x: number; y: number }[];
  selected?: { x: number; y: number } | null;
  xLabel: string;
  yLabel: string;
}) {
  const width = 460;
  const height = 260;
  if (points.length < 2) return <p className="empty">No frontier computed.</p>;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (v: number) => ((v - minX) / (maxX - minX || 1)) * width;
  const sy = (v: number) => height - ((v - minY) / (maxY - minY || 1)) * height;

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${yLabel} against ${xLabel}`}>
        {[0, 1, 2, 3, 4].map((i) => (
          <line key={i} className="grid" x1={0} y1={(i / 4) * height} x2={width} y2={(i / 4) * height} />
        ))}
        <path className="line portfolio"
              d={points.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ")} />
        {selected && <circle className="selected-point" cx={sx(selected.x)} cy={sy(selected.y)} r={6} />}
      </svg>
      <figcaption>
        {xLabel} → · {yLabel} ↑
      </figcaption>
    </figure>
  );
}

/** Horizontal bars, for weights and ratio families. */
export function BarList({
  rows,
  format = (v: number) => `${(v * 100).toFixed(1)}%`,
}: {
  rows: { label: string; value: number; hint?: string }[];
  format?: (value: number) => string;
}) {
  if (rows.length === 0) return <p className="empty">Nothing to show.</p>;
  const peak = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);

  return (
    <ul className="bars">
      {rows.map((row) => (
        <li key={row.label} title={row.hint}>
          <span className="bar-label">{row.label}</span>
          <span className="bar-track">
            <span
              className={`bar-fill ${row.value < 0 ? "negative" : ""}`}
              style={{ width: `${(Math.abs(row.value) / peak) * 100}%` }}
            />
          </span>
          <span className="bar-value">{format(row.value)}</span>
        </li>
      ))}
    </ul>
  );
}

/** Tiny inline trend line for table rows. */
export function Sparkline({ values, up }: { values: number[]; up: boolean }) {
  if (values.length < 2) return <span className="empty">—</span>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  return (
    <svg className="sparkline" viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden="true">
      <path className={up ? "up" : "down"} d={path(values, 100, 24, min, max)} />
    </svg>
  );
}
