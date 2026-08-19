import type { ReactNode } from "react";
import {
  fixed,
  percent,
  type Attribution,
  type Audit,
  type Capabilities,
  type Frontier,
  type Performance,
  type Portfolio,
  type Vetoes,
} from "../api";

export function Panel({ title, subtitle, children }: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      <header>
        <h2>{title}</h2>
        {subtitle && <p className="subtitle">{subtitle}</p>}
      </header>
      {children}
    </section>
  );
}

/** this is the screen to demo first. */
export function VetoPanel({ data }: { data: Vetoes }) {
  const codes = Object.entries(data.by_code).sort((a, b) => b[1] - a[1]);
  return (
    <Panel
      title="Vetoed trades"
      subtitle="Every proposal the risk engine refused, with the rule that stopped it."
    >
      {data.total === 0 ? (
        <p className="empty">No proposals were vetoed in this run.</p>
      ) : (
        <>
          <div className="chips">
            {codes.map(([code, count]) => (
              <span className="chip" key={code}>
                <code>{code}</code>
                <b>{count}</b>
              </span>
            ))}
          </div>
          <table>
            <thead>
              <tr>
                <th>When</th>
                <th>Rule</th>
                <th>Symbol</th>
                <th className="num">Observed</th>
                <th className="num">Limit</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {data.vetoes.slice(0, 40).map((veto, index) => (
                <tr key={`${veto.timestamp}-${veto.code}-${index}`}>
                  <td>{veto.timestamp.slice(0, 10)}</td>
                  <td><code>{veto.code}</code></td>
                  <td>{veto.symbol ?? "—"}</td>
                  <td className="num">{fixed(veto.observed, 4)}</td>
                  <td className="num">{fixed(veto.limit, 4)}</td>
                  <td className="detail">{veto.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  );
}

export function PerformancePanel({ data }: { data: Performance }) {
  const beat = Number(data.annualized_twr) >= Number(data.annualized_benchmark_twr);
  return (
    <Panel
      title="Performance"
      subtitle="TWR is the headline: GIPS requires it because chain-linking removes the effect of cash-flow timing. MWR sits beside it so the gap can be explained."
    >
      <div className="stats">
        <Stat label="Annualized TWR" value={percent(data.annualized_twr)} emphasis />
        <Stat label="Benchmark TWR" value={percent(data.annualized_benchmark_twr)} />
        <Stat label="MWR" value={percent(data.mwr)} />
        <Stat label="Volatility" value={percent(data.annualized_volatility)} />
        <Stat label="Max drawdown" value={percent(data.max_drawdown)} />
        <Stat label="Sharpe" value={fixed(data.sharpe)} />
        <Stat label="Treynor" value={fixed(data.treynor, 4)} />
        <Stat label="Information ratio" value={fixed(data.information_ratio)} />
        <Stat label="Beta" value={fixed(data.beta)} />
        <Stat label="R²" value={fixed(data.r_squared)} />
      </div>

      <div className={`alpha ${data.alpha_is_significant ? "significant" : "insignificant"}`}>
        <div>
          <span className="alpha-label">Jensen&rsquo;s α</span>
          <span className="alpha-value">{percent(data.jensens_alpha)}</span>
        </div>
        <div>
          <span className="alpha-label">t-statistic</span>
          <span className="alpha-value">{fixed(data.alpha_t_stat)}</span>
        </div>
        <p>
          {data.alpha_is_significant
            ? "Distinguishable from zero at roughly 5% — in sample. That is not the same as the strategy having alpha out of sample."
            : "Not distinguishable from zero. The point estimate above should not be read as a result."}
        </p>
      </div>

      <EquityChart data={data} />
      <p className="note">
        {beat
          ? "The portfolio outperformed its benchmark over this window."
          : "The portfolio underperformed its benchmark over this window — the expected outcome under the semi-strong efficiency assumption this system states up front."}
      </p>
    </Panel>
  );
}

function EquityChart({ data }: { data: Performance }) {
  const equity = data.equity_curve.map(Number);
  const benchmark = data.benchmark_curve.map(Number);
  if (equity.length < 2) return null;

  const all = [...equity, ...benchmark];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const width = 720;
  const height = 220;

  const path = (series: number[]) =>
    series
      .map((value, index) => {
        const x = (index / (series.length - 1)) * width;
        const y = height - ((value - min) / span) * height;
        return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Equity curve versus benchmark">
        <path d={path(benchmark)} className="line benchmark" />
        <path d={path(equity)} className="line portfolio" />
      </svg>
      <figcaption>
        <span className="key portfolio" /> Portfolio
        <span className="key benchmark" /> Benchmark
      </figcaption>
    </figure>
  );
}

export function FrontierPanel({ data }: { data: Frontier }) {
  if (!data.points.length) {
    return (
      <Panel title="Efficient frontier">
        <p className="empty">No frontier was computed in this run.</p>
      </Panel>
    );
  }

  const xs = data.points.map((p) => Number(p.standard_deviation));
  const ys = data.points.map((p) => Number(p.expected_return));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = 460;
  const height = 260;
  const sx = (v: number) => ((v - minX) / (maxX - minX || 1)) * width;
  const sy = (v: number) => height - ((v - minY) / (maxY - minY || 1)) * height;

  const selected = data.selected;
  return (
    <Panel
      title="Efficient frontier"
      subtitle={`Long-only and capped per name, so every point shown is a portfolio the risk engine could actually approve. Selection: ${data.method}.`}
    >
      <figure className="chart">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Efficient frontier">
          <path
            d={data.points
              .map((p, i) => `${i === 0 ? "M" : "L"}${sx(Number(p.standard_deviation)).toFixed(1)},${sy(Number(p.expected_return)).toFixed(1)}`)
              .join(" ")}
            className="line portfolio"
          />
          {selected && (
            <circle
              cx={sx(Number(selected.standard_deviation))}
              cy={sy(Number(selected.expected_return))}
              r={6}
              className="selected-point"
            />
          )}
        </svg>
        <figcaption>Risk (σ) → · Expected return ↑</figcaption>
      </figure>
      {selected && (
        <table>
          <thead>
            <tr><th>Symbol</th><th className="num">Target weight</th></tr>
          </thead>
          <tbody>
            {Object.entries(selected.weights).map(([symbol, weight]) => (
              <tr key={symbol}>
                <td>{symbol}</td>
                <td className="num">{percent(weight)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

export function HoldingsPanel({ data }: { data: Portfolio }) {
  return (
    <Panel title="Holdings" subtitle={`As of ${data.as_of.slice(0, 10)}`}>
      <div className="stats">
        <Stat label="Total value" value={data.total_value} emphasis />
        <Stat label="Cash" value={data.cash} />
        <Stat label="Cash weight" value={percent(data.cash_weight)} />
      </div>
      {data.holdings.length === 0 ? (
        <p className="empty">The book is entirely in cash.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Sector</th>
              <th className="num">Shares</th>
              <th className="num">Value</th>
              <th className="num">Weight</th>
            </tr>
          </thead>
          <tbody>
            {data.holdings.map((h) => (
              <tr key={h.symbol}>
                <td>{h.symbol}</td>
                <td>{h.sector ?? "—"}</td>
                <td className="num">{h.quantity}</td>
                <td className="num">{h.market_value}</td>
                <td className="num">{percent(h.weight)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

export function AttributionPanel({ data }: { data: Attribution }) {
  const share = Number(data.systematic_share);
  return (
    <Panel
      title="Risk attribution"
      subtitle="Total variance split into the part explained by market exposure and the part diversification could remove. Under CAPM the second earns no expected return."
    >
      <div className="bar">
        <div className="systematic" style={{ width: `${(share * 100).toFixed(1)}%` }}>
          Systematic {percent(data.systematic_share)}
        </div>
        <div className="unsystematic">Residual</div>
      </div>
      <div className="stats">
        <Stat label="Portfolio beta" value={fixed(data.beta)} />
        <Stat label="Total variance" value={fixed(data.total_variance, 5)} />
        <Stat label="Systematic" value={fixed(data.systematic_variance, 5)} />
        <Stat label="Residual" value={fixed(data.unsystematic_variance, 5)} />
      </div>
    </Panel>
  );
}

export function AuditPanel({ data }: { data: Audit }) {
  return (
    <Panel
      title="Audit trail"
      subtitle="Every consequential act, tagged with the CFA Standard it implements."
    >
      {data.total === 0 ? (
        <p className="empty">No audit events were recorded in this run.</p>
      ) : (
        <table>
          <thead>
            <tr><th>When</th><th>Actor</th><th>Code</th><th>Standard</th><th>Detail</th></tr>
          </thead>
          <tbody>
            {data.entries.slice(0, 40).map((entry, index) => (
              <tr key={`${entry.timestamp}-${index}`}>
                <td>{entry.timestamp.slice(0, 10)}</td>
                <td>{entry.actor}</td>
                <td><code>{entry.code}</code></td>
                <td>{entry.standard}</td>
                <td className="detail">{entry.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

export function CapabilitiesPanel({ data }: { data: Capabilities }) {
  return (
    <Panel
      title="Executor capabilities"
      subtitle="What the configured engine can actually honor. A constraint it cannot respect is advisory, and saying so beats assuming it was applied."
    >
      <div className="stats">
        <Stat label="Engine" value={data.engine_name} />
        <Stat label="Version" value={data.engine_version} />
        <Stat label="Intraday" value={data.supports_intraday ? "yes" : "no"} />
        <Stat label="Participation limits" value={data.supports_participation_limits ? "yes" : "no"} />
      </div>
      {data.advisory_constraints.length > 0 && (
        <p className="warn">
          Advisory only — not enforced by this executor:{" "}
          {data.advisory_constraints.map((c) => <code key={c}>{c}</code>)}
        </p>
      )}
    </Panel>
  );
}

function Stat({ label, value, emphasis }: { label: string; value: string; emphasis?: boolean }) {
  return (
    <div className={`stat ${emphasis ? "emphasis" : ""}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}
