import { useEffect, useMemo, useState } from "react";
import {
  compact,
  direction,
  fixed,
  percent,
  ratioLabel,
  research as researchApi,
  type Research as ResearchData,
  type Screen,
} from "../api";
import { BarList, LineChart } from "./Charts";

/** Deep dive on a single instrument. */
export function Research({
  screen,
  symbol,
  onSelect,
}: {
  screen: Screen;
  symbol: string;
  onSelect: (symbol: string) => void;
}) {
  const [data, setData] = useState<ResearchData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    setData(null);
    setError(null);
    researchApi
      .symbol(symbol)
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, [symbol]);

  const matches = useMemo(() => {
    const needle = query.trim().toUpperCase();
    if (!needle) return [];
    return screen.symbols.filter((s) => s.symbol.startsWith(needle)).slice(0, 8);
  }, [screen.symbols, query]);

  const families = useMemo(() => {
    if (!data) return [];
    const grouped = new Map<string, typeof data.ratios>();
    for (const row of data.ratios) {
      const list = grouped.get(row.family) ?? [];
      list.push(row);
      grouped.set(row.family, list);
    }
    return [...grouped.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [data]);

  return (
    <section className="panel research">
      <header>
        <h2>Research</h2>
        <p className="subtitle">
          Everything the system knows about one instrument, read <strong>as of</strong> the end of
          the recorded window — the same point-in-time discipline the backtest runs under.
        </p>
      </header>

      <div className="controls">
        <div className="typeahead">
          <input
            className="search"
            type="search"
            placeholder="Search a ticker… (e.g. AAPL)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search instrument"
          />
          {matches.length > 0 && (
            <ul className="suggestions">
              {matches.map((m) => (
                <li key={m.symbol}>
                  <button onClick={() => { onSelect(m.symbol); setQuery(""); }}>
                    <b>{m.symbol}</b> <span className="muted">{m.sector.replace(/_/g, " ")}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="quick">
          {screen.symbols.slice(0, 8).map((s) => (
            <button key={s.symbol} className={s.symbol === symbol ? "chip active" : "chip"}
                    onClick={() => onSelect(s.symbol)}>
              {s.symbol}
            </button>
          ))}
        </div>
      </div>

      {error && <p className="error">Could not load {symbol}: {error}</p>}
      {!data && !error && <p className="empty">Loading {symbol}…</p>}

      {data && (
        <>
          <div className="quote">
            <div className="quote-main">
              <span className="quote-ticker">{data.profile.symbol}</span>
              <span className="quote-sector">{data.profile.sector.replace(/_/g, " ")}</span>
              <span className="quote-cat">{data.profile.category}</span>
            </div>
            <div className="quote-price">
              <span className="price">{data.profile.latest_price ?? "—"}</span>
              <span className={`delta ${direction(data.profile.change_1d)}`}>
                {percent(data.profile.change_1d)} 1D
              </span>
              <span className={`delta ${direction(data.profile.change_ytd)}`}>
                {percent(data.profile.change_ytd)} YTD
              </span>
            </div>
          </div>

          <div className="stats">
            <Stat label="Beta (regressed)" value={fixed(data.profile.beta)} />
            <Stat label="Volatility (ann.)" value={percent(data.profile.volatility, 1)} />
            <Stat label="CAPM required return" value={percent(data.capm_required_return)} />
            <Stat label="Portfolio weight" value={percent(data.profile.current_weight)} />
            <Stat label="Enterprise value" value={compact(data.enterprise_value)} />
            <Stat label="Filings through" value={data.fundamentals_period ?? "none"} />
          </div>

          <LineChart
            height={220}
            labels={data.prices.map((p) => p.t)}
            format={(v) => v.toFixed(2)}
            series={[
              {
                label: `${data.profile.symbol} adjusted close`,
                className: "portfolio",
                values: data.prices.map((p) => Number(p.adjusted)),
              },
            ]}
          />

          {data.veto_codes.length > 0 && (
            <p className="warn">
              The risk engine has vetoed trades in this name:{" "}
              {[...new Set(data.veto_codes)].map((c) => <code key={c}>{c}</code>)}
            </p>
          )}

          {data.valuation && (
            <div className="valuation">
              <h3>Valuation — {data.valuation.method}</h3>
              <p className="valuation-value">{data.valuation.value ?? "no value"}</p>
              <p className="muted">{data.valuation.reason}</p>
            </div>
          )}

          {families.length > 0 ? (
            <div className="ratio-grid">
              {families.map(([family, rows]) => (
                <div key={family} className="ratio-family">
                  <h3>{family}</h3>
                  <BarList
                    rows={rows.map((r) => ({
                      label: ratioLabel(r.name),
                      value: Number(r.value),
                      hint: r.name,
                    }))}
                    format={(v) => v.toFixed(4)}
                  />
                </div>
              ))}
            </div>
          ) : (
            <p className="empty">No ratios — this instrument has no SEC filings recorded.</p>
          )}

          {data.notes.length > 0 && (
            <div className="notes">
              <h3>Caveats for this instrument</h3>
              <ul>
                {data.notes.map((note) => <li key={note}>{note}</li>)}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}
