import { useMemo, useState } from "react";
import { compact, direction, fixed, percent, type Screen, type SymbolCard } from "../api";

type SortKey = "symbol" | "change_ytd" | "beta" | "volatility" | "current_weight";

/** Browse and filter the investable universe. */
export function Screener({
  data,
  onSelect,
}: {
  data: Screen;
  onSelect: (symbol: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [sector, setSector] = useState("ALL");
  const [heldOnly, setHeldOnly] = useState(false);
  const [sort, setSort] = useState<SortKey>("current_weight");

  const rows = useMemo(() => {
    const needle = query.trim().toUpperCase();
    const filtered = data.symbols.filter((s) => {
      if (needle && !s.symbol.includes(needle) && !s.sector.includes(needle)) return false;
      if (sector !== "ALL" && s.sector !== sector) return false;
      if (heldOnly && Number(s.current_weight) <= 0) return false;
      return true;
    });

    const numeric = (card: SymbolCard, key: SortKey): number => {
      const raw = card[key];
      return raw === null ? Number.NEGATIVE_INFINITY : Number(raw);
    };

    return [...filtered].sort((a, b) =>
      sort === "symbol" ? a.symbol.localeCompare(b.symbol) : numeric(b, sort) - numeric(a, sort),
    );
  }, [data.symbols, query, sector, heldOnly, sort]);

  return (
    <section className="panel">
      <header>
        <h2>Screener</h2>
        <p className="subtitle">
          {data.count} instruments in the investable universe. Betas are regressed from realized
          excess returns, not supplied.
        </p>
      </header>

      <div className="controls">
        <input
          className="search"
          type="search"
          placeholder="Filter by ticker or sector…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Filter instruments"
        />
        <select value={sector} onChange={(e) => setSector(e.target.value)} aria-label="Sector">
          <option value="ALL">All sectors</option>
          {data.sectors.map((s) => (
            <option key={s} value={s}>{s.replace(/_/g, " ")}</option>
          ))}
        </select>
        <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} aria-label="Sort by">
          <option value="current_weight">Sort: weight</option>
          <option value="change_ytd">Sort: YTD</option>
          <option value="beta">Sort: beta</option>
          <option value="volatility">Sort: volatility</option>
          <option value="symbol">Sort: ticker</option>
        </select>
        <label className="toggle">
          <input type="checkbox" checked={heldOnly} onChange={(e) => setHeldOnly(e.target.checked)} />
          Held only
        </label>
      </div>

      {rows.length === 0 ? (
        <p className="empty">No instrument matches that filter.</p>
      ) : (
        <table className="screener">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Sector</th>
              <th className="num">Last</th>
              <th className="num">1D</th>
              <th className="num">YTD</th>
              <th className="num">β</th>
              <th className="num">σ</th>
              <th className="num">Weight</th>
              <th>Filings</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((card) => (
              <tr key={card.symbol} onClick={() => onSelect(card.symbol)} className="clickable"
                  tabIndex={0}
                  onKeyDown={(e) => e.key === "Enter" && onSelect(card.symbol)}>
                <td className="ticker">{card.symbol}</td>
                <td className="muted">{card.sector.replace(/_/g, " ")}</td>
                <td className="num">{card.latest_price ? compact(card.latest_price) : "—"}</td>
                <td className={`num ${direction(card.change_1d)}`}>{percent(card.change_1d)}</td>
                <td className={`num ${direction(card.change_ytd)}`}>{percent(card.change_ytd)}</td>
                <td className="num">{fixed(card.beta)}</td>
                <td className="num">{percent(card.volatility, 1)}</td>
                <td className="num">
                  {Number(card.current_weight) > 0 ? percent(card.current_weight) : "—"}
                </td>
                <td>{card.has_fundamentals ? <span className="tag">SEC</span> : <span className="muted">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
