import { useEffect, useState } from "react";
import {
  api,
  direction,
  percent,
  research as researchApi,
  type Attribution,
  type Audit,
  type Capabilities,
  type Frontier,
  type Performance,
  type Portfolio,
  type Screen,
  type Status,
  type Vetoes,
} from "./api";
import {
  AttributionPanel,
  AuditPanel,
  CapabilitiesPanel,
  FrontierPanel,
  HoldingsPanel,
  PerformancePanel,
  VetoPanel,
} from "./components/Panels";
import { Research } from "./components/Research";
import { Screener } from "./components/Screener";

type View = "overview" | "screener" | "research" | "risk" | "audit";

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: "overview", label: "Overview", hint: "Performance and holdings" },
  { id: "screener", label: "Screener", hint: "Browse the universe" },
  { id: "research", label: "Research", hint: "Deep dive on one name" },
  { id: "risk", label: "Risk", hint: "Vetoes, frontier, attribution" },
  { id: "audit", label: "Audit", hint: "Decision trail" },
];

interface Data {
  status: Status;
  portfolio: Portfolio;
  performance: Performance;
  frontier: Frontier;
  vetoes: Vetoes;
  attribution: Attribution;
  audit: Audit;
  capabilities: Capabilities;
  screen: Screen | null;
}

export default function App() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("overview");
  const [symbol, setSymbol] = useState("AAPL");

  useEffect(() => {
    Promise.all([
      api.status(), api.portfolio(), api.performance(), api.frontier(),
      api.vetoes(), api.attribution(), api.audit(), api.capabilities(),
      // The screener needs recorded data; a synthetic run has no universe.
      researchApi.screen().catch(() => null),
    ])
      .then(([status, portfolio, performance, frontier, vetoes, attribution, audit, capabilities, screen]) =>
        setData({ status, portfolio, performance, frontier, vetoes, attribution, audit, capabilities, screen }),
      )
      .catch((err: Error) => setError(err.message));
  }, []);

  function open(ticker: string) {
    setSymbol(ticker);
    setView("research");
  }

  if (error) return <main className="shell"><p className="error">Could not load: {error}</p></main>;
  if (!data) return <main className="shell"><p className="empty loading">Loading market data…</p></main>;

  const beat =
    Number(data.performance.annualized_twr) >= Number(data.performance.annualized_benchmark_twr);

  return (
    <>
      {/* SPEC §1: persistent, not dismissible, above everything else. */}
      <div className="banner" role="note">{data.status.disclaimer}</div>

      {/* Ticker tape: live-looking, but every figure is from the recorded run. */}
      {data.screen && (
        <div className="tape" aria-hidden="true">
          <div className="tape-track">
            {[...data.screen.symbols, ...data.screen.symbols].map((s, i) => (
              <span className="tape-item" key={`${s.symbol}-${i}`}>
                <b>{s.symbol}</b>
                <i className={direction(s.change_ytd)}>{percent(s.change_ytd, 1)}</i>
              </span>
            ))}
          </div>
        </div>
      )}

      <header className="masthead">
        <div className="brand">
          <span className="mark" aria-hidden="true">▚</span>
          <div>
            <h1>Agentic Portfolio Manager</h1>
            <p className="lede">
              LLM agents propose categories; every number is computed in Python under a
              deterministic risk engine.
            </p>
          </div>
        </div>
        <div className="chips">
          <span className="chip"><code>data</code><b>{data.status.data_source}</b></span>
          <span className="chip"><code>llm</code><b>{data.status.llm_provider}</b></span>
          <span className="chip"><code>executor</code><b>{data.status.executor}</b></span>
          <span className={`chip ${beat ? "good" : "bad"}`}>
            <code>vs bench</code>
            <b>
              {percent(
                String(
                  Number(data.performance.annualized_twr) -
                    Number(data.performance.annualized_benchmark_twr),
                ),
              )}
            </b>
          </span>
          <span className="chip"><code>vetoed</code><b>{data.status.vetoed}</b></span>
        </div>
      </header>

      <nav className="tabs" aria-label="Views">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            className={view === v.id ? "tab active" : "tab"}
            onClick={() => setView(v.id)}
            title={v.hint}
            aria-current={view === v.id}
          >
            {v.label}
          </button>
        ))}
      </nav>

      <main className="shell">
        {view === "overview" && (
          <>
            <PerformancePanel data={data.performance} />
            <HoldingsPanel data={data.portfolio} />
          </>
        )}

        {view === "screener" &&
          (data.screen ? (
            <Screener data={data.screen} onSelect={open} />
          ) : (
            <section className="panel">
              <p className="empty">
                The screener needs recorded market data. Run <code>make backfill</code> with
                credentials in <code>.env</code>.
              </p>
            </section>
          ))}

        {view === "research" &&
          (data.screen ? (
            <Research screen={data.screen} symbol={symbol} onSelect={setSymbol} />
          ) : (
            <section className="panel">
              <p className="empty">
                Research needs recorded market data. Run <code>make backfill</code>.
              </p>
            </section>
          ))}

        {view === "risk" && (
          <>
            {/* Vetoes first: SPEC §7 names it the screen to demo first. */}
            <VetoPanel data={data.vetoes} />
            <div className="two-up">
              <FrontierPanel data={data.frontier} />
              <AttributionPanel data={data.attribution} />
            </div>
            <CapabilitiesPanel data={data.capabilities} />
          </>
        )}

        {view === "audit" && <AuditPanel data={data.audit} />}

        <footer className="footer">
          <p><strong>{data.status.disclaimer}</strong></p>
          {/* SPEC §4.4 requires the survivorship limitation in the footer. */}
          <p>{data.status.survivorship_notice}</p>
        </footer>
      </main>
    </>
  );
}
