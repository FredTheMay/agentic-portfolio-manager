import { useEffect, useState } from "react";
import {
  api,
  type Attribution,
  type Audit,
  type Capabilities,
  type Frontier,
  type Performance,
  type Portfolio,
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

interface Data {
  status: Status;
  portfolio: Portfolio;
  performance: Performance;
  frontier: Frontier;
  vetoes: Vetoes;
  attribution: Attribution;
  audit: Audit;
  capabilities: Capabilities;
}

export default function App() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.status(),
      api.portfolio(),
      api.performance(),
      api.frontier(),
      api.vetoes(),
      api.attribution(),
      api.audit(),
      api.capabilities(),
    ])
      .then(([status, portfolio, performance, frontier, vetoes, attribution, audit, capabilities]) =>
        setData({ status, portfolio, performance, frontier, vetoes, attribution, audit, capabilities }),
      )
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error) return <main className="shell"><p className="error">Could not load: {error}</p></main>;
  if (!data) return <main className="shell"><p className="empty">Loading…</p></main>;

  return (
    <>
      {/* SPEC §1: persistent, not dismissible, and above everything else. */}
      <div className="banner" role="note">{data.status.disclaimer}</div>

      <main className="shell">
        <header className="masthead">
          <h1>Agentic Portfolio Manager</h1>
          <p className="lede">
            Multi-agent research over a deterministic risk engine. The LLM proposes
            categories; every number is computed in Python.
          </p>
          <div className="chips">
            <span className="chip"><code>LLM</code><b>{data.status.llm_provider}</b></span>
            <span className="chip"><code>executor</code><b>{data.status.executor}</b></span>
            <span className="chip"><code>data</code><b>{data.status.data_source}</b></span>
            <span className="chip"><code>cycles</code><b>{data.status.cycles}</b></span>
            <span className="chip"><code>executed</code><b>{data.status.executed}</b></span>
            <span className="chip"><code>vetoed</code><b>{data.status.vetoed}</b></span>
          </div>
        </header>

        {/* Vetoes first: SPEC §7 names it the screen to demo first. */}
        <VetoPanel data={data.vetoes} />
        <PerformancePanel data={data.performance} />
        <div className="two-up">
          <FrontierPanel data={data.frontier} />
          <AttributionPanel data={data.attribution} />
        </div>
        <HoldingsPanel data={data.portfolio} />
        <CapabilitiesPanel data={data.capabilities} />
        <AuditPanel data={data.audit} />

        <footer className="footer">
          <p><strong>{data.status.disclaimer}</strong></p>
          {/* SPEC §4.4 requires the survivorship limitation in the footer. */}
          <p>{data.status.survivorship_notice}</p>
        </footer>
      </main>
    </>
  );
}
