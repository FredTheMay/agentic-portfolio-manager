// Typed client for the read-only dashboard API (SPEC §9).
//
// Every monetary and ratio value arrives as a decimal STRING, deliberately.
// Parsing one into a JS number is lossy — that is the whole reason the API
// emits strings — so values are formatted for display without ever being
// turned back into floats for arithmetic.

export interface Holding {
  symbol: string;
  quantity: number;
  market_value: string;
  weight: string;
  sector: string | null;
}

export interface Portfolio {
  as_of: string;
  total_value: string;
  cash: string;
  cash_weight: string;
  holdings: Holding[];
  disclaimer: string;
}

export interface Performance {
  periods: number;
  annualized_twr: string;
  annualized_benchmark_twr: string;
  mwr: string | null;
  annualized_volatility: string;
  max_drawdown: string;
  sharpe: string | null;
  treynor: string | null;
  information_ratio: string | null;
  jensens_alpha: string;
  alpha_t_stat: string | null;
  alpha_is_significant: boolean;
  beta: string;
  r_squared: string;
  tracking_error: string;
  equity_curve: string[];
  benchmark_curve: string[];
  timestamps: string[];
}

export interface FrontierPoint {
  expected_return: string;
  standard_deviation: string;
  weights: Record<string, string>;
}

export interface Frontier {
  points: FrontierPoint[];
  selected: FrontierPoint | null;
  method: string;
}

export interface Veto {
  timestamp: string;
  code: string;
  symbol: string | null;
  detail: string;
  observed: string | null;
  limit: string | null;
}

export interface Vetoes {
  total: number;
  by_code: Record<string, number>;
  vetoes: Veto[];
}

export interface Attribution {
  total_variance: string;
  systematic_variance: string;
  unsystematic_variance: string;
  systematic_share: string;
  beta: string;
}

export interface AuditEntry {
  timestamp: string;
  actor: string;
  code: string;
  standard: string;
  symbol: string | null;
  detail: string;
}

export interface Audit {
  total: number;
  by_code: Record<string, number>;
  entries: AuditEntry[];
}

export interface Capabilities {
  engine_name: string;
  engine_version: string;
  supports_intraday: boolean;
  supports_participation_limits: boolean;
  supports_streaming_updates: boolean;
  advisory_constraints: string[];
}

export interface Status {
  llm_provider: string;
  executor: string;
  cycles: number;
  executed: number;
  vetoed: number;
  data_source: string;
  disclaimer: string;
  survivorship_notice: string;
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return (await response.json()) as T;
}

export const api = {
  status: () => get<Status>("/api/status"),
  portfolio: () => get<Portfolio>("/api/portfolio"),
  performance: () => get<Performance>("/api/performance"),
  frontier: () => get<Frontier>("/api/frontier"),
  vetoes: () => get<Vetoes>("/api/vetoes"),
  attribution: () => get<Attribution>("/api/attribution"),
  audit: () => get<Audit>("/api/audit"),
  capabilities: () => get<Capabilities>("/api/capabilities"),
};

/** Format a decimal string as a percentage for display only. */
export function percent(value: string | null, places = 2): string {
  if (value === null) return "n/a";
  const parsed = Number(value);
  if (Number.isNaN(parsed)) return value;
  return `${(parsed * 100).toFixed(places)}%`;
}

/** Format a decimal string with fixed places, for display only. */
export function fixed(value: string | null, places = 2): string {
  if (value === null) return "n/a";
  const parsed = Number(value);
  return Number.isNaN(parsed) ? value : parsed.toFixed(places);
}
