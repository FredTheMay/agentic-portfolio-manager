# Dashboard (SPEC §9, M11)

React + TypeScript + Vite over the read-only FastAPI surface in [`src/api/`](../src/api).

```bash
make serve                              # terminal 1 — the API on :8000
cd web && npm install && npm run dev    # terminal 2 — the UI on :5173
```

Vite proxies `/api` in both `dev` and `preview`, so the origin is single and the read-only
service needs no CORS configuration.

## Views

| Tab | Shows |
|---|---|
| **Overview** | TWR headline, MWR beside it, α **with its t-statistic**, equity curve, holdings |
| **Screener** | All 28 instruments — filter by ticker/sector, sort by weight, YTD, β, σ |
| **Research** | One instrument: price history, ratios by CFA family, valuation, veto history |
| **Risk** | Vetoed trades first, efficient frontier, risk attribution, executor capabilities |
| **Audit** | Decision trail tagged with CFA Standards |

## Design notes

**Decimal strings are never parsed for arithmetic.** The API emits money and weights as strings
because JSON numbers are IEEE-754 doubles. `percent()`, `fixed()` and `compact()` convert for
*display* only; nothing computes on a parsed value.

**No charting library.** The whole chart set is a few hundred lines of inline SVG in
[`components/Charts.tsx`](src/components/Charts.tsx). It ships nothing extra to the client and
keeps the app deployable behind a strict CSP.

**The disclaimer never scrolls away** — a sticky banner plus a required field on every API
response, so no screen can render without it (SPEC §1). The survivorship caveat sits in the
footer, as SPEC §4.4 requires.

**Finance conventions carry meaning, not decoration**: green up / red down, tabular figures so
digits align down a column, monospace for anything a reader compares. The ticker tape respects
`prefers-reduced-motion`.

## Research view

Everything comes from `/api/research/{symbol}` and is computed by `src/cfa/` from recorded data.
Where a figure cannot be produced honestly the page says why rather than degrading silently — a
REIT ETF shows no ratios and a note explaining it has no SEC filings; a name whose sustainable
growth exceeds its required return shows no DDM value and says the model does not converge.
