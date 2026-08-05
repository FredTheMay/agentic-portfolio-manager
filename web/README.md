# Dashboard (SPEC §9, M9)

React + TypeScript + Vite over the read-only FastAPI surface in [`src/api/`](../src/api).

```bash
# terminal 1 — the API
make serve

# terminal 2 — the dashboard
cd web && npm install && npm run dev
```

Vite proxies `/api` to `127.0.0.1:8000`, so the dev origin is single and the read-only
service needs no CORS configuration.

## Screens

Ordered as SPEC §9 lists them, with one deliberate change: **vetoed trades comes first**,
because SPEC §7 names it the screen to demo first and it is the one that shows the risk
engine actually doing something.

| Panel | Shows |
|---|---|
| Vetoed trades | Every refused proposal, with the rule and the margin |
| Performance | TWR headline, MWR beside it, α **with its t-statistic** |
| Efficient frontier | Long-only, per-name capped — every point is approvable |
| Risk attribution | Systematic vs diversifiable variance |
| Holdings | Positions, weights, sectors |
| Executor capabilities | What the engine can honor; advisory constraints named |
| Audit trail | Consequential acts tagged with their CFA Standard |

## Two things the UI is careful about

**The disclaimer is not dismissible.** It is a sticky banner and a required field on every
API response, so no screen can render without it (SPEC §1).

**Decimal strings are never parsed for arithmetic.** The API emits money and weights as
strings precisely because JSON numbers are IEEE-754 doubles. `percent()` and `fixed()`
convert for *display* only; nothing computes on a parsed value.
