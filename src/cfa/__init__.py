"""CFA Level I quantitative core (SPEC §6). Pure functions, zero I/O.

Every public function names its CFA topic area in its docstring and is covered
by a hand-computed golden test in ``tests/test_cfa_golden.py``.

| Module | SPEC | Topic area |
|---|---|---|
| :mod:`~src.cfa.returns` | §6.1 | Quantitative Methods |
| :mod:`~src.cfa.portfolio` | §6.2 | Portfolio Management |
| :mod:`~src.cfa.ratios` | §6.4 | Financial Statement Analysis |
| :mod:`~src.cfa.valuation` | §6.5 | Equity Investments |
| :mod:`~src.cfa.fixed_income` | §6.6 | Fixed Income |
| :mod:`~src.cfa.derivatives` | §6.7 | Derivatives |
| :mod:`~src.cfa.alternatives` | §6.8 | Alternative Investments |

``Decimal`` at every public boundary. Matrix algebra, regression, and
root-finding run in float64 and convert back through :mod:`src.cfa._numeric`,
which is the only place the two representations meet.

Import-linter forbids this package from importing :mod:`src.llm`,
:mod:`src.execution`, :mod:`src.data`, or :mod:`src.api`.
"""
