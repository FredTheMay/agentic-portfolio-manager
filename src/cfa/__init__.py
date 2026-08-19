"""CFA Level I quantitative core. Pure functions, zero I/O.

``Decimal`` at every public boundary. Matrix algebra, regression and
root-finding run in float64 and convert back through :mod:`src.cfa._numeric`,
the only place the two representations meet.
"""
