"""Decision layer: optimizer and mandate emission (SPEC §2.2, §3.1).

Decides **what to hold** and emits target weights. Knows nothing about orders,
venues, slicing, fills, or brokers.

Nothing in this package may import ``src.execution``. Enforced by
``tests/test_layer_isolation.py`` and by an import-linter contract in CI.

Built at M4.
"""
