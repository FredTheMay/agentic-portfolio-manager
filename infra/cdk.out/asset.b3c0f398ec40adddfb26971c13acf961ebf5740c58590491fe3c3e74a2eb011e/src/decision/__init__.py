"""Decision layer: optimizer and mandate emission.

Decides what to hold and emits target weights. Knows nothing about orders,
venues, fills or brokers; an import-linter contract enforces that.
"""
