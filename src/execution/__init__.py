"""Execution layer (SPEC §3).

Everything below the boundary: sizing, orders, venues, brokers, fill models.
The only package permitted to name a broker or an order type.

Built at M5 (``SimulatedExecutor``) and M8 (``NaiveExecutor``). ``grpc_client``
stays a stub — the C++ engine is a separate project.
"""
