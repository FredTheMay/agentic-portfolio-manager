#!/usr/bin/env python3
"""CDK application entry point."""

from __future__ import annotations

import aws_cdk as cdk

from stack import PortfolioManagerStack

app = cdk.App()
PortfolioManagerStack(
    app,
    "AgenticPortfolioManager",
    description="Educational paper-trading simulation. Not investment advice.",
)
app.synth()
