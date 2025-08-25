"""
AI Investment Advisor - Multi-Agent System for Investment Management

A sophisticated multi-agent AI system that combines cutting-edge artificial intelligence
with quantitative finance to deliver personalized investment advice at scale.

Features:
- Real-time market data integration (yfinance, Alpha Vantage, FRED)
- Advanced portfolio optimization using modern portfolio theory
- Comprehensive risk assessment with scenario analysis
- Regulatory compliance validation (SEC/IRS rules)
- Multi-agent coordination for complex financial decisions
- Synthetic data generation for backtesting and validation

Architecture:
- Multi-Task Agent: Portfolio analysis, risk evaluation, customer engagement
- Execution Agent: Trade execution with compliance checks and user approval
- Compliance Reviewer: Policy validation and client-facing rationale

This system bridges the gap between robo-advisory automation and human-like
personalized financial guidance, making sophisticated investment management
accessible to investors at all levels.
"""

__version__ = "1.0.0"
__author__ = "AI Investment Advisor Team"
__description__ = "Multi-Agent AI Investment Advisor with Real Financial Data"

from .core.config import Config

# Core imports will be available once implemented
try:
    from .core.supervisor import InvestmentAdvisorSupervisor
    __all__ = ["Config", "InvestmentAdvisorSupervisor"]
except ImportError:
    # During development, some modules may not be ready yet
    __all__ = ["Config"]
