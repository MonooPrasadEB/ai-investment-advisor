"""Advanced portfolio analysis with real financial data and quantitative methods."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import optimize
import yfinance as yf
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import warnings

# Financial analysis libraries
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("cvxpy not available - advanced optimization disabled")

try:
    import empyrical as emp
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False
    warnings.warn("empyrical not available - some performance metrics disabled")

from ..core.config import Config
from .market_data import MarketDataService, market_data_service

config = Config.get_instance()
logger = logging.getLogger(__name__)


class Asset(BaseModel):
    """Enhanced asset model with comprehensive financial data."""
    symbol: str
    name: str
    quantity: float
    current_price: float
    market_value: float
    allocation: float  # percentage of total portfolio
    sector: Optional[str] = None
    industry: Optional[str] = None
    asset_type: str = "stock"  # stock, bond, etf, mutual_fund, etc.
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None


class Portfolio(BaseModel):
    """Comprehensive portfolio model with enhanced metrics."""
    total_value: float
    assets: List[Asset]
    last_updated: datetime
    
    # Risk metrics
    portfolio_beta: float
    portfolio_volatility: float  # annualized
    value_at_risk_95: float  # 95% VaR
    expected_shortfall_95: float  # Conditional VaR
    
    # Performance metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    
    # Diversification metrics
    effective_number_stocks: float  # Inverse of sum of squared weights
    sector_concentration: float  # Herfindahl index for sectors
    
    # ESG and sustainability metrics (placeholder for future)
    esg_score: Optional[float] = None


class PortfolioOptimization(BaseModel):
    """Portfolio optimization results."""
    method: str  # "mean_variance", "risk_parity", "black_litterman"
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    optimized_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    
    # Trade recommendations
    trades: List[Dict[str, Union[str, float]]]
    total_turnover: float


class RiskDecomposition(BaseModel):
    """Risk decomposition analysis."""
    total_risk: float
    systematic_risk: float  # Market risk
    idiosyncratic_risk: float  # Stock-specific risk
    sector_risks: Dict[str, float]
    factor_exposures: Dict[str, float]  # Size, Value, Momentum, Quality, etc.


class AdvancedPortfolioAnalyzer:
    """Advanced portfolio analyzer with quantitative finance methods."""
    
    def __init__(self):
        self.market_data = market_data_service
        self.risk_free_rate = config.risk_free_rate
        
    async def analyze_portfolio(
        self, 
        assets: List[Dict[str, Union[str, float]]], 
        benchmark: str = "SPY",
        lookback_days: int = 252
    ) -> Dict:
        """Comprehensive portfolio analysis with modern portfolio theory."""
        try:
            # Build portfolio with current market data
            portfolio_assets = []
            symbols = [asset["symbol"] for asset in assets]
            
            # Get bulk market data
            prices = self.market_data.get_multiple_prices(symbols)
            total_value = 0.0
            
            for asset_input in assets:
                symbol = asset_input["symbol"]
                quantity = float(asset_input["quantity"])
                current_price = prices.get(symbol, 0.0)
                market_value = quantity * current_price
                total_value += market_value
                
                # Get additional stock info
                stock_info = self.market_data.get_stock_info(symbol)
                
                asset = Asset(
                    symbol=symbol,
                    name=stock_info.name if stock_info else symbol,
                    quantity=quantity,
                    current_price=current_price,
                    market_value=market_value,
                    allocation=0.0,  # Will calculate below
                    sector=stock_info.sector if stock_info else None,
                    industry=stock_info.industry if stock_info else None,
                    asset_type=asset_input.get("asset_type", "stock"),
                    beta=stock_info.beta if stock_info else None,
                    dividend_yield=stock_info.dividend_yield if stock_info else None,
                    pe_ratio=stock_info.pe_ratio if stock_info else None,
                    market_cap=stock_info.market_cap if stock_info else None,
                )
                portfolio_assets.append(asset)
            
            # Calculate allocations
            for asset in portfolio_assets:
                asset.allocation = (asset.market_value / total_value) * 100
            
            # Get historical data for risk analysis
            historical_data = await self._get_historical_returns(symbols + [benchmark], lookback_days)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                portfolio_assets, historical_data, benchmark
            )
            
            # Create comprehensive portfolio object
            portfolio = Portfolio(
                total_value=total_value,
                assets=portfolio_assets,
                last_updated=datetime.now(),
                **portfolio_metrics
            )
            
            # Perform advanced analysis
            risk_analysis = self._analyze_risk_decomposition(portfolio, historical_data)
            diversification_analysis = self._analyze_diversification(portfolio)
            performance_attribution = self._analyze_performance_attribution(
                portfolio, historical_data, benchmark
            )
            
            # Generate optimization recommendations
            optimization_results = await self._optimize_portfolio(
                portfolio, historical_data, method="mean_variance"
            )
            
            return {
                "portfolio": portfolio.dict(),
                "risk_analysis": risk_analysis,
                "diversification": diversification_analysis,
                "performance_attribution": performance_attribution,
                "optimization": optimization_results,
                "recommendations": self._generate_recommendations(portfolio, optimization_results),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}")
            return {"error": str(e)}
    
    async def _get_historical_returns(
        self, symbols: List[str], lookback_days: int
    ) -> pd.DataFrame:
        """Get historical returns for portfolio analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 50)  # Extra buffer
        
        try:
            # Download data for all symbols
            data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=True)
            
            if len(symbols) == 1:
                returns = data['Close'].pct_change().dropna()
                returns.name = symbols[0]
                returns_df = returns.to_frame()
            else:
                close_prices = data['Close']
                returns_df = close_prices.pct_change().dropna()
            
            # Ensure we have the requested lookback period
            returns_df = returns_df.tail(lookback_days)
            
            return returns_df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=symbols)
    
    def _calculate_portfolio_metrics(
        self, assets: List[Asset], returns_data: pd.DataFrame, benchmark: str
    ) -> Dict:
        """Calculate comprehensive portfolio risk and performance metrics."""
        weights = np.array([asset.allocation / 100 for asset in assets])
        asset_symbols = [asset.symbol for asset in assets]
        
        # Get returns for portfolio assets only
        asset_returns = returns_data[asset_symbols] if len(asset_symbols) > 1 else returns_data[[asset_symbols[0]]]
        
        if asset_returns.empty:
            logger.warning("No historical data available for portfolio metrics")
            return self._get_default_metrics()
        
        # Calculate portfolio returns
        portfolio_returns = (asset_returns * weights).sum(axis=1)
        
        # Risk metrics
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Portfolio beta (vs benchmark)
        if benchmark in returns_data.columns:
            benchmark_returns = returns_data[benchmark]
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = benchmark_returns.var()
            portfolio_beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        else:
            portfolio_beta = 1.0
        
        # VaR and Expected Shortfall
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)  # Annualized
        es_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        
        # Performance metrics
        mean_return = portfolio_returns.mean() * 252  # Annualized
        sharpe_ratio = (mean_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Diversification metrics
        effective_n_stocks = 1 / np.sum(weights ** 2)  # Inverse of sum of squared weights
        
        # Sector concentration (Herfindahl index)
        sector_weights = {}
        for asset in assets:
            sector = asset.sector or "Unknown"
            sector_weights[sector] = sector_weights.get(sector, 0) + asset.allocation / 100
        
        sector_concentration = sum(weight ** 2 for weight in sector_weights.values())
        
        return {
            "portfolio_beta": float(portfolio_beta),
            "portfolio_volatility": float(portfolio_volatility),
            "value_at_risk_95": float(var_95),
            "expected_shortfall_95": float(es_95),
            "sharpe_ratio": float(sharpe_ratio) if not np.isnan(sharpe_ratio) else None,
            "sortino_ratio": float(sortino_ratio) if not np.isnan(sortino_ratio) else None,
            "max_drawdown": float(max_drawdown),
            "effective_number_stocks": float(effective_n_stocks),
            "sector_concentration": float(sector_concentration)
        }
    
    def _get_default_metrics(self) -> Dict:
        """Return default metrics when data is unavailable."""
        return {
            "portfolio_beta": 1.0,
            "portfolio_volatility": 0.15,  # 15% default volatility
            "value_at_risk_95": -0.05,
            "expected_shortfall_95": -0.08,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": 0.0,
            "effective_number_stocks": 1.0,
            "sector_concentration": 1.0
        }
    
    def _analyze_risk_decomposition(
        self, portfolio: Portfolio, returns_data: pd.DataFrame
    ) -> Dict:
        """Decompose portfolio risk into systematic and idiosyncratic components."""
        # Simplified risk decomposition
        systematic_risk_ratio = portfolio.portfolio_beta ** 2 * 0.16  # Assume market vol = 16%
        total_risk = portfolio.portfolio_volatility ** 2
        systematic_risk = systematic_risk_ratio * total_risk
        idiosyncratic_risk = total_risk - systematic_risk
        
        # Sector risk contribution
        sector_risks = {}
        for asset in portfolio.assets:
            sector = asset.sector or "Unknown"
            asset_weight = asset.allocation / 100
            asset_vol = 0.25  # Default assumption
            
            sector_risks[sector] = sector_risks.get(sector, 0) + (asset_weight * asset_vol) ** 2
        
        return {
            "total_risk": float(total_risk),
            "systematic_risk": float(systematic_risk),
            "idiosyncratic_risk": float(idiosyncratic_risk),
            "sector_risks": {k: float(v) for k, v in sector_risks.items()},
            "risk_decomposition_method": "simplified_beta_based"
        }
    
    def _analyze_diversification(self, portfolio: Portfolio) -> Dict:
        """Analyze portfolio diversification across multiple dimensions."""
        # Asset type diversification
        asset_type_allocation = {}
        for asset in portfolio.assets:
            asset_type = asset.asset_type
            asset_type_allocation[asset_type] = asset_type_allocation.get(asset_type, 0) + asset.allocation
        
        # Sector diversification
        sector_allocation = {}
        for asset in portfolio.assets:
            sector = asset.sector or "Unknown"
            sector_allocation[sector] = sector_allocation.get(sector, 0) + asset.allocation
        
        # Market cap diversification
        mcap_allocation = {"Large": 0, "Mid": 0, "Small": 0, "Unknown": 0}
        for asset in portfolio.assets:
            if asset.market_cap:
                if asset.market_cap > 10e9:  # > $10B
                    mcap_allocation["Large"] += asset.allocation
                elif asset.market_cap > 2e9:  # > $2B
                    mcap_allocation["Mid"] += asset.allocation
                else:
                    mcap_allocation["Small"] += asset.allocation
            else:
                mcap_allocation["Unknown"] += asset.allocation
        
        # Diversification scores
        sector_hhi = sum((v/100)**2 for v in sector_allocation.values())
        asset_type_hhi = sum((v/100)**2 for v in asset_type_allocation.values())
        
        return {
            "sector_allocation": sector_allocation,
            "asset_type_allocation": asset_type_allocation,
            "market_cap_allocation": mcap_allocation,
            "sector_concentration_index": float(sector_hhi),
            "asset_type_concentration_index": float(asset_type_hhi),
            "effective_diversification_ratio": float(1 / sector_hhi),
            "recommendations": self._generate_diversification_recommendations(
                sector_allocation, asset_type_allocation
            )
        }
    
    def _analyze_performance_attribution(
        self, portfolio: Portfolio, returns_data: pd.DataFrame, benchmark: str
    ) -> Dict:
        """Analyze what drives portfolio performance vs benchmark."""
        # This is a simplified performance attribution
        # In practice, you'd use factor models (Fama-French, etc.)
        
        asset_contributions = []
        for asset in portfolio.assets:
            weight = asset.allocation / 100
            # Simplified: assume asset return = beta * market return + alpha
            expected_return = (asset.beta or 1.0) * 0.10 + 0.02  # Simplified calculation
            contribution = weight * expected_return
            
            asset_contributions.append({
                "symbol": asset.symbol,
                "weight": weight,
                "expected_return": expected_return,
                "contribution": contribution,
                "sector": asset.sector
            })
        
        return {
            "asset_contributions": asset_contributions,
            "total_expected_return": sum(item["contribution"] for item in asset_contributions),
            "benchmark_return": 0.10,  # Assume 10% benchmark return
            "active_return": sum(item["contribution"] for item in asset_contributions) - 0.10,
            "attribution_method": "simplified_beta_based"
        }
    
    async def _optimize_portfolio(
        self, portfolio: Portfolio, returns_data: pd.DataFrame, method: str = "mean_variance"
    ) -> Dict:
        """Optimize portfolio using modern portfolio theory."""
        if not CVXPY_AVAILABLE:
            logger.warning("CVXPY not available - using simplified optimization")
            return self._simplified_optimization(portfolio)
        
        try:
            # Prepare data
            asset_symbols = [asset.symbol for asset in portfolio.assets]
            asset_returns = returns_data[asset_symbols] if len(asset_symbols) > 1 else returns_data[[asset_symbols[0]]]
            
            if asset_returns.empty or len(asset_returns) < 20:
                logger.warning("Insufficient data for optimization")
                return self._simplified_optimization(portfolio)
            
            # Calculate expected returns and covariance matrix
            expected_returns = asset_returns.mean() * 252  # Annualized
            cov_matrix = asset_returns.cov() * 252  # Annualized
            
            n_assets = len(asset_symbols)
            weights = cp.Variable(n_assets)
            
            # Objective: maximize Sharpe ratio (equivalent to min variance for given return)
            portfolio_return = expected_returns.values @ weights
            portfolio_variance = cp.quad_form(weights, cov_matrix.values)
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,  # Fully invested
                weights >= 0,  # Long-only
                weights <= config.max_position_size  # Position size limit
            ]
            
            # Solve optimization
            if method == "mean_variance":
                # Target return = current portfolio return + 2%
                current_weights = np.array([asset.allocation / 100 for asset in portfolio.assets])
                current_return = expected_returns.values @ current_weights
                target_return = current_return + 0.02
                
                constraints.append(portfolio_return >= target_return)
                objective = cp.Minimize(portfolio_variance)
            else:
                # Default: minimum variance
                objective = cp.Minimize(portfolio_variance)
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                optimized_return = expected_returns.values @ optimal_weights
                optimized_variance = optimal_weights @ cov_matrix.values @ optimal_weights
                optimized_volatility = np.sqrt(optimized_variance)
                sharpe_ratio = (optimized_return - self.risk_free_rate) / optimized_volatility
                
                # Generate trade recommendations
                current_weights = np.array([asset.allocation / 100 for asset in portfolio.assets])
                weight_changes = optimal_weights - current_weights
                
                trades = []
                total_turnover = 0
                
                for i, asset in enumerate(portfolio.assets):
                    weight_change = weight_changes[i]
                    if abs(weight_change) > 0.01:  # Only recommend trades > 1%
                        dollar_change = weight_change * portfolio.total_value
                        shares_change = dollar_change / asset.current_price
                        
                        trades.append({
                            "symbol": asset.symbol,
                            "action": "buy" if weight_change > 0 else "sell",
                            "current_weight": asset.allocation / 100,
                            "target_weight": optimal_weights[i],
                            "weight_change": weight_change,
                            "dollar_amount": abs(dollar_change),
                            "shares": abs(shares_change),
                            "reason": f"Optimize to {optimal_weights[i]:.1%} allocation"
                        })
                        total_turnover += abs(weight_change)
                
                return {
                    "method": method,
                    "status": "success",
                    "optimized_weights": {asset.symbol: float(w) for asset, w in zip(portfolio.assets, optimal_weights)},
                    "expected_return": float(optimized_return),
                    "expected_volatility": float(optimized_volatility),
                    "sharpe_ratio": float(sharpe_ratio),
                    "trades": trades,
                    "total_turnover": float(total_turnover),
                    "improvement_metrics": {
                        "return_improvement": float(optimized_return - (expected_returns.values @ current_weights)),
                        "volatility_change": float(optimized_volatility - portfolio.portfolio_volatility),
                        "sharpe_improvement": float(sharpe_ratio - (portfolio.sharpe_ratio or 0))
                    }
                }
            else:
                logger.error(f"Optimization failed with status: {problem.status}")
                return self._simplified_optimization(portfolio)
                
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return self._simplified_optimization(portfolio)
    
    def _simplified_optimization(self, portfolio: Portfolio) -> Dict:
        """Simplified optimization when advanced methods fail."""
        # Simple rebalancing rules
        trades = []
        
        for asset in portfolio.assets:
            if asset.allocation > config.max_position_size * 100:
                # Reduce overweight positions
                target_weight = config.max_position_size * 0.8  # 80% of max
                weight_change = (target_weight - asset.allocation / 100)
                dollar_change = weight_change * portfolio.total_value
                
                trades.append({
                    "symbol": asset.symbol,
                    "action": "sell",
                    "current_weight": asset.allocation / 100,
                    "target_weight": target_weight,
                    "weight_change": weight_change,
                    "dollar_amount": abs(dollar_change),
                    "shares": abs(dollar_change / asset.current_price),
                    "reason": f"Reduce concentration risk (currently {asset.allocation:.1f}%)"
                })
        
        return {
            "method": "simplified_rebalancing",
            "status": "fallback",
            "trades": trades,
            "total_turnover": sum(abs(trade["weight_change"]) for trade in trades),
            "note": "Advanced optimization unavailable - using simple rebalancing rules"
        }
    
    def _generate_diversification_recommendations(
        self, sector_allocation: Dict, asset_type_allocation: Dict
    ) -> List[str]:
        """Generate recommendations to improve diversification."""
        recommendations = []
        
        # Check sector concentration
        for sector, allocation in sector_allocation.items():
            if allocation > config.max_sector_allocation * 100:
                recommendations.append(
                    f"Reduce {sector} exposure (currently {allocation:.1f}%, "
                    f"recommended max {config.max_sector_allocation*100:.0f}%)"
                )
        
        # Check asset type diversity
        if asset_type_allocation.get("stock", 0) > 80:
            recommendations.append("Consider adding bonds or other asset classes for stability")
        
        if len(sector_allocation) < 4:
            recommendations.append("Diversify across more sectors to reduce concentration risk")
        
        return recommendations
    
    def _generate_recommendations(self, portfolio: Portfolio, optimization: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Risk-based recommendations
        if portfolio.portfolio_volatility > 0.25:
            recommendations.append("Portfolio volatility is high - consider reducing risk through diversification")
        
        if portfolio.sharpe_ratio and portfolio.sharpe_ratio < 0.5:
            recommendations.append("Poor risk-adjusted returns - review asset selection and allocation")
        
        if portfolio.max_drawdown < -0.3:
            recommendations.append("Maximum drawdown exceeds 30% - consider defensive assets")
        
        # Diversification recommendations
        if portfolio.effective_number_stocks < 5:
            recommendations.append("Portfolio lacks diversification - consider adding more positions")
        
        if portfolio.sector_concentration > 0.4:
            recommendations.append("High sector concentration risk - diversify across industries")
        
        # Optimization-based recommendations
        if optimization.get("total_turnover", 0) > 0.1:
            recommendations.append("Consider rebalancing to improve risk-adjusted returns")
        
        return recommendations


# Tool instances
portfolio_analyzer = AdvancedPortfolioAnalyzer()


class PortfolioAnalysisInput(BaseModel):
    """Input model for portfolio analysis tool."""
    assets: List[Dict[str, Union[str, float]]] = Field(
        ..., 
        description="List of assets with symbol, quantity, and optional asset_type"
    )
    benchmark: str = Field("SPY", description="Benchmark symbol for comparison")
    optimize: bool = Field(True, description="Whether to include optimization recommendations")


class PortfolioAnalysisTool(BaseTool):
    """Advanced portfolio analysis tool with quantitative methods."""
    name: str = "analyze_portfolio_advanced"
    description: str = """
    Perform comprehensive portfolio analysis including:
    - Risk decomposition and performance attribution
    - Modern portfolio theory optimization
    - Diversification analysis across sectors and asset types
    - Value-at-Risk and expected shortfall calculations
    - Sharpe ratio and other performance metrics
    - Actionable rebalancing recommendations
    """
    args_schema: type = PortfolioAnalysisInput

    def _run(
        self, 
        assets: List[Dict[str, Union[str, float]]], 
        benchmark: str = "SPY",
        optimize: bool = True
    ) -> str:
        """Execute comprehensive portfolio analysis."""
        try:
            import asyncio
            
            # Run the async analysis
            if hasattr(asyncio, 'run'):
                result = asyncio.run(portfolio_analyzer.analyze_portfolio(assets, benchmark))
            else:
                # Fallback for older Python versions
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(portfolio_analyzer.analyze_portfolio(assets, benchmark))
            
            if "error" in result:
                return f"Portfolio analysis failed: {result['error']}"
            
            # Format output for the agent
            portfolio = result["portfolio"]
            summary = f"""
Portfolio Analysis Summary:
=========================
Total Value: ${portfolio['total_value']:,.2f}
Risk Level: {portfolio['portfolio_volatility']*100:.1f}% volatility
Sharpe Ratio: {portfolio['sharpe_ratio']:.2f if portfolio['sharpe_ratio'] else 'N/A'}
Max Drawdown: {portfolio['max_drawdown']*100:.1f}%
Diversification Score: {1/portfolio['sector_concentration']:.1f}/10

Asset Allocation:
"""
            for asset in portfolio["assets"]:
                summary += f"- {asset['symbol']}: {asset['allocation']:.1f}% (${asset['market_value']:,.2f})\n"
            
            summary += f"\nKey Recommendations:\n"
            for rec in result["recommendations"]:
                summary += f"• {rec}\n"
            
            if optimize and result.get("optimization", {}).get("trades"):
                summary += f"\nOptimization Trades:\n"
                for trade in result["optimization"]["trades"][:5]:  # Top 5 trades
                    summary += f"• {trade['action'].upper()} {trade['symbol']}: {trade['reason']}\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis tool: {e}")
            return f"Portfolio analysis error: {str(e)}"


# Tool instance
portfolio_analysis_tool = PortfolioAnalysisTool()
