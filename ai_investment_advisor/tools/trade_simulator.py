"""
Trade Simulator for Portfolio Rebalancing and Individual Stock Trades
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from .market_data import market_data_service
from .portfolio_analyzer import portfolio_analyzer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..core.config import Config

logger = logging.getLogger(__name__)


class TradeRecommendation(BaseModel):
    """Single trade recommendation model."""
    symbol: str
    action: str  # "buy" or "sell"
    quantity: int
    current_price: float
    estimated_cost: float
    current_allocation: float
    target_allocation: float
    reason: str
    impact_on_portfolio: str


class RebalancingPlan(BaseModel):
    """Complete rebalancing plan."""
    portfolio_value: float
    total_trades: int
    total_cost: float
    total_turnover: float
    trades: List[TradeRecommendation]
    net_cash_impact: float
    projected_allocation: Dict[str, float]
    risk_impact: Dict[str, float]


class TradeSimulator:
    """Simulates trades and generates rebalancing recommendations."""
    
    def __init__(self):
        self.commission_per_trade = 0.65  # Typical online broker commission
        config = Config.get_instance()
        self.llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.1,  # Lower for more consistent trade analysis
            api_key=config.openai_api_key
        )
    
    def generate_rebalancing_plan(
        self,
        portfolio_data: Dict,
        client_profile: Optional[Dict] = None,
        target_allocation: Optional[Dict[str, float]] = None,
        rebalancing_threshold: float = 0.05  # 5% deviation threshold
    ) -> RebalancingPlan:
        """
        Generate specific trade recommendations to rebalance portfolio.
        
        Args:
            portfolio_data: Current portfolio data
            client_profile: User's risk profile and preferences (optional)
            target_allocation: Target allocations by symbol (optional) 
            rebalancing_threshold: Minimum deviation to trigger rebalancing
        """
        try:
            assets = portfolio_data.get('assets', [])
            portfolio_value = portfolio_data.get('total_value', 0)
            
            if not assets:
                raise ValueError("No assets found in portfolio")
            
            # If no target allocation provided, use LLM-powered optimization
            if not target_allocation:
                target_allocation = self._get_llm_optimized_allocation(assets, client_profile, portfolio_data)
            
            trades = []
            total_cost = 0
            total_turnover = 0
            projected_allocation = {}
            
            for asset in assets:
                symbol = asset.get('symbol', '')
                current_allocation = asset.get('allocation', 0) / 100
                current_value = asset.get('market_value', 0)
                current_price = asset.get('current_price', 0)
                current_quantity = asset.get('quantity', 0)
                
                # Get target allocation for this symbol
                target_alloc = target_allocation.get(symbol, current_allocation)
                allocation_diff = target_alloc - current_allocation
                
                # Check if rebalancing is needed
                if abs(allocation_diff) > rebalancing_threshold:
                    # Calculate trade details
                    target_value = target_alloc * portfolio_value
                    value_diff = target_value - current_value
                    shares_diff = int(value_diff / current_price) if current_price > 0 else 0
                    
                    if shares_diff != 0:
                        action = "buy" if shares_diff > 0 else "sell"
                        estimated_cost = abs(shares_diff) * current_price + self.commission_per_trade
                        
                        # Determine impact
                        if abs(allocation_diff) > 0.15:  # >15% change
                            impact = "High - Major portfolio restructuring"
                        elif abs(allocation_diff) > 0.05:  # >5% change
                            impact = "Moderate - Notable allocation shift"
                        else:
                            impact = "Low - Minor adjustment"
                        
                        # Generate reason
                        if allocation_diff > 0:
                            reason = f"Increase allocation from {current_allocation:.1%} to {target_alloc:.1%}"
                        else:
                            reason = f"Reduce allocation from {current_allocation:.1%} to {target_alloc:.1%}"
                        
                        trade = TradeRecommendation(
                            symbol=symbol,
                            action=action,
                            quantity=abs(shares_diff),
                            current_price=current_price,
                            estimated_cost=estimated_cost,
                            current_allocation=current_allocation,
                            target_allocation=target_alloc,
                            reason=reason,
                            impact_on_portfolio=impact
                        )
                        
                        trades.append(trade)
                        total_cost += estimated_cost
                        total_turnover += abs(allocation_diff)
                
                projected_allocation[symbol] = target_alloc
            
            # Calculate net cash impact
            net_cash_impact = sum(
                (trade.estimated_cost - self.commission_per_trade) * (-1 if trade.action == "buy" else 1)
                for trade in trades
            )
            
            # Calculate risk impact using actual sector data
            risk_impact = self._calculate_risk_impact(assets, projected_allocation, total_turnover)
            
            return RebalancingPlan(
                portfolio_value=portfolio_value,
                total_trades=len(trades),
                total_cost=total_cost,
                total_turnover=total_turnover,
                trades=trades,
                net_cash_impact=net_cash_impact,
                projected_allocation=projected_allocation,
                risk_impact=risk_impact
            )
            
        except Exception as e:
            logger.error(f"Error generating rebalancing plan: {e}")
            raise
    
    def simulate_trade(
        self,
        portfolio_data: Dict,
        symbol: str,
        action: str,
        quantity: int
    ) -> Dict:
        """
        Simulate buying or selling a specific stock.
        
        Args:
            portfolio_data: Current portfolio data
            symbol: Stock symbol to trade
            action: "buy" or "sell"
            quantity: Number of shares
        """
        try:
            # Get current price
            stock_info = market_data_service.get_stock_info(symbol)
            if not stock_info:
                return {"error": f"Could not fetch data for {symbol}"}
            
            current_price = stock_info.current_price
            trade_value = quantity * current_price
            commission = self.commission_per_trade
            total_cost = trade_value + commission
            
            # Find current position
            assets = portfolio_data.get('assets', [])
            current_position = None
            for asset in assets:
                if asset.get('symbol') == symbol:
                    current_position = asset
                    break
            
            # Calculate impact
            portfolio_value = portfolio_data.get('total_value', 0)
            current_shares = current_position.get('quantity', 0) if current_position else 0
            current_value = current_position.get('market_value', 0) if current_position else 0
            current_allocation = (current_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            if action.lower() == "buy":
                new_shares = current_shares + quantity
                new_value = current_value + trade_value
                cash_impact = -total_cost
            else:  # sell
                if quantity > current_shares:
                    return {"error": f"Cannot sell {quantity} shares - only own {current_shares}"}
                new_shares = current_shares - quantity
                new_value = current_value - trade_value
                cash_impact = trade_value - commission
            
            new_allocation = (new_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            allocation_change = new_allocation - current_allocation
            
            # Create simulated result
            return {
                "trade_details": {
                    "symbol": symbol,
                    "action": action.upper(),
                    "quantity": quantity,
                    "price": current_price,
                    "trade_value": trade_value,
                    "commission": commission,
                    "total_cost": total_cost,
                    "cash_impact": cash_impact
                },
                "position_impact": {
                    "current_shares": current_shares,
                    "new_shares": new_shares,
                    "current_value": current_value,
                    "new_value": new_value,
                    "current_allocation": current_allocation,
                    "new_allocation": new_allocation,
                    "allocation_change": allocation_change
                },
                "portfolio_impact": {
                    "portfolio_value": portfolio_value,
                    "cash_required": total_cost if action.lower() == "buy" else 0,
                    "cash_generated": trade_value - commission if action.lower() == "sell" else 0,
                    "allocation_impact": f"{allocation_change:+.2f}%"
                },
                "market_data": {
                    "current_price": current_price,
                    "market_cap": stock_info.market_cap,
                    "sector": stock_info.sector,
                    "pe_ratio": stock_info.pe_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"Error simulating trade: {e}")
            return {"error": str(e)}
    
    def _get_llm_optimized_allocation(
        self, 
        assets: List[Dict], 
        client_profile: Optional[Dict],
        portfolio_data: Dict
    ) -> Dict[str, float]:
        """Use LLM to generate optimized allocation based on portfolio analysis and user profile."""
        try:
            # Prepare portfolio context for LLM
            portfolio_summary = []
            total_value = portfolio_data.get('total_value', 0)
            sector_breakdown = {}
            
            for asset in assets:
                symbol = asset.get('symbol', '')
                allocation = asset.get('allocation', 0)
                sector = asset.get('sector', 'Unknown')
                market_value = asset.get('market_value', 0)
                
                portfolio_summary.append(f"- {symbol}: {allocation:.1f}% (${market_value:,.0f}) - {sector}")
                sector_breakdown[sector] = sector_breakdown.get(sector, 0) + allocation
            
            # Prepare client context
            client_context = "No specific risk profile provided"
            if client_profile:
                risk_tolerance = client_profile.get('risk_tolerance', 'moderate')
                age = client_profile.get('age', 'unknown')
                time_horizon = client_profile.get('time_horizon', 'unknown')
                primary_goal = client_profile.get('primary_goal', 'unknown')
                
                client_context = f"""
Risk Tolerance: {risk_tolerance}
Age: {age}
Time Horizon: {time_horizon} years
Primary Goal: {primary_goal}
"""
            
            # Create LLM prompt for rebalancing analysis
            prompt = ChatPromptTemplate.from_template("""
You are an expert portfolio manager analyzing a client's portfolio for rebalancing opportunities.

CURRENT PORTFOLIO (Total: ${total_value:,.0f}):
{portfolio_summary}

SECTOR BREAKDOWN:
{sector_breakdown}

CLIENT PROFILE:
{client_context}

TASK: Analyze this portfolio and recommend optimized allocations. Consider:
1. Current concentration risks and diversification
2. Client's risk tolerance and investment goals
3. Sector/stock-specific risks and opportunities
4. Appropriate position sizing for portfolio size

Provide your allocation recommendations as a JSON object where keys are stock symbols and values are decimal percentages (0.05 = 5%).

IMPORTANT: 
- Only include symbols that are currently in the portfolio
- Allocations must sum to 1.0 (100%)
- Consider the client's risk profile when making recommendations
- Explain your reasoning briefly

Response format:
{{
  "allocations": {{
    "SYMBOL1": 0.15,
    "SYMBOL2": 0.10,
    ...
  }},
  "reasoning": "Brief explanation of the rebalancing strategy"
}}
""")
            
            # Generate LLM response
            chain = prompt | self.llm
            response = chain.invoke({
                "total_value": total_value,
                "portfolio_summary": "\n".join(portfolio_summary),
                "sector_breakdown": "\n".join([f"- {sector}: {alloc:.1f}%" for sector, alloc in sector_breakdown.items()]),
                "client_context": client_context
            })
            
            # Parse LLM response
            try:
                import json
                import re
                
                # Extract JSON from response
                response_text = response.content
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_response = json.loads(json_str)
                    
                    allocations = parsed_response.get("allocations", {})
                    reasoning = parsed_response.get("reasoning", "LLM-generated allocation")
                    
                    logger.info(f"LLM Rebalancing Reasoning: {reasoning}")
                    
                    # Validate allocations
                    total_allocation = sum(allocations.values())
                    if abs(total_allocation - 1.0) > 0.1:  # Allow 10% tolerance
                        logger.warning(f"LLM allocations sum to {total_allocation}, normalizing...")
                        allocations = {k: v / total_allocation for k, v in allocations.items()}
                    
                    return allocations
                
            except Exception as parse_error:
                logger.error(f"Error parsing LLM response: {parse_error}")
            
            # Fallback to simple diversification
            logger.warning("Using fallback allocation due to LLM parsing error")
            return self._get_fallback_allocation(assets, client_profile)
            
        except Exception as e:
            logger.error(f"Error getting LLM optimized allocation: {e}")
            return self._get_fallback_allocation(assets, client_profile)
    
    def _get_fallback_allocation(self, assets: List[Dict], client_profile: Optional[Dict]) -> Dict[str, float]:
        """Fallback allocation strategy when LLM fails."""
        num_assets = len(assets)
        
        # Simple risk-based allocation
        if client_profile and client_profile.get('risk_tolerance') == 'conservative':
            # Conservative: reduce concentration, prefer equal weighting
            equal_weight = 1.0 / num_assets
            max_position = min(0.20, equal_weight * 2)  # Max 20% or 2x equal weight
        elif client_profile and client_profile.get('risk_tolerance') == 'aggressive':
            # Aggressive: allow concentration, but still diversify
            max_position = 0.25  # Max 25% per position
        else:
            # Moderate: balanced approach
            max_position = 0.15  # Max 15% per position
        
        # Create allocation with position limits
        allocations = {}
        for asset in assets:
            symbol = asset.get('symbol', '')
            current_allocation = asset.get('allocation', 0) / 100
            allocations[symbol] = min(current_allocation, max_position)
        
        # Normalize to sum to 1
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: v / total for k, v in allocations.items()}
        
        return allocations
    
    def _calculate_risk_impact(
        self, 
        assets: List[Dict], 
        projected_allocation: Dict[str, float], 
        total_turnover: float
    ) -> Dict[str, float]:
        """Calculate risk impact using actual portfolio data."""
        try:
            # Calculate current sector concentrations
            current_sectors = {}
            projected_sectors = {}
            
            for asset in assets:
                sector = asset.get('sector', 'Unknown')
                symbol = asset.get('symbol', '')
                current_alloc = asset.get('allocation', 0) / 100
                projected_alloc = projected_allocation.get(symbol, 0)
                
                current_sectors[sector] = current_sectors.get(sector, 0) + current_alloc
                projected_sectors[sector] = projected_sectors.get(sector, 0) + projected_alloc
            
            # Calculate concentration changes
            max_current_sector = max(current_sectors.values()) if current_sectors else 0
            max_projected_sector = max(projected_sectors.values()) if projected_sectors else 0
            
            return {
                "concentration_change": max_projected_sector - max_current_sector,
                "sector_diversification": len(projected_sectors),
                "turnover_ratio": total_turnover,
                "largest_sector_allocation": max_projected_sector
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk impact: {e}")
            return {
                "concentration_change": 0.0,
                "sector_diversification": len(set(asset.get('sector', 'Unknown') for asset in assets)),
                "turnover_ratio": total_turnover,
                "largest_sector_allocation": 0.0
            }
    
    def export_rebalancing_plan(
        self,
        rebalancing_plan: RebalancingPlan,
        output_file: str = "rebalancing_plan.json"
    ) -> str:
        """Export rebalancing plan to JSON file."""
        try:
            # Convert to exportable format
            export_data = {
                "generated_at": datetime.now().isoformat(),
                "portfolio_value": rebalancing_plan.portfolio_value,
                "summary": {
                    "total_trades": rebalancing_plan.total_trades,
                    "total_cost": rebalancing_plan.total_cost,
                    "total_turnover": rebalancing_plan.total_turnover,
                    "net_cash_impact": rebalancing_plan.net_cash_impact
                },
                "trades": [trade.dict() for trade in rebalancing_plan.trades],
                "projected_allocation": rebalancing_plan.projected_allocation,
                "risk_impact": rebalancing_plan.risk_impact
            }
            
            # Write to file
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return str(output_path.absolute())
            
        except Exception as e:
            logger.error(f"Error exporting rebalancing plan: {e}")
            return f"Error: {str(e)}"


# Global instance
trade_simulator = TradeSimulator()
