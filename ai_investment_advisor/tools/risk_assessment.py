"""Advanced risk assessment with quantitative models and scenario analysis."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import warnings

# Financial risk libraries
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch package not available - GARCH models disabled")

try:
    import empyrical as emp
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False
    warnings.warn("empyrical not available - some risk metrics disabled")

from ..core.config import Config
from .market_data import market_data_service

config = Config.get_instance()
logger = logging.getLogger(__name__)


class RiskProfile(BaseModel):
    """Comprehensive risk profile model."""
    user_id: str
    assessment_date: datetime
    
    # Demographics
    age: int
    annual_income: float
    net_worth: float
    investment_experience: str  # beginner, intermediate, advanced, expert
    
    # Risk preferences (1-10 scales)
    risk_comfort: int = Field(..., ge=1, le=10)
    volatility_tolerance: int = Field(..., ge=1, le=10)
    liquidity_needs: int = Field(..., ge=1, le=10)  # 1=low, 10=high
    
    # Investment objectives
    time_horizon: int  # years
    primary_goal: str  # retirement, wealth_building, income, preservation
    secondary_goals: List[str] = []
    
    # Calculated risk metrics
    risk_score: float = Field(..., ge=1, le=10)
    risk_tolerance: str  # conservative, moderate, aggressive
    recommended_equity_allocation: float = Field(..., ge=0, le=1)
    
    # Behavioral factors
    loss_aversion_score: float = 1.0  # 1=normal, >1=more loss averse
    overconfidence_score: float = 1.0  # 1=normal, >1=overconfident
    home_bias_score: float = 1.0  # 1=no bias, >1=prefers domestic


class RiskScenario(BaseModel):
    """Risk scenario for stress testing."""
    scenario_name: str
    description: str
    probability: float = Field(..., ge=0, le=1)
    time_horizon: str  # "1 month", "6 months", "1 year", etc.
    
    # Market impacts
    equity_return: float  # Expected return for scenario
    bond_return: float
    commodity_return: float = 0.0
    fx_impact: float = 0.0  # USD impact
    
    # Volatility changes
    volatility_multiplier: float = 1.0  # 1=no change, >1=higher vol
    
    # Economic indicators
    interest_rate_change: float = 0.0  # basis points
    inflation_change: float = 0.0  # percentage points
    gdp_impact: float = 0.0  # percentage change


class VaRResult(BaseModel):
    """Value at Risk calculation results."""
    method: str  # "historical", "parametric", "monte_carlo"
    confidence_level: float
    time_horizon_days: int
    
    var_absolute: float  # Dollar amount at risk
    var_percentage: float  # Percentage of portfolio at risk
    expected_shortfall: float  # Expected loss beyond VaR
    
    # Additional metrics
    worst_case_scenario: float
    best_case_scenario: float
    probability_of_loss: float


class StressTesting(BaseModel):
    """Stress testing results."""
    base_portfolio_value: float
    scenarios: Dict[str, Dict[str, float]]  # scenario_name -> {value, return, loss}
    summary_statistics: Dict[str, float]
    
    # Risk decomposition
    systematic_risk_contribution: float
    idiosyncratic_risk_contribution: float
    
    # Recommendations
    risk_mitigation_suggestions: List[str]


class AdvancedRiskAssessmentService:
    """Advanced risk assessment with quantitative models."""
    
    def __init__(self):
        self.market_data = market_data_service
        self.risk_free_rate = config.risk_free_rate
        
        # Standard risk scenarios
        self.standard_scenarios = self._create_standard_scenarios()
    
    def assess_user_risk_profile(
        self,
        age: int,
        annual_income: float,
        net_worth: float,
        investment_experience: str,
        risk_comfort: int,
        volatility_tolerance: int,
        liquidity_needs: int,
        time_horizon: int,
        primary_goal: str,
        scenario_responses: Optional[List[Dict]] = None
    ) -> RiskProfile:
        """Assess comprehensive user risk profile."""
        
        # Calculate base risk score from demographics
        risk_score = self._calculate_base_risk_score(
            age, annual_income, net_worth, investment_experience, 
            risk_comfort, volatility_tolerance, time_horizon
        )
        
        # Adjust based on scenario responses
        if scenario_responses:
            behavioral_adjustment = self._analyze_scenario_responses(scenario_responses)
            risk_score = max(1, min(10, risk_score + behavioral_adjustment))
        
        # Determine risk tolerance category
        if risk_score <= 3.5:
            risk_tolerance = "conservative"
            equity_allocation = 0.3 + (age < 40) * 0.1  # 30-40%
        elif risk_score <= 7:
            risk_tolerance = "moderate" 
            equity_allocation = 0.6 + (age < 35) * 0.1  # 60-70%
        else:
            risk_tolerance = "aggressive"
            equity_allocation = 0.8 + (age < 30) * 0.1  # 80-90%
        
        # Adjust for liquidity needs
        if liquidity_needs > 7:
            equity_allocation = max(0.2, equity_allocation - 0.2)
        
        # Calculate behavioral factors
        loss_aversion = self._calculate_loss_aversion(risk_comfort, scenario_responses or [])
        overconfidence = self._calculate_overconfidence(investment_experience, risk_comfort)
        
        return RiskProfile(
            user_id=f"user_{datetime.now().timestamp()}",
            assessment_date=datetime.now(),
            age=age,
            annual_income=annual_income,
            net_worth=net_worth,
            investment_experience=investment_experience,
            risk_comfort=risk_comfort,
            volatility_tolerance=volatility_tolerance,
            liquidity_needs=liquidity_needs,
            time_horizon=time_horizon,
            primary_goal=primary_goal,
            risk_score=risk_score,
            risk_tolerance=risk_tolerance,
            recommended_equity_allocation=equity_allocation,
            loss_aversion_score=loss_aversion,
            overconfidence_score=overconfidence,
            home_bias_score=1.0  # Default, would be assessed through additional questions
        )
    
    def calculate_portfolio_var(
        self,
        portfolio_returns: pd.Series,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
        method: str = "historical"
    ) -> VaRResult:
        """Calculate Value at Risk using multiple methods."""
        
        if portfolio_returns.empty:
            logger.warning("No return data provided for VaR calculation")
            return self._get_default_var_result(confidence_level, time_horizon_days)
        
        # Scale returns for time horizon
        scaled_returns = portfolio_returns * np.sqrt(time_horizon_days)
        
        if method == "historical":
            var_percentage = np.percentile(scaled_returns, (1 - confidence_level) * 100)
            expected_shortfall = scaled_returns[scaled_returns <= var_percentage].mean()
        
        elif method == "parametric":
            # Assume normal distribution
            mean_return = scaled_returns.mean()
            std_return = scaled_returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var_percentage = mean_return + z_score * std_return
            expected_shortfall = mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence_level)
        
        elif method == "monte_carlo":
            # Monte Carlo simulation
            np.random.seed(42)  # For reproducibility
            n_simulations = 10000
            
            mean_return = scaled_returns.mean()
            std_return = scaled_returns.std()
            
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            var_percentage = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            expected_shortfall = simulated_returns[simulated_returns <= var_percentage].mean()
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Calculate additional metrics
        worst_case = scaled_returns.min()
        best_case = scaled_returns.max()
        prob_loss = (scaled_returns < 0).mean()
        
        return VaRResult(
            method=method,
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            var_absolute=0.0,  # Will be calculated when portfolio value is known
            var_percentage=float(var_percentage),
            expected_shortfall=float(expected_shortfall),
            worst_case_scenario=float(worst_case),
            best_case_scenario=float(best_case),
            probability_of_loss=float(prob_loss)
        )
    
    def perform_stress_testing(
        self,
        portfolio_assets: List[Dict],
        portfolio_value: float,
        custom_scenarios: Optional[List[RiskScenario]] = None
    ) -> StressTesting:
        """Perform comprehensive stress testing."""
        
        scenarios_to_test = custom_scenarios or self.standard_scenarios
        results = {}
        
        for scenario in scenarios_to_test:
            scenario_result = self._apply_scenario_to_portfolio(
                portfolio_assets, portfolio_value, scenario
            )
            results[scenario.scenario_name] = scenario_result
        
        # Calculate summary statistics
        scenario_values = [result["value"] for result in results.values()]
        scenario_returns = [result["return"] for result in results.values()]
        
        summary_stats = {
            "worst_case_value": min(scenario_values),
            "best_case_value": max(scenario_values),
            "worst_case_return": min(scenario_returns),
            "best_case_return": max(scenario_returns),
            "average_scenario_return": np.mean(scenario_returns),
            "scenario_volatility": np.std(scenario_returns),
            "scenarios_with_loss": sum(1 for r in scenario_returns if r < 0),
            "max_drawdown": min(scenario_returns)
        }
        
        # Risk decomposition (simplified)
        systematic_risk = 0.7  # Assume 70% systematic risk
        idiosyncratic_risk = 0.3  # 30% idiosyncratic risk
        
        # Generate risk mitigation suggestions
        suggestions = self._generate_risk_mitigation_suggestions(results, portfolio_assets)
        
        return StressTesting(
            base_portfolio_value=portfolio_value,
            scenarios=results,
            summary_statistics=summary_stats,
            systematic_risk_contribution=systematic_risk,
            idiosyncratic_risk_contribution=idiosyncratic_risk,
            risk_mitigation_suggestions=suggestions
        )
    
    def _calculate_base_risk_score(
        self, age: int, income: float, net_worth: float, 
        experience: str, comfort: int, volatility_tolerance: int, time_horizon: int
    ) -> float:
        """Calculate base risk score from demographic and preference data."""
        
        score = 5.0  # Start with moderate risk
        
        # Age factor (Rule of thumb: 100 - age = % in stocks)
        if age < 30:
            score += 1.5
        elif age < 45:
            score += 0.5
        elif age > 60:
            score -= 1.0
        elif age > 70:
            score -= 2.0
        
        # Income and net worth factor
        income_to_networth_ratio = income / (net_worth + 1)  # Avoid division by zero
        if net_worth > income * 10:  # High net worth relative to income
            score += 0.5
        if income > 150000:  # High income
            score += 0.5
        if income_to_networth_ratio > 0.5:  # Income dependent
            score -= 0.5
        
        # Experience factor
        experience_scores = {
            "beginner": -1.0,
            "intermediate": 0.0,
            "advanced": 0.5,
            "expert": 1.0
        }
        score += experience_scores.get(experience, 0.0)
        
        # Comfort and volatility tolerance (scale 1-10 to -2 to +2)
        score += (comfort - 5.5) / 2.5
        score += (volatility_tolerance - 5.5) / 2.5
        
        # Time horizon factor
        if time_horizon > 20:
            score += 1.0
        elif time_horizon > 10:
            score += 0.5
        elif time_horizon < 5:
            score -= 1.0
        
        return max(1.0, min(10.0, score))
    
    def _analyze_scenario_responses(self, responses: List[Dict]) -> float:
        """Analyze behavioral responses to scenarios."""
        adjustment = 0.0
        
        for response in responses:
            action = response.get("action", "hold").lower()
            scenario_severity = response.get("severity", 0.5)  # 0-1 scale
            
            if action == "buy_more":
                adjustment += 0.5 + scenario_severity  # More aggressive with worse scenarios
            elif action == "hold":
                adjustment += 0.1
            elif action == "sell":
                adjustment -= 0.5 + scenario_severity  # More conservative with worse scenarios
        
        return adjustment / len(responses) if responses else 0.0
    
    def _calculate_loss_aversion(self, risk_comfort: int, scenario_responses: List[Dict]) -> float:
        """Calculate loss aversion coefficient."""
        base_aversion = 2.0  # Standard loss aversion coefficient
        
        # Adjust based on risk comfort
        comfort_adjustment = (6 - risk_comfort) * 0.2  # Lower comfort = higher aversion
        
        # Adjust based on scenario responses
        sell_responses = sum(1 for r in scenario_responses if r.get("action") == "sell")
        total_responses = len(scenario_responses)
        
        if total_responses > 0:
            sell_ratio = sell_responses / total_responses
            scenario_adjustment = sell_ratio * 0.5  # More selling = higher aversion
        else:
            scenario_adjustment = 0.0
        
        return base_aversion + comfort_adjustment + scenario_adjustment
    
    def _calculate_overconfidence(self, experience: str, risk_comfort: int) -> float:
        """Calculate overconfidence score."""
        base_confidence = 1.0
        
        # Experience can lead to overconfidence
        experience_multiplier = {
            "beginner": 0.8,
            "intermediate": 1.0,
            "advanced": 1.2,
            "expert": 1.3
        }
        
        confidence = base_confidence * experience_multiplier.get(experience, 1.0)
        
        # High risk comfort can indicate overconfidence
        if risk_comfort > 8:
            confidence += 0.2
        
        return confidence
    
    def _create_standard_scenarios(self) -> List[RiskScenario]:
        """Create standard stress test scenarios."""
        return [
            RiskScenario(
                scenario_name="Market Correction",
                description="20% market decline over 3 months",
                probability=0.1,
                time_horizon="3 months",
                equity_return=-0.20,
                bond_return=0.02,
                volatility_multiplier=1.5
            ),
            RiskScenario(
                scenario_name="Tech Crash",
                description="Technology sector crash similar to 2000",
                probability=0.05,
                time_horizon="1 year",
                equity_return=-0.35,
                bond_return=0.05,
                volatility_multiplier=2.0
            ),
            RiskScenario(
                scenario_name="Interest Rate Shock",
                description="Rapid rate increase of 300 basis points",
                probability=0.15,
                time_horizon="6 months",
                equity_return=-0.15,
                bond_return=-0.10,
                interest_rate_change=300,
                volatility_multiplier=1.3
            ),
            RiskScenario(
                scenario_name="Inflation Surge",
                description="Inflation rises to 8% annually",
                probability=0.2,
                time_horizon="1 year",
                equity_return=-0.05,
                bond_return=-0.08,
                commodity_return=0.25,
                inflation_change=0.05
            ),
            RiskScenario(
                scenario_name="Recession",
                description="Economic recession with GDP decline",
                probability=0.3,
                time_horizon="18 months",
                equity_return=-0.25,
                bond_return=0.08,
                gdp_impact=-0.03,
                volatility_multiplier=1.6
            )
        ]
    
    def _apply_scenario_to_portfolio(
        self, assets: List[Dict], portfolio_value: float, scenario: RiskScenario
    ) -> Dict[str, float]:
        """Apply stress scenario to portfolio."""
        
        new_value = 0.0
        
        for asset in assets:
            asset_type = asset.get("asset_type", "stock").lower()
            weight = asset.get("allocation", 0) / 100
            asset_value = weight * portfolio_value
            
            # Apply scenario returns based on asset type
            if asset_type in ["stock", "equity", "etf"]:
                scenario_return = scenario.equity_return
            elif asset_type in ["bond", "fixed_income"]:
                scenario_return = scenario.bond_return
            elif asset_type in ["commodity", "gold", "real_estate"]:
                scenario_return = scenario.commodity_return
            else:
                scenario_return = scenario.equity_return * 0.5  # Conservative assumption
            
            new_asset_value = asset_value * (1 + scenario_return)
            new_value += new_asset_value
        
        portfolio_return = (new_value / portfolio_value) - 1
        portfolio_loss = portfolio_value - new_value
        
        return {
            "value": new_value,
            "return": portfolio_return,
            "loss": portfolio_loss,
            "scenario_probability": scenario.probability
        }
    
    def _generate_risk_mitigation_suggestions(
        self, scenario_results: Dict, portfolio_assets: List[Dict]
    ) -> List[str]:
        """Generate suggestions to mitigate identified risks."""
        suggestions = []
        
        # Analyze worst-case scenarios
        worst_scenarios = sorted(
            scenario_results.items(),
            key=lambda x: x[1]["return"]
        )[:2]  # Two worst scenarios
        
        for scenario_name, result in worst_scenarios:
            if result["return"] < -0.2:  # More than 20% loss
                if "tech" in scenario_name.lower():
                    suggestions.append("Reduce technology sector concentration")
                elif "rate" in scenario_name.lower():
                    suggestions.append("Consider adding floating-rate bonds or TIPS")
                elif "recession" in scenario_name.lower():
                    suggestions.append("Increase allocation to defensive sectors and cash")
        
        # Check asset concentration
        asset_allocations = {asset["symbol"]: asset.get("allocation", 0) for asset in portfolio_assets}
        max_allocation = max(asset_allocations.values()) if asset_allocations else 0
        
        if max_allocation > config.max_position_size * 100:
            suggestions.append(f"Reduce individual position sizes (max currently {max_allocation:.1f}%)")
        
        # General diversification
        if len(portfolio_assets) < 10:
            suggestions.append("Increase portfolio diversification with more positions")
        
        return suggestions
    
    def _get_default_var_result(self, confidence_level: float, time_horizon_days: int) -> VaRResult:
        """Return default VaR result when calculation fails."""
        return VaRResult(
            method="default",
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            var_absolute=0.0,
            var_percentage=-0.05,  # Assume 5% VaR
            expected_shortfall=-0.08,  # Assume 8% expected shortfall
            worst_case_scenario=-0.20,
            best_case_scenario=0.15,
            probability_of_loss=0.4
        )


# Service instance
risk_service = AdvancedRiskAssessmentService()


# Tools
class RiskProfileInput(BaseModel):
    """Input for risk profile assessment."""
    age: int = Field(..., ge=18, le=100, description="User age")
    annual_income: float = Field(..., ge=0, description="Annual income in USD")
    net_worth: float = Field(..., ge=0, description="Total net worth in USD")
    investment_experience: str = Field(..., description="Investment experience level")
    risk_comfort: int = Field(..., ge=1, le=10, description="Risk comfort level (1-10)")
    volatility_tolerance: int = Field(..., ge=1, le=10, description="Market volatility tolerance (1-10)")
    liquidity_needs: int = Field(..., ge=1, le=10, description="Liquidity needs (1-10)")
    time_horizon: int = Field(..., ge=1, description="Investment time horizon in years")
    primary_goal: str = Field(..., description="Primary investment goal")
    scenario_responses: Optional[List[Dict]] = Field(None, description="Responses to risk scenarios")


class RiskAssessmentTool(BaseTool):
    """Comprehensive risk assessment tool."""
    name: str = "assess_risk_profile"
    description: str = """
    Conduct comprehensive risk profile assessment including:
    - Demographic and financial situation analysis
    - Behavioral risk preferences evaluation  
    - Loss aversion and overconfidence scoring
    - Recommended asset allocation based on risk tolerance
    - Time horizon and liquidity considerations
    """
    args_schema: type = RiskProfileInput

    def _run(
        self,
        age: int,
        annual_income: float, 
        net_worth: float,
        investment_experience: str,
        risk_comfort: int,
        volatility_tolerance: int,
        liquidity_needs: int,
        time_horizon: int,
        primary_goal: str,
        scenario_responses: Optional[List[Dict]] = None
    ) -> str:
        """Execute risk profile assessment."""
        try:
            profile = risk_service.assess_user_risk_profile(
                age=age,
                annual_income=annual_income,
                net_worth=net_worth,
                investment_experience=investment_experience,
                risk_comfort=risk_comfort,
                volatility_tolerance=volatility_tolerance,
                liquidity_needs=liquidity_needs,
                time_horizon=time_horizon,
                primary_goal=primary_goal,
                scenario_responses=scenario_responses
            )
            
            return f"""
Risk Profile Assessment Results:
================================

Risk Score: {profile.risk_score:.1f}/10
Risk Tolerance: {profile.risk_tolerance.title()}
Recommended Equity Allocation: {profile.recommended_equity_allocation:.0%}

Key Metrics:
- Investment Time Horizon: {profile.time_horizon} years
- Primary Goal: {profile.primary_goal.replace('_', ' ').title()}
- Loss Aversion Score: {profile.loss_aversion_score:.1f} (1.0 = normal)
- Overconfidence Score: {profile.overconfidence_score:.1f} (1.0 = normal)

Recommended Asset Allocation:
- Stocks/Equity: {profile.recommended_equity_allocation:.0%}
- Bonds/Fixed Income: {(1-profile.recommended_equity_allocation):.0%}

Risk Assessment Notes:
- Based on age {profile.age}, this allocation balances growth potential with risk management
- {profile.risk_tolerance.title()} investors typically handle market volatility {'well' if profile.risk_tolerance == 'aggressive' else 'moderately' if profile.risk_tolerance == 'moderate' else 'cautiously'}
- Time horizon of {profile.time_horizon} years {'allows for' if profile.time_horizon > 10 else 'requires careful consideration of'} recovery from market downturns
"""
        except Exception as e:
            return f"Risk assessment error: {str(e)}"


class StressTestInput(BaseModel):
    """Input for portfolio stress testing."""
    portfolio_assets: List[Dict] = Field(..., description="Portfolio assets with allocations")
    portfolio_value: float = Field(..., description="Total portfolio value")
    include_custom_scenarios: bool = Field(False, description="Whether to include custom scenarios")


class StressTestTool(BaseTool):
    """Portfolio stress testing tool."""
    name: str = "stress_test_portfolio"  
    description: str = """
    Perform comprehensive stress testing on portfolio including:
    - Market correction scenarios (20% decline, tech crash, etc.)
    - Interest rate shock and inflation surge scenarios
    - Recession and economic downturn analysis
    - Value-at-Risk calculations
    - Risk mitigation recommendations
    """
    args_schema: type = StressTestInput

    def _run(
        self,
        portfolio_assets: List[Dict],
        portfolio_value: float,
        include_custom_scenarios: bool = False
    ) -> str:
        """Execute stress testing."""
        try:
            stress_results = risk_service.perform_stress_testing(
                portfolio_assets, portfolio_value
            )
            
            output = f"""
Portfolio Stress Testing Results:
=================================

Base Portfolio Value: ${portfolio_value:,.2f}

Scenario Analysis:
"""
            
            # Sort scenarios by impact
            sorted_scenarios = sorted(
                stress_results.scenarios.items(),
                key=lambda x: x[1]["return"]
            )
            
            for scenario_name, result in sorted_scenarios:
                return_pct = result["return"] * 100
                loss_amount = result["loss"]
                color = "🟢" if return_pct > 0 else "🟡" if return_pct > -10 else "🔴"
                
                output += f"{color} {scenario_name}:\n"
                output += f"   Portfolio Value: ${result['value']:,.2f}\n"
                output += f"   Return: {return_pct:+.1f}%\n"
                output += f"   Loss Amount: ${loss_amount:,.2f}\n\n"
            
            output += f"""
Summary Statistics:
- Worst Case Loss: {stress_results.summary_statistics['worst_case_return']*100:.1f}%
- Best Case Return: {stress_results.summary_statistics['best_case_return']*100:.1f}%
- Average Scenario Return: {stress_results.summary_statistics['average_scenario_return']*100:.1f}%
- Scenarios with Loss: {stress_results.summary_statistics['scenarios_with_loss']} out of {len(stress_results.scenarios)}

Risk Mitigation Recommendations:
"""
            for suggestion in stress_results.risk_mitigation_suggestions:
                output += f"• {suggestion}\n"
            
            return output
            
        except Exception as e:
            return f"Stress testing error: {str(e)}"


# Tool instances
risk_assessment_tool = RiskAssessmentTool()
stress_test_tool = StressTestTool()


class RiskScenarioGeneratorTool(BaseTool):
    """Tool to generate interactive risk scenarios for user assessment."""
    name: str = "generate_risk_scenarios"
    description: str = """
    Generate interactive risk scenarios for user assessment including:
    - Market correction scenarios with different severities
    - Sector-specific crashes and recoveries  
    - Interest rate and inflation scenarios
    - Economic recession scenarios
    - Custom scenarios based on portfolio composition
    """

    def _run(self, focus_area: str = "general") -> str:
        """Generate risk scenarios for user interaction."""
        scenarios = risk_service.standard_scenarios
        
        output = """
Interactive Risk Assessment Scenarios:
=====================================

Please consider how you would respond to each of the following scenarios:

"""
        
        for i, scenario in enumerate(scenarios, 1):
            output += f"""
Scenario {i}: {scenario.scenario_name}
Description: {scenario.description}
Probability: {scenario.probability:.0%} chance in next 12 months
Expected Impact: {scenario.equity_return*100:+.0f}% on stock portion of portfolio

Question: If this scenario occurred, would you:
A) Sell investments to avoid further losses
B) Hold your current positions and wait for recovery  
C) Buy more investments at lower prices

---
"""
        
        output += """
Instructions: 
1. Consider each scenario carefully
2. Think about your emotional reaction to potential losses
3. Your responses help determine your true risk tolerance
4. There are no "right" answers - only honest ones

Use the risk assessment tool with your responses to get a comprehensive risk profile.
"""
        
        return output


# Additional tool instance
risk_scenario_tool = RiskScenarioGeneratorTool()
