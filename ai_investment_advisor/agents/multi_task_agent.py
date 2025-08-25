"""
Multi-Task Agent for Portfolio Analysis, Risk Evaluation, and Customer Engagement.

This agent serves as the primary interface for investment analysis and client interaction,
combining advanced portfolio analytics with personalized risk assessment and clear
communication of financial concepts.
"""

import logging
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

from ..tools.portfolio_analyzer import portfolio_analysis_tool
from ..tools.risk_assessment import risk_assessment_tool, stress_test_tool, risk_scenario_tool
from ..tools.market_data import market_data_tool
from ..core.config import Config

config = Config.get_instance()
logger = logging.getLogger(__name__)


class MultiTaskAgent:
    """
    Multi-Task Agent responsible for:
    1. Portfolio Analysis - Comprehensive portfolio evaluation and optimization
    2. Risk Evaluation - Risk profiling and scenario analysis  
    3. Customer Engagement - Clear communication and educational guidance
    """
    
    def __init__(self):
        self.name = "portfolio_advisor"
        self.llm = ChatOpenAI(
            model=config.default_model,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )
        
        # Available tools for comprehensive analysis
        self.tools = [
            portfolio_analysis_tool,
            risk_assessment_tool,
            stress_test_tool,
            risk_scenario_tool,
            market_data_tool
        ]
        
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt for multi-task agent."""
        return """
You are an AI Investment Advisor specializing in portfolio analysis, risk evaluation, and client education. 
You combine the analytical rigor of a quantitative analyst with the communication skills of a personal financial advisor.

## Your Core Responsibilities:

### 1. Portfolio Analysis
- Perform comprehensive portfolio evaluation using modern portfolio theory
- Calculate risk-adjusted returns, Sharpe ratios, and diversification metrics
- Identify concentration risks and rebalancing opportunities
- Compare performance against benchmarks and peer groups
- Provide actionable optimization recommendations

### 2. Risk Evaluation  
- Assess client risk tolerance through demographic and behavioral analysis
- Conduct stress testing against various market scenarios
- Calculate Value-at-Risk and expected shortfall metrics
- Evaluate portfolio resilience to market corrections, sector crashes, and economic downturns
- Recommend appropriate risk management strategies

### 3. Customer Engagement & Education
- Explain complex financial concepts in clear, accessible language
- Provide context for market conditions and economic factors
- Help clients understand the rationale behind recommendations
- Address emotional and behavioral aspects of investing
- Build confidence through education and transparency

## Your Approach:
- **Data-Driven**: Base all recommendations on quantitative analysis and market data
- **Client-Centric**: Always consider the client's unique situation, goals, and constraints
- **Educational**: Explain the "why" behind recommendations to build understanding
- **Transparent**: Clearly communicate risks, assumptions, and limitations
- **Adaptive**: Adjust recommendations based on changing market conditions and client needs

## Tools Available:
1. Portfolio Analysis - Comprehensive portfolio evaluation with optimization
2. Risk Assessment - Risk profiling and tolerance evaluation
3. Stress Testing - Scenario analysis and stress testing
4. Market Data - Real-time market data and historical analysis
5. Risk Scenarios - Interactive risk scenario generation

## Communication Style:
- Professional yet approachable
- Use analogies and examples to explain complex concepts
- Provide specific, actionable recommendations
- Include relevant data and metrics to support recommendations
- Acknowledge uncertainty and explain assumptions clearly

## Key Principles:
- Fiduciary duty: Always act in the client's best interest
- Diversification: Emphasize proper risk management through diversification
- Long-term focus: Encourage long-term wealth building over short-term speculation
- Education: Help clients become more informed investors
- Transparency: Full disclosure of risks, costs, and potential conflicts

When analyzing portfolios or assessing risk, always:
1. Gather necessary data about the client and portfolio
2. Perform comprehensive quantitative analysis using available tools
3. Consider the client's personal situation and goals
4. Provide clear explanations of findings and recommendations
5. Discuss implementation steps and ongoing monitoring

Remember: You're not just providing analysis - you're helping clients make informed decisions
that align with their financial goals and risk tolerance.
"""
    
    def get_tools(self) -> List[BaseTool]:
        """Get list of available tools for the agent."""
        return self.tools
    
    def get_system_message(self) -> str:
        """Get the system prompt for this agent."""
        return self.system_prompt
    
    def analyze_portfolio_comprehensive(
        self,
        portfolio_assets: List[Dict],
        client_profile: Dict,
        analysis_type: str = "full"
    ) -> Dict:
        """
        Perform comprehensive portfolio analysis including risk evaluation.
        
        Args:
            portfolio_assets: List of portfolio assets with symbols and quantities
            client_profile: Client demographic and preference information
            analysis_type: Type of analysis - "full", "risk_only", "performance_only"
        
        Returns:
            Dictionary containing analysis results and recommendations
        """
        try:
            # This would typically be called by the LangGraph system
            # Here we provide a direct interface for testing
            
            results = {
                "portfolio_analysis": None,
                "risk_assessment": None, 
                "stress_testing": None,
                "recommendations": [],
                "educational_content": []
            }
            
            if analysis_type in ["full", "performance_only"]:
                # Perform portfolio analysis
                logger.info("Performing portfolio analysis...")
                # Tool would be called by LLM in actual system
                
            if analysis_type in ["full", "risk_only"]:
                # Perform risk assessment
                logger.info("Assessing risk profile...")
                # Risk assessment tool would be called
                
            if analysis_type == "full":
                # Perform stress testing
                logger.info("Running stress tests...")
                # Stress testing tool would be called
                
            # Generate educational content and recommendations
            results["educational_content"] = self._generate_educational_content(
                portfolio_assets, client_profile
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive portfolio analysis: {e}")
            return {"error": str(e)}
    
    def assess_client_risk_interactively(self, client_info: Dict) -> Dict:
        """
        Conduct interactive risk assessment with scenario-based questions.
        
        Args:
            client_info: Basic client demographic information
            
        Returns:
            Risk assessment results and scenario questions for further evaluation
        """
        try:
            # Generate risk scenarios based on client profile
            scenarios = self._generate_personalized_scenarios(client_info)
            
            # Prepare interactive assessment
            assessment = {
                "demographic_analysis": self._analyze_demographics(client_info),
                "risk_scenarios": scenarios,
                "preliminary_risk_score": self._calculate_preliminary_risk_score(client_info),
                "next_steps": [
                    "Review and respond to risk scenarios",
                    "Discuss investment time horizon and goals",
                    "Evaluate liquidity needs and constraints",
                    "Consider behavioral factors and past experiences"
                ]
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in interactive risk assessment: {e}")
            return {"error": str(e)}
    
    def explain_market_conditions(self, focus_areas: List[str] = None) -> Dict:
        """
        Provide educational content about current market conditions and their implications.
        
        Args:
            focus_areas: Optional list of areas to focus on (e.g., "interest_rates", "inflation")
            
        Returns:
            Educational content about market conditions and investment implications
        """
        try:
            focus_areas = focus_areas or ["general_market", "economic_indicators", "sector_rotation"]
            
            market_education = {
                "market_overview": self._generate_market_overview(),
                "key_factors": self._identify_key_market_factors(),
                "sector_analysis": self._analyze_sector_trends(),
                "investment_implications": self._explain_investment_implications(focus_areas),
                "action_items": self._suggest_investor_actions()
            }
            
            return market_education
            
        except Exception as e:
            logger.error(f"Error explaining market conditions: {e}")
            return {"error": str(e)}
    
    def _generate_educational_content(
        self, portfolio_assets: List[Dict], client_profile: Dict
    ) -> List[Dict]:
        """Generate personalized educational content based on portfolio and client profile."""
        
        content = []
        
        # Portfolio diversification education
        if len(portfolio_assets) < 10:
            content.append({
                "topic": "Portfolio Diversification",
                "message": "Your portfolio has fewer than 10 holdings. Diversification helps reduce risk by spreading investments across different assets, sectors, and geographies.",
                "action": "Consider adding more holdings or using broad-market ETFs for instant diversification",
                "learn_more": "Research shows that portfolios with 15-20 well-chosen stocks can achieve most of the benefits of diversification."
            })
        
        # Risk management education
        client_age = client_profile.get("age", 40)
        if client_age > 50:
            content.append({
                "topic": "Age-Based Asset Allocation", 
                "message": f"At age {client_age}, many financial advisors suggest a balanced approach between growth and stability.",
                "action": "Consider the 'age in bonds' rule as a starting point, then adjust based on your risk tolerance and goals",
                "learn_more": "Asset allocation becomes more important as you approach retirement, focusing on preservation of capital."
            })
        
        # Market timing education
        content.append({
            "topic": "Time in Market vs. Timing the Market",
            "message": "Studies show that missing just the 10 best trading days over 20 years can cut returns in half.",
            "action": "Focus on staying invested for the long term rather than trying to time market movements",
            "learn_more": "Dollar-cost averaging can help reduce the impact of market volatility on your investments."
        })
        
        return content
    
    def _generate_personalized_scenarios(self, client_info: Dict) -> List[Dict]:
        """Generate risk scenarios personalized to client's situation."""
        
        age = client_info.get("age", 40)
        time_horizon = client_info.get("time_horizon", 20)
        
        scenarios = []
        
        if age < 35:
            scenarios.append({
                "scenario": "Early Career Market Crash",
                "description": "A major market crash occurs early in your career when you have decades to recover",
                "question": "Would you continue investing, reduce contributions, or stop investing temporarily?",
                "educational_note": "Young investors often benefit from market downturns as they can buy more shares at lower prices"
            })
        
        if age > 55:
            scenarios.append({
                "scenario": "Pre-Retirement Market Decline", 
                "description": "A 30% market decline occurs 5 years before your planned retirement",
                "question": "Would you delay retirement, reduce withdrawal plans, or maintain your timeline?",
                "educational_note": "Sequence of returns risk is highest near retirement when portfolios are largest"
            })
        
        if time_horizon < 10:
            scenarios.append({
                "scenario": "Short-Term Goal Jeopardy",
                "description": "Market volatility threatens a financial goal you need to achieve within 5 years",
                "question": "How would you adjust your investment strategy?",
                "educational_note": "Short-term goals typically require more conservative investment approaches"
            })
        
        return scenarios
    
    def _analyze_demographics(self, client_info: Dict) -> Dict:
        """Analyze demographic information for risk assessment."""
        
        age = client_info.get("age", 40)
        income = client_info.get("annual_income", 75000)
        net_worth = client_info.get("net_worth", 200000)
        
        analysis = {
            "age_factor": "Young investor with long time horizon" if age < 35 
                         else "Mid-career with moderate time horizon" if age < 55
                         else "Pre-retirement requiring capital preservation",
            
            "capacity_for_risk": "High" if net_worth > income * 10
                                else "Moderate" if net_worth > income * 5  
                                else "Limited",
            
            "investment_timeline": f"Approximately {max(65 - age, 5)} years to traditional retirement age"
        }
        
        return analysis
    
    def _calculate_preliminary_risk_score(self, client_info: Dict) -> float:
        """Calculate preliminary risk score based on demographics."""
        
        score = 5.0  # Base moderate score
        
        age = client_info.get("age", 40)
        if age < 30:
            score += 1.5
        elif age < 45:
            score += 0.5
        elif age > 60:
            score -= 1.5
        
        income = client_info.get("annual_income", 75000)
        net_worth = client_info.get("net_worth", 200000)
        
        if net_worth > income * 10:
            score += 1.0
        elif net_worth < income * 3:
            score -= 0.5
        
        return max(1.0, min(10.0, score))
    
    def _generate_market_overview(self) -> str:
        """Generate current market overview (would use real data in production)."""
        return """
        Current market conditions reflect a complex interplay of factors including:
        - Federal Reserve monetary policy and interest rate environment
        - Inflation trends and their impact on different asset classes
        - Geopolitical developments and their effect on global markets
        - Corporate earnings growth and valuation levels
        - Sector rotation patterns and emerging investment themes
        
        Understanding these factors helps inform investment decisions and risk management.
        """
    
    def _identify_key_market_factors(self) -> List[str]:
        """Identify key factors currently driving markets."""
        return [
            "Interest Rate Environment",
            "Inflation and Fed Policy", 
            "Corporate Earnings Growth",
            "Geopolitical Tensions",
            "Technology Sector Trends",
            "Energy and Commodity Prices",
            "Consumer Spending Patterns",
            "Employment and Wage Growth"
        ]
    
    def _analyze_sector_trends(self) -> Dict:
        """Analyze current sector trends and rotation patterns."""
        return {
            "growth_sectors": ["Technology", "Healthcare", "Consumer Discretionary"],
            "value_sectors": ["Financials", "Energy", "Materials"],
            "defensive_sectors": ["Utilities", "Consumer Staples", "Real Estate"],
            "rotation_theme": "Market showing mixed signals with both growth and value performing well in different periods"
        }
    
    def _explain_investment_implications(self, focus_areas: List[str]) -> Dict:
        """Explain investment implications of current market conditions."""
        implications = {}
        
        for area in focus_areas:
            if area == "interest_rates":
                implications[area] = {
                    "impact": "Higher rates generally benefit banks and hurt long-duration bonds",
                    "strategy": "Consider shorter-duration bonds and rate-sensitive sectors"
                }
            elif area == "inflation":
                implications[area] = {
                    "impact": "Inflation erodes purchasing power but may benefit real assets",
                    "strategy": "Consider TIPS, commodities, and companies with pricing power"
                }
            elif area == "general_market":
                implications[area] = {
                    "impact": "Market volatility creates both risks and opportunities",
                    "strategy": "Maintain diversified portfolio and focus on long-term goals"
                }
        
        return implications
    
    def _suggest_investor_actions(self) -> List[str]:
        """Suggest actionable steps for investors in current environment."""
        return [
            "Review and rebalance portfolio allocations quarterly",
            "Maintain emergency fund of 3-6 months expenses",
            "Consider tax-loss harvesting opportunities",
            "Stay informed but avoid emotional decision-making",
            "Focus on asset allocation rather than individual stock picking",
            "Review beneficiaries and estate planning regularly"
        ]


# Create agent instance
multi_task_agent = MultiTaskAgent()
