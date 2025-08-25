"""Compliance checking for SEC, FINRA, and IRS regulations."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from ..core.config import Config

config = Config.get_instance()
logger = logging.getLogger(__name__)


class ComplianceRule(BaseModel):
    """Individual compliance rule model."""
    rule_id: str
    regulation_source: str  # SEC, FINRA, IRS, etc.
    rule_name: str
    description: str
    severity: str  # "critical", "warning", "advisory"
    applies_to: List[str]  # ["individual", "institutional", "advisor"]
    effective_date: datetime
    last_updated: datetime


class ComplianceViolation(BaseModel):
    """Compliance violation detected."""
    rule_id: str
    violation_type: str
    severity: str
    description: str
    recommended_action: str
    potential_penalty: Optional[str] = None
    auto_correctable: bool = False


class TradeComplianceCheck(BaseModel):
    """Trade compliance check result."""
    trade_approved: bool
    violations: List[ComplianceViolation]
    warnings: List[str]
    recommendations: List[str]
    requires_disclosure: bool = False
    cooling_off_period: Optional[int] = None  # Days


class PortfolioComplianceCheck(BaseModel):
    """Portfolio-wide compliance check."""
    overall_compliant: bool
    violations: List[ComplianceViolation]
    warnings: List[str]
    recommendations: List[str]
    next_review_date: datetime


class ComplianceChecker:
    """Comprehensive compliance checking service."""
    
    def __init__(self):
        self.rules = self._load_compliance_rules()
    
    def check_trade_compliance(
        self,
        trade_type: str,  # "buy", "sell"
        symbol: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        client_type: str = "individual",
        account_type: str = "taxable"
    ) -> TradeComplianceCheck:
        """Check individual trade for compliance violations."""
        
        violations = []
        warnings = []
        recommendations = []
        
        trade_value = quantity * price
        position_size = trade_value / portfolio_value
        
        # Position Size Limits (SEC/FINRA Guidelines)
        if position_size > config.max_position_size:
            violations.append(ComplianceViolation(
                rule_id="CONC-001",
                violation_type="concentration_risk",
                severity="warning",
                description=f"Trade would create {position_size:.1%} position, exceeding {config.max_position_size:.1%} guideline",
                recommended_action="Reduce trade size or seek additional diversification",
                auto_correctable=True
            ))
        
        # Wash Sale Rule (IRS)
        if trade_type == "buy" and account_type == "taxable":
            warnings.append("Verify no wash sale violation if selling similar security at loss within 30 days")
        
        # Pattern Day Trader Rule (FINRA)
        if trade_value < 25000 and client_type == "individual":
            warnings.append("Account under $25K - monitor day trading activity to avoid PDT violations")
        
        # Penny Stock Rules (SEC Rule 15g)
        if price < 5.0:
            violations.append(ComplianceViolation(
                rule_id="PENNY-001", 
                violation_type="penny_stock",
                severity="advisory",
                description=f"Trading penny stock (${price:.2f})",
                recommended_action="Ensure proper disclosure and suitability analysis",
                requires_disclosure=True
            ))
        
        # Market Manipulation (SEC Rule 10b-5)
        if trade_value > portfolio_value * 0.5:  # Very large trade
            warnings.append("Large trade size - ensure no market manipulation concerns")
        
        trade_approved = len([v for v in violations if v.severity == "critical"]) == 0
        requires_disclosure = any(v.requires_disclosure for v in violations)
        
        return TradeComplianceCheck(
            trade_approved=trade_approved,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            requires_disclosure=requires_disclosure
        )
    
    def check_portfolio_compliance(
        self,
        portfolio_assets: List[Dict],
        portfolio_value: float,
        client_profile: Dict,
        account_type: str = "taxable"
    ) -> PortfolioComplianceCheck:
        """Check entire portfolio for compliance issues."""
        
        violations = []
        warnings = []
        recommendations = []
        
        # Concentration Risk Analysis
        for asset in portfolio_assets:
            allocation = asset.get("allocation", 0) / 100
            if allocation > config.max_position_size:
                violations.append(ComplianceViolation(
                    rule_id="CONC-001",
                    violation_type="concentration_risk", 
                    severity="warning",
                    description=f"{asset['symbol']} position ({allocation:.1%}) exceeds concentration limit",
                    recommended_action="Reduce position size through rebalancing"
                ))
        
        # Sector Concentration
        sector_allocations = {}
        for asset in portfolio_assets:
            sector = asset.get("sector", "Unknown")
            allocation = asset.get("allocation", 0) / 100
            sector_allocations[sector] = sector_allocations.get(sector, 0) + allocation
        
        for sector, allocation in sector_allocations.items():
            if allocation > config.max_sector_allocation:
                violations.append(ComplianceViolation(
                    rule_id="CONC-002",
                    violation_type="sector_concentration",
                    severity="warning", 
                    description=f"{sector} sector allocation ({allocation:.1%}) exceeds limit",
                    recommended_action="Diversify across more sectors"
                ))
        
        # Suitability Analysis (FINRA Rule 2111)
        client_age = client_profile.get("age", 50)
        client_experience = client_profile.get("investment_experience", "intermediate")
        risk_tolerance = client_profile.get("risk_tolerance", "moderate")
        
        # Age-based suitability
        equity_allocation = sum(
            asset.get("allocation", 0) / 100 
            for asset in portfolio_assets 
            if asset.get("asset_type", "stock") in ["stock", "equity", "etf"]
        )
        
        if client_age > 65 and equity_allocation > 0.6:
            warnings.append(f"High equity allocation ({equity_allocation:.1%}) may not be suitable for age {client_age}")
        
        # Experience-based suitability
        if client_experience == "beginner":
            complex_assets = [
                asset for asset in portfolio_assets 
                if asset.get("asset_type") in ["options", "futures", "derivatives"]
            ]
            if complex_assets:
                violations.append(ComplianceViolation(
                    rule_id="SUIT-001",
                    violation_type="suitability",
                    severity="critical",
                    description="Complex instruments not suitable for beginner investor",
                    recommended_action="Replace with more suitable investments"
                ))
        
        # Diversification Requirements
        if len(portfolio_assets) < 5:
            recommendations.append("Consider additional diversification with more holdings")
        
        # ESG/Social Responsibility Screening (if applicable)
        # This would integrate with external ESG databases
        
        overall_compliant = len([v for v in violations if v.severity == "critical"]) == 0
        next_review_date = datetime.now() + timedelta(days=90)  # Quarterly review
        
        return PortfolioComplianceCheck(
            overall_compliant=overall_compliant,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            next_review_date=next_review_date
        )
    
    def check_advisor_compliance(
        self,
        advisor_id: str,
        recommendations: List[Dict],
        client_interactions: List[Dict]
    ) -> Dict:
        """Check investment advisor compliance (SEC Investment Advisers Act)."""
        
        violations = []
        warnings = []
        
        # Fiduciary Duty Checks
        for rec in recommendations:
            if rec.get("conflicts_of_interest") and not rec.get("disclosure_provided"):
                violations.append(ComplianceViolation(
                    rule_id="FIDU-001",
                    violation_type="fiduciary_breach",
                    severity="critical",
                    description="Conflict of interest not properly disclosed",
                    recommended_action="Provide full disclosure to client"
                ))
        
        # Record Keeping Requirements (SEC Rule 204-2)
        for interaction in client_interactions:
            if not interaction.get("documented"):
                warnings.append("Client interaction may need proper documentation")
        
        # Form ADV Updates
        warnings.append("Ensure Form ADV is updated annually and material changes disclosed")
        
        return {
            "violations": [v.dict() for v in violations],
            "warnings": warnings,
            "next_compliance_review": (datetime.now() + timedelta(days=365)).isoformat()
        }
    
    def validate_investment_recommendation(
        self,
        recommendation: Dict,
        client_profile: Dict,
        portfolio_context: Dict
    ) -> Dict:
        """Validate investment recommendation against regulations and best practices."""
        
        violations = []
        warnings = []
        approved = True
        
        # Suitability Analysis
        client_risk_tolerance = client_profile.get("risk_tolerance", "moderate") 
        recommendation_risk = recommendation.get("risk_level", "moderate")
        
        risk_mapping = {"conservative": 1, "moderate": 2, "aggressive": 3}
        client_risk_score = risk_mapping.get(client_risk_tolerance, 2)
        rec_risk_score = risk_mapping.get(recommendation_risk, 2)
        
        if rec_risk_score > client_risk_score + 1:  # Allow slight mismatch
            violations.append(ComplianceViolation(
                rule_id="SUIT-002",
                violation_type="suitability",
                severity="critical", 
                description=f"Recommendation risk level ({recommendation_risk}) exceeds client tolerance ({client_risk_tolerance})",
                recommended_action="Adjust recommendation to match client risk profile"
            ))
            approved = False
        
        # Best Interest Standard (Reg BI)
        if not recommendation.get("best_interest_analysis"):
            warnings.append("Ensure recommendation meets best interest standard under Regulation BI")
        
        # Reasonable Basis Analysis
        if not recommendation.get("research_basis"):
            violations.append(ComplianceViolation(
                rule_id="SUIT-003", 
                violation_type="reasonable_basis",
                severity="warning",
                description="Insufficient research basis for recommendation",
                recommended_action="Provide detailed analysis supporting recommendation"
            ))
        
        return {
            "approved": approved,
            "violations": [v.dict() for v in violations],
            "warnings": warnings,
            "compliance_score": max(0, 100 - len(violations) * 20 - len(warnings) * 5)
        }
    
    def _load_compliance_rules(self) -> Dict[str, ComplianceRule]:
        """Load compliance rules database."""
        # In production, this would load from a database
        # Here we define key rules as examples
        
        rules = {
            "CONC-001": ComplianceRule(
                rule_id="CONC-001",
                regulation_source="SEC",
                rule_name="Position Concentration Limit",
                description="Individual position should not exceed 25% of portfolio",
                severity="warning",
                applies_to=["individual", "institutional"],
                effective_date=datetime(2000, 1, 1),
                last_updated=datetime.now()
            ),
            "SUIT-001": ComplianceRule(
                rule_id="SUIT-001", 
                regulation_source="FINRA",
                rule_name="Suitability Rule 2111",
                description="Recommendations must be suitable for client", 
                severity="critical",
                applies_to=["advisor"],
                effective_date=datetime(2012, 7, 9),
                last_updated=datetime.now()
            ),
            "WASH-001": ComplianceRule(
                rule_id="WASH-001",
                regulation_source="IRS", 
                rule_name="Wash Sale Rule",
                description="Cannot claim loss if repurchasing substantially identical security within 30 days",
                severity="warning",
                applies_to=["individual", "institutional"],
                effective_date=datetime(1921, 1, 1),
                last_updated=datetime.now()
            )
        }
        
        return rules


# Service instance
compliance_checker = ComplianceChecker()


# Tools
class TradeComplianceInput(BaseModel):
    """Input for trade compliance check."""
    trade_type: str = Field(..., description="Type of trade: buy or sell")
    symbol: str = Field(..., description="Security symbol")
    quantity: float = Field(..., description="Number of shares")
    price: float = Field(..., description="Price per share")
    portfolio_value: float = Field(..., description="Total portfolio value")
    client_type: str = Field("individual", description="Client type: individual or institutional")
    account_type: str = Field("taxable", description="Account type: taxable, ira, 401k")


class TradeComplianceTool(BaseTool):
    """Tool to check trade compliance before execution."""
    name: str = "check_trade_compliance"
    description: str = """
    Check individual trade for regulatory compliance including:
    - Position concentration limits
    - Wash sale rule considerations  
    - Pattern day trader rules
    - Penny stock regulations
    - Market manipulation safeguards
    - Suitability requirements
    """
    args_schema: type = TradeComplianceInput

    def _run(
        self,
        trade_type: str,
        symbol: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        client_type: str = "individual", 
        account_type: str = "taxable"
    ) -> str:
        """Execute trade compliance check."""
        try:
            result = compliance_checker.check_trade_compliance(
                trade_type=trade_type,
                symbol=symbol,
                quantity=quantity,
                price=price,
                portfolio_value=portfolio_value,
                client_type=client_type,
                account_type=account_type
            )
            
            trade_value = quantity * price
            position_size = (trade_value / portfolio_value) * 100
            
            output = f"""
Trade Compliance Check Results:
===============================

Trade Details:
- Action: {trade_type.upper()} {quantity:,.0f} shares of {symbol}
- Price: ${price:.2f} per share
- Total Value: ${trade_value:,.2f}
- Position Size: {position_size:.1f}% of portfolio

Compliance Status: {"✅ APPROVED" if result.trade_approved else "❌ REQUIRES REVIEW"}
"""
            
            if result.violations:
                output += "\n🚨 VIOLATIONS DETECTED:\n"
                for violation in result.violations:
                    severity_icon = "🔴" if violation.severity == "critical" else "🟡" if violation.severity == "warning" else "🔵"
                    output += f"{severity_icon} {violation.description}\n"
                    output += f"   Recommended Action: {violation.recommended_action}\n"
            
            if result.warnings:
                output += "\n⚠️  WARNINGS:\n"
                for warning in result.warnings:
                    output += f"• {warning}\n"
            
            if result.requires_disclosure:
                output += "\n📋 DISCLOSURE REQUIRED: Additional client disclosures needed before execution\n"
            
            if result.recommendations:
                output += "\n💡 RECOMMENDATIONS:\n"
                for rec in result.recommendations:
                    output += f"• {rec}\n"
            
            return output
            
        except Exception as e:
            return f"Trade compliance check error: {str(e)}"


class PortfolioComplianceInput(BaseModel):
    """Input for portfolio compliance check."""
    portfolio_assets: List[Dict] = Field(..., description="Portfolio assets with allocations")
    portfolio_value: float = Field(..., description="Total portfolio value")
    client_age: int = Field(..., description="Client age")
    risk_tolerance: str = Field(..., description="Client risk tolerance")
    investment_experience: str = Field(..., description="Client investment experience")
    account_type: str = Field("taxable", description="Account type")


class PortfolioComplianceTool(BaseTool):
    """Tool to check portfolio-wide compliance."""
    name: str = "check_portfolio_compliance"
    description: str = """
    Check entire portfolio for regulatory compliance including:
    - Position and sector concentration limits
    - Suitability analysis based on client profile
    - Diversification requirements
    - Age-appropriate asset allocation
    - Complex instrument suitability
    - Best practices adherence
    """
    args_schema: type = PortfolioComplianceInput

    def _run(
        self,
        portfolio_assets: List[Dict],
        portfolio_value: float,
        client_age: int,
        risk_tolerance: str,
        investment_experience: str,
        account_type: str = "taxable"
    ) -> str:
        """Execute portfolio compliance check.""" 
        try:
            client_profile = {
                "age": client_age,
                "risk_tolerance": risk_tolerance,
                "investment_experience": investment_experience
            }
            
            result = compliance_checker.check_portfolio_compliance(
                portfolio_assets=portfolio_assets,
                portfolio_value=portfolio_value,
                client_profile=client_profile,
                account_type=account_type
            )
            
            output = f"""
Portfolio Compliance Analysis:
==============================

Portfolio Value: ${portfolio_value:,.2f}
Client Profile: {client_age} years old, {risk_tolerance} risk tolerance, {investment_experience} experience

Overall Status: {"✅ COMPLIANT" if result.overall_compliant else "❌ VIOLATIONS DETECTED"}
Next Review Date: {result.next_review_date.strftime("%Y-%m-%d")}
"""
            
            if result.violations:
                output += "\n🚨 COMPLIANCE VIOLATIONS:\n"
                for violation in result.violations:
                    severity_icon = "🔴" if violation.severity == "critical" else "🟡" if violation.severity == "warning" else "🔵"
                    output += f"{severity_icon} {violation.violation_type.replace('_', ' ').title()}\n"
                    output += f"   Issue: {violation.description}\n"
                    output += f"   Action: {violation.recommended_action}\n\n"
            
            if result.warnings:
                output += "⚠️  COMPLIANCE WARNINGS:\n"
                for warning in result.warnings:
                    output += f"• {warning}\n"
                output += "\n"
            
            if result.recommendations:
                output += "💡 COMPLIANCE RECOMMENDATIONS:\n"
                for rec in result.recommendations:
                    output += f"• {rec}\n"
            
            return output
            
        except Exception as e:
            return f"Portfolio compliance check error: {str(e)}"


class RecommendationValidationInput(BaseModel):
    """Input for investment recommendation validation."""
    recommendation_type: str = Field(..., description="Type of recommendation")
    symbol: str = Field(..., description="Security symbol")
    action: str = Field(..., description="Recommended action")
    rationale: str = Field(..., description="Investment rationale")
    risk_level: str = Field(..., description="Risk level of recommendation")
    client_risk_tolerance: str = Field(..., description="Client risk tolerance")
    research_basis: bool = Field(True, description="Whether recommendation has research basis")


class RecommendationValidationTool(BaseTool):
    """Tool to validate investment recommendations for compliance."""
    name: str = "validate_investment_recommendation"
    description: str = """
    Validate investment recommendations for regulatory compliance including:
    - Suitability analysis against client profile
    - Best interest standard compliance (Reg BI)
    - Reasonable basis requirements
    - Risk level appropriateness
    - Disclosure requirements
    - Fiduciary duty compliance
    """
    args_schema: type = RecommendationValidationInput

    def _run(
        self,
        recommendation_type: str,
        symbol: str,
        action: str,
        rationale: str,
        risk_level: str,
        client_risk_tolerance: str,
        research_basis: bool = True
    ) -> str:
        """Execute recommendation validation."""
        try:
            recommendation = {
                "type": recommendation_type,
                "symbol": symbol,
                "action": action,
                "rationale": rationale,
                "risk_level": risk_level,
                "research_basis": research_basis,
                "best_interest_analysis": True  # Assume provided
            }
            
            client_profile = {
                "risk_tolerance": client_risk_tolerance
            }
            
            result = compliance_checker.validate_investment_recommendation(
                recommendation=recommendation,
                client_profile=client_profile,
                portfolio_context={}
            )
            
            output = f"""
Investment Recommendation Validation:
====================================

Recommendation: {action.upper()} {symbol} ({recommendation_type})
Risk Level: {risk_level} (Client Tolerance: {client_risk_tolerance})
Compliance Score: {result['compliance_score']}/100

Status: {"✅ APPROVED" if result['approved'] else "❌ REQUIRES REVISION"}

Rationale: {rationale}
"""
            
            if result['violations']:
                output += "\n🚨 COMPLIANCE ISSUES:\n"
                for violation in result['violations']:
                    severity_icon = "🔴" if violation['severity'] == "critical" else "🟡"
                    output += f"{severity_icon} {violation['description']}\n"
                    output += f"   Action Required: {violation['recommended_action']}\n\n"
            
            if result['warnings']:
                output += "⚠️  REGULATORY WARNINGS:\n"
                for warning in result['warnings']:
                    output += f"• {warning}\n"
            
            return output
            
        except Exception as e:
            return f"Recommendation validation error: {str(e)}"


# Tool instances
trade_compliance_tool = TradeComplianceTool()
portfolio_compliance_tool = PortfolioComplianceTool()
recommendation_validation_tool = RecommendationValidationTool()


# Main compliance tool for general use
class ComplianceCheckTool(BaseTool):
    """General compliance checking tool."""
    name: str = "check_compliance"
    description: str = """
    General compliance checker that can validate:
    - Individual trades before execution
    - Portfolio-wide compliance status  
    - Investment recommendation suitability
    - Regulatory requirement adherence
    """

    def _run(self, check_type: str, **kwargs) -> str:
        """Route to appropriate compliance check."""
        if check_type == "trade":
            return trade_compliance_tool._run(**kwargs)
        elif check_type == "portfolio": 
            return portfolio_compliance_tool._run(**kwargs)
        elif check_type == "recommendation":
            return recommendation_validation_tool._run(**kwargs)
        else:
            return f"Unknown compliance check type: {check_type}"


compliance_check_tool = ComplianceCheckTool()
