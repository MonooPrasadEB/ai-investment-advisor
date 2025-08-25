"""
Compliance Reviewer Agent for Policy Validation and Client-Facing Communication.

This agent reviews all investment recommendations and communications to ensure regulatory
compliance and rewrites content for clear, compliant client communication.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..tools.compliance_checker import (
    recommendation_validation_tool,
    trade_compliance_tool,
    portfolio_compliance_tool
)
from ..core.config import Config

config = Config.get_instance()
logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Review status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"
    ESCALATED = "escalated"


class ComplianceIssue(BaseModel):
    """Compliance issue model."""
    issue_id: str
    severity: str  # "critical", "major", "minor"
    category: str  # "disclosure", "suitability", "fiduciary", "record_keeping"
    description: str
    regulation_reference: str
    suggested_resolution: str
    auto_correctable: bool = False


class ReviewResult(BaseModel):
    """Document review result."""
    review_id: str
    original_content: str
    revised_content: Optional[str] = None
    status: ReviewStatus
    compliance_issues: List[ComplianceIssue] = []
    
    # Review metadata
    reviewed_by: str
    review_timestamp: datetime
    review_duration_seconds: float
    
    # Approval tracking
    approver: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    final_approval_required: bool = False


class ClientCommunication(BaseModel):
    """Client communication model."""
    communication_id: str
    communication_type: str  # "recommendation", "report", "alert", "education"
    original_content: str
    reviewed_content: str
    
    # Compliance elements
    required_disclosures: List[str] = []
    risk_warnings: List[str] = []
    disclaimers: List[str] = []
    
    # Client-friendly elements
    plain_english_summary: str
    key_takeaways: List[str] = []
    next_steps: List[str] = []


class ComplianceReviewerAgent:
    """
    Compliance Reviewer Agent responsible for:
    1. Policy validation against SEC, FINRA, IRS regulations
    2. Content review and revision for client communications
    3. Disclosure and disclaimer management
    4. Audit trail and documentation compliance
    """
    
    def __init__(self):
        self.name = "compliance_reviewer"
        self.llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.0,  # Very low temperature for compliance consistency
            api_key=config.openai_api_key
        )
        
        # Available tools for compliance review
        self.tools = [
            recommendation_validation_tool,
            trade_compliance_tool,
            portfolio_compliance_tool
        ]
        
        # Review tracking
        self.pending_reviews: Dict[str, ReviewResult] = {}
        self.completed_reviews: Dict[str, ReviewResult] = {}
        
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for compliance reviewer agent."""
        return """
You are an AI Compliance Reviewer Agent specializing in investment advisory compliance and client communication. Your role is to ensure all investment recommendations and client communications meet regulatory requirements while being clear and understandable.

## Your Core Responsibilities:

### 1. Regulatory Compliance Review
- Validate all content against SEC Investment Advisers Act requirements
- Ensure FINRA suitability and best interest standards are met
- Check for required disclosures under Regulation BI
- Verify fiduciary duty compliance and conflict of interest disclosures
- Ensure proper record-keeping and documentation standards

### 2. Risk Disclosure Management
- Identify and include all required risk warnings
- Ensure material risks are prominently disclosed
- Validate that risk disclosures match investment recommendations
- Check that disclaimers are appropriate and complete
- Verify that past performance disclaimers are included where required

### 3. Client Communication Enhancement
- Rewrite technical content into plain English
- Ensure key information is prominently featured
- Add clear calls-to-action and next steps
- Make complex financial concepts accessible to average investors
- Maintain professional tone while being approachable

### 4. Documentation and Audit Trail
- Create complete audit trails for all reviews
- Document compliance decisions and rationale
- Maintain version control for all revisions
- Ensure all required documentation is complete
- Flag items requiring manual review or escalation

## Key Compliance Requirements:

### SEC Investment Advisers Act
- Form ADV disclosures must be current and complete
- Fiduciary duty to act in client's best interest
- Material conflicts of interest must be disclosed
- Investment advice must be suitable for the client
- Records must be maintained per Rule 204-2

### FINRA Rules
- Rule 2111: Suitability requirements
- Rule 2090: Know Your Customer
- Rule 3270: Outside business activities disclosure
- Communications must be fair and balanced
- Supervision and review requirements

### Regulation BI (Best Interest)
- Care obligation: Due diligence in recommendations
- Disclosure obligation: Material facts and conflicts
- Conflict of interest mitigation
- Documentation of best interest analysis

## Review Categories:

### Critical Issues (Must Fix)
- Missing required disclosures
- Unsuitable recommendations
- Fiduciary duty violations
- Misleading or false statements
- Regulatory requirement violations

### Major Issues (Should Fix)
- Incomplete risk disclosures
- Unclear or confusing language
- Missing disclaimers
- Potential conflict of interest concerns
- Documentation deficiencies

### Minor Issues (Good Practice)
- Language clarity improvements
- Additional helpful disclosures
- Enhanced client education
- Better formatting and presentation

## Client Communication Standards:

### Plain English Requirements
- Use common words instead of jargon
- Explain technical terms when they must be used
- Use active voice and short sentences
- Organize information logically
- Include examples and analogies when helpful

### Required Elements
- Clear statement of recommendation or advice
- Rationale for the recommendation
- Material risks prominently disclosed
- Any conflicts of interest
- Required regulatory disclaimers
- Next steps or actions for the client

### Formatting Guidelines
- Use headers and bullet points for clarity
- Highlight key information
- Keep paragraphs short
- Include white space for readability
- Use consistent terminology throughout

## Tools Available:
1. Recommendation Validation - Check investment recommendations for compliance
2. Trade Compliance - Validate trades against regulatory requirements
3. Portfolio Compliance - Check portfolio-wide compliance issues

## Review Process:
1. Analyze content for regulatory compliance
2. Identify required disclosures and disclaimers
3. Check for suitability and best interest compliance
4. Rewrite content for clarity and compliance
5. Add necessary risk warnings and disclosures
6. Create audit documentation
7. Route for appropriate approval if needed

## Communication Tone:
- Professional but accessible
- Transparent about risks and limitations
- Helpful and educational
- Compliant with regulatory requirements
- Client-focused and clear

Remember: Your primary responsibility is ensuring regulatory compliance while making investment advice clear and understandable for clients. When in doubt, always err on the side of more disclosure and clearer communication.
"""
    
    def get_tools(self) -> List[BaseTool]:
        """Get list of available tools for the agent."""
        return self.tools
    
    def get_system_message(self) -> str:
        """Get the system prompt for this agent."""
        return self.system_prompt
    
    def review_investment_recommendation(
        self,
        recommendation_content: str,
        client_profile: Dict,
        recommendation_context: Dict
    ) -> Dict:
        """
        Review investment recommendation for compliance and client clarity.
        
        Args:
            recommendation_content: Original recommendation text
            client_profile: Client demographic and risk information
            recommendation_context: Context about the recommendation
            
        Returns:
            Dictionary containing review results and revised content
        """
        try:
            review_id = f"rec_review_{int(datetime.now().timestamp())}"
            start_time = datetime.now()
            
            # Step 1: Analyze original content for compliance issues
            compliance_issues = self._identify_compliance_issues(
                recommendation_content, client_profile, recommendation_context
            )
            
            # Step 2: Check suitability and best interest compliance
            suitability_check = self._validate_suitability(
                recommendation_context, client_profile
            )
            
            # Step 3: Identify required disclosures
            required_disclosures = self._get_required_disclosures(
                recommendation_context, compliance_issues
            )
            
            # Step 4: Rewrite content for compliance and clarity
            revised_content = self._rewrite_recommendation(
                recommendation_content,
                compliance_issues,
                required_disclosures,
                client_profile
            )
            
            # Step 5: Create review result
            review_duration = (datetime.now() - start_time).total_seconds()
            
            review_result = ReviewResult(
                review_id=review_id,
                original_content=recommendation_content,
                revised_content=revised_content,
                status=ReviewStatus.APPROVED if len(compliance_issues) == 0 else ReviewStatus.REQUIRES_REVISION,
                compliance_issues=compliance_issues,
                reviewed_by=self.name,
                review_timestamp=datetime.now(),
                review_duration_seconds=review_duration,
                final_approval_required=any(issue.severity == "critical" for issue in compliance_issues)
            )
            
            # Store review
            self.pending_reviews[review_id] = review_result
            
            return {
                "review_id": review_id,
                "status": review_result.status.value,
                "compliance_score": self._calculate_compliance_score(compliance_issues),
                "original_content": recommendation_content,
                "revised_content": revised_content,
                "compliance_issues": [issue.dict() for issue in compliance_issues],
                "suitability_check": suitability_check,
                "required_disclosures": required_disclosures,
                "final_approval_required": review_result.final_approval_required
            }
            
        except Exception as e:
            logger.error(f"Error reviewing investment recommendation: {e}")
            return {"error": str(e)}
    
    def create_client_communication(
        self,
        content_type: str,
        raw_content: str,
        client_profile: Dict,
        include_education: bool = True
    ) -> Dict:
        """
        Create compliant client communication from raw content.
        
        Args:
            content_type: Type of communication (recommendation, report, alert, education)
            raw_content: Raw content to be processed
            client_profile: Client information for personalization
            include_education: Whether to include educational content
            
        Returns:
            Dictionary containing final client communication
        """
        try:
            comm_id = f"comm_{content_type}_{int(datetime.now().timestamp())}"
            
            # Step 1: Analyze content and determine required compliance elements
            compliance_elements = self._analyze_content_for_compliance(raw_content, content_type)
            
            # Step 2: Create plain English version
            plain_english_content = self._convert_to_plain_english(raw_content, client_profile)
            
            # Step 3: Add required disclosures and disclaimers
            final_content = self._add_compliance_elements(
                plain_english_content, compliance_elements
            )
            
            # Step 4: Create structured communication
            client_comm = ClientCommunication(
                communication_id=comm_id,
                communication_type=content_type,
                original_content=raw_content,
                reviewed_content=final_content,
                required_disclosures=compliance_elements["disclosures"],
                risk_warnings=compliance_elements["risk_warnings"],
                disclaimers=compliance_elements["disclaimers"],
                plain_english_summary=self._create_summary(final_content),
                key_takeaways=self._extract_key_takeaways(final_content),
                next_steps=self._suggest_next_steps(content_type, final_content)
            )
            
            # Step 5: Add educational content if requested
            if include_education:
                educational_content = self._generate_educational_content(
                    content_type, client_profile
                )
                final_content += "\n\n" + educational_content
            
            return {
                "communication_id": comm_id,
                "final_content": final_content,
                "plain_english_summary": client_comm.plain_english_summary,
                "key_takeaways": client_comm.key_takeaways,
                "next_steps": client_comm.next_steps,
                "compliance_elements": {
                    "disclosures": client_comm.required_disclosures,
                    "risk_warnings": client_comm.risk_warnings,
                    "disclaimers": client_comm.disclaimers
                },
                "readability_score": self._calculate_readability_score(final_content),
                "compliance_complete": True
            }
            
        except Exception as e:
            logger.error(f"Error creating client communication: {e}")
            return {"error": str(e)}
    
    def validate_trade_communication(
        self,
        trade_details: Dict,
        client_profile: Dict,
        communication_draft: str
    ) -> Dict:
        """
        Validate trade-related client communication for compliance.
        
        Args:
            trade_details: Details about the proposed trade
            client_profile: Client information
            communication_draft: Draft communication to client
            
        Returns:
            Dictionary containing validation results and recommendations
        """
        try:
            # Step 1: Validate trade suitability
            suitability_issues = self._check_trade_suitability(trade_details, client_profile)
            
            # Step 2: Check required trade disclosures
            missing_disclosures = self._check_trade_disclosures(
                communication_draft, trade_details
            )
            
            # Step 3: Validate risk communication
            risk_communication_issues = self._validate_risk_communication(
                communication_draft, trade_details
            )
            
            # Step 4: Check for required approvals
            approval_requirements = self._determine_approval_requirements(
                trade_details, suitability_issues
            )
            
            all_issues = suitability_issues + missing_disclosures + risk_communication_issues
            
            return {
                "validation_passed": len(all_issues) == 0,
                "suitability_issues": suitability_issues,
                "missing_disclosures": missing_disclosures,
                "risk_communication_issues": risk_communication_issues,
                "approval_requirements": approval_requirements,
                "recommended_revisions": self._suggest_communication_revisions(all_issues),
                "compliance_checklist": self._generate_compliance_checklist(trade_details)
            }
            
        except Exception as e:
            logger.error(f"Error validating trade communication: {e}")
            return {"error": str(e)}
    
    def generate_disclosure_library(self) -> Dict:
        """Generate library of standard disclosures and disclaimers."""
        return {
            "investment_risks": {
                "general": "All investments involve risk, including the potential loss of principal. Past performance does not guarantee future results.",
                "market_risk": "Market risk is the possibility that securities will decline in value due to general market conditions.",
                "sector_concentration": "Investments concentrated in particular sectors may be subject to greater volatility than more diversified investments.",
                "small_cap": "Small-cap securities may be subject to greater volatility and less liquidity than larger-cap securities."
            },
            "advisor_disclosures": {
                "fiduciary": "As your investment advisor, we have a fiduciary duty to act in your best interest.",
                "conflicts": "We may receive compensation from third parties in connection with your investments. We will disclose any material conflicts of interest.",
                "fees": "Our advisory fees are described in our Form ADV Part 2A, which is available upon request."
            },
            "regulatory_disclaimers": {
                "sec_registration": "This firm is registered as an investment advisor with the Securities and Exchange Commission.",
                "past_performance": "Past performance is not indicative of future results and does not guarantee future performance.",
                "forward_looking": "Forward-looking statements are based on current expectations and are subject to change without notice."
            },
            "client_obligations": {
                "notify_changes": "Please notify us promptly of any changes to your financial situation, investment objectives, or risk tolerance.",
                "review_statements": "Please review your account statements and report any discrepancies immediately.",
                "understand_risks": "Ensure you understand the risks associated with any investment before proceeding."
            }
        }
    
    def _identify_compliance_issues(
        self, content: str, client_profile: Dict, context: Dict
    ) -> List[ComplianceIssue]:
        """Identify compliance issues in content."""
        issues = []
        
        # Check for missing risk disclosures
        if "risk" not in content.lower():
            issues.append(ComplianceIssue(
                issue_id="RISK-001",
                severity="major",
                category="disclosure",
                description="No risk disclosure found in recommendation",
                regulation_reference="SEC Investment Advisers Act Rule 206(4)-1",
                suggested_resolution="Add appropriate risk disclosure for recommended investments",
                auto_correctable=True
            ))
        
        # Check for suitability analysis
        if "suitable" not in content.lower() and "appropriate" not in content.lower():
            issues.append(ComplianceIssue(
                issue_id="SUIT-001",
                severity="critical",
                category="suitability", 
                description="No suitability analysis provided",
                regulation_reference="FINRA Rule 2111",
                suggested_resolution="Include clear suitability analysis based on client profile",
                auto_correctable=False
            ))
        
        # Check for conflict of interest disclosure
        if context.get("potential_conflicts") and "conflict" not in content.lower():
            issues.append(ComplianceIssue(
                issue_id="COI-001",
                severity="critical",
                category="fiduciary",
                description="Potential conflicts of interest not disclosed",
                regulation_reference="SEC Investment Advisers Act Section 206",
                suggested_resolution="Add full disclosure of any conflicts of interest",
                auto_correctable=True
            ))
        
        return issues
    
    def _validate_suitability(self, recommendation_context: Dict, client_profile: Dict) -> Dict:
        """Validate suitability of recommendation."""
        client_risk_tolerance = client_profile.get("risk_tolerance", "moderate")
        recommendation_risk = recommendation_context.get("risk_level", "moderate")
        
        risk_mapping = {"conservative": 1, "moderate": 2, "aggressive": 3}
        client_risk_score = risk_mapping.get(client_risk_tolerance, 2)
        rec_risk_score = risk_mapping.get(recommendation_risk, 2)
        
        suitability_gap = rec_risk_score - client_risk_score
        
        return {
            "suitable": suitability_gap <= 1,  # Allow slight mismatch
            "suitability_gap": suitability_gap,
            "client_risk_tolerance": client_risk_tolerance,
            "recommendation_risk": recommendation_risk,
            "suitability_analysis": self._generate_suitability_analysis(
                client_profile, recommendation_context
            )
        }
    
    def _get_required_disclosures(
        self, recommendation_context: Dict, compliance_issues: List[ComplianceIssue]
    ) -> List[str]:
        """Determine required disclosures based on context and issues."""
        disclosures = []
        
        # Always include general investment risks
        disclosures.append("All investments involve risk, including the potential loss of principal.")
        
        # Add specific disclosures based on investment type
        investment_type = recommendation_context.get("investment_type", "stock")
        if investment_type == "stock":
            disclosures.append("Stock prices can be volatile and may fluctuate significantly.")
        elif investment_type == "bond":
            disclosures.append("Bond prices may decline due to interest rate changes and credit risk.")
        
        # Add disclosures based on compliance issues
        for issue in compliance_issues:
            if issue.category == "disclosure":
                disclosures.append(issue.suggested_resolution)
        
        return disclosures
    
    def _rewrite_recommendation(
        self, original_content: str, compliance_issues: List[ComplianceIssue], 
        required_disclosures: List[str], client_profile: Dict
    ) -> str:
        """Rewrite recommendation content for compliance and clarity."""
        
        # Start with plain English version of original content
        revised_content = self._convert_to_plain_english(original_content, client_profile)
        
        # Add suitability analysis if missing
        if any(issue.issue_id.startswith("SUIT") for issue in compliance_issues):
            suitability_section = self._generate_suitability_section(client_profile)
            revised_content += "\n\n" + suitability_section
        
        # Add risk disclosures
        if required_disclosures:
            risk_section = "\n\nIMPORTANT RISK DISCLOSURES:\n"
            for disclosure in required_disclosures:
                risk_section += f"• {disclosure}\n"
            revised_content += risk_section
        
        # Add standard disclaimers
        revised_content += "\n\n" + self._get_standard_disclaimers()
        
        return revised_content
    
    def _calculate_compliance_score(self, compliance_issues: List[ComplianceIssue]) -> int:
        """Calculate compliance score based on issues."""
        base_score = 100
        
        for issue in compliance_issues:
            if issue.severity == "critical":
                base_score -= 30
            elif issue.severity == "major":
                base_score -= 15
            else:  # minor
                base_score -= 5
        
        return max(0, base_score)
    
    def _convert_to_plain_english(self, content: str, client_profile: Dict) -> str:
        """Convert technical content to plain English."""
        # This is a simplified implementation
        # In production, this would use more sophisticated language processing
        
        # Replace common jargon
        replacements = {
            "alpha": "excess return above market",
            "beta": "market sensitivity",
            "sharpe ratio": "risk-adjusted return measure",
            "volatility": "price fluctuation",
            "diversification": "spreading investments to reduce risk",
            "asset allocation": "how investments are divided among different types",
            "rebalancing": "adjusting portfolio back to target percentages"
        }
        
        plain_content = content
        for technical, plain in replacements.items():
            plain_content = plain_content.replace(technical, f"{plain} ({technical})")
        
        return plain_content
    
    def _generate_suitability_section(self, client_profile: Dict) -> str:
        """Generate suitability analysis section."""
        age = client_profile.get("age", "unknown")
        risk_tolerance = client_profile.get("risk_tolerance", "moderate")
        time_horizon = client_profile.get("time_horizon", "medium-term")
        
        return f"""
SUITABILITY ANALYSIS:
Based on your profile (age {age}, {risk_tolerance} risk tolerance, {time_horizon} investment horizon), 
this recommendation is considered suitable because it aligns with your stated investment objectives 
and risk tolerance. We have considered your financial situation, investment experience, and 
investment timeline in making this recommendation.
"""
    
    def _get_standard_disclaimers(self) -> str:
        """Get standard regulatory disclaimers."""
        return """
IMPORTANT DISCLAIMERS:
• Past performance does not guarantee future results
• All investments involve risk, including potential loss of principal
• This recommendation is based on information available at the time and may change
• Please consult with your tax advisor regarding tax implications
• We are registered investment advisors and have a fiduciary duty to act in your best interest
"""
    
    def _create_summary(self, content: str) -> str:
        """Create plain English summary of content."""
        # Simplified implementation
        lines = content.split('\n')
        key_lines = [line for line in lines[:3] if line.strip()]
        return " ".join(key_lines)[:200] + "..."
    
    def _extract_key_takeaways(self, content: str) -> List[str]:
        """Extract key takeaways from content."""
        # Simplified implementation
        return [
            "Review the recommendation and ensure you understand the risks",
            "Consider how this fits with your overall investment strategy",
            "Contact us if you have questions or need clarification"
        ]
    
    def _suggest_next_steps(self, content_type: str, content: str) -> List[str]:
        """Suggest next steps based on content type."""
        if content_type == "recommendation":
            return [
                "Review the recommendation details carefully",
                "Consider the risks and how they fit your tolerance", 
                "Contact us to discuss or approve the recommendation",
                "We will handle the execution once approved"
            ]
        elif content_type == "report":
            return [
                "Review your portfolio performance",
                "Note any rebalancing recommendations",
                "Schedule a review meeting if desired",
                "Update us on any changes to your situation"
            ]
        else:
            return ["Contact us if you have questions or need assistance"]


# Create agent instance
compliance_reviewer_agent = ComplianceReviewerAgent()
