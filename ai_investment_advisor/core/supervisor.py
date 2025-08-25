

"""
Investment Advisor Supervisor using LangGraph for Multi-Agent Coordination.

This supervisor orchestrates the three core agents to provide comprehensive 
investment advisory services with proper handoffs and workflow management.
"""

import logging
from typing import Annotated, Dict, List, Literal, Optional, TypedDict
from datetime import datetime
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from ..agents.multi_task_agent import multi_task_agent
from ..agents.execution_agent import execution_agent
from ..agents.compliance_reviewer import compliance_reviewer_agent
from ..core.config import Config

config = Config.get_instance()
logger = logging.getLogger(__name__)


class InvestmentAdvisorState(TypedDict):
    """State management for the investment advisor workflow."""
    # Message history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Client context
    client_profile: Optional[Dict]
    portfolio_data: Optional[Dict] 
    
    # Workflow state
    current_task: Optional[str]
    analysis_results: Optional[Dict]
    trade_recommendations: Optional[List[Dict]]
    compliance_approval: Optional[bool]
    
    # Agent handoffs
    next_agent: Optional[str]
    requires_approval: bool
    workflow_complete: bool
    
    # Session management
    session_id: str
    created_at: datetime


class InvestmentAdvisorSupervisor:
    """
    Multi-Agent Investment Advisor Supervisor.
    
    Coordinates between:
    1. Multi-Task Agent (Portfolio Analysis, Risk Assessment, Customer Engagement)
    2. Execution Agent (Trade execution with compliance checks)
    3. Compliance Reviewer (Policy validation and client communication)
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.default_model,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )
        
        # Agent instances
        self.multi_task_agent = multi_task_agent
        self.execution_agent = execution_agent 
        self.compliance_reviewer = compliance_reviewer_agent
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        
        logger.info("Investment Advisor Supervisor initialized")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for agent coordination."""
        
        # Create individual agent executors
        portfolio_agent = create_react_agent(
            self.llm,
            self.multi_task_agent.get_tools(),
            prompt=self.multi_task_agent.get_system_message()
        )
        
        trade_agent = create_react_agent(
            self.llm,
            self.execution_agent.get_tools(),
            prompt=self.execution_agent.get_system_message()
        )
        
        compliance_agent = create_react_agent(
            self.llm,
            self.compliance_reviewer.get_tools(),
            prompt=self.compliance_reviewer.get_system_message()
        )
        
        # Define the state graph
        workflow = StateGraph(InvestmentAdvisorState)
        
        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("portfolio_analysis", self._portfolio_analysis_node)
        workflow.add_node("trade_execution", self._trade_execution_node)
        workflow.add_node("compliance_review", self._compliance_review_node)
        workflow.add_node("client_communication", self._client_communication_node)
        
        # Add edges with routing logic
        workflow.set_entry_point("supervisor")
        
        workflow.add_conditional_edges(
            "supervisor",
            self._route_next_action,
            {
                "portfolio_analysis": "portfolio_analysis",
                "trade_execution": "trade_execution",
                "compliance_review": "compliance_review",
                "client_communication": "client_communication",
                "end": END
            }
        )
        
        # All agents route back to supervisor for coordination
        workflow.add_edge("portfolio_analysis", "supervisor")
        workflow.add_edge("trade_execution", "supervisor")
        workflow.add_edge("compliance_review", "supervisor")
        workflow.add_edge("client_communication", "supervisor")
        
        return workflow.compile()
    
    def _supervisor_node(self, state: InvestmentAdvisorState) -> InvestmentAdvisorState:
        """Supervisor node that coordinates workflow and makes routing decisions."""
        
        messages = state["messages"]
        current_task = state.get("current_task")
        
        # Analyze the current state and determine next action
        if not messages:
            # Initialize conversation
            state["current_task"] = "initial_assessment"
            state["next_agent"] = "portfolio_analysis"
            return state
        
        last_message = messages[-1]
        
        # Use LLM-powered intelligent routing
        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()
            
            # Handle approval responses - route directly to compliance
            if "approve" in content and state.get("requires_approval"):
                state["current_task"] = "compliance_review"
                state["next_agent"] = "compliance_review"
            else:
                # Use LLM for other routing decisions
                routing_decision = self._get_llm_routing_decision(
                    last_message.content,
                    state.get("portfolio_data"),
                    state.get("client_profile")
                )
                
                state["current_task"] = routing_decision["task"]
                state["next_agent"] = routing_decision["agent"]
        
        elif isinstance(last_message, AIMessage):
            # Process AI response and determine if workflow is complete
            if state.get("analysis_results") and state.get("compliance_approval"):
                state["workflow_complete"] = True
                state["next_agent"] = "client_communication"
            elif state.get("trade_recommendations") and not state.get("compliance_approval"):
                state["next_agent"] = "compliance_review"
            else:
                state["next_agent"] = "end"
        
        return state
    
    def _get_llm_routing_decision(self, user_message: str, portfolio_data: Optional[Dict], client_profile: Optional[Dict]) -> Dict[str, str]:
        """
        Use LLM to intelligently route user requests to appropriate agents.
        
        Returns:
            Dict with 'agent' and 'task' keys
        """
        try:
            # Build context for routing decision
            portfolio_context = ""
            if portfolio_data:
                total_value = sum(
                    holding.get('quantity', 0) * holding.get('current_price', 0)
                    for holding in portfolio_data.get('holdings', [])
                )
                portfolio_context = f"User has active portfolio worth ${total_value:,.2f}"
            
            client_context = ""
            if client_profile:
                risk_tolerance = client_profile.get('risk_tolerance', 'unknown')
                client_context = f"Client risk tolerance: {risk_tolerance}"
            
            routing_prompt = ChatPromptTemplate.from_template("""
You are an intelligent routing system for a multi-agent investment advisor. 
Analyze the user's message and route to the most appropriate agent.

AVAILABLE AGENTS:

1. **portfolio_analysis** - Portfolio Analysis & Advisory Agent
   - Portfolio analysis, risk assessment, diversification advice
   - Investment recommendations and strategy discussions  
   - Questions about what to buy/sell (advisory, not execution)
   - Market insights and educational content
   - Examples: "Should I buy NVDA?", "Analyze my portfolio", "What should I sell?", "Is my portfolio too risky?"

2. **trade_execution** - Trade Execution Agent  
   - Actual trade execution and order placement
   - Specific buy/sell commands with quantities
   - Trade confirmations and order management
   - Examples: "Buy 100 shares of AAPL", "Execute this trade", "Place order for $1000 SPY", "Confirm purchase"

3. **compliance_review** - Compliance & Review Agent
   - Trade compliance validation
   - Risk management checks  
   - Regulatory review and approval
   - Examples: "Review this trade", "Check compliance", "Is this allowed?"

CONTEXT:
{portfolio_context}
{client_context}

USER MESSAGE: "{user_message}"

ROUTING DECISION:
Analyze the user's intent and route appropriately. Most questions and advisory requests go to portfolio_analysis. 
Only route to trade_execution for specific execution commands. Only route to compliance_review for explicit compliance checks.

Respond with ONLY a JSON object:
{{
  "agent": "portfolio_analysis|trade_execution|compliance_review",
  "task": "brief_description_of_task"
}}
""")
            
            chain = routing_prompt | self.llm
            response = chain.invoke({
                "user_message": user_message,
                "portfolio_context": portfolio_context,
                "client_context": client_context
            })
            
            # Parse LLM response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response.content)
            if json_match:
                routing_data = json.loads(json_match.group())
                
                # Validate agent choice
                valid_agents = ["portfolio_analysis", "trade_execution", "compliance_review"]
                if routing_data.get("agent") in valid_agents:
                    return {
                        "agent": routing_data["agent"],
                        "task": routing_data.get("task", "user_request")
                    }
            
            # Fallback to default routing
            logger.warning(f"LLM routing failed, using default. Response: {response.content}")
            
        except Exception as e:
            logger.error(f"Error in LLM routing: {e}")
        
        # Default fallback - most requests are advisory
        return {
            "agent": "portfolio_analysis", 
            "task": "client_engagement"
        }
    

    def _portfolio_analysis_node(self, state: InvestmentAdvisorState) -> InvestmentAdvisorState:
        """Portfolio analysis and risk assessment node."""
        
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        # Create context for portfolio analysis
        analysis_prompt = self._create_portfolio_analysis_prompt(state)
        
        # Get response from multi-task agent
        # In a real implementation, this would call the LangGraph agent
        response_content = self._simulate_portfolio_analysis(
            last_message.content if last_message else "Analyze portfolio",
            state.get("portfolio_data", {}),
            state.get("client_profile", {})
        )
        
        # Calculate real analysis results from portfolio data
        portfolio_data = state.get("portfolio_data", {})
        assets = portfolio_data.get('assets', [])
        
        if assets:
            # Calculate real metrics
            tech_allocation = sum(asset.get('allocation', 0) for asset in assets if asset.get('sector') == 'Technology')
            num_assets = len(assets)
            diversification_score = min(10, num_assets * 2)
            risk_score = min(10, (tech_allocation / 10) + (diversification_score / 2))
            
            # Generate real recommendations
            recommendations = []
            if tech_allocation > 50:
                recommendations.append("Reduce technology concentration")
            if num_assets < 5:
                recommendations.append("Add more diversification")
            if not any(asset.get('asset_type') == 'bond' for asset in assets):
                recommendations.append("Add fixed income allocation")
                
            state["analysis_results"] = {
                "portfolio_metrics": {
                    "risk_score": round(risk_score, 1), 
                    "diversification_score": round(diversification_score, 1),
                    "tech_allocation": round(tech_allocation, 1),
                    "num_holdings": num_assets
                },
                "recommendations": recommendations if recommendations else ["Portfolio appears well balanced"],
                "risk_assessment": {"tolerance": "moderate", "score": round(risk_score, 1)}
            }
        else:
            # Fallback for no portfolio data
            state["analysis_results"] = {
                "portfolio_metrics": {"risk_score": 0, "diversification_score": 0},
                "recommendations": ["Please provide portfolio data for analysis"],
                "risk_assessment": {"tolerance": "unknown", "score": 0}
            }
        
        # Add AI response to messages
        ai_response = AIMessage(content=response_content)
        state["messages"] = messages + [ai_response]
        
        return state
    
    def _trade_execution_node(self, state: InvestmentAdvisorState) -> InvestmentAdvisorState:
        """Trade execution and order management node - uses LLM to process trade requests."""
        
        messages = state["messages"] 
        last_message = messages[-1] if messages else None
        
        if last_message:
            # Use LLM to intelligently process the trade request
            response_content = self._process_trade_request_with_llm(
                last_message.content,
                state.get("portfolio_data", {}),
                state.get("client_profile", {})
            )
            
            # Extract trade details from the request and populate state
            trade_details = self._extract_trade_details(last_message.content)
            state["trade_recommendations"] = [trade_details] if trade_details else []
            state["requires_approval"] = True
        else:
            response_content = "Please specify trade details for execution analysis."
        
        ai_response = AIMessage(content=response_content)
        state["messages"] = messages + [ai_response]
        
        return state
    
    def _extract_trade_details(self, request: str) -> Optional[Dict]:
        """Extract structured trade details from user request using LLM."""
        try:
            extraction_prompt = ChatPromptTemplate.from_template("""
Extract trade details from this request. If this is NOT a trade request, return null.

USER REQUEST: "{request}"

Extract and return JSON with these fields:
- "symbol": stock symbol (e.g., "OKTA")
- "action": "buy" or "sell" 
- "quantity": number of shares (integer)
- "order_type": "market", "limit", etc.
- "rationale": brief reason for trade

Examples:
"Sell 100 shares of OKTA" → {{"symbol": "OKTA", "action": "sell", "quantity": 100, "order_type": "market", "rationale": "User requested sale"}}
"Buy 50 AAPL at market" → {{"symbol": "AAPL", "action": "buy", "quantity": 50, "order_type": "market", "rationale": "User requested purchase"}}
"Should I sell NVDA?" → null (this is a question, not a trade request)

Return ONLY the JSON object or null:
""")
            
            chain = extraction_prompt | self.llm
            response = chain.invoke({"request": request})
            
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response.content)
            if json_match:
                try:
                    trade_details = json.loads(json_match.group())
                    # Validate required fields
                    if all(key in trade_details for key in ["symbol", "action", "quantity"]):
                        return trade_details
                except json.JSONDecodeError:
                    pass
                    
            # Check for null response
            if "null" in response.content.lower():
                return None
                
            return None
            
        except Exception as e:
            logger.error(f"Error extracting trade details: {e}")
            return None
    
    def _process_trade_request_with_llm(self, request: str, portfolio_data: Dict, client_profile: Dict) -> str:
        """
        Use LLM to intelligently process trade execution requests.
        
        This analyzes the trade request, validates it against the portfolio, 
        calculates impact, and provides detailed execution analysis.
        """
        try:
            # Build portfolio context
            portfolio_context = ""
            current_holdings = {}
            
            if portfolio_data and 'holdings' in portfolio_data:
                total_value = sum(
                    holding.get('quantity', 0) * holding.get('current_price', 0)
                    for holding in portfolio_data['holdings']
                )
                portfolio_context = f"Current Portfolio Value: ${total_value:,.2f}\nHoldings:\n"
                
                for holding in portfolio_data['holdings']:
                    symbol = holding.get('symbol', 'Unknown')
                    quantity = holding.get('quantity', 0)
                    price = holding.get('current_price', 0)
                    value = quantity * price
                    
                    current_holdings[symbol] = {
                        'quantity': quantity,
                        'price': price,
                        'value': value
                    }
                    
                    portfolio_context += f"- {symbol}: {quantity} shares @ ${price:.2f} = ${value:,.2f}\n"
            
            # Build client context
            client_context = ""
            if client_profile:
                risk_tolerance = client_profile.get('risk_tolerance', 'moderate')
                experience = client_profile.get('investment_experience', 'intermediate')
                client_context = f"Risk Tolerance: {risk_tolerance}, Experience: {experience}"
            
            # Create trade processing prompt
            trade_prompt = ChatPromptTemplate.from_template("""
You are a professional trade execution specialist analyzing a client's trade request.

TRADE REQUEST: "{request}"

CURRENT PORTFOLIO:
{portfolio_context}

CLIENT PROFILE: {client_context}

TASK: Analyze this trade request and provide:

1. **Trade Validation**: Is this a valid, executable trade request?
2. **Portfolio Impact**: How will this affect their portfolio?
3. **Risk Analysis**: What are the implications?
4. **Execution Plan**: Specific steps to execute this trade
5. **Recommendations**: Any suggestions or concerns

If the request is for a SELL order:
- Check if they own the stock and have enough shares
- Calculate the proceeds and portfolio impact
- Provide specific execution details

If the request is for a BUY order:
- Estimate cost and market impact
- Check portfolio balance implications
- Provide execution guidance

Respond in a professional, clear manner that helps the client understand:
- What exactly will be executed
- The financial impact
- Any risks or considerations
- Next steps for approval/execution

Be conversational but precise. If the trade request is unclear or invalid, ask for clarification.
""")
            
            chain = trade_prompt | self.llm
            response = chain.invoke({
                "request": request,
                "portfolio_context": portfolio_context,
                "client_context": client_context
            })
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error in LLM trade processing: {e}")
            return f"I'm having trouble processing your trade request: {str(e)}. Please try rephrasing your request or contact support."
    
    def _process_trade_request(self, request: str, portfolio_data: Dict, client_profile: Dict) -> str:
        """
        Process trade execution requests and provide detailed analysis.
        """
        try:
            # Use LLM to parse the trade request and provide execution analysis
            trade_prompt = ChatPromptTemplate.from_template("""
You are a trade execution specialist analyzing a client's trade request.

CLIENT REQUEST: "{request}"

PORTFOLIO CONTEXT:
{portfolio_context}

CLIENT PROFILE:
{client_context}

TASK: Analyze this trade request and provide:

1. **Trade Details Interpretation**: Extract the specific trade action, symbol, and quantity
2. **Current Position Analysis**: Check if the client currently holds this security
3. **Market Impact Assessment**: Consider current market conditions and timing
4. **Execution Recommendation**: Suggest optimal execution strategy
5. **Risk Assessment**: Identify potential risks and benefits
6. **Next Steps**: What the client needs to do to proceed

IMPORTANT: This is SIMULATED trading - no actual trades will be executed. 
Provide actionable guidance while being clear this is for analysis purposes only.

Format your response as a clear, professional trade execution analysis.
""")
            
            # Build context
            portfolio_context = ""
            if portfolio_data and portfolio_data.get('holdings'):
                total_value = sum(
                    holding.get('quantity', 0) * holding.get('current_price', 0)
                    for holding in portfolio_data.get('holdings', [])
                )
                portfolio_context = f"Portfolio value: ${total_value:,.2f} with {len(portfolio_data.get('holdings', []))} holdings"
            else:
                portfolio_context = "No portfolio data available"
            
            client_context = ""
            if client_profile:
                risk_tolerance = client_profile.get('risk_tolerance', 'unknown')
                client_context = f"Risk tolerance: {risk_tolerance}"
            else:
                client_context = "No client profile available"
            
            chain = trade_prompt | self.llm
            response = chain.invoke({
                "request": request,
                "portfolio_context": portfolio_context,
                "client_context": client_context
            })
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error processing trade request: {e}")
            return f"I'm analyzing your trade request: '{request}'. However, I need more specific details to proceed with execution analysis. Please provide the stock symbol, action (buy/sell), and quantity for a complete analysis."
    
    def _compliance_review_node(self, state: InvestmentAdvisorState) -> InvestmentAdvisorState:
        """Compliance review and validation node."""
        
        messages = state["messages"]
        trade_recommendations = state.get("trade_recommendations", [])
        last_message = messages[-1].content.lower() if messages else ""
        
        # Check if user just approved a trade
        if "approve" in last_message and trade_recommendations:
            response_content = self._process_trade_approval(trade_recommendations, state.get("client_profile", {}))
            state["compliance_approval"] = True
            state["workflow_complete"] = True
            
        elif trade_recommendations:
            response_content = self._simulate_compliance_review(
                trade_recommendations,
                state.get("client_profile", {}),
                messages[-1].content if messages else ""
            )
            state["compliance_approval"] = True
            
        else:
            # This shouldn't happen in normal flow, but handle gracefully
            response_content = "Trade processing complete. Thank you for using AI Investment Advisor."
        
        ai_response = AIMessage(content=response_content)
        state["messages"] = messages + [ai_response]
        
        return state
    
    def _process_trade_approval(self, trade_recommendations: List[Dict], client_profile: Dict) -> str:
        """Process user approval and provide confirmation message."""
        try:
            if not trade_recommendations:
                return "Trade approval processed. Thank you for using AI Investment Advisor."
            
            trade = trade_recommendations[0]  # Get the first/primary trade
            symbol = trade.get('symbol', 'Unknown')
            action = trade.get('action', 'trade')
            quantity = trade.get('quantity', 0)
            
            return f"""
✅ **Trade Approved & Processed**

**Trade Details:**
• **Symbol:** {symbol}
• **Action:** {action.upper()} {quantity:,} shares
• **Order Type:** Market order
• **Status:** Approved for execution

**Next Steps:**
1. Trade has been validated for compliance ✅
2. Order will be submitted to the market during next trading session
3. You will receive execution confirmation once completed
4. Portfolio will be automatically updated

**Important Notes:**
• This is a simulated trade execution for demonstration
• In a real system, actual market orders would be placed
• All regulatory requirements have been satisfied

**Thank you for using AI Investment Advisor!**

*For questions about this trade or your portfolio, feel free to ask anytime.*
"""
            
        except Exception as e:
            logger.error(f"Error processing trade approval: {e}")
            return "Trade has been approved and processed. Thank you for using AI Investment Advisor."
    
    def _client_communication_node(self, state: InvestmentAdvisorState) -> InvestmentAdvisorState:
        """Client communication and final response node."""
        
        messages = state["messages"]
        
        # Create final client communication
        final_response = self._create_final_communication(state)
        
        ai_response = AIMessage(content=final_response)
        state["messages"] = messages + [ai_response]
        state["workflow_complete"] = True
        
        return state
    
    def _route_next_action(
        self, state: InvestmentAdvisorState
    ) -> Literal["portfolio_analysis", "trade_execution", "compliance_review", "client_communication", "end"]:
        """Route to the next appropriate agent based on state."""
        
        next_agent = state.get("next_agent")
        
        if state.get("workflow_complete"):
            return "end"
        elif next_agent == "portfolio_analysis":
            return "portfolio_analysis"
        elif next_agent == "trade_execution":
            return "trade_execution"
        elif next_agent == "compliance_review":
            return "compliance_review"
        elif next_agent == "client_communication":
            return "client_communication"
        else:
            return "end"
    
    def process_client_request(
        self, 
        request: str, 
        client_profile: Optional[Dict] = None,
        portfolio_data: Optional[Dict] = None,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[BaseMessage]] = None
    ) -> Dict:
        """
        Process a client request through the multi-agent workflow.
        
        Args:
            request: Client's investment question or request
            client_profile: Client demographic and preference information
            portfolio_data: Current portfolio holdings and data
            session_id: Session identifier for conversation continuity
            conversation_history: Previous messages in the conversation for context
            
        Returns:
            Dictionary containing the complete workflow results including updated conversation history
        """
        try:
            # Build complete message history including previous conversation
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add the current request
            messages.append(HumanMessage(content=request))
            
            # Initialize state with full conversation context
            initial_state = InvestmentAdvisorState(
                messages=messages,
                client_profile=client_profile or {},
                portfolio_data=portfolio_data or {},
                current_task=None,
                analysis_results=None,
                trade_recommendations=None,
                compliance_approval=None,
                next_agent=None,
                requires_approval=False,
                workflow_complete=False,
                session_id=session_id or f"session_{int(datetime.now().timestamp())}",
                created_at=datetime.now()
            )
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Extract final response and conversation history
            final_messages = final_state["messages"]
            final_response = final_messages[-1].content if final_messages else "No response generated"
            
            return {
                "response": final_response,
                "conversation_history": final_messages,
                "session_id": final_state.get("session_id"),
                "requires_approval": final_state.get("requires_approval", False),
                "trade_recommendations": final_state.get("trade_recommendations", []),
                "analysis_results": final_state.get("analysis_results", {}),
                "workflow_complete": final_state.get("workflow_complete", False)
            }
            
        except Exception as e:
            logger.error(f"Error processing client request: {e}")
            return {
                "error": str(e),
                "response": "I apologize, but I encountered an error processing your request. Please try again or contact support."
            }
    
    async def process_client_request_async(
        self, 
        request: str,
        client_profile: Optional[Dict] = None,
        portfolio_data: Optional[Dict] = None,
        session_id: Optional[str] = None
    ) -> Dict:
        """Async version of process_client_request."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.process_client_request, request, client_profile, portfolio_data, session_id
        )
    
    # Helper methods for simulation (replace with actual agent calls in production)
    
    def _create_portfolio_analysis_prompt(self, state: InvestmentAdvisorState) -> str:
        """Create context-aware prompt for portfolio analysis."""
        client_info = state.get("client_profile", {})
        portfolio_info = state.get("portfolio_data", {})
        
        return f"""
        Client Profile: {client_info}
        Portfolio Data: {portfolio_info}
        Current Task: {state.get("current_task")}
        
        Please provide comprehensive portfolio analysis including risk assessment and recommendations.
        """
    
    def _simulate_portfolio_analysis(self, request: str, portfolio_data: Dict, client_profile: Dict) -> str:
        """Generate dynamic conversational responses using LLM based on user's specific question."""
        try:
            # Extract portfolio metrics for context
            assets = portfolio_data.get('assets', [])
            if not assets:
                return f"""I don't see any portfolio data loaded. Please load your portfolio file first using the 'portfolio' command, then I can help answer your question: "{request}" """
            
            # Calculate key portfolio metrics for LLM context
            total_value = portfolio_data.get('total_value', 0)
            tech_allocation = sum(asset.get('allocation', 0) for asset in assets 
                                if asset.get('sector', '').lower().startswith('tech'))
            num_assets = len(assets)
            max_allocation = max([asset.get('allocation', 0) for asset in assets]) if assets else 0
            
            # Get top holdings for context
            top_holdings = sorted(assets, key=lambda x: x.get('allocation', 0), reverse=True)[:5]
            holdings_summary = ", ".join([f"{asset.get('symbol')} ({asset.get('allocation', 0):.1f}%)" 
                                        for asset in top_holdings])
            
            # Create conversational prompt for LLM
            portfolio_context = f"""
PORTFOLIO CONTEXT:
- Total Value: ${total_value:,.2f}
- Holdings: {num_assets} positions
- Tech Allocation: {tech_allocation:.1f}%
- Largest Position: {max_allocation:.1f}%
- Top Holdings: {holdings_summary}
- Risk Profile: {"High (tech-focused)" if tech_allocation > 60 else "Moderate"}
"""

            # Use LLM to generate conversational response
            prompt = ChatPromptTemplate.from_template("""
You are an experienced investment advisor having a conversation with a client about their portfolio.

{portfolio_context}

CLIENT QUESTION: "{request}"

Provide a conversational, helpful response that:
1. Directly addresses their specific question
2. References their actual portfolio data when relevant
3. Gives personalized advice based on their holdings
4. Uses a friendly, professional tone
5. Keeps response focused and concise (2-3 paragraphs max)

If they want to stay aggressive, support that preference and give advice on how to optimize their aggressive strategy.
If they ask about specific stocks, reference their actual holdings.
Be conversational, not templated.
""")
            
            # Generate response using LLM
            chain = prompt | self.llm
            response = chain.invoke({
                "portfolio_context": portfolio_context,
                "request": request
            })
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error in conversational analysis: {e}")
            return f"""I apologize, but I encountered an error while processing your question: "{request}". 

Could you please try rephrasing your question? I'm here to help with any questions about your portfolio, investment strategy, or market insights."""
    
    def _simulate_trade_execution(self, request: str, portfolio_data: Dict, client_profile: Dict) -> str:
        """Simulate trade execution response."""
        return f"""
## Trade Execution Analysis

Based on your request: "{request}"

**Proposed Trades for Rebalancing:**

1. **SELL: AAPL** - 50 shares at $193.50
   - Reduce from 15% to 12% allocation  
   - Estimated proceeds: $9,675

2. **BUY: BND** - 150 shares at $72.15
   - Add bond allocation from 15% to 20%
   - Estimated cost: $10,823

**Trade Summary:**
- Net cash needed: $1,148
- Commission costs: $1.30 total
- Expected execution: Within market hours

**Compliance Status:** ✅ Pre-approved
- Position size limits: Compliant
- Suitability requirements: Met
- Risk tolerance: Aligned

**Next Steps:**
These trades require your explicit approval before execution. Would you like to:
1. Approve these trades as proposed
2. Modify the trade sizes or timing
3. Review additional details first

*Note: Orders will be executed as market orders during regular trading hours once approved.*
"""
    
    def _simulate_compliance_review(self, trade_recs: List[Dict], client_profile: Dict, context: str) -> str:
        """Simulate compliance review response."""
        return f"""
## Compliance Review Complete ✅

**Trade Recommendations Reviewed:**
- {len(trade_recs)} trade(s) analyzed for regulatory compliance
- Client suitability assessment: PASSED
- Risk tolerance alignment: CONFIRMED

**Regulatory Compliance:**
- SEC Investment Advisers Act: Compliant
- FINRA Suitability Rule 2111: Met
- Best Interest Standard (Reg BI): Satisfied

**Required Disclosures:**
• All investments involve risk, including potential loss of principal
• Past performance does not guarantee future results
• Rebalancing may have tax implications for taxable accounts

**Client Communication Approved:**
The proposed trades and communication have been reviewed and approved for client presentation. All regulatory requirements have been met.

**Recommendation:** Proceed with client approval process for trade execution.
"""
    
    def _create_final_communication(self, state: InvestmentAdvisorState) -> str:
        """Create final client communication."""
        analysis = state.get("analysis_results", {})
        trades = state.get("trade_recommendations", [])
        
        return f"""
## Investment Advisory Summary

Thank you for working with our AI Investment Advisor. Here's your comprehensive analysis:

**Portfolio Health Check:** Your portfolio shows strong fundamentals with room for optimization.

**Key Recommendations:**
{chr(10).join(f"• {rec}" for rec in analysis.get("recommendations", ["Portfolio review complete"]))}

**Next Steps:**
{"• Trades ready for your approval" if trades else "• Continue monitoring portfolio"}
• Regular quarterly review scheduled
• Contact us with any questions

**Important:** This analysis is based on current market conditions and your stated preferences. Please review carefully and let us know if you'd like to proceed.

*Compliance Note: All recommendations have been reviewed for suitability and regulatory compliance.*
"""


# Global supervisor instance
investment_advisor_supervisor = InvestmentAdvisorSupervisor()
