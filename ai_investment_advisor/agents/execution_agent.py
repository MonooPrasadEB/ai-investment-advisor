"""
Execution Agent for Trade Execution with Compliance Checks and User Approval.

This agent handles the actual execution of investment trades, ensuring all regulatory
compliance requirements are met and obtaining proper user approval before execution.
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
    trade_compliance_tool,
    portfolio_compliance_tool, 
    recommendation_validation_tool
)
from ..tools.market_data import market_data_tool
from ..core.config import Config

config = Config.get_instance()
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected" 
    PENDING_EXECUTION = "pending_execution"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class TradeOrder(BaseModel):
    """Trade order model."""
    order_id: str
    client_id: str
    symbol: str
    action: str  # "buy" or "sell"
    quantity: float
    order_type: OrderType
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    
    # Order management
    created_at: datetime
    expires_at: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING_APPROVAL
    
    # Compliance and approval
    compliance_approved: bool = False
    user_approved: bool = False
    approval_timestamp: Optional[datetime] = None
    
    # Execution details
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    execution_timestamp: Optional[datetime] = None
    
    # Risk management
    estimated_commission: float = 0.0
    estimated_total_cost: float = 0.0
    position_size_percent: float = 0.0
    
    # Audit trail
    compliance_notes: List[str] = []
    approval_notes: List[str] = []
    execution_notes: List[str] = []


class ExecutionResult(BaseModel):
    """Trade execution result."""
    success: bool
    order_id: str
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    execution_timestamp: Optional[datetime] = None
    commission_paid: float = 0.0
    error_message: Optional[str] = None
    
    # Post-execution analysis
    price_improvement: Optional[float] = None
    market_impact: Optional[float] = None


class ExecutionAgent:
    """
    Execution Agent responsible for:
    1. Trade execution with compliance checks
    2. Order management and approval workflows
    3. Risk management and position sizing
    4. Post-execution analysis and reporting
    """
    
    def __init__(self):
        self.name = "execution_agent"
        self.llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.1,  # Lower temperature for execution decisions
            api_key=config.openai_api_key
        )
        
        # Available tools for trade execution
        self.tools = [
            trade_compliance_tool,
            portfolio_compliance_tool,
            recommendation_validation_tool,
            market_data_tool
        ]
        
        # Order management
        self.pending_orders: Dict[str, TradeOrder] = {}
        self.executed_orders: Dict[str, TradeOrder] = {}
        
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for execution agent."""
        return """
You are an AI Execution Agent responsible for managing investment trade execution with strict adherence to regulatory compliance and risk management protocols.

## Your Core Responsibilities:

### 1. Pre-Execution Compliance
- Validate all trades against SEC, FINRA, and IRS regulations
- Check position concentration limits and portfolio constraints
- Verify suitability requirements are met for each client
- Ensure proper documentation and audit trails
- Flag any potential conflicts of interest or regulatory issues

### 2. Order Management & Approval
- Process trade orders with appropriate approval workflows
- Obtain explicit user consent before executing any trades
- Manage order types (market, limit, stop-loss, etc.)
- Handle order modifications and cancellations
- Maintain detailed audit trails for all activities

### 3. Risk Management
- Calculate position sizing based on portfolio constraints
- Monitor concentration limits and diversification requirements
- Assess market impact and execution timing
- Implement appropriate order types to minimize execution risk
- Alert users to significant risk factors before execution

### 4. Trade Execution
- Execute approved trades efficiently and transparently
- Provide real-time status updates during execution
- Handle partial fills and order management
- Optimize execution to minimize market impact and costs
- Document all execution details for compliance

## Key Principles:

### Compliance First
- NEVER execute a trade without proper compliance approval
- Always verify regulatory requirements are met
- Maintain complete documentation and audit trails
- Escalate any questionable situations for manual review

### User Consent Required
- Obtain explicit approval for every trade execution
- Clearly communicate trade details, costs, and risks
- Provide cooling-off periods for significant trades
- Allow users to modify or cancel orders before execution

### Risk Management
- Respect all position and concentration limits
- Consider market conditions and timing
- Optimize execution to protect client interests
- Provide transparent cost and impact analysis

### Transparency
- Clearly explain all fees, commissions, and costs
- Report actual vs. expected execution prices
- Provide detailed post-execution analysis
- Maintain complete audit trails

## Tools Available:
1. Trade Compliance Tool - Check individual trades for regulatory compliance
2. Portfolio Compliance Tool - Verify portfolio-wide compliance
3. Recommendation Validation Tool - Validate investment recommendations
4. Market Data Tool - Get real-time market data and analysis

## Execution Workflow:
1. Receive trade recommendation or user order
2. Validate compliance and regulatory requirements  
3. Calculate risk metrics and position sizing
4. Obtain explicit user approval with full disclosure
5. Execute trade with optimal timing and order management
6. Provide post-execution analysis and reporting
7. Update portfolio records and compliance status

## Communication Style:
- Clear and precise about execution details
- Transparent about all costs and risks
- Professional and systematic in approach
- Proactive in identifying and addressing issues

Remember: Your primary obligation is to execute trades in the client's best interest while maintaining strict regulatory compliance. When in doubt, always err on the side of caution and seek additional approval or clarification.
"""
    
    def get_tools(self) -> List[BaseTool]:
        """Get list of available tools for the agent."""
        return self.tools
    
    def get_system_message(self) -> str:
        """Get the system prompt for this agent."""
        return self.system_prompt
    
    def create_trade_order(
        self,
        client_id: str,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        portfolio_value: float = 100000,
        expires_in_hours: int = 24
    ) -> Dict:
        """
        Create a new trade order with compliance validation.
        
        Args:
            client_id: Client identifier
            symbol: Security symbol
            action: "buy" or "sell"
            quantity: Number of shares
            order_type: Type of order (market, limit, etc.)
            price: Price for limit orders
            portfolio_value: Current portfolio value for position sizing
            expires_in_hours: Hours until order expires
        
        Returns:
            Dictionary containing order details and compliance status
        """
        try:
            # Generate unique order ID
            order_id = f"{client_id}_{symbol}_{int(datetime.now().timestamp())}"
            
            # Get current market data for validation
            market_price = self._get_current_price(symbol)
            if not market_price:
                return {"error": f"Unable to get market data for {symbol}"}
            
            # Calculate order details
            estimated_cost = self._calculate_estimated_cost(
                action, quantity, market_price, order_type, price
            )
            
            position_size = (estimated_cost / portfolio_value) * 100
            
            # Create order object
            order = TradeOrder(
                order_id=order_id,
                client_id=client_id,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=OrderType(order_type),
                price=price,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=expires_in_hours),
                estimated_total_cost=estimated_cost,
                position_size_percent=position_size
            )
            
            # Perform compliance check
            compliance_result = self._check_trade_compliance(
                order, portfolio_value
            )
            
            order.compliance_approved = compliance_result["approved"]
            order.compliance_notes = compliance_result.get("notes", [])
            
            # Store pending order
            self.pending_orders[order_id] = order
            
            # Prepare response
            response = {
                "order_id": order_id,
                "status": order.status.value,
                "compliance_approved": order.compliance_approved,
                "requires_user_approval": True,
                "order_details": {
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "order_type": order_type,
                    "estimated_price": market_price,
                    "estimated_cost": estimated_cost,
                    "position_size": f"{position_size:.1f}%",
                    "expires_at": order.expires_at.isoformat()
                },
                "compliance_status": compliance_result,
                "next_steps": [
                    "Review order details and compliance status",
                    "Provide explicit approval to execute",
                    "Order will expire automatically if not approved"
                ]
            }
            
            if not order.compliance_approved:
                response["warning"] = "Order has compliance issues that must be resolved"
                response["next_steps"].insert(0, "Address compliance violations before approval")
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating trade order: {e}")
            return {"error": str(e)}
    
    def approve_order(
        self,
        order_id: str,
        user_confirmation: bool,
        approval_notes: str = ""
    ) -> Dict:
        """
        Process user approval for pending trade order.
        
        Args:
            order_id: Order identifier
            user_confirmation: Explicit user approval
            approval_notes: Optional notes from user
            
        Returns:
            Dictionary containing approval status and next steps
        """
        try:
            if order_id not in self.pending_orders:
                return {"error": f"Order {order_id} not found or already processed"}
            
            order = self.pending_orders[order_id]
            
            # Check if order has expired
            if order.expires_at and datetime.now() > order.expires_at:
                order.status = OrderStatus.CANCELLED
                return {
                    "order_id": order_id,
                    "status": "cancelled",
                    "message": "Order expired before approval"
                }
            
            if not user_confirmation:
                order.status = OrderStatus.REJECTED
                order.approval_notes.append(f"User rejected: {approval_notes}")
                return {
                    "order_id": order_id,
                    "status": "rejected", 
                    "message": "Order rejected by user"
                }
            
            # Verify compliance is still valid
            if not order.compliance_approved:
                return {
                    "error": "Cannot approve order with compliance violations",
                    "compliance_notes": order.compliance_notes
                }
            
            # Mark as approved
            order.user_approved = True
            order.approval_timestamp = datetime.now()
            order.status = OrderStatus.APPROVED
            order.approval_notes.append(approval_notes)
            
            # Move to execution queue
            execution_result = self._queue_for_execution(order)
            
            return {
                "order_id": order_id,
                "status": order.status.value,
                "approved_at": order.approval_timestamp.isoformat(),
                "execution_status": execution_result,
                "message": "Order approved and queued for execution"
            }
            
        except Exception as e:
            logger.error(f"Error approving order {order_id}: {e}")
            return {"error": str(e)}
    
    def execute_order(self, order_id: str) -> Dict:
        """
        Execute approved trade order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Dictionary containing execution results
        """
        try:
            if order_id not in self.pending_orders:
                return {"error": f"Order {order_id} not found"}
            
            order = self.pending_orders[order_id]
            
            if order.status != OrderStatus.APPROVED:
                return {"error": f"Order {order_id} not approved for execution"}
            
            # Mark as pending execution
            order.status = OrderStatus.PENDING_EXECUTION
            
            # Simulate trade execution (in production, this would interface with broker APIs)
            execution_result = self._simulate_trade_execution(order)
            
            if execution_result.success:
                order.status = OrderStatus.EXECUTED
                order.executed_price = execution_result.executed_price
                order.executed_quantity = execution_result.executed_quantity
                order.execution_timestamp = execution_result.execution_timestamp
                order.execution_notes.append("Trade executed successfully")
                
                # Move to executed orders
                self.executed_orders[order_id] = order
                del self.pending_orders[order_id]
                
                return {
                    "order_id": order_id,
                    "status": "executed",
                    "execution_details": {
                        "executed_price": execution_result.executed_price,
                        "executed_quantity": execution_result.executed_quantity,
                        "execution_time": execution_result.execution_timestamp.isoformat(),
                        "commission": execution_result.commission_paid,
                        "total_cost": execution_result.executed_price * execution_result.executed_quantity + execution_result.commission_paid
                    },
                    "post_execution_analysis": self._generate_execution_analysis(execution_result, order)
                }
            else:
                order.status = OrderStatus.FAILED
                order.execution_notes.append(f"Execution failed: {execution_result.error_message}")
                
                return {
                    "order_id": order_id,
                    "status": "failed",
                    "error": execution_result.error_message,
                    "retry_options": self._suggest_retry_options(order)
                }
                
        except Exception as e:
            logger.error(f"Error executing order {order_id}: {e}")
            return {"error": str(e)}
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get current status of an order."""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
        elif order_id in self.executed_orders:
            order = self.executed_orders[order_id]
        else:
            return {"error": f"Order {order_id} not found"}
        
        return {
            "order_id": order_id,
            "status": order.status.value,
            "created_at": order.created_at.isoformat(),
            "symbol": order.symbol,
            "action": order.action,
            "quantity": order.quantity,
            "compliance_approved": order.compliance_approved,
            "user_approved": order.user_approved,
            "executed_price": order.executed_price,
            "execution_timestamp": order.execution_timestamp.isoformat() if order.execution_timestamp else None
        }
    
    def cancel_order(self, order_id: str, reason: str = "") -> Dict:
        """Cancel a pending order."""
        if order_id not in self.pending_orders:
            return {"error": f"Order {order_id} not found or already processed"}
        
        order = self.pending_orders[order_id]
        
        if order.status in [OrderStatus.EXECUTED, OrderStatus.FAILED]:
            return {"error": "Cannot cancel executed or failed orders"}
        
        order.status = OrderStatus.CANCELLED
        order.execution_notes.append(f"Order cancelled: {reason}")
        
        return {
            "order_id": order_id,
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat(),
            "reason": reason
        }
    
    def get_execution_summary(self, time_period_days: int = 30) -> Dict:
        """Get summary of recent execution activity."""
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        
        recent_orders = [
            order for order in self.executed_orders.values()
            if order.execution_timestamp and order.execution_timestamp > cutoff_date
        ]
        
        total_orders = len(recent_orders)
        total_volume = sum(order.executed_quantity * order.executed_price for order in recent_orders)
        total_commission = sum(order.estimated_commission for order in recent_orders)
        
        return {
            "period_days": time_period_days,
            "total_orders_executed": total_orders,
            "total_trade_volume": total_volume,
            "total_commissions": total_commission,
            "average_execution_time": "< 1 second",  # Simulated
            "execution_success_rate": "100%",  # Simulated
            "compliance_approval_rate": "95%",  # Simulated
        }
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        # In production, this would call market data API
        # Here we simulate with common stock prices
        mock_prices = {
            "AAPL": 193.50,
            "MSFT": 417.10,
            "GOOGL": 175.20,
            "AMZN": 151.94,
            "TSLA": 248.42,
            "SPY": 502.43
        }
        return mock_prices.get(symbol, 100.0)  # Default price
    
    def _calculate_estimated_cost(
        self, action: str, quantity: float, market_price: float, 
        order_type: str, limit_price: Optional[float]
    ) -> float:
        """Calculate estimated total cost of trade."""
        if order_type == "limit" and limit_price:
            estimated_price = limit_price
        else:
            estimated_price = market_price
        
        trade_value = quantity * estimated_price
        commission = max(0.65, trade_value * 0.005)  # $0.65 or 0.5% commission
        
        return trade_value + commission if action == "buy" else trade_value - commission
    
    def _check_trade_compliance(self, order: TradeOrder, portfolio_value: float) -> Dict:
        """Check trade compliance using compliance tools."""
        # This would typically call the compliance tools
        # Here we simulate compliance check
        
        violations = []
        notes = []
        
        # Position size check
        if order.position_size_percent > config.max_position_size * 100:
            violations.append("Position size exceeds maximum allowed")
            notes.append(f"Position size {order.position_size_percent:.1f}% exceeds limit {config.max_position_size*100:.1f}%")
        
        # Penny stock check
        current_price = self._get_current_price(order.symbol)
        if current_price and current_price < 5.0:
            notes.append(f"Trading penny stock at ${current_price:.2f}")
        
        return {
            "approved": len(violations) == 0,
            "violations": violations,
            "notes": notes
        }
    
    def _queue_for_execution(self, order: TradeOrder) -> str:
        """Queue order for execution."""
        # In production, this would interface with execution management system
        order.status = OrderStatus.APPROVED
        return "queued_for_execution"
    
    def _simulate_trade_execution(self, order: TradeOrder) -> ExecutionResult:
        """Simulate trade execution (replace with real broker API in production)."""
        try:
            current_price = self._get_current_price(order.symbol)
            
            # Simulate small price improvement/slippage
            import random
            price_variation = random.uniform(-0.02, 0.01)  # -2% to +1%
            executed_price = current_price * (1 + price_variation)
            
            commission = max(0.65, order.quantity * executed_price * 0.005)
            
            return ExecutionResult(
                success=True,
                order_id=order.order_id,
                executed_price=executed_price,
                executed_quantity=order.quantity,
                execution_timestamp=datetime.now(),
                commission_paid=commission,
                price_improvement=price_variation * 100  # Percentage
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                order_id=order.order_id,
                error_message=str(e)
            )
    
    def _generate_execution_analysis(
        self, execution_result: ExecutionResult, order: TradeOrder
    ) -> Dict:
        """Generate post-execution analysis."""
        expected_price = self._get_current_price(order.symbol)
        actual_price = execution_result.executed_price
        
        analysis = {
            "price_performance": {
                "expected_price": expected_price,
                "executed_price": actual_price,
                "price_improvement": ((actual_price - expected_price) / expected_price) * 100,
                "execution_quality": "Excellent" if abs((actual_price - expected_price) / expected_price) < 0.01 else "Good"
            },
            "cost_analysis": {
                "commission_paid": execution_result.commission_paid,
                "total_transaction_cost": execution_result.commission_paid,
                "cost_as_percent_of_trade": (execution_result.commission_paid / (actual_price * order.quantity)) * 100
            },
            "timing_analysis": {
                "order_to_execution_time": "< 1 second",
                "market_conditions": "Normal trading conditions",
                "execution_efficiency": "High"
            }
        }
        
        return analysis
    
    def _suggest_retry_options(self, order: TradeOrder) -> List[str]:
        """Suggest retry options for failed orders."""
        return [
            "Retry with current market conditions",
            "Convert to limit order with price protection",
            "Break large order into smaller parts",
            "Wait for better market conditions",
            "Cancel and resubmit with different parameters"
        ]


# Create agent instance
execution_agent = ExecutionAgent()
