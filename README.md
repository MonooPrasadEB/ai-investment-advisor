# AI Investment Advisor - Multi-Agent System

A sophisticated multi-agent AI system that combines cutting-edge artificial intelligence with quantitative finance to deliver personalized investment advice at scale. Built with Python, LangGraph, and real financial data integration.

## 🎯 Overview

This system implements a **3-agent architecture** for comprehensive investment management, replacing traditional robo-advisors' static recommendations with dynamic, conversational financial guidance.

### Core Agents

1. **🧠 Multi-Task Agent**: Portfolio analysis, risk evaluation, and customer engagement
2. **⚡ Execution Agent**: Trade execution with regulatory compliance and user approval workflows
3. **✅ Compliance Reviewer**: SEC/FINRA/IRS policy validation and client communication optimization

## 🚀 Key Features

- **Real Financial Data**: Integration with yfinance, Alpha Vantage, and FRED for live market data
- **Quantitative Analysis**: Modern portfolio theory, VaR calculations, and stress testing
- **Regulatory Compliance**: Automated SEC, FINRA, and IRS compliance checking
- **Risk Assessment**: Interactive risk profiling with scenario analysis
- **Multi-Agent Coordination**: LangGraph-powered agent orchestration
- **Plain English Communication**: Complex financial concepts explained clearly

## 🛠 Technology Stack

- **Python 3.9+**: Core application language
- **LangChain/LangGraph**: Multi-agent framework and LLM orchestration
- **yfinance**: Real-time market data
- **pandas/numpy**: Financial data analysis
- **scipy**: Portfolio optimization
- **Rich**: Beautiful CLI interface
- **OpenAI GPT-4**: Conversational AI capabilities

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-investment-advisor

# Install dependencies
pip install -e .

# Or install with optional advanced features
pip install -e ".[dev,jupyter,advanced-finance]"
```

### 2. Configuration

Create a `.env` file with your API keys:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional for enhanced data
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FRED_API_KEY=your_fred_key
```

### 3. Run Examples

```bash
# Quick demo
ai-advisor demo "Analyze my portfolio with 70% tech stocks"

# Interactive mode
ai-advisor interactive

# Portfolio analysis
ai-advisor portfolio --assets sample_portfolio.json

# Risk assessment
ai-advisor risk --client-profile sample_client.json

# Validate configuration
ai-advisor validate
```

## 📊 Usage Examples

### Portfolio Analysis
```python
from ai_investment_advisor import investment_advisor_supervisor

# Sample portfolio data
portfolio_data = {
    "total_value": 100000,
    "assets": [
        {"symbol": "AAPL", "quantity": 100, "allocation": 40},
        {"symbol": "SPY", "quantity": 200, "allocation": 60}
    ]
}

# Client profile
client_profile = {
    "age": 35,
    "risk_tolerance": "moderate", 
    "time_horizon": 20,
    "annual_income": 85000
}

# Get comprehensive analysis
result = investment_advisor_supervisor.process_client_request(
    request="Analyze my portfolio and suggest improvements",
    client_profile=client_profile,
    portfolio_data=portfolio_data
)

print(result["response"])
```

## 🏗 Architecture

### Multi-Agent Workflow
```
   User Request
        ↓
   Supervisor Agent
        ↓
  ┌─────┼─────┐
  ↓     ↓     ↓
Multi- Exec- Comp-
Task   ution liance
Agent  Agent Reviewer
        ↓
   Final Response
```

### Project Structure
```
ai_investment_advisor/
├── agents/                     # Core AI agents
│   ├── multi_task_agent.py    # Portfolio analysis & customer engagement  
│   ├── execution_agent.py     # Trade execution & order management
│   └── compliance_reviewer.py # Regulatory compliance & communication
├── tools/                      # Financial analysis tools
│   ├── market_data.py         # Real-time market data integration
│   ├── portfolio_analyzer.py  # Advanced portfolio analysis
│   ├── risk_assessment.py     # Risk profiling & stress testing
│   └── compliance_checker.py  # SEC/FINRA/IRS compliance validation
├── core/                       # System core
│   ├── config.py              # Configuration management
│   └── supervisor.py          # Multi-agent coordination
└── cli.py                      # Command-line interface
```

## 🔬 Advanced Features

### Quantitative Finance
- Modern Portfolio Theory optimization
- Value-at-Risk (VaR) and Expected Shortfall calculations
- Monte Carlo simulations for stress testing
- Sharpe ratio, Sortino ratio, and other performance metrics
- Factor model analysis and attribution

### Regulatory Compliance
- **SEC Investment Advisers Act** compliance checking
- **FINRA Rule 2111** suitability requirements
- **Regulation BI** best interest standards
- Automated disclosure generation
- Audit trail maintenance

### Risk Management
- Interactive risk tolerance assessment
- Scenario-based stress testing (market crashes, interest rate shocks)
- Behavioral finance considerations (loss aversion, overconfidence)
- Position sizing and concentration limits
- Diversification analysis across sectors and asset classes

## 🧪 Development

### Setup Development Environment
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Code formatting
black ai_investment_advisor/
isort ai_investment_advisor/

# Type checking
mypy ai_investment_advisor/
```

### Example Data Files

Create sample JSON files for testing:

**sample_portfolio.json**:
```json
{
  "total_value": 250000,
  "assets": [
    {"symbol": "AAPL", "quantity": 100, "allocation": 20, "sector": "Technology"},
    {"symbol": "MSFT", "quantity": 50, "allocation": 15, "sector": "Technology"},
    {"symbol": "SPY", "quantity": 300, "allocation": 45, "sector": "Diversified"},
    {"symbol": "BND", "quantity": 200, "allocation": 20, "sector": "Fixed Income"}
  ]
}
```

**sample_client.json**:
```json
{
  "age": 35,
  "annual_income": 85000,
  "net_worth": 250000,
  "investment_experience": "intermediate",
  "risk_tolerance": "moderate",
  "time_horizon": 25,
  "primary_goal": "wealth_building"
}
```

## 📈 Real Financial Data

The system integrates with multiple financial data providers:

- **yfinance**: Real-time stock prices and historical data
- **Alpha Vantage**: Professional-grade financial data API
- **FRED**: Economic data from the Federal Reserve
- **Market sectors**: Real-time sector performance analysis
- **Risk-free rates**: Current Treasury rates for calculations

## 🎓 Masters Capstone Project

This project demonstrates:

- **Advanced AI Integration**: Multi-agent systems with LangGraph
- **Financial Domain Expertise**: Quantitative finance and regulatory compliance
- **Real-world Application**: Production-ready financial advisory system
- **Scalable Architecture**: Handles millions of users without human advisors
- **Educational Impact**: Makes sophisticated investment management accessible


## Architecture
graph TB
    subgraph "LangChain Components"
        A1["ChatOpenAI<br/>(GPT-4 Interface)"]
        A2["ChatPromptTemplate<br/>(Structured Prompts)"]
        A3["BaseTool<br/>(Custom Tools)"]
        A4["Messages<br/>(HumanMessage, AIMessage)"]
    end
    
    subgraph "LangGraph Orchestration"
        B1["StateGraph<br/>(Workflow Manager)"]
        B2["create_react_agent<br/>(Agent Factory)"]
        B3["InvestmentAdvisorState<br/>(Shared State)"]
        B4["Workflow Nodes<br/>(supervisor, portfolio_analysis, etc.)"]
    end
    
    subgraph "Your Custom Agents"
        C1["MultiTaskAgent<br/>(Portfolio Analysis)"]
        C2["ExecutionAgent<br/>(Trade Execution)"]
        C3["ComplianceReviewer<br/>(Regulatory Checks)"]
    end
    
    subgraph "Custom Tools"
        D1["portfolio_analysis_tool"]
        D2["risk_assessment_tool"]
        D3["trade_compliance_tool"]
        D4["market_data_tool"]
    end
    
    A1 --> B2
    A2 --> C1
    A2 --> C2
    A2 --> C3
    A3 --> D1
    A3 --> D2
    A3 --> D3
    A3 --> D4
    
    B2 --> C1
    B2 --> C2
    B2 --> C3
    
    C1 --> D1
    C1 --> D2
    C1 --> D4
    C2 --> D3
    C3 --> D3
    
    B1 --> B4
    B3 --> B4
    A4 --> B3

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch  
5. Create a Pull Request

## 📧 Contact

For questions about this Masters Capstone project or the AI Investment Advisor system, please reach out through the repository issues.
