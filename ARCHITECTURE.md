# AI Investment Advisor - System Architecture

## Overview
This system integrates **LangChain** (AI components) with **LangGraph** (multi-agent orchestration) to create a sophisticated investment advisory platform.

## Architecture Diagram

```mermaid
graph TB
    subgraph "🧠 LangChain Foundation"
        A1["ChatOpenAI<br/>GPT-4 Interface"]
        A2["ChatPromptTemplate<br/>Structured Prompts"]
        A3["BaseTool<br/>Custom Tool Framework"]
        A4["Messages<br/>HumanMessage, AIMessage"]
    end
    
    subgraph "🔄 LangGraph Orchestration"
        B1["StateGraph<br/>Workflow Manager"]
        B2["create_react_agent<br/>Agent Factory"]
        B3["InvestmentAdvisorState<br/>Shared Context"]
        B4["Routing Logic<br/>Intelligent Handoffs"]
    end
    
    subgraph "🤖 Specialized Agents"
        C1["MultiTaskAgent<br/>📊 Portfolio Analysis<br/>🎯 Risk Assessment<br/>👥 Client Engagement"]
        C2["ExecutionAgent<br/>⚡ Trade Execution<br/>📋 Order Management<br/>✅ Pre-trade Validation"]
        C3["ComplianceReviewer<br/>🔍 Regulatory Checks<br/>📜 SEC/FINRA Compliance<br/>📞 Client Communication"]
    end
    
    subgraph "🛠️ Financial Tools"
        D1["portfolio_analysis_tool<br/>Modern Portfolio Theory<br/>Sharpe Ratio, Beta"]
        D2["risk_assessment_tool<br/>VaR Calculations<br/>Stress Testing"]
        D3["market_data_tool<br/>Real-time Prices<br/>yfinance Integration"]
        D4["trade_compliance_tool<br/>Position Limits<br/>Suitability Rules"]
        D5["trade_simulator<br/>Impact Analysis<br/>Cost Estimation"]
    end
    
    subgraph "💾 Data Layer"
        E1["Market Data APIs<br/>yfinance, Alpha Vantage"]
        E2["Portfolio Data<br/>JSON Configuration"]
        E3["Client Profiles<br/>Risk Tolerance, Goals"]
    end
    
    %% LangChain to LangGraph Integration
    A1 --> B2
    A2 --> C1
    A2 --> C2  
    A2 --> C3
    A3 --> D1
    A3 --> D2
    A3 --> D3
    A3 --> D4
    A3 --> D5
    
    %% LangGraph Agent Creation
    B2 --> C1
    B2 --> C2
    B2 --> C3
    
    %% Agent Tool Usage
    C1 --> D1
    C1 --> D2
    C1 --> D3
    C2 --> D4
    C2 --> D5
    C3 --> D4
    
    %% State Management
    B1 --> B3
    B3 --> B4
    A4 --> B3
    
    %% Data Integration
    D3 --> E1
    C1 --> E2
    C1 --> E3
    
    %% Styling
    classDef langchain fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef langgraph fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef agents fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef tools fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class A1,A2,A3,A4 langchain
    class B1,B2,B3,B4 langgraph
    class C1,C2,C3 agents
    class D1,D2,D3,D4,D5 tools
    class E1,E2,E3 data
```

## Component Details

### 🧠 LangChain Foundation
- **ChatOpenAI**: Interface to GPT-4 for all natural language processing
- **ChatPromptTemplate**: Structured prompt engineering for consistent responses
- **BaseTool**: Framework for creating domain-specific financial tools
- **Messages**: Conversation history and state management

### 🔄 LangGraph Orchestration  
- **StateGraph**: Manages the multi-agent workflow and decision routing
- **create_react_agent**: Combines LangChain LLM with specialized tools
- **InvestmentAdvisorState**: Shared context across all agents
- **Routing Logic**: Intelligent agent handoffs based on request type

### 🤖 Specialized Agents
1. **MultiTaskAgent**: Primary interface for portfolio analysis and client interaction
2. **ExecutionAgent**: Handles trade execution with compliance validation
3. **ComplianceReviewer**: Ensures regulatory compliance and manages communications

### 🛠️ Financial Tools
- Portfolio analysis with modern portfolio theory
- Risk assessment including VaR and stress testing
- Real-time market data integration
- Trade compliance and suitability checking
- Trade simulation and impact analysis

## Workflow Example

```mermaid
sequenceDiagram
    participant User
    participant Supervisor
    participant MultiTask as MultiTaskAgent
    participant Execution as ExecutionAgent
    participant Compliance as ComplianceReviewer
    
    User->>Supervisor: "Should I sell 100 shares of AAPL?"
    Supervisor->>Supervisor: Route to appropriate agent
    Supervisor->>MultiTask: Analyze portfolio impact
    MultiTask->>MultiTask: Use portfolio_analysis_tool
    MultiTask->>Supervisor: Analysis results
    Supervisor->>Execution: Prepare trade execution
    Execution->>Execution: Use trade_simulator
    Execution->>Supervisor: Trade plan ready
    Supervisor->>Compliance: Review for compliance
    Compliance->>Compliance: Use trade_compliance_tool
    Compliance->>Supervisor: Approved ✅
    Supervisor->>User: Complete recommendation with execution plan
```

## Key Integration Points

### File Locations
```
ai_investment_advisor/
├── core/
│   └── supervisor.py          # LangGraph + LangChain integration
├── agents/
│   ├── multi_task_agent.py    # LangChain agents
│   ├── execution_agent.py     # LangChain agents  
│   └── compliance_reviewer.py # LangChain agents
└── tools/
    ├── portfolio_analyzer.py  # LangChain BaseTool
    ├── risk_assessment.py     # LangChain BaseTool
    ├── market_data.py         # LangChain BaseTool
    └── trade_simulator.py     # LangChain BaseTool
```

### State Flow
1. **User Input** → LangChain Messages
2. **Routing** → LangGraph StateGraph  
3. **Processing** → LangChain Agents + Tools
4. **Coordination** → LangGraph State Management
5. **Response** → LangChain Formatted Output

## Benefits of This Architecture

- **Modularity**: Each agent specializes in specific financial domains
- **Scalability**: Easy to add new agents or tools
- **Reliability**: Built-in state management and error handling
- **Compliance**: Regulatory checks built into the workflow
- **Conversational**: Natural language interface for complex financial operations
