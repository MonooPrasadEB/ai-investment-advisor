#!/usr/bin/env python3
"""
AI Investment Advisor CLI Demo

A command-line interface to demonstrate the multi-agent investment advisory system
with real financial data integration and comprehensive compliance checking.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel  
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.text import Text

from .core.supervisor import investment_advisor_supervisor
from .core.config import Config
from .tools.trade_simulator import trade_simulator

console = Console()
config = Config.get_instance()


class InvestmentAdvisorCLI:
    """Command-line interface for the AI Investment Advisor."""
    
    def __init__(self):
        self.supervisor = investment_advisor_supervisor
        self.session_id = None
        self.client_profile = None
        
    def run(self):
        """Main CLI entry point."""
        parser = self._create_parser()
        args = parser.parse_args()
        
        if args.command == "demo":
            self.run_demo(args.query)
        elif args.command == "interactive":
            self.run_interactive()
        elif args.command == "portfolio":
            self.analyze_portfolio(args.assets, args.client_profile)
        elif args.command == "risk":
            self.assess_risk(args.client_profile)
        elif args.command == "validate":
            self.validate_config()
        elif args.command == "rebalance":
            self.generate_rebalancing_plan(args.portfolio, args.client_profile, args.output)
        elif args.command == "simulate":
            self.simulate_trade(args.portfolio, args.symbol, args.action, args.quantity)
        else:
            parser.print_help()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command line argument parser."""
        parser = argparse.ArgumentParser(
            description="AI Investment Advisor - Multi-Agent Financial Advisory System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s demo "Analyze my portfolio with 70%% tech stocks"
  %(prog)s interactive
  %(prog)s portfolio --assets assets.json --client-profile profile.json
  %(prog)s risk --client-profile profile.json
  %(prog)s rebalance --portfolio portfolio.json --output rebalancing_plan.json
  %(prog)s simulate --portfolio portfolio.json --symbol NVDA --action buy --quantity 10
  %(prog)s validate
            """
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Demo command
        demo_parser = subparsers.add_parser("demo", help="Quick demo with sample query")
        demo_parser.add_argument("query", help="Investment question or request")
        
        # Interactive command
        subparsers.add_parser("interactive", help="Interactive conversation mode")
        
        # Portfolio analysis command
        portfolio_parser = subparsers.add_parser("portfolio", help="Analyze portfolio")
        portfolio_parser.add_argument("--assets", help="Path to JSON file with portfolio assets")
        portfolio_parser.add_argument("--client-profile", help="Path to JSON file with client profile")
        
        # Risk assessment command
        risk_parser = subparsers.add_parser("risk", help="Assess risk profile")
        risk_parser.add_argument("--client-profile", help="Path to JSON file with client profile")
        
        # Validate configuration
        subparsers.add_parser("validate", help="Validate API keys and configuration")
        
        # Rebalancing command
        rebalance_parser = subparsers.add_parser("rebalance", help="Generate specific rebalancing trade recommendations")
        rebalance_parser.add_argument("--portfolio", required=True, help="Path to portfolio JSON file")
        rebalance_parser.add_argument("--client-profile", help="Path to client risk profile JSON file (optional)")
        rebalance_parser.add_argument("--output", help="Output file for trade recommendations (optional)")
        
        # Trade simulation command  
        simulate_parser = subparsers.add_parser("simulate", help="Simulate buying or selling specific stocks")
        simulate_parser.add_argument("--portfolio", required=True, help="Path to portfolio JSON file")
        simulate_parser.add_argument("--symbol", required=True, help="Stock symbol (e.g., NVDA, AAPL)")
        simulate_parser.add_argument("--action", required=True, choices=["buy", "sell"], help="Trade action")
        simulate_parser.add_argument("--quantity", required=True, type=int, help="Number of shares")
        
        return parser
    
    def run_demo(self, query: str):
        """Run a quick demo with the provided query."""
        console.print(Panel.fit("🤖 AI Investment Advisor - Demo Mode", style="bold blue"))
        console.print(f"📝 Query: {query}\n")
        
        # Sample client profile and portfolio for demo
        sample_client = self._get_sample_client_profile()
        sample_portfolio = self._get_sample_portfolio()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing request...", total=None)
            
            try:
                result = self.supervisor.process_client_request(
                    request=query,
                    client_profile=sample_client,
                    portfolio_data=sample_portfolio
                )
                
                progress.stop()
                self._display_results(result)
                
            except Exception as e:
                progress.stop()
                console.print(f"[red]❌ Error: {str(e)}[/red]")
                console.print("\n💡 Make sure you have configured your API keys in .env file")
    
    def run_interactive(self):
        """Run interactive conversation mode."""
        console.print(Panel.fit("🤖 AI Investment Advisor - Interactive Mode", style="bold green"))
        console.print("Type 'exit' to quit, 'help' for commands, or ask any investment question.\n")
        
        # Initialize session
        self.session_id = f"interactive_{int(datetime.now().timestamp())}"
        
        # Optional: Load or create client profile - with exit handling
        profile_response = Prompt.ask("Would you like to set up a client profile for personalized advice? (y/n/profile/exit/help)", choices=["y", "yes", "n", "no", "exit", "quit", "help", "profile"], show_choices=False, default="n")
        
        if profile_response.lower() in ['exit', 'quit']:
            console.print("👋 Thank you for using AI Investment Advisor!")
            return
        elif profile_response.lower() == 'help':
            self._show_help()
            return
        elif profile_response.lower() in ['y', 'yes', 'profile']:
            self.client_profile = self._collect_client_profile()
            if self.client_profile is None:
                console.print("💡 Continuing without client profile")
        else:
            self.client_profile = None
        
        # Optional: Load portfolio data - with exit handling
        self.portfolio_data = None
        portfolio_response = Prompt.ask("Would you like to load a portfolio file for personalized analysis? (y/n/portfolio/exit/help)", choices=["y", "yes", "n", "no", "exit", "quit", "help", "portfolio"], show_choices=False, default="n")
        
        if portfolio_response.lower() in ['exit', 'quit']:
            console.print("👋 Thank you for using AI Investment Advisor!")
            return
        elif portfolio_response.lower() == 'help':
            self._show_help()
            return
        elif portfolio_response.lower() in ['y', 'yes', 'portfolio']:
            portfolio_file = Prompt.ask("Enter portfolio file path (or press Enter for sample)", default="")
            if portfolio_file:
                self.portfolio_data = self._load_json_file(portfolio_file)
                if self.portfolio_data:
                    console.print(f"✅ Loaded portfolio: {len(self.portfolio_data.get('assets', []))} holdings, ${self.portfolio_data.get('total_value', 0):,.2f} total value")
                else:
                    console.print("❌ Failed to load portfolio file")
            else:
                self.portfolio_data = self._get_sample_portfolio()
                console.print("📊 Using sample portfolio data")
        else:
            console.print("💡 Proceeding without portfolio data - you can load one later using 'portfolio' command\n")
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    console.print("👋 Thank you for using AI Investment Advisor!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'profile':
                    profile_result = self._collect_client_profile()
                    if profile_result is not None:
                        self.client_profile = profile_result
                        console.print("✅ Client profile updated")
                    else:
                        console.print("💡 Profile setup cancelled")
                    continue
                elif user_input.lower() == 'portfolio':
                    portfolio_file = Prompt.ask("Enter portfolio file path", default="portfolio_aggressive.json")
                    self.portfolio_data = self._load_json_file(portfolio_file)
                    if self.portfolio_data:
                        console.print(f"✅ Loaded portfolio: {len(self.portfolio_data.get('assets', []))} holdings, ${self.portfolio_data.get('total_value', 0):,.2f} total value")
                    else:
                        console.print("❌ Failed to load portfolio file")
                    continue
                
                # Process the request
                with console.status("[bold green]AI Advisor is thinking..."):
                    result = self.supervisor.process_client_request(
                        request=user_input,
                        client_profile=self.client_profile,
                        portfolio_data=self.portfolio_data,
                        session_id=self.session_id
                    )
                
                # Display response
                self._display_interactive_response(result)
                
            except KeyboardInterrupt:
                console.print("\n👋 Thank you for using AI Investment Advisor! (Ctrl+C)")
                break
            except EOFError:
                console.print("\n👋 Thank you for using AI Investment Advisor!")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {str(e)}[/red]")
                console.print("Please try again or type 'help' for assistance.")
    
    def analyze_portfolio(self, assets_file: Optional[str], profile_file: Optional[str]):
        """Analyze portfolio from file inputs."""
        console.print(Panel.fit("📊 Portfolio Analysis", style="bold yellow"))
        
        # Load data from files
        assets_data = self._load_json_file(assets_file) if assets_file else self._get_sample_portfolio()
        client_data = self._load_json_file(profile_file) if profile_file else self._get_sample_client_profile()
        
        query = "Please provide a comprehensive analysis of my portfolio including risk assessment and rebalancing recommendations."
        
        with console.status("[bold green]Analyzing portfolio..."):
            result = self.supervisor.process_client_request(
                request=query,
                client_profile=client_data,
                portfolio_data=assets_data
            )
        
        self._display_results(result)
    
    def assess_risk(self, profile_file: Optional[str]):
        """Assess client risk profile."""
        console.print(Panel.fit("⚖️ Risk Assessment", style="bold red"))
        
        client_data = self._load_json_file(profile_file) if profile_file else None
        
        if not client_data:
            console.print("Collecting risk profile information...\n")
            client_data = self._collect_client_profile()
        
        query = "Please assess my risk tolerance and recommend an appropriate asset allocation based on my profile."
        
        with console.status("[bold green]Assessing risk profile..."):
            result = self.supervisor.process_client_request(
                request=query,
                client_profile=client_data
            )
        
        self._display_results(result)
    
    def validate_config(self):
        """Validate API keys and configuration."""
        console.print(Panel.fit("🔧 Configuration Validation", style="bold magenta"))
        
        # Check API keys
        api_keys = config.validate_api_keys()
        
        table = Table(title="API Key Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Required", style="yellow")
        
        for service, configured in api_keys.items():
            status = "✅ Configured" if configured else "❌ Missing"
            required = "Yes" if service == "openai" else "Optional"
            table.add_row(service.title(), status, required)
        
        console.print(table)
        
        # Check market data configuration
        market_config = config.get_market_data_config()
        console.print(f"\n📈 Market Data Provider: {market_config['provider']}")
        
        # Provide guidance
        if not api_keys["openai"]:
            console.print("\n[red]⚠️ OpenAI API key is required for the system to function.[/red]")
            console.print("Add OPENAI_API_KEY to your .env file")
        else:
            console.print("\n[green]✅ Configuration looks good![/green]")
    
    def _display_results(self, result: Dict):
        """Display comprehensive results from the investment advisor."""
        if "error" in result:
            console.print(f"[red]❌ Error: {result['error']}[/red]")
            return
        
        # Main response
        console.print(Panel(result["response"], title="🤖 AI Investment Advisor", border_style="blue"))
        
        # Analysis results
        if result.get("analysis_results"):
            analysis = result["analysis_results"]
            
            # Portfolio metrics table
            metrics_table = Table(title="📊 Portfolio Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            
            portfolio_metrics = analysis.get("portfolio_metrics", {})
            for metric, value in portfolio_metrics.items():
                metrics_table.add_row(metric.replace('_', ' ').title(), f"{value}")
            
            console.print(metrics_table)
            
            # Recommendations
            if analysis.get("recommendations"):
                console.print("\n💡 Recommendations:")
                for i, rec in enumerate(analysis["recommendations"], 1):
                    console.print(f"   {i}. {rec}")
        
        # Trade recommendations
        if result.get("trade_recommendations"):
            trades_table = Table(title="📈 Trade Recommendations")
            trades_table.add_column("Symbol", style="cyan")
            trades_table.add_column("Action", style="yellow")
            trades_table.add_column("Quantity", style="green")
            trades_table.add_column("Rationale", style="white")
            
            for trade in result["trade_recommendations"]:
                trades_table.add_row(
                    trade["symbol"],
                    trade["action"].upper(),
                    str(trade["quantity"]),
                    trade["rationale"]
                )
            
            console.print(trades_table)
        
        # Approval required
        if result.get("requires_user_approval"):
            console.print("\n[yellow]⚠️ User approval required before trade execution[/yellow]")
        
        # Compliance status
        if result.get("compliance_approved"):
            console.print("\n[green]✅ Compliance approved[/green]")
    
    def _display_interactive_response(self, result: Dict):
        """Display response in interactive mode."""
        if "error" in result:
            console.print(f"[red]❌ {result['error']}[/red]")
            return
        
        # AI response
        ai_text = Text("AI Advisor: ", style="bold blue")
        response_md = Markdown(result["response"])
        console.print(ai_text)
        console.print(response_md)
        
        # Show additional info if available
        if result.get("requires_user_approval"):
            console.print("\n[yellow]💬 This requires your approval. Type 'approve' to proceed.[/yellow]")
    
    def _collect_client_profile(self) -> Optional[Dict]:
        """Collect client profile information interactively."""
        console.print("📋 Client Profile Setup")
        
        try:
            age_response = Prompt.ask("Age", default="35")
            if age_response.lower() in ['exit', 'quit']:
                console.print("👋 Cancelled profile setup")
                return None
            age = int(age_response)
            
            income_response = Prompt.ask("Annual income (USD)", default="75000")
            if income_response.lower() in ['exit', 'quit']:
                console.print("👋 Cancelled profile setup")
                return None
            income = float(income_response)
            
            net_worth_response = Prompt.ask("Net worth (USD)", default="250000")
            if net_worth_response.lower() in ['exit', 'quit']:
                console.print("👋 Cancelled profile setup")
                return None
            net_worth = float(net_worth_response)
        except ValueError as e:
            console.print(f"❌ Invalid input: {e}")
            console.print("Using default values...")
            age, income, net_worth = 35, 75000, 250000
        
        experience = Prompt.ask(
            "Investment experience",
            choices=["beginner", "intermediate", "advanced", "expert", "exit", "quit"],
            default="intermediate"
        )
        if experience.lower() in ['exit', 'quit']:
            console.print("👋 Cancelled profile setup")
            return None
        
        risk_tolerance = Prompt.ask(
            "Risk tolerance",
            choices=["conservative", "moderate", "aggressive", "exit", "quit"],
            default="moderate"
        )
        if risk_tolerance.lower() in ['exit', 'quit']:
            console.print("👋 Cancelled profile setup")
            return None
        
        try:
            horizon_response = Prompt.ask("Investment time horizon (years)", default="20")
            if horizon_response.lower() in ['exit', 'quit']:
                console.print("👋 Cancelled profile setup")
                return None
            time_horizon = int(horizon_response)
        except ValueError:
            console.print("❌ Invalid time horizon, using default (20 years)")
            time_horizon = 20
        
        primary_goal = Prompt.ask(
            "Primary investment goal",
            choices=["retirement", "wealth_building", "income", "preservation", "exit", "quit"],
            default="wealth_building"
        )
        if primary_goal.lower() in ['exit', 'quit']:
            console.print("👋 Cancelled profile setup")
            return None
        
        profile = {
            "age": age,
            "annual_income": income,
            "net_worth": net_worth,
            "investment_experience": experience,
            "risk_tolerance": risk_tolerance,
            "time_horizon": time_horizon,
            "primary_goal": primary_goal
        }
        
        console.print("[green]✅ Profile created successfully![/green]\n")
        return profile
    
    def _show_help(self):
        """Show help information."""
        help_text = """
## Available Commands:
- **help** - Show this help message
- **profile** - Set up or update client profile  
- **portfolio** - Load a portfolio file for analysis
- **exit/quit/bye** - Exit the program
- **Ctrl+C** - Quick exit

## Example Questions:
- "Analyze my portfolio and suggest improvements"
- "What should my asset allocation be for my age?"
- "Should I invest more in tech stocks?"
- "How risky is my current portfolio?"
- "I want to buy 100 shares of AAPL, is that a good idea?"
- "Create a retirement portfolio for someone my age"

## Tips:
- Be specific about your investment goals and timeframe
- Mention your current holdings if you want portfolio analysis
- Ask about specific stocks, sectors, or investment strategies
"""
        console.print(Markdown(help_text))
    
    def _get_sample_client_profile(self) -> Dict:
        """Get sample client profile for demo."""
        return {
            "age": 32,
            "annual_income": 85000,
            "net_worth": 150000,
            "investment_experience": "intermediate",
            "risk_tolerance": "moderate",
            "time_horizon": 25,
            "primary_goal": "wealth_building"
        }
    
    def _get_sample_portfolio(self) -> Dict:
        """Get sample portfolio for demo."""
        return {
            "total_value": 75000,
            "assets": [
                {"symbol": "AAPL", "quantity": 50, "allocation": 25.0, "sector": "Technology"},
                {"symbol": "MSFT", "quantity": 30, "allocation": 20.0, "sector": "Technology"},
                {"symbol": "GOOGL", "quantity": 40, "allocation": 15.0, "sector": "Technology"},
                {"symbol": "SPY", "quantity": 100, "allocation": 25.0, "sector": "Diversified"},
                {"symbol": "BND", "quantity": 150, "allocation": 15.0, "sector": "Fixed Income"}
            ]
        }
    
    def _load_json_file(self, filename: str) -> Dict:
        """Load data from JSON file."""
        try:
            path = Path(filename)
            if not path.exists():
                console.print(f"[red]File not found: {filename}[/red]")
                return {}
            
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading {filename}: {str(e)}[/red]")
            return {}
    
    def generate_rebalancing_plan(self, portfolio_file: str, client_profile_file: Optional[str] = None, output_file: Optional[str] = None):
        """Generate and display specific rebalancing trade recommendations."""
        console.print(Panel.fit("📊 Portfolio Rebalancing Analysis", style="bold green"))
        
        # Load portfolio data
        portfolio_data = self._load_json_file(portfolio_file)
        if not portfolio_data:
            console.print(f"[red]❌ Could not load portfolio from {portfolio_file}[/red]")
            return
        
        # Load client profile if provided
        client_profile = None
        if client_profile_file:
            client_profile = self._load_json_file(client_profile_file)
            if client_profile:
                risk_tolerance = client_profile.get('risk_tolerance', 'unknown')
                console.print(f"👤 Client Profile: {risk_tolerance.title()} risk tolerance")
            else:
                console.print("[yellow]⚠️ Could not load client profile - using default analysis[/yellow]")
        else:
            console.print("[yellow]💡 No client profile provided - using general rebalancing logic[/yellow]")
        
        console.print(f"📁 Portfolio: {len(portfolio_data.get('assets', []))} holdings, ${portfolio_data.get('total_value', 0):,.2f} total value\n")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Analyzing portfolio and generating trade recommendations...", total=None)
                
                # Generate rebalancing plan (with user profile if available)
                rebalancing_plan = trade_simulator.generate_rebalancing_plan(
                    portfolio_data, 
                    client_profile=client_profile
                )
                
                progress.update(task, description="Analysis complete!")
            
            # Display results
            console.print("\n" + "="*60)
            console.print(f"[bold blue]REBALANCING RECOMMENDATIONS[/bold blue]")
            console.print("="*60)
            
            console.print(f"💰 Portfolio Value: [bold]${rebalancing_plan.portfolio_value:,.2f}[/bold]")
            console.print(f"🔄 Recommended Trades: [bold]{rebalancing_plan.total_trades}[/bold]")
            console.print(f"💸 Total Estimated Cost: [bold]${rebalancing_plan.total_cost:,.2f}[/bold]")
            console.print(f"📈 Portfolio Turnover: [bold]{rebalancing_plan.total_turnover:.1%}[/bold]")
            console.print(f"💵 Net Cash Impact: [bold]${rebalancing_plan.net_cash_impact:,.2f}[/bold]")
            console.print()
            
            if rebalancing_plan.trades:
                # Create trades table
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Symbol", style="cyan", width=8)
                table.add_column("Action", style="green", width=6)
                table.add_column("Quantity", justify="right", width=10)
                table.add_column("Price", justify="right", width=10)
                table.add_column("Cost", justify="right", width=12)
                table.add_column("Current %", justify="right", width=10)
                table.add_column("Target %", justify="right", width=10)
                table.add_column("Reason", width=25)
                
                for trade in rebalancing_plan.trades:
                    action_color = "green" if trade.action == "buy" else "red"
                    table.add_row(
                        trade.symbol,
                        f"[{action_color}]{trade.action.upper()}[/{action_color}]",
                        f"{trade.quantity:,}",
                        f"${trade.current_price:.2f}",
                        f"${trade.estimated_cost:,.2f}",
                        f"{trade.current_allocation:.1%}",
                        f"{trade.target_allocation:.1%}",
                        trade.reason
                    )
                
                console.print(table)
                console.print()
                
                # Risk impact summary
                risk_impact = rebalancing_plan.risk_impact
                console.print("[bold yellow]📊 RISK IMPACT ANALYSIS[/bold yellow]")
                console.print(f"• Tech Concentration Change: {risk_impact['tech_concentration_change']:+.1%}")
                console.print(f"• Diversification Improvement: {risk_impact['diversification_improvement']:.1%}")
                console.print(f"• Portfolio Turnover: {risk_impact['turnover_ratio']:.1%}")
                
            else:
                console.print("[green]✅ Portfolio is already well-balanced - no trades recommended![/green]")
            
            # User approval process
            if rebalancing_plan.trades:
                console.print("\n" + "="*60)
                console.print("[bold yellow]💭 AI ADVISOR REASONING[/bold yellow]")
                
                # Try to get the LLM reasoning from logs or regenerate summary
                console.print("The AI analyzed your portfolio and risk profile to generate these recommendations.")
                console.print("Key considerations:")
                console.print("• Current sector concentration and diversification")
                console.print("• Your risk tolerance and investment timeline") 
                console.print("• Position sizing appropriate for portfolio value")
                console.print("• Regulatory compliance and best practices")
                console.print()
                
                # Ask for user approval
                if Confirm.ask("[bold blue]Do you want to proceed with these trade recommendations?[/bold blue]"):
                    console.print("\n[green]✅ Trade recommendations approved![/green]")
                    console.print("[yellow]⚠️ Note: These are simulated recommendations. You would need to execute these trades manually through your broker.[/yellow]")
                    
                    # Export option
                    if output_file:
                        export_path = trade_simulator.export_rebalancing_plan(rebalancing_plan, output_file)
                        console.print(f"\n[green]✅ Rebalancing plan exported to: {export_path}[/green]")
                    else:
                        if Confirm.ask("Export approved rebalancing plan to file?"):
                            output_name = f"approved_rebalancing_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            export_path = trade_simulator.export_rebalancing_plan(rebalancing_plan, output_name)
                            console.print(f"[green]✅ Exported to: {export_path}[/green]")
                else:
                    console.print("\n[red]❌ Trade recommendations declined.[/red]")
                    console.print("You can:")
                    console.print("• Run the analysis again with different parameters")
                    console.print("• Try the interactive mode for more discussion")
                    console.print("• Adjust your risk profile and re-run")
            else:
                # No trades case - still offer export
                if output_file:
                    export_path = trade_simulator.export_rebalancing_plan(rebalancing_plan, output_file)
                    console.print(f"\n[green]✅ Analysis exported to: {export_path}[/green]")
                    
        except Exception as e:
            console.print(f"[red]❌ Error generating rebalancing plan: {str(e)}[/red]")
    
    def simulate_trade(self, portfolio_file: str, symbol: str, action: str, quantity: int):
        """Simulate a specific buy or sell trade."""
        console.print(Panel.fit(f"🎯 Trade Simulation: {action.upper()} {quantity} shares of {symbol.upper()}", style="bold yellow"))
        
        # Load portfolio data
        portfolio_data = self._load_json_file(portfolio_file)
        if not portfolio_data:
            console.print(f"[red]❌ Could not load portfolio from {portfolio_file}[/red]")
            return
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching market data and simulating trade...", total=None)
                
                # Simulate the trade
                result = trade_simulator.simulate_trade(portfolio_data, symbol, action, quantity)
                
                progress.update(task, description="Simulation complete!")
            
            if "error" in result:
                console.print(f"[red]❌ {result['error']}[/red]")
                return
            
            # Display results
            trade_details = result["trade_details"]
            position_impact = result["position_impact"]
            portfolio_impact = result["portfolio_impact"]
            market_data = result["market_data"]
            
            console.print("\n" + "="*50)
            console.print("[bold blue]TRADE SIMULATION RESULTS[/bold blue]")
            console.print("="*50)
            
            # Trade Details
            console.print("[bold green]💼 TRADE DETAILS[/bold green]")
            action_color = "green" if action == "buy" else "red"
            console.print(f"• Action: [{action_color}]{trade_details['action']}[/{action_color}]")
            console.print(f"• Symbol: [bold]{trade_details['symbol']}[/bold]")
            console.print(f"• Quantity: [bold]{trade_details['quantity']:,}[/bold] shares")
            console.print(f"• Price: [bold]${trade_details['price']:.2f}[/bold] per share")
            console.print(f"• Trade Value: [bold]${trade_details['trade_value']:,.2f}[/bold]")
            console.print(f"• Commission: [bold]${trade_details['commission']:.2f}[/bold]")
            console.print(f"• Total Cost: [bold]${trade_details['total_cost']:,.2f}[/bold]")
            
            if trade_details['cash_impact'] < 0:
                console.print(f"• Cash Required: [red]${abs(trade_details['cash_impact']):,.2f}[/red]")
            else:
                console.print(f"• Cash Generated: [green]${trade_details['cash_impact']:,.2f}[/green]")
            
            console.print()
            
            # Position Impact
            console.print("[bold yellow]📊 POSITION IMPACT[/bold yellow]")
            console.print(f"• Current Shares: [bold]{position_impact['current_shares']:,}[/bold]")
            console.print(f"• New Shares: [bold]{position_impact['new_shares']:,}[/bold]")
            console.print(f"• Current Value: [bold]${position_impact['current_value']:,.2f}[/bold]")
            console.print(f"• New Value: [bold]${position_impact['new_value']:,.2f}[/bold]")
            console.print(f"• Current Allocation: [bold]{position_impact['current_allocation']:.2f}%[/bold]")
            console.print(f"• New Allocation: [bold]{position_impact['new_allocation']:.2f}%[/bold]")
            
            change_color = "green" if position_impact['allocation_change'] > 0 else "red"
            console.print(f"• Allocation Change: [{change_color}]{position_impact['allocation_change']:+.2f}%[/{change_color}]")
            console.print()
            
            # Market Data
            console.print("[bold cyan]📈 MARKET DATA[/bold cyan]")
            console.print(f"• Current Price: [bold]${market_data['current_price']:.2f}[/bold]")
            if market_data['market_cap']:
                console.print(f"• Market Cap: [bold]${market_data['market_cap']/1e9:.1f}B[/bold]")
            if market_data['sector']:
                console.print(f"• Sector: [bold]{market_data['sector']}[/bold]")
            if market_data['pe_ratio']:
                console.print(f"• P/E Ratio: [bold]{market_data['pe_ratio']:.1f}[/bold]")
            
            console.print()
            
            # User approval for trade simulation
            console.print("[bold yellow]💭 SIMULATION COMPLETE[/bold yellow]")
            console.print("This simulation shows the expected impact of your proposed trade.")
            console.print()
            
            if Confirm.ask(f"[bold blue]Would you like to proceed with this {action.upper()} order?[/bold blue]"):
                console.print(f"\n[green]✅ {action.upper()} order approved![/green]")
                console.print(f"[yellow]⚠️ Note: This is a simulation. To execute this trade:[/yellow]")
                console.print(f"• Log into your broker (Schwab, Fidelity, etc.)")
                console.print(f"• Place a {action.upper()} order for {quantity:,} shares of {symbol.upper()}")
                console.print(f"• Consider using a limit order at ~${trade_details['price']:.2f}")
                console.print(f"• Expected total cost: ${trade_details['total_cost']:,.2f}")
            else:
                console.print(f"\n[red]❌ {action.upper()} order declined.[/red]")
                console.print("You can:")
                console.print("• Simulate different quantities or prices")
                console.print("• Try the rebalancing feature for comprehensive recommendations")
                console.print("• Use interactive mode to discuss this trade")
            
        except Exception as e:
            console.print(f"[red]❌ Error simulating trade: {str(e)}[/red]")


def main():
    """Main entry point for CLI."""
    try:
        cli = InvestmentAdvisorCLI()
        cli.run()
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
