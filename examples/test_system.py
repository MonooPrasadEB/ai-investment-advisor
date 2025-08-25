#!/usr/bin/env python3
"""
Test script for the AI Investment Advisor system.

This script validates the system setup and runs basic functionality tests.
"""

import json
import sys
from pathlib import Path
import logging

# Add the parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from ai_investment_advisor.core.config import Config
from ai_investment_advisor.core.supervisor import investment_advisor_supervisor
from ai_investment_advisor.tools.market_data import market_data_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_configuration():
    """Test system configuration."""
    print("🔧 Testing Configuration...")
    
    try:
        config = Config.get_instance()
        api_keys = config.validate_api_keys()
        
        print(f"  ✅ Configuration loaded successfully")
        print(f"  📊 Market data provider: {config.market_data_provider}")
        
        # Check API keys
        if api_keys["openai"]:
            print("  ✅ OpenAI API key configured")
        else:
            print("  ❌ OpenAI API key missing (required)")
            return False
            
        for service, configured in api_keys.items():
            if service != "openai" and configured:
                print(f"  ✅ {service.title()} API key configured")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def test_market_data():
    """Test market data functionality."""
    print("\n📈 Testing Market Data...")
    
    try:
        # Test getting stock info
        stock_info = market_data_service.get_stock_info("AAPL")
        if stock_info:
            print(f"  ✅ Retrieved AAPL data: ${stock_info.current_price:.2f}")
        else:
            print("  ⚠️  Could not retrieve AAPL data (may be after hours)")
        
        # Test bulk price retrieval
        prices = market_data_service.get_multiple_prices(["AAPL", "MSFT", "GOOGL"])
        if prices:
            print(f"  ✅ Retrieved multiple stock prices: {len(prices)} symbols")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Market data error: {e}")
        return False


def test_portfolio_analysis():
    """Test portfolio analysis functionality."""
    print("\n📊 Testing Portfolio Analysis...")
    
    try:
        # Load sample portfolio
        sample_portfolio_path = Path(__file__).parent / "sample_portfolio.json"
        sample_client_path = Path(__file__).parent / "sample_client_moderate.json"
        
        if not sample_portfolio_path.exists():
            print("  ⚠️  Sample portfolio file not found - creating basic test data")
            portfolio_data = {
                "total_value": 100000,
                "assets": [
                    {"symbol": "AAPL", "quantity": 100, "allocation": 50},
                    {"symbol": "SPY", "quantity": 100, "allocation": 50}
                ]
            }
            client_profile = {
                "age": 35,
                "risk_tolerance": "moderate",
                "time_horizon": 20
            }
        else:
            with open(sample_portfolio_path) as f:
                portfolio_data = json.load(f)
            
            if sample_client_path.exists():
                with open(sample_client_path) as f:
                    client_profile = json.load(f)
            else:
                client_profile = {"age": 35, "risk_tolerance": "moderate"}
        
        # Test the supervisor system
        result = investment_advisor_supervisor.process_client_request(
            request="Please analyze my portfolio and provide recommendations.",
            client_profile=client_profile,
            portfolio_data=portfolio_data
        )
        
        if result and "response" in result and not result.get("error"):
            print("  ✅ Portfolio analysis completed successfully")
            print(f"  📝 Response length: {len(result['response'])} characters")
            return True
        else:
            print(f"  ❌ Portfolio analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"  ❌ Portfolio analysis error: {e}")
        return False


def run_interactive_test():
    """Run an interactive test with user input."""
    print("\n🤖 Interactive Test Mode")
    print("Ask the AI Investment Advisor a question (or press Enter to skip):")
    
    user_question = input("> ").strip()
    
    if not user_question:
        print("  ⏭️  Skipping interactive test")
        return True
    
    try:
        print("  🤔 AI is thinking...")
        result = investment_advisor_supervisor.process_client_request(
            request=user_question,
            client_profile={"age": 35, "risk_tolerance": "moderate"}
        )
        
        if result and "response" in result:
            print(f"\n  🤖 AI Response:")
            print(f"  {result['response']}")
            return True
        else:
            print(f"  ❌ Failed to get response: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"  ❌ Interactive test error: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 AI Investment Advisor System Test")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Market Data", test_market_data), 
        ("Portfolio Analysis", test_portfolio_analysis),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Run interactive test
    interactive_result = run_interactive_test()
    results.append(("Interactive Test", interactive_result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have created a .env file with your OpenAI API key")
        print("2. Install all required dependencies: pip install -e .")
        print("3. Check your internet connection for market data")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
