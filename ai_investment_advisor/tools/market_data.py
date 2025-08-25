"""Market data service with real financial data integration."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import yfinance as yf
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.config import Config

config = Config.get_instance()

logger = logging.getLogger(__name__)

class StockInfo(BaseModel):
    """Stock information model."""
    symbol: str
    name: str
    current_price: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None

class HistoricalData(BaseModel):
    """Historical price data model."""
    symbol: str
    start_date: datetime
    end_date: datetime
    prices: List[Dict[str, Union[float, str]]]  # Date, Open, High, Low, Close, Volume
    returns: List[float]
    volatility: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: float

class MarketDataService:
    """Service for fetching real market data."""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=config.cache_duration_hours)
    
    def get_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """Get comprehensive stock information."""
        cache_key = f"info_{symbol}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            stock_info = StockInfo(
                symbol=symbol.upper(),
                name=info.get('longName', symbol),
                current_price=info.get('currentPrice', 0.0),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                dividend_yield=info.get('dividendYield'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                beta=info.get('beta'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
                fifty_two_week_low=info.get('fiftyTwoWeekLow')
            )
            
            # Cache result
            self.cache[cache_key] = (stock_info, datetime.now())
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return None
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[HistoricalData]:
        """Get historical price data and calculate metrics."""
        cache_key = f"hist_{symbol}_{period}_{start_date}_{end_date}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                hist = ticker.history(start=start_date, end=end_date)
            else:
                hist = ticker.history(period=period)
            
            if hist.empty:
                return None
            
            # Calculate returns and metrics
            hist['Returns'] = hist['Close'].pct_change().dropna()
            returns = hist['Returns'].dropna().tolist()
            
            # Calculate volatility (annualized)
            volatility = float(hist['Returns'].std() * np.sqrt(252))
            
            # Calculate Sharpe ratio
            mean_return = float(hist['Returns'].mean() * 252)  # Annualized
            sharpe_ratio = (mean_return - config.risk_free_rate) / volatility if volatility > 0 else None
            
            # Calculate maximum drawdown
            cumulative = (1 + hist['Returns']).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = float(drawdown.min())
            
            # Convert to list of dictionaries
            prices = []
            for date, row in hist.iterrows():
                prices.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                })
            
            historical_data = HistoricalData(
                symbol=symbol.upper(),
                start_date=hist.index[0].to_pydatetime(),
                end_date=hist.index[-1].to_pydatetime(),
                prices=prices,
                returns=returns,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            
            # Cache result
            self.cache[cache_key] = (historical_data, datetime.now())
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols."""
        prices = {}
        
        try:
            # Use yfinance's bulk download for efficiency
            data = yf.download(symbols, period="1d", interval="1d", group_by="ticker", auto_adjust=True, prepost=True)
            
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        current_price = float(data['Close'].iloc[-1])
                    else:
                        current_price = float(data[symbol]['Close'].iloc[-1])
                    prices[symbol.upper()] = current_price
                except Exception as e:
                    logger.warning(f"Could not get price for {symbol}: {e}")
                    prices[symbol.upper()] = 0.0
                    
        except Exception as e:
            logger.error(f"Error fetching bulk prices: {e}")
            # Fallback to individual requests
            for symbol in symbols:
                info = self.get_stock_info(symbol)
                prices[symbol.upper()] = info.current_price if info else 0.0
        
        return prices
    
    def get_sector_performance(self) -> Dict[str, Dict[str, float]]:
        """Get sector ETF performance data."""
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Communication Services': 'XLC',
            'Industrials': 'XLI',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }
        
        performance = {}
        
        for sector, etf in sector_etfs.items():
            hist_data = self.get_historical_data(etf, period="1y")
            if hist_data:
                # Calculate YTD return
                ytd_return = (hist_data.prices[-1]['close'] / hist_data.prices[0]['close']) - 1
                performance[sector] = {
                    'ytd_return': float(ytd_return),
                    'volatility': hist_data.volatility,
                    'sharpe_ratio': hist_data.sharpe_ratio or 0.0,
                    'max_drawdown': hist_data.max_drawdown
                }
        
        return performance
    
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for stocks by name or symbol."""
        # This is a simplified implementation
        # In production, you'd use a proper search API
        common_stocks = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
            {'symbol': 'BRK-B', 'name': 'Berkshire Hathaway Inc.'},
            {'symbol': 'V', 'name': 'Visa Inc.'},
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson'},
        ]
        
        query_lower = query.lower()
        matches = []
        
        for stock in common_stocks:
            if (query_lower in stock['symbol'].lower() or 
                query_lower in stock['name'].lower()):
                matches.append(stock)
        
        return matches[:limit]

# Global instance
market_data_service = MarketDataService()

class MarketDataInput(BaseModel):
    """Input model for market data tool."""
    symbols: List[str] = Field(..., description="List of stock symbols to analyze")
    period: str = Field("1y", description="Time period for historical data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)")
    include_historical: bool = Field(True, description="Whether to include historical data analysis")

class MarketDataTool(BaseTool):
    """Tool for fetching real market data."""
    name: str = "get_market_data"
    description: str = """
    Fetch comprehensive market data including current prices, historical performance,
    and key financial metrics for stocks, ETFs, and other securities.
    """
    args_schema: type = MarketDataInput

    def _run(self, symbols: List[str], period: str = "1y", include_historical: bool = True) -> str:
        """Execute the market data fetch."""
        try:
            result = {
                'current_data': [],
                'historical_analysis': {},
                'sector_performance': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Get current stock info
            for symbol in symbols:
                stock_info = market_data_service.get_stock_info(symbol)
                if stock_info:
                    result['current_data'].append(stock_info.dict())
            
            # Get historical data if requested
            if include_historical:
                for symbol in symbols:
                    hist_data = market_data_service.get_historical_data(symbol, period)
                    if hist_data:
                        # Summarize historical data (don't return all price points)
                        result['historical_analysis'][symbol] = {
                            'symbol': hist_data.symbol,
                            'period': period,
                            'start_date': hist_data.start_date.isoformat(),
                            'end_date': hist_data.end_date.isoformat(),
                            'total_return': ((hist_data.prices[-1]['close'] / hist_data.prices[0]['close']) - 1) * 100,
                            'volatility': hist_data.volatility * 100,  # Convert to percentage
                            'sharpe_ratio': hist_data.sharpe_ratio,
                            'max_drawdown': hist_data.max_drawdown * 100,  # Convert to percentage
                            'latest_price': hist_data.prices[-1]['close']
                        }
            
            # Add sector performance context
            result['sector_performance'] = market_data_service.get_sector_performance()
            
            return f"Market data retrieved successfully:\n{pd.DataFrame(result['current_data']).to_string()}\n\nHistorical Analysis:\n{result['historical_analysis']}"
            
        except Exception as e:
            logger.error(f"Error in market data tool: {e}")
            return f"Error fetching market data: {str(e)}"

# Tool instance
market_data_tool = MarketDataTool()
