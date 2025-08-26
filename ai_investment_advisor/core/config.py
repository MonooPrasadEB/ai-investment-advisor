"""Configuration management for AI Investment Advisor."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config(BaseSettings):
    """Configuration settings for the AI Investment Advisor."""
    
    # AI Model Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Default model settings
    default_model: str = Field("gpt-3.5-turbo", env="DEFAULT_MODEL")  # Faster model
    temperature: float = Field(0.0, env="TEMPERATURE")  # More deterministic
    max_tokens: int = Field(800, env="MAX_TOKENS")  # Lower latency
    
    # Financial Data APIs
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    fred_api_key: Optional[str] = Field(None, env="FRED_API_KEY")
    polygon_api_key: Optional[str] = Field(None, env="POLYGON_API_KEY")
    
    # Data Settings
    market_data_provider: str = Field("yfinance", env="MARKET_DATA_PROVIDER")  # yfinance, alpha_vantage, etc.
    cache_duration_hours: int = Field(1, env="CACHE_DURATION_HOURS")
    
    # Risk Management
    max_position_size: float = Field(0.25, env="MAX_POSITION_SIZE")  # 25% max position
    max_sector_allocation: float = Field(0.40, env="MAX_SECTOR_ALLOCATION")  # 40% max sector
    risk_free_rate: float = Field(0.03, env="RISK_FREE_RATE")  # 3% risk-free rate
    
    # Compliance Settings
    enable_compliance_checks: bool = Field(True, env="ENABLE_COMPLIANCE_CHECKS")
    require_trade_approval: bool = Field(True, env="REQUIRE_TRADE_APPROVAL")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    # Development Settings
    debug: bool = Field(False, env="DEBUG")
    enable_synthetic_data: bool = Field(True, env="ENABLE_SYNTHETIC_DATA")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }
    
    @classmethod
    def get_instance(cls) -> "Config":
        """Get singleton instance of configuration."""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance
    
    def validate_api_keys(self) -> dict[str, bool]:
        """Validate which API keys are configured."""
        return {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "alpha_vantage": bool(self.alpha_vantage_api_key),
            "fred": bool(self.fred_api_key),
            "polygon": bool(self.polygon_api_key),
        }
    
    def get_market_data_config(self) -> dict:
        """Get market data provider configuration."""
        config = {
            "provider": self.market_data_provider,
            "cache_duration": self.cache_duration_hours,
        }
        
        if self.market_data_provider == "alpha_vantage" and self.alpha_vantage_api_key:
            config["api_key"] = self.alpha_vantage_api_key
        elif self.market_data_provider == "polygon" and self.polygon_api_key:
            config["api_key"] = self.polygon_api_key
            
        return config

# Global configuration instance
config = Config.get_instance()
