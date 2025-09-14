"""
Comprehensive test script to verify the yfinance MCP server functionality
"""

import json
import sys
import os
import asyncio

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yfinance_mcp.server import (
    get_stock_price,
    get_stock_info,
    get_historical_stock_prices,
    get_financial_statement,
    get_moving_averages,
    get_rsi,
    get_macd,
    get_bollinger_bands,
    get_technical_summary,
    add_to_watchlist,
    remove_from_watchlist,
    get_watchlist,
    get_watchlist_prices,
    get_yahoo_finance_news,
    get_recommendations,
    compare_stocks,
    get_holder_info
)

def print_result(title, result):
    data = json.loads(result)
    if data.get("success"):
        print(f"✓ {title}")
        return data["data"]
    else:
        print(f"✗ {title} | Error: {data.get('message', 'Unknown error')}")
        return None
    
  
async def test_all_tools():
    print("🧪 Testing yfinance MCP server tools")
    print("=" * 60)

    # 1. Stock price
    print_result("1. get_stock_price (AAPL)", get_stock_price("AAPL"))

    # 2. Stock info
    print_result("2. get_stock_info (MSFT)", get_stock_info("MSFT"))

    # 3. Historical prices
    print_result("3. get_historical_stock_prices (GOOGL)", get_historical_stock_prices("GOOGL", period="1mo"))

    # 4. Financial statement
    print_result("4. get_financial_statement (AMZN, income_stmt)", get_financial_statement("AMZN", "income_stmt"))

    # 5. Moving averages
    print_result("5. get_moving_averages (NFLX)", get_moving_averages("NFLX"))

    # 6. RSI
    print_result("6. get_rsi (NVDA)", get_rsi("NVDA"))

    # 7. MACD
    print_result("7. get_macd (META)", get_macd("META"))

    # 8. Bollinger Bands
    print_result("8. get_bollinger_bands (TSLA)", get_bollinger_bands("TSLA"))

    # 9. Technical Summary
    print_result("9. get_technical_summary (AMD)", get_technical_summary("AMD"))

    # 10. Add to watchlist
    print_result("10. add_to_watchlist (TSLA)", add_to_watchlist("TSLA"))

    # 11. Get watchlist
    print_result("11. get_watchlist", get_watchlist())

    # 12. Get watchlist prices
    print_result("12. get_watchlist_prices", get_watchlist_prices())

    # 13. Realtime watchlist prices (cached)
    result = await get_holder_info("TSLA", "major_holders")
    print_result("13. get_holder_info", result)

    # 14. Remove from watchlist
    print_result("14. remove_from_watchlist (TSLA)", remove_from_watchlist("TSLA"))

    # 15. News
    print_result("15. get_yahoo_finance_news (AAPL)", get_yahoo_finance_news("AAPL"))

    # 16. Recommendations
    result = await get_recommendations("MSFT", "recommendations")
    print_result("16. get_recommendations (MSFT)", result)

    # 17. Compare stocks
    print_result("17. compare_stocks (AAPL vs MSFT)", compare_stocks("AAPL", "MSFT"))

    print("=" * 60)
    print("✅ All tools tested!")


if __name__ == "__main__":
    asyncio.run(test_all_tools())