from mcp.server.fastmcp.server import FastMCP
import yfinance as yf

server = FastMCP("YFinance-MCP")

@server.tool
def ping():
    return {"status": "ok"}

@server.tool
def get_stock_price(ticker: str):
    return {"ticker": ticker, "price": yf.Ticker(ticker).info["regularMarketPrice"]}
