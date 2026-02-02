# AI_trading_base

An autonomous AI trading agent designed for the Base L2 blockchain. It trades KVCM/USDC on Aerodrome using a hybrid strategy that combines RSI (Wilder's Smoothing) with Trend Analysis (SMA-50) to capture medium-term moves while minimizing noise.

Powered by LangGraph, Grok-3 (xAI), and Web3.py.

🚀 Key Features
Intelligent Strategy:

Trend Aware: Uses a 50-period SMA to determine if the market is in an Uptrend or Downtrend.

Contextual RSI: Adjusts buy/sell thresholds dynamically based on the trend (e.g., buys dips earlier in uptrends, holds longer for profits).

EWMA Smoothing: Uses Exponential Weighted Moving Average (Wilder's Smoothing) for RSI to filter out short-term price jitters.


Robust Execution:

Retry Logic: Automatically retries failed transactions with exponential backoff to handle network spikes.

Nonce Management: Local state tracking prevents "Nonce too low" errors during high-frequency checks.

Route Handling: Standardized Aerodrome V2 routing ensures correct Factory address usage.


Safety First:

Capital Protection: Hardcoded $100 USD limit per trade (configurable).

Gas Monitoring: Checks Native ETH balance before every cycle.

Slippage Protection: Defaults to 0.5% max slippage.

Allowance Optimization: Checks token allowance before approving to save gas.

Persistent State: Saves price history to a local file to maintain context across restarts.

🛠️ Tech Stack
Language: Python 3.10+

Blockchain: Base (Coinbase L2)

DEX: Aerodrome V2

AI Core: LangGraph + LangChain + xAI (Grok-3)

Data Analysis: Pandas (EWMA/SMA), NumPy
