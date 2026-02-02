import os
import time
import schedule
import traceback
import pickle
import warnings
import requests
import numpy as np  # For RSI calculation
import random
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain_core.tools import Tool
from langchain_xai import ChatXAI
from langchain_core.callbacks import StdOutCallbackHandler
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Other libraries 
from web3 import Web3
from collections import deque

load_dotenv()

# ────────────────────────────────────────────────
# 1. Environment & Setup
# ────────────────────────────────────────────────
private_key = os.getenv('PRIVATE_KEY')
rpc_url = os.getenv('BASE_RPC_URL')
xai_key = os.getenv('XAI_API_KEY')

if not private_key:
    raise ValueError("PRIVATE_KEY not found in .env file! Please add it.")
if not rpc_url:
    raise ValueError("BASE_RPC_URL not found in .env! Please add it.")
if not xai_key:
    raise ValueError("XAI_API_KEY not found in .env! Please add it.")

print("Environment variables loaded successfully:")
print(f"  RPC URL: {rpc_url[:30]}...")
print(f"  XAI key: {'present' if xai_key else 'missing'}")

w3 = Web3(Web3.HTTPProvider(rpc_url))
account = w3.eth.account.from_key(private_key)

MAX_USD_VALUE = 100

# Tokens & Addresses
KVCM_ADDRESS = "0x00fBAC94Fec8D4089d3fe979F39454F48c71A65d"
USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
FACTORY_ADDRESS_RAW = "0x420DD381b31aEf6683db6B902084cB0FFECe40Da"
AERODROME_ROUTER_ADDRESS = "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"

# Checksums
USDC_ADDRESS_CHECKSUM = Web3.to_checksum_address(USDC_ADDRESS)
KVCM_ADDRESS_CHECKSUM = Web3.to_checksum_address(KVCM_ADDRESS)
FACTORY_ADDRESS_CHECKSUM = Web3.to_checksum_address(FACTORY_ADDRESS_RAW)

# ────────────────────────────────────────────────
# 2. ABIs (Fixed balanceOf issue)
# ────────────────────────────────────────────────

# Complete ERC20 ABI including balanceOf and allowance
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}, {"name": "_spender", "type": "address"}],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]

# Partial Aerodrome ABI
AERODROME_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"components":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"bool","name":"stable","type":"bool"},{"internalType":"address","name":"factory","type":"address"}],"internalType":"struct IRouter.Route[]","name":"routes","type":"tuple[]"}],"name":"getAmountsOut","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"components":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"bool","name":"stable","type":"bool"},{"internalType":"address","name":"factory","type":"address"}],"internalType":"struct IRouter.Route[]","name":"routes","type":"tuple[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

router = w3.eth.contract(address=AERODROME_ROUTER_ADDRESS, abi=AERODROME_ABI)

# Fetch Decimals (Safe Fetch)
KVCM_DECIMALS = 18 
try:
    kvcm = w3.eth.contract(KVCM_ADDRESS_CHECKSUM, abi=ERC20_ABI)
    KVCM_DECIMALS = kvcm.functions.decimals().call()
    print(f"[Startup] kVCM decimals detected: {KVCM_DECIMALS}")
except Exception as e:
    print(f"[Startup] Could not fetch kVCM decimals: {e} → using fallback option of 18")

# Price Persistence
prices = deque(maxlen=300) ##stores 5 hours of data (300 minutes)
PERSIST_FILE = "prices_RSI.pkl"
if os.path.exists(PERSIST_FILE):
    try:
        with open(PERSIST_FILE, 'rb') as f:
            loaded_list = pickle.load(f)
            prices = deque(loaded_list, maxlen=20)
        print(f"[Startup] Loaded {len(prices)} persisted prices from {PERSIST_FILE}")
    except Exception as e:
        print(f"[WARNING] Failed to load persisted prices: {str(e)} → starting fresh")
else:
    print("[Startup] No persisted prices file found → starting fresh")


# ────────────────────────────────────────────────
# 3. Helper Functions (Defined BEFORE use)
# ────────────────────────────────────────────────

def get_eip1559_gas_params():
    try:
        base_fee = w3.eth.get_block('latest')['baseFeePerGas']
        priority_fee = w3.eth.max_priority_fee
        return {
            'maxFeePerGas': base_fee * 2 + priority_fee,
            'maxPriorityFeePerGas': priority_fee
        }
    except Exception as e:
        print(f"[WARNING] Failed to fetch dynamic gas params: {e} → using fallback")
        return {
            'maxFeePerGas': w3.to_wei('0.5', 'gwei'),
            'maxPriorityFeePerGas': w3.to_wei('0.05', 'gwei')
        }

class NonceManager:
    def __init__(self, w3, account_address):
        self.w3 = w3
        self.address = account_address
        self.reset()

    def get_nonce(self):
        current_pending = self.w3.eth.get_transaction_count(self.address, 'pending')
        if current_pending > self._nonce:
            self._nonce = current_pending
        nonce = self._nonce
        self._nonce += 1
        return nonce

    def reset(self):
        self._nonce = self.w3.eth.get_transaction_count(self.address, 'pending')

nonce_manager = NonceManager(w3, account.address)

def reset_nonce_manager(_):
    nonce_manager.reset()
    return "Nonce manager resynced with blockchain."

def send_transaction_with_retries(signed_tx, max_retries=3): 
    attempt = 0
    base_delay = 2 
    while attempt < max_retries:
        try:
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            print(f"[Retry Loop] Attempt {attempt + 1}: Transaction sent. Hash: {tx_hash.hex()}")
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
            
            if receipt.status == 1:
                print(f"[Retry Loop] Success! Transaction confirmed in block {receipt.blockNumber}")
                return receipt
            else:
                print(f"[Retry Loop] Attempt {attempt + 1} REVERTED.")
                
        except Exception as e:
            error_msg = str(e).lower()
            print(f"[Retry Loop] Attempt {attempt + 1} failed: {e}")
            
            if "nonce too low" in error_msg or "already known" in error_msg:
                print("[Retry Loop] Critical Error: Nonce sync issue. Aborting loop.")
                break

        delay = (base_delay * (2 ** attempt)) + (random.uniform(0, 1))
        print(f"[Retry Loop] Waiting {delay:.2f}s before next attempt...")
        time.sleep(delay)
        attempt += 1
        
    return None

def get_aerodrome_route(from_token, to_token):
    return [{
        "from": Web3.to_checksum_address(from_token),
        "to": Web3.to_checksum_address(to_token),
        "stable": False,
        "factory": FACTORY_ADDRESS_CHECKSUM
    }]

def get_expected_out(amount_in, from_token, to_token):
    routes = get_aerodrome_route(from_token, to_token)
    try:
        amounts = router.functions.getAmountsOut(amount_in, routes).call()
        return amounts[-1]
    except Exception as e:
        raise Exception(f"Aerodrome Quote failed: {str(e)}")

def approve_token(token_address, amount):
    try:
        token = w3.eth.contract(address=token_address, abi=ERC20_ABI)
        gas_params = get_eip1559_gas_params()
        current_nonce = nonce_manager.get_nonce()
        
        tx = token.functions.approve(AERODROME_ROUTER_ADDRESS, amount).build_transaction({
            'from': account.address,
            'gas': 80000,
            **gas_params,
            'nonce': current_nonce
        })
        
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        return "Approval successful" if receipt.status == 1 else "Approval failed"
    except Exception as e:
        return f"Approval failed: {str(e)}"

# ────────────────────────────────────────────────
# 4. Market & Trading Functions
# ────────────────────────────────────────────────

def get_current_kvcm_price_usd():
    """Best-effort price lookup for calculation logic."""
    try:
        url = "https://api.dexscreener.com/latest/dex/pairs/base/0x5c0d76fab1822bdeb47308ed6028231761ed723e"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if 'pair' not in data or 'priceUsd' not in data['pair']:
            raise ValueError("Invalid DexScreener response structure")
        price_usd = float(data['pair']['priceUsd'])
        return price_usd
    except Exception as e:
        print(f"[DexScreener price fetch error] {str(e)}")
        if len(prices) > 0:
            return prices[-1]
        return None

def get_current_price(_):
    """Fetch current KVCM price, update history, and persist (Agent Tool)."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = "https://api.dexscreener.com/latest/dex/pairs/base/0x5c0d76fab1822bdeb47308ed6028231761ed723e"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if 'pair' not in data: raise ValueError("Missing 'pair' key")
            if 'priceUsd' not in data['pair']: raise ValueError("Missing 'priceUsd'")
            
            price_usd = float(data['pair']['priceUsd'])
            latest_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            
            prices.append(price_usd)
            print(f"[DexScreener OK @ {latest_time}] KVCM = ${price_usd:.6f}  (attempt {attempt+1})")
            print(f"   → Window: {len(prices)}/20")
            
            try:
                with open(PERSIST_FILE, 'wb') as f:
                    pickle.dump(list(prices), f)
            except Exception as save_err:
                print(f"[WARNING] Failed to save prices: {str(save_err)}")
            
            time.sleep(2)
            return f"Current KVCM: ${price_usd:.6f} USD"
        
        except Exception as e:
            print(f"[DexScreener attempt {attempt+1} error] {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return f"Price fetch failed after {max_retries} tries"

def evaluate_movement(_):
    # We require at least 50 data points to establish the SMA-50 trend
    if len(prices) < 50:
        return f"Collecting data... ({len(prices)}/50 needed for trend analysis)"

    # Convert deque to pandas Series for efficient vector math
    data = pd.Series(list(prices))
    current_price = data.iloc[-1]

    # --- 1. Calculate RSI with EWMA (Wilder's Smoothing) ---
    delta = data.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    # Use 'com=13' (Center of Mass) which corresponds to alpha=1/14.
    # 'adjust=False' ensures the smoothing mimics Wilder's original recursive formula.
    avg_gain = gain.ewm(com=13, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(com=13, adjust=False).mean().iloc[-1]

    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # --- 2. Calculate Trend (SMA-50) ---
    # Simple Moving Average of the last 50 periods to determine trend direction
    sma_50 = data.rolling(window=50).mean().iloc[-1]
    
    # --- 3. Determine Context & Signal ---
    trend = "UPTREND" if current_price > sma_50 else "DOWNTREND"
    
    print(f"Stats: RSI(ewm)={rsi:.2f} | Price=${current_price:.6f} | SMA(50)=${sma_50:.6f} | Trend={trend}")

    if trend == "UPTREND":
        # Strategy: Hold winners, buy dips aggressively
        if rsi > 85:
            return f"UPTREND (Euphoria): RSI {rsi:.2f} is extreme (>85). Take partial profit."
        elif rsi < 45:
            # In uptrends, we buy earlier (RSI 45) rather than waiting for 30
            return f"UPTREND (Dip): RSI {rsi:.2f} is a strong buy opportunity."
        else:
            return f"UPTREND (Hold): Price is above SMA-50. RSI {rsi:.2f} is healthy. Hold."

    else: # DOWNTREND
        # Strategy: Sell rallies, buy only deep oversold
        if rsi > 70:
            return f"DOWNTREND (Rally): Price below SMA-50 and RSI {rsi:.2f} is high. Sell."
        elif rsi < 30:
            return f"DOWNTREND (Oversold): RSI {rsi:.2f} is low. Speculative buy."
        else:
            return f"DOWNTREND (Wait): Price below SMA-50. No signal."

    # 4. Smart Signals based on Trend
    if trend == "UPTREND":
        # In an uptrend, we DO NOT sell at RSI 70. We let winners run.
        # We only sell if things get euphoric (RSI > 85).
        if rsi > 85:
            return f"UPTREND (Euphoria): RSI {rsi:.2f} is extreme. Take partial profit."
        elif rsi < 45:
            # In an uptrend, even a mild dip (RSI < 45) is a buy.
            return f"UPTREND (Dip): RSI {rsi:.2f} is a strong buy opportunity."
        else:
            return f"UPTREND (Hold): Price is above SMA-50. RSI {rsi:.2f} is healthy. Hold."

    else: # DOWNTREND
        # In a downtrend, we play defensive.
        if rsi > 70:
            return f"DOWNTREND (Rally): Price below SMA-50 and RSI {rsi:.2f} is high. Sell."
        elif rsi < 30:
            return f"DOWNTREND (Oversold): RSI {rsi:.2f} is low. Speculative buy."
        else:
            return f"DOWNTREND (Wait): Price below SMA-50. No signal."

def execute_buy(amount_usdc_str):
    try:
        requested_usdc = float(amount_usdc_str)
        actual_usdc = min(requested_usdc, MAX_USD_VALUE)
        
        if actual_usdc <= 0: return "Buy aborted: invalid amount."

        amount_in = int(actual_usdc * 10**6)

        usdc_contract = w3.eth.contract(address=USDC_ADDRESS_CHECKSUM, abi=ERC20_ABI)
        balance = usdc_contract.functions.balanceOf(account.address).call()
        
        if balance < amount_in:
            return (f"Insufficient USDC balance.\nHas: {balance / 10**6:.2f}, Needed: {actual_usdc:.2f}")

        # Allowance Check
        allowance = usdc_contract.functions.allowance(account.address, AERODROME_ROUTER_ADDRESS).call()
        if allowance < amount_in:
            print(f"[Buy] Low USDC allowance. Requesting approval...")
            approve_res = approve_token(USDC_ADDRESS_CHECKSUM, 500 * 10**6)
            if "failed" in approve_res.lower():
                return f"Buy aborted: USDC {approve_res}"

        routes = get_aerodrome_route(USDC_ADDRESS_CHECKSUM, KVCM_ADDRESS_CHECKSUM)
        
        try:
            expected_out = get_expected_out(amount_in, USDC_ADDRESS_CHECKSUM, KVCM_ADDRESS_CHECKSUM)
            amount_out_min = int(expected_out * 0.995)
        except Exception as quote_err:
            return f"Buy aborted: Price quote failed. {str(quote_err)}"

        if amount_out_min == 0: return "Buy aborted: Quote returned 0 output."

        deadline = int(time.time()) + 1200
        gas_params = get_eip1559_gas_params()

        tx = router.functions.swapExactTokensForTokens(
            amount_in, amount_out_min, routes, account.address, deadline
        ).build_transaction({
            'from': account.address, 'gas': 300000, **gas_params, 'nonce': nonce_manager.get_nonce()
        })

        receipt = send_transaction_with_retries(account.sign_transaction(tx))

        if receipt and receipt.status == 1:
            return f"BUY SUCCESS: https://basescan.org/tx/{receipt.transactionHash.hex()}"
        else:
            nonce_manager.reset()
            return "BUY FAILED: Transaction failed/reverted."

    except Exception as e:
        return f"Buy Execution Error: {str(e)}"

def execute_sell(amount_kvcm_str):
    try:
        requested_kvcm = float(amount_kvcm_str)
        price_usd = get_current_kvcm_price_usd()
        
        if not price_usd: return "Sell aborted: No reliable KVCM price."

        actual_kvcm = min(requested_kvcm, MAX_USD_VALUE / price_usd)
        if actual_kvcm <= 0: return "Sell aborted: invalid amount."

        amount_in = int(actual_kvcm * 10**KVCM_DECIMALS)

        kvcm_contract = w3.eth.contract(address=KVCM_ADDRESS_CHECKSUM, abi=ERC20_ABI)
        balance = kvcm_contract.functions.balanceOf(account.address).call()
        if balance < amount_in:
            return f"Insufficient balance. Have: {balance / 10**KVCM_DECIMALS:.4f}, Need: {actual_kvcm:.4f}"

        allowance = kvcm_contract.functions.allowance(account.address, AERODROME_ROUTER_ADDRESS).call()
        if allowance < amount_in:
            print(f"[Sell] Low allowance. Requesting approval...")
            approve_res = approve_token(KVCM_ADDRESS_CHECKSUM, amount_in * 10) 
            if "failed" in approve_res.lower(): return f"Sell aborted: {approve_res}"

        try:
            expected_out = get_expected_out(amount_in, KVCM_ADDRESS_CHECKSUM, USDC_ADDRESS_CHECKSUM)
            amount_out_min = int(expected_out * 0.995)
        except Exception as e:
            return f"Sell aborted: Quote failed. {str(e)}"

        routes = get_aerodrome_route(KVCM_ADDRESS_CHECKSUM, USDC_ADDRESS_CHECKSUM)
        deadline = int(time.time()) + 1200
        gas_params = get_eip1559_gas_params()

        tx = router.functions.swapExactTokensForTokens(
            amount_in, amount_out_min, routes, account.address, deadline
        ).build_transaction({
            'from': account.address, 'gas': 300000, **gas_params, 'nonce': nonce_manager.get_nonce() 
        })

        receipt = send_transaction_with_retries(account.sign_transaction(tx))

        if receipt and receipt.status == 1:
            return f"SELL SUCCESS: https://basescan.org/tx/{receipt.transactionHash.hex()}"
        else:
            nonce_manager.reset()
            return "SELL FAILED: Transaction failed/reverted."

    except Exception as e:
        return f"Sell Execution Error: {str(e)}"

def display_balances():
    """Fetches and prints current ETH, USDC, and KVCM balances."""
    try:
        eth_balance_wei = w3.eth.get_balance(account.address)
        eth_balance = w3.from_wei(eth_balance_wei, 'ether')

        usdc_contract = w3.eth.contract(address=USDC_ADDRESS_CHECKSUM, abi=ERC20_ABI)
        usdc_balance = usdc_contract.functions.balanceOf(account.address).call() / 10**6

        kvcm_contract = w3.eth.contract(address=KVCM_ADDRESS_CHECKSUM, abi=ERC20_ABI)
        kvcm_balance = kvcm_contract.functions.balanceOf(account.address).call() / 10**KVCM_DECIMALS

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*40}")
        print(f"💰 WALLET BALANCE UPDATE [@ {timestamp}]")
        print(f"{'='*40}")
        print(f"  ⛽ Base ETH:  {eth_balance:.6f} ETH")
        print(f"  💵 USDC:      {usdc_balance:.2f} USDC")
        print(f"  🪙 kVCM:      {kvcm_balance:.4f} kVCM")
        print(f"{'='*40}\n")
        
        if eth_balance < 0.001:
            print("⚠️ WARNING: Your ETH balance is very low! Transactions may fail.")
            
    except Exception as e:
        print(f"[Balance Check Error] {str(e)}")

# ────────────────────────────────────────────────
# 5. Agent Setup
# ────────────────────────────────────────────────

tools = [
    Tool(name="GetPrice", func=get_current_price, description="Fetch current KVCM price"),
    Tool(name="EvaluateMovement", func=evaluate_movement, description="Check price movement for signals"),
    Tool(name="Buy", func=lambda x: execute_buy(x), description="Buy USDC for kVCM with specified USDC amount"),
    Tool(name="Sell", func=lambda x: execute_sell(x), description="Sell kVCM for USDC with specified kVCM amount"),
    Tool(name="ResetConnection", func=reset_nonce_manager, description="Resyncs nonces if errors occur")
]

llm = ChatXAI(
    xai_api_key=os.getenv('XAI_API_KEY'),
    model="grok-3",
    temperature=0
)

# Global System Prompt
SYSTEM_PROMPT = """You are a smart trading agent on Base network for KVCM/USDC.
You have access to RSI and Trend (SMA-50) analysis.

Rules:
1. TREND IS KING: 
   - If the analysis says "UPTREND", do NOT sell unless the signal is explicitly "Euphoria" or "Take profit".
   - In an uptrend, holding is preferred over trading in and out.
   
2. Trade Size Constraints:
   - Buy: Max 100 USDC input.
   - Sell: Max 100 USDC worth of kVCM.

3. Signal Interpretation:
   - "UPTREND (Dip)": Strong BUY signal. Executing a buy is high priority.
   - "UPTREND (Hold)": DO NOT TRADE. Just wait.
   - "DOWNTREND (Rally)": Strong SELL signal.
   - "DOWNTREND (Oversold)": Cautious BUY (smaller size).

4. Execution:
   - Use the provided tools.
   - 0.5% slippage on all trades."""

memory = MemorySaver()

# Fix for Agent Creation:
# Removed 'state_modifier' as your version does not support it.
# We will inject the prompt in the messages list instead.
agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)

# ────────────────────────────────────────────────
# 6. Execution Loop
# ────────────────────────────────────────────────

def collect_price():
    try:
        price_info = get_current_price(None)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        if "error" in price_info.lower() or "failed" in price_info.lower():
            print(f"[PRICE FAILED @ {timestamp}] {price_info}")
        else:
            print(f"[PRICE COLLECTED @ {timestamp}] {price_info}")
        print(f"   → Window size: {len(prices)} / 20")
    except Exception as e:
        print(f"Collection crashed: {e}")

def evaluate_and_trade():
    """Run full analysis + potential trade – runs every 5 minutes"""
    display_balances()
    
    if len(prices) < 15:
        print(f"[EVAL SKIPPED @ {time.strftime('%Y-%m-%d %H:%M:%S')}] Not enough data yet ({len(prices)}/15)")
        return

    print(f"[EVALUATION START @ {time.strftime('%Y-%m-%d %H:%M:%S')}] Window size: {len(prices)}")
    
    try:
        user_query = (
            "Analyze the latest price movement using the collected data "
            "and execute a trade ONLY if there is a strong RSI signal "
            "(with 0.5% slippage limit)."
        )

        # Inject System Prompt manually here since state_modifier failed
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        config = {
            "configurable": {"thread_id": "kvcm-trading-agent-001"},
            "callbacks": [StdOutCallbackHandler()]
        }

        result = agent_executor.invoke(
            {"messages": messages},
            config=config
        )

        if "messages" in result and result["messages"]:
            final_msg = result["messages"][-1]
            output = getattr(final_msg, "content", str(final_msg))
            print("[AGENT OUTPUT]", output)
        else:
            print("[AGENT OUTPUT] No response or messages returned from agent")

    except Exception as e:
        print(f"Agent execution failed: {str(e)}")
        traceback.print_exc()

# Schedule the jobs
schedule.every(1).minutes.do(collect_price)
schedule.every(5).minutes.do(evaluate_and_trade)

# Trigger first collection immediately
print("Environment checks passed. Starting first collection...")
display_balances()
collect_price()

print("kVCM/USDC Aerodrome agent started")
print("• Collecting price ............ every 1 minute")
print("• Evaluating & possible trade .... every 5 minutes")
print("Running forever. Press Ctrl+C to stop.\n")

try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down gracefully...")
    try:
        with open(PERSIST_FILE, 'wb') as f:
            pickle.dump(list(prices), f)
        print(f"[Shutdown] Saved {len(prices)} prices")
    except Exception as e:
        print(f"[Shutdown] Failed to save prices: {e}")
    print("Goodbye.")