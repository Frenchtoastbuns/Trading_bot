# nn_screener_enhanced.py

import os
import math
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import joblib

# Data
import yfinance as yf
from typing import Optional, Dict, List, Tuple

# Web Scraping (for news sentiment)
try:
    from bs4 import BeautifulSoup
    import requests
    import time
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    
from insider_fetcher import get_insider_trades

# Technical Analysis
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Trading
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# =====================
# Configuration
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- API Keys ---
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

# --- Model & Data Config ---
EV_WINDOW = 20
DEFAULT_RR = 2.0
LOOKBACK_PERIOD = "2y" # Used for training
FORWARD_DAYS = 10
TARGET_RETURN = 0.03 

# --- Stock Universe & Filters ---
PRICE_MIN = 2.0
PRICE_MAX = 100.0
MIN_MARKET_CAP = 200_000_000
MIN_AVG_DAILY_VOL = 200_000
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM',
    'PYPL', 'INTC', 'AMD', 'ORCL', 'CSCO', 'AVGO', 'TXN', 'QCOM', 'MU', 'AMAT',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP', 'COF',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'MDT', 'AMGN',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'HAL', 'BKR', 'DVN',
    'SPY', 'QQQ', 'IWM'
]

# --- Neural Network Config ---
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 100
PATIENCE = 15

# --- Trading Config ---
MAX_POSITION_SIZE = 0.05
MIN_CONFIDENCE = 0.55

# =====================
# Helper Functions
# =====================

def _safe_ratio(numerator: pd.Series, denominator: pd.Series, default_value: float = 0.0) -> pd.Series:
    ratio = numerator / denominator
    ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
    ratio.fillna(default_value, inplace=True)
    return ratio

def normalize_score_01(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    x = 0.0 if x is None or np.isnan(x) else x
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

# =====================
# Data Collection & Feature Engineering
# =====================

def get_symbol_fast_info(symbol: str) -> dict:
    info = {'last_price': None, 'market_cap': None, 'avg_volume': None}
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t, 'fast_info', None)
        if fi:
            info['last_price'] = getattr(fi, 'last_price', None)
            info['market_cap'] = getattr(fi, 'market_cap', None)
            info['avg_volume'] = getattr(fi, 'three_month_average_volume', None) or getattr(fi, 'ten_day_average_volume', None)
        if any(v is None for v in info.values()):
            di = t.info or {}
            info['last_price'] = info['last_price'] or di.get('regularMarketPrice') or di.get('currentPrice')
            info['market_cap'] = info['market_cap'] or di.get('marketCap')
            info['avg_volume'] = info['avg_volume'] or di.get('averageVolume') or di.get('averageDailyVolume10Day')
    except Exception: pass
    return info

def prefilter_symbol(symbol: str) -> bool:
    fi = get_symbol_fast_info(symbol)
    price_ok = (fi['last_price'] is not None and PRICE_MIN <= fi['last_price'] <= PRICE_MAX)
    mcap_ok = (fi['market_cap'] is not None and fi['market_cap'] >= MIN_MARKET_CAP)
    vol_ok = (fi['avg_volume'] is not None and fi['avg_volume'] >= MIN_AVG_DAILY_VOL)
    return bool(price_ok and mcap_ok and vol_ok)

def get_stock_data(symbol: str, period: str = None, start: str = None, end: str = None) -> Optional[pd.DataFrame]:
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, start=start, end=end, auto_adjust=True)
        if df is None or df.empty:
            # Quieter warning for backtesting where some stocks might not exist in the period
            if start: pass
            else: print(f"Warning: No data found for {symbol}, it may be delisted.")
            return None
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df['symbol'] = symbol
        return df
    except Exception as e:
        print(f"Could not get stock data for {symbol}: {e}")
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['sma200'] = SMAIndicator(df['close'], window=200).sma_indicator()
    df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema50_slope'] = df['ema50'].diff()
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    atr = AverageTrueRange(df['high'], df['low'], df['close'])
    df['atr'] = atr.average_true_range()
    df['atr_avg'] = df['atr'].rolling(window=20).mean()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_width'] = _safe_ratio(bb.bollinger_hband() - bb.bollinger_lband(), bb.bollinger_mavg())
    df['volume_sma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = _safe_ratio(df['volume'], df['volume_sma20'])
    df['price_to_sma_ratio'] = _safe_ratio(df['close'], df['sma200'], 1.0)
    df['above_sma'] = (df['close'] > df['sma200']).astype(int)
    df['hh_hl_pattern'] = ((df['high'] > df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
    df['lh_ll_pattern'] = ((df['high'] < df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)
    df['rsi_condition'] = (df['rsi'] > 40).astype(int)
    df['macd_bullish'] = ((df['macd'] < 0) & (df['macd'].shift(1) < df['macd_signal'].shift(1)) & (df['macd'] > df['macd_signal'])).astype(int)
    df['atr_condition'] = (df['atr'] > df['atr_avg']).astype(int)
    df['bbw_median50'] = df['bb_width'].rolling(window=50, min_periods=25).median()
    df['bbw_condition_high'] = (df['bb_width'] > (df['bbw_median50'] * 1.2)).astype(int)
    df['bbw_condition_low'] = (df['bb_width'] < (df['bbw_median50'] * 0.8)).astype(int)
    df['return_20d'] = df['close'].pct_change(20)
    df['relative_strength_20d'] = df['return_20d'] - df['spy_return']
    return df

def assign_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    regime = np.zeros(len(df), dtype=int)
    trending_up = ((df['ema50_slope'] > 0) & ((df['adx'] > 20) | (df['hh_hl_pattern'] == 1)))
    trending_down = ((df['ema50_slope'] < 0) & ((df['adx'] > 20) | (df['lh_ll_pattern'] == 1)))
    high_vol_breakout = ((df['bbw_condition_high'] == 1) & (df['atr_condition'] == 1))
    low_vol_drift = ((df['bbw_condition_low'] == 1) & (df['adx'] < 20) & (df['ema50_slope'].abs() < df['ema50'].rolling(20).std().fillna(0) * 0.1))
    regime[trending_up.fillna(False).values] = 1
    regime[trending_down.fillna(False).values] = 2
    regime[high_vol_breakout.fillna(False).values] = 3
    regime[low_vol_drift.fillna(False).values] = 4
    df['regime_label'] = regime
    for k in range(5):
        df[f'regime_{k}'] = (df['regime_label'] == k).astype(int)
    return df

def calculate_forward_returns(df: pd.DataFrame, forward_days: int = FORWARD_DAYS) -> pd.DataFrame:
    df['future_close'] = df['close'].shift(-forward_days)
    df['forward_return'] = _safe_ratio(df['future_close'] - df['close'], df['close'])
    df['target'] = (df['forward_return'] > TARGET_RETURN).astype(int)
    return df

def compute_ev_features(df: pd.DataFrame, window: int = EV_WINDOW) -> pd.DataFrame:
    DEFAULT_RR = 2.0 
    def _rolling_rr(x: pd.Series) -> float:
        pos = x[x > 0]
        neg = x[x <= 0]
        if len(pos) == 0 or len(neg) == 0: return DEFAULT_RR
        reward = pos.mean()
        risk = abs(neg.mean())
        if risk == 0: return DEFAULT_RR
        return reward / risk
    df = df.sort_index()
    for symbol, g in df.groupby('symbol'):
        if g.empty: continue
        win_rate = g['target'].rolling(window=window, min_periods=window//2).mean()
        rr = g['forward_return'].rolling(window=window, min_periods=window//2).apply(_rolling_rr, raw=False)
        ev = (win_rate * rr) - (1 - win_rate)
        df.loc[g.index, 'rolling_win_rate'] = win_rate
        df.loc[g.index, 'rolling_rr'] = rr
        df.loc[g.index, 'ev'] = ev
    return df

def collect_training_data() -> pd.DataFrame:
    rows = []
    print("Fetching market data (SPY)...")
    spy_data = get_stock_data(symbol='SPY', period=LOOKBACK_PERIOD)
    if spy_data is None: raise ValueError("Could not fetch SPY data.")
    spy_data['spy_return'] = spy_data['close'].pct_change(20)
    for idx, symbol in enumerate(STOCK_UNIVERSE):
        print(f"Processing {symbol} ({idx+1}/{len(STOCK_UNIVERSE)})...")
        if not prefilter_symbol(symbol): continue
        df = get_stock_data(symbol, period=LOOKBACK_PERIOD)
        if df is None or df.empty: continue
        df = pd.merge(df, spy_data[['spy_return']], left_index=True, right_index=True, how='left')
        df['spy_return'] = df['spy_return'].ffill()
        df = calculate_technical_indicators(df)
        df = assign_market_regime(df)
        df = calculate_forward_returns(df)
        df = df.dropna(subset=['sma200', 'relative_strength_20d'])
        if len(df) < 250: continue
        rows.append(df)
    if not rows: raise ValueError("No valid training data collected. Check filters.")
    data = pd.concat(rows, axis=0)
    data = compute_ev_features(data)
    return data.replace([np.inf, -np.inf], np.nan).dropna()

def prepare_features(df: pd.DataFrame) -> tuple:
    feature_columns = [
        'close', 'price_to_sma_ratio', 'above_sma', 'ema50_slope', 'adx', 'rsi', 'macd', 
        'macd_signal', 'macd_histogram', 'rsi_condition', 'macd_bullish', 'atr', 'atr_avg', 
        'bb_width', 'bbw_condition_high', 'bbw_condition_low', 'atr_condition', 
        'hh_hl_pattern', 'lh_ll_pattern', 'volume_ratio', 'rolling_win_rate', 'rolling_rr', 'ev',
        'regime_0', 'regime_1', 'regime_2', 'regime_3', 'regime_4',
        'relative_strength_20d'
    ]
    existing = [c for c in feature_columns if c in df.columns]
    X = df[existing].astype(np.float32).values
    y = df['target'].astype(np.float32).values.reshape(-1, 1)
    return X, y, existing

# =====================
# Model Definition, Training & Evaluation
# =====================

class EnhancedStockNN(nn.Module):
    def __init__(self, input_size: int, hidden_sizes=None, dropout_rate: float = 0.3):
        super().__init__()
        if hidden_sizes is None: hidden_sizes = [128, 64, 32, 16]
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size), nn.BatchNorm1d(size),
                nn.ReLU(), nn.Dropout(dropout_rate)
            ])
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def train_model(model: nn.Module, train_loader, val_loader, epochs: int, patience: int, pos_weight: torch.Tensor):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss = criterion(pred, yb)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        if (epoch + 1) % 10 == 0: print(f"Epoch {epoch+1}/{epochs}.. Val loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    if best_model_state: model.load_state_dict(best_model_state)
    return model

def evaluate_model(model: nn.Module, X_test_s: np.ndarray, y_test: np.ndarray):
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        preds = (probs > 0.5).astype(int)
        y_true = y_test.squeeze()
        print("\n--- Model Evaluation ---")
        print(f"Accuracy: {np.mean(preds == y_true):.4f}")
        try: print(f"ROC AUC: {roc_auc_score(y_true, probs):.4f}")
        except ValueError: print("ROC AUC could not be calculated (only one class in test set).")
        print("Classification Report:\n", classification_report(y_true, preds, digits=4, zero_division=0))

# =====================
# Live Screening & Trading
# =====================

def compute_live_signals(symbols: List[str]) -> pd.DataFrame:
    signals = []
    print("\n--- Fetching Live Signals ---")
    insider_df = get_insider_trades(tickers=symbols, days_back=90, api_key=FMP_API_KEY)
    insider_scores = {}
    if not insider_df.empty:
        for symbol, group in insider_df.groupby('ticker'):
            buy_val = group[group['transaction_type'] == 'Buy']['transaction_value'].sum()
            sell_val = group[group['transaction_type'] == 'Sell']['transaction_value'].sum()
            total_val = buy_val + sell_val
            if total_val > 0: insider_scores[symbol] = (buy_val - sell_val) / total_val
    for symbol in symbols:
        inst_trend = 0.0
        try:
            holders = yf.Ticker(symbol).institutional_holders
            if holders is not None and not holders.empty:
                pct_held = holders['% Out'].str.rstrip('%').astype(float).sum() / 100
                inst_trend = float(np.clip((pct_held - 0.5) * 2, -1.0, 1.0))
        except Exception: pass
        signals.append({
            'symbol': symbol,
            'insider_signal': insider_scores.get(symbol, 0.0),
            'institutional_trend': inst_trend
        })
    return pd.DataFrame(signals).set_index('symbol')

def compute_factor_scores(row: pd.Series) -> dict:
    ev = 0.0 if pd.isna(row.get('ev')) else row['ev']
    ev_score = max(ev, 0.0) / (1.0 + max(ev, 0.0))
    momentum_flags = [row.get('above_sma', 0), row.get('rsi_condition', 0), row.get('macd_bullish', 0), row.get('atr_condition', 0)]
    momentum_score = float(np.mean(momentum_flags))
    regime_map = {0: 0.4, 1: 1.0, 2: 0.0, 3: 0.8, 4: 0.6}
    regime_score = regime_map.get(int(row.get('regime_label', 0)), 0.4)
    insider = normalize_score_01(row.get('insider_signal', 0.0))
    instit = normalize_score_01(row.get('institutional_trend', 0.0))
    sentiment_score = float(np.mean([insider, instit]))
    return {'ev_score': ev_score, 'momentum_score': momentum_score, 'regime_score': regime_score, 'sentiment_score': sentiment_score}

def weighted_total_score(scores: dict, weights: dict = None) -> float:
    if weights is None: weights = {'ev': 0.35, 'momentum': 0.35, 'regime': 0.15, 'sentiment': 0.15}
    return (scores['ev_score'] * weights['ev'] + scores['momentum_score'] * weights['momentum'] +
            scores['regime_score'] * weights['regime'] + scores['sentiment_score'] * weights['sentiment'])

def screen_and_rank_current(top_n: int = 20) -> pd.DataFrame:
    print("--- Starting Daily Screen ---")
    df = collect_training_data()
    latest = df.groupby('symbol').tail(1).copy()
    symbols_to_screen = latest['symbol'].unique().tolist()
    if symbols_to_screen:
        live_signals_df = compute_live_signals(symbols_to_screen)
        latest.set_index('symbol', inplace=True)
        latest.update(live_signals_df)
        latest.reset_index(inplace=True)
    factors = latest.apply(compute_factor_scores, axis=1, result_type='expand')
    latest = pd.concat([latest, factors], axis=1)
    latest['total_score'] = latest.apply(lambda row: weighted_total_score(row.to_dict()), axis=1)
    return latest.sort_values('total_score', ascending=False).head(top_n)

def predict_trade_quality(model: nn.Module, scaler, row: pd.Series, feature_cols: list) -> float:
    model.to(device)
    model.eval()
    x = row[feature_cols].astype(np.float32).values.reshape(1, -1)
    xs = scaler.transform(x)
    xt = torch.tensor(xs, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(xt)
        prob = torch.sigmoid(logits).item()
    return float(prob)

class AlpacaTrader:
    def __init__(self):
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        self.account = self.trading_client.get_account()
        print(f"\nConnected to Alpaca. Buying power: ${self.account.buying_power}")

    def execute_trades(self, candidates: pd.DataFrame, model: nn.Module, scaler, feature_cols: list):
        print("\n--- Executing Trades ---")
        positions = {p.symbol: p for p in self.trading_client.get_all_positions()}
        for _, candidate in candidates.iterrows():
            symbol = candidate['symbol']
            confidence = predict_trade_quality(model, scaler, candidate, feature_cols)
            print(f"Analyzing {symbol}: Score={candidate['total_score']:.2f}, Confidence={confidence:.2f}")
            if confidence >= MIN_CONFIDENCE and symbol not in positions:
                target_value = float(self.account.portfolio_value) * MAX_POSITION_SIZE
                qty = int(target_value / candidate['close'])
                if qty > 0:
                    print(f"  --> Placing BUY order for {qty} shares of {symbol}")
                    try:
                        self.trading_client.submit_order(
                            order_data=MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                        )
                    except Exception as e:
                        print(f"    Order failed for {symbol}: {e}")
                else:
                    print(f"  Skipping {symbol}: Calculated quantity is zero.")

# =====================
# Main Execution Logic
# =====================

def run_training_and_save():
    print("--- Starting Training Pipeline ---")
    df = collect_training_data()
    print(f"Collected {len(df):,} total samples for training.")
    X, y, feature_columns = prepare_features(df)
    n = len(X)
    tr_end, va_end = int(n * 0.6), int(n * 0.8)
    X_train, y_train = X[:tr_end], y[:tr_end]
    X_val, y_val = X[tr_end:va_end], y[tr_end:va_end]
    X_test, y_test = X[va_end:], y[va_end:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    train_data = TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train))
    val_data = TensorDataset(torch.tensor(X_val_s), torch.tensor(y_val))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    if num_pos > 0:
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
        print(f"\nClass Imbalance: 'No Buy' is {pos_weight.item():.2f}x more frequent.")
        print("Applying weighted loss to compensate.")
    else:
        print("Warning: No positive samples found. Cannot apply weights.")
        pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)

    model = EnhancedStockNN(input_size=X_train_s.shape[1])
    model.to(device)
    
    print("\n--- Training Model ---")
    model = train_model(model, train_loader, val_loader, EPOCHS, PATIENCE, pos_weight=pos_weight)
    evaluate_model(model, X_test_s, y_test)
    
    model.to("cpu")
    torch.save({'model_state_dict': model.state_dict(), 'feature_columns': feature_columns}, 'stock_trading_model.pth')
    joblib.dump(scaler, 'feature_scaler.joblib')
    print("\nSaved model and scaler to disk.")

def auto_trade_daily():
    print("--- Starting Daily Trading ---")
    if not all(os.path.exists(f) for f in ['stock_trading_model.pth', 'feature_scaler.joblib']):
        print("Model or scaler not found. Please run --mode train first.")
        return
    
    checkpoint = torch.load('stock_trading_model.pth')
    scaler = joblib.load('feature_scaler.joblib')
    feature_columns = checkpoint['feature_columns']
    
    model = EnhancedStockNN(input_size=len(feature_columns))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    ranked_candidates = screen_and_rank_current()
    if ranked_candidates.empty:
        print("No attractive candidates found today.")
        return
        
    print("\n--- Top Ranked Candidates ---")
    print(ranked_candidates[['symbol', 'close', 'total_score']])

    trader = AlpacaTrader()
    trader.execute_trades(ranked_candidates, model, scaler, feature_columns)

def backtest_strategy(start_date: str = "2023-01-01", end_date: str = "2024-01-01"):
    print("\n--- Starting Backtest ---")
    if not all(os.path.exists(f) for f in ['stock_trading_model.pth', 'feature_scaler.joblib']):
        print("Model or scaler not found. Please run --mode train first.")
        return
    checkpoint = torch.load('stock_trading_model.pth')
    scaler = joblib.load('feature_scaler.joblib')
    feature_columns = checkpoint['feature_columns']
    model = EnhancedStockNN(input_size=len(feature_columns))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Collecting historical data for backtest...")
    fetch_start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=300)).strftime('%Y-%m-%d')
    rows = []
    spy_data = get_stock_data(symbol='SPY', start=fetch_start_date, end=end_date)
    if spy_data is None: raise ValueError("Could not fetch SPY data for backtest.")
    spy_data['spy_return'] = spy_data['close'].pct_change(20)

    for symbol in STOCK_UNIVERSE:
        df = get_stock_data(symbol=symbol, start=fetch_start_date, end=end_date)
        if df is None or df.empty: continue
        
        df = pd.merge(df, spy_data[['spy_return']], left_index=True, right_index=True, how='left')
        df['spy_return'] = df['spy_return'].ffill()
        
        ## FIXED: Added missing function calls to ensure data matches training data
        df = calculate_technical_indicators(df)
        df = assign_market_regime(df)
        df = calculate_forward_returns(df)
        rows.append(df)
    
    if not rows:
        print("No historical data collected for the backtest period.")
        return
        
    all_data = pd.concat(rows, axis=0)
    ## FIXED: Added missing function call
    all_data = compute_ev_features(all_data)
    all_data = all_data.dropna()
    all_data.index = pd.to_datetime(all_data.index.date)
    
    backtest_data = all_data[(all_data.index >= start_date) & (all_data.index <= end_date)]
    if backtest_data.empty:
        print("No data available for the specified backtest date range after cleaning.")
        return

    initial_capital = 100_000
    capital = initial_capital
    portfolio_history = pd.Series(index=pd.to_datetime(backtest_data.index.unique()).sort_values(), dtype=float)
    active_positions = {}
    trade_log = []

    print(f"Running simulation from {start_date} to {end_date}...")
    for date, daily_data in backtest_data.groupby(level=0):
        current_value = capital
        for symbol, pos in list(active_positions.items()):
            if symbol in daily_data['symbol'].values:
                current_price = daily_data[daily_data['symbol'] == symbol]['close'].iloc[0]
                current_value += pos['shares'] * current_price
            else:
                current_value += pos['shares'] * pos['entry_price']
        portfolio_history[date] = current_value

        positions_to_exit = [s for s, p in active_positions.items() if (date.to_pydatetime().date() - p['entry_date'].date()).days >= FORWARD_DAYS]
        for symbol in positions_to_exit:
            if symbol in daily_data['symbol'].values:
                exit_price = daily_data[daily_data['symbol'] == symbol]['close'].iloc[0]
                pos = active_positions.pop(symbol)
                trade_profit = (exit_price - pos['entry_price']) * pos['shares']
                capital += pos['shares'] * exit_price
                trade_log.append({'symbol': symbol, 'pnl': trade_profit})

        for _, row in daily_data.iterrows():
            symbol = row['symbol']
            if symbol in active_positions: continue
            confidence = predict_trade_quality(model, scaler, row, feature_columns)
            if confidence > MIN_CONFIDENCE:
                position_size = portfolio_history[date] * MAX_POSITION_SIZE
                shares_to_buy = int(position_size / row['close'])
                if shares_to_buy > 0 and capital >= shares_to_buy * row['close']:
                    capital -= shares_to_buy * row['close']
                    active_positions[symbol] = {'shares': shares_to_buy, 'entry_price': row['close'], 'entry_date': date.to_pydatetime()}

    portfolio_history.dropna(inplace=True)
    if len(portfolio_history) < 2:
        print("Not enough portfolio history to calculate metrics.")
        return
        
    daily_returns = portfolio_history.pct_change().dropna()
    total_return = (portfolio_history.iloc[-1] / initial_capital) - 1
    annualized_return = daily_returns.mean() * 252
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = _safe_ratio(pd.Series([annualized_return]), pd.Series([annualized_volatility])).iloc[0]
    running_max = portfolio_history.cummax()
    drawdown = (portfolio_history - running_max) / running_max
    max_drawdown = drawdown.min() if not drawdown.empty else 0
    wins = sum(1 for trade in trade_log if trade['pnl'] > 0)
    win_rate = _safe_ratio(pd.Series([wins]), pd.Series([len(trade_log)])).iloc[0] if trade_log else 0

    print("\n--- Backtest Performance Metrics ---")
    print(f"Period: {start_date} to {end_date}")
    print(f"Final Portfolio Value: ${portfolio_history.iloc[-1]:,.2f}")
    print("-" * 35)
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility (Risk): {annualized_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Total Trades: {len(trade_log)}")
    print(f"Win Rate: {win_rate:.2%}")
    print("-" * 35)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Stock Screener and Trading Bot')
    parser.add_argument('--mode', choices=['train', 'screen', 'trade', 'backtest'], default='train', help='Operation mode')
    args = parser.parse_args()

    if not FMP_API_KEY or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        warnings.warn("API keys are not set. Configure FMP_API_KEY, ALPACA_API_KEY, and ALPACA_SECRET_KEY in your environment.")

    if args.mode == 'train':
        run_training_and_save()
    elif args.mode == 'screen':
        ranked = screen_and_rank_current()
        print("\n--- Top Ranked Candidates ---")
        print(ranked[['symbol', 'close', 'total_score']])
    elif args.mode == 'trade':
        auto_trade_daily()
    elif args.mode == 'backtest':
        backtest_strategy(start_date="2024-08-23", end_date="2025-08-23")
