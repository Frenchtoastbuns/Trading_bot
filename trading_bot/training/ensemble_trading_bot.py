import os
import math
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
# Data
import yfinance as yf
from typing import Optional, Dict, List, Tuple
# ML
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier # For Feature Selection
# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Alpaca API
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Technical Analysis
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# =====================
# Configuration
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def artifact_path(filename: str) -> Path:
    return ARTIFACT_DIR / filename

# --- API KEYS (Replace with your own) ---
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

# --- Model & Data Config ---
LOOKBACK_PERIOD = "2y"
FORWARD_DAYS = 20
TARGET_RETURN = 0.0 # Any positive return is a win
NUM_MODELS = 5
NUM_FEATURES_TO_SELECT = 15 # Select the top 15 features

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
MAX_POSITION_SIZE = 1
MIN_CONFIDENCE = 0.55

# =====================
# Data Collection & Feature Engineering
# =====================

def get_stock_data(symbol: str, period: str = None, start: str = None, end: str = None) -> Optional[pd.DataFrame]:
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, start=start, end=end, auto_adjust=True)
        if df is None or df.empty: return None
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df['symbol'] = symbol
        return df
    except Exception: return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['sma200'] = SMAIndicator(df['close'], window=200).sma_indicator()
    df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema50_slope'] = df['ema50'].diff()
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    atr = AverageTrueRange(df['high'], df['low'], df['close'])
    df['atr'] = atr.average_true_range()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['price_to_sma_ratio'] = df['close'] / df['sma200']
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)
    df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
    return df

def calculate_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['forward_return'] = df['close'].shift(-FORWARD_DAYS) / df['close'] - 1
    df['target'] = (df['forward_return'] > TARGET_RETURN).astype(int)
    return df

def collect_data(period: str = None, start: str = None, end: str = None) -> pd.DataFrame:
    rows = []
    print("Fetching data...")
    for idx, symbol in enumerate(STOCK_UNIVERSE):
        df = get_stock_data(symbol, period=period, start=start, end=end)
        if df is None or df.empty: continue
        print(f"Processing {symbol} ({idx+1}/{len(STOCK_UNIVERSE)})...")
        df = calculate_technical_indicators(df)
        df = calculate_forward_returns(df)
        rows.append(df)
    if not rows: raise ValueError("No valid data collected.")
    data = pd.concat(rows, axis=0)
    return data.replace([np.inf, -np.inf], np.nan).dropna()

def prepare_full_features(df: pd.DataFrame) -> tuple:
    feature_columns = [
        'sma200', 'ema50', 'ema50_slope', 'rsi', 'macd', 'macd_signal', 'macd_diff',
        'atr', 'adx', 'bb_width', 'volume_ratio', 'price_to_sma_ratio',
        'return_5d', 'return_20d', 'volatility_20d'
    ]
    df = df.dropna()
    X = df[feature_columns].astype(np.float32)
    y = df['target'].astype(np.int64)
    return X, y, feature_columns

def select_best_features(X_train: pd.DataFrame, y_train: pd.Series, original_cols: list) -> list:
    print("\n--- Performing Feature Selection ---")
    selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector.fit(X_train, y_train)
    
    importances = selector.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': original_cols,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print("Top features found:")
    print(feature_importance_df.head(NUM_FEATURES_TO_SELECT))
    
    best_features = feature_importance_df['feature'].head(NUM_FEATURES_TO_SELECT).tolist()
    return best_features

# =====================
# Model Definition, Training & Evaluation
# =====================

class EnhancedStockNN(nn.Module):
    def __init__(self, input_size: int, hidden_sizes=None, dropout_rate: float = 0.4):
        super().__init__()
        if hidden_sizes is None: hidden_sizes = [64, 32]
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

def load_ensemble_models(num_models: int) -> Tuple[List[EnhancedStockNN], List[str]]:
    ensemble = []
    checkpoint_0 = torch.load(artifact_path('model_0.pth'), map_location=device)
    best_features = checkpoint_0['feature_columns']
    input_size = len(best_features)
    print(f"\n--- Loading Ensemble of {num_models} Models (trained on {input_size} features) ---")

    for i in range(num_models):
        model_path = artifact_path(f'model_{i}.pth')
        if not os.path.exists(model_path): raise FileNotFoundError(f"{model_path} not found.")
        model = EnhancedStockNN(input_size=input_size)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['feature_columns'] != best_features:
            raise ValueError(f"Feature mismatch in model_{i}.pth")
        model.to(device)
        model.eval()
        ensemble.append(model)
    return ensemble, best_features

def predict_with_ensemble(ensemble_models: List[EnhancedStockNN], scaler, row: pd.Series, feature_cols: list) -> float:
    all_probs = []
    x_df = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
    xs = scaler.transform(x_df)
    xt = torch.tensor(xs, dtype=torch.float32).to(device)
    with torch.no_grad():
        for model in ensemble_models:
            logits = model(xt)
            prob = torch.sigmoid(logits).item()
            all_probs.append(prob)
    return np.mean(all_probs)

def evaluate_ensemble(ensemble_models: List[EnhancedStockNN], X_test_s: np.ndarray, y_test: np.ndarray):
    all_probs = []
    X_test_tensor = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    with torch.no_grad():
        for model in ensemble_models:
            logits = model(X_test_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    avg_probs = np.mean(all_probs, axis=0).squeeze()
    preds = (avg_probs > 0.5).astype(int)
    y_true = y_test.squeeze()
    print("\n--- Ensemble Model Evaluation ---")
    print(f"Accuracy: {np.mean(preds == y_true):.4f}")
    try:
        print(f"ROC AUC: {roc_auc_score(y_true, avg_probs):.4f}")
    except ValueError:
        print("ROC AUC could not be calculated.")
    print("Classification Report:\n", classification_report(y_true, preds, digits=4, zero_division=0))

# =====================
# Main Execution Logic
# =====================

def run_training_and_save():
    print("--- Starting Training Pipeline ---")
    df = collect_data(period=LOOKBACK_PERIOD)
    print(f"Collected {len(df):,} total samples.")
    X, y, feature_columns = prepare_full_features(df)
    
    n = len(X)
    tr_end = int(n * 0.7)
    X_train, y_train = X.iloc[:tr_end], y.iloc[:tr_end]
    
    best_features = select_best_features(X_train, y_train, feature_columns)
    
    X = X[best_features]
    
    X_train, y_train = X.iloc[:tr_end], y.iloc[:tr_end]
    va_end = int(n * 0.85)
    X_val, y_val = X.iloc[tr_end:va_end], y.iloc[tr_end:va_end]
    X_test, y_test = X.iloc[va_end:], y.iloc[va_end:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    train_data = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train.values.reshape(-1, 1)).float())
    val_data = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val.values.reshape(-1, 1)).float())
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device) if num_pos > 0 else torch.tensor([1.0]).to(device)

    for i in range(NUM_MODELS):
        print(f"\n--- Training Model {i+1}/{NUM_MODELS} on {len(best_features)} Features ---")
        model = EnhancedStockNN(input_size=len(best_features))
        model.to(device)
        model = train_model(model, train_loader, val_loader, EPOCHS, PATIENCE, pos_weight=pos_weight)
        
        model.to("cpu")
        torch.save({
            'model_state_dict': model.state_dict(), 
            'feature_columns': best_features
        }, artifact_path(f'model_{i}.pth'))
        print(f"Saved model_{i}.pth to disk.")

    joblib.dump(scaler, artifact_path('feature_scaler.joblib'))
    print("\nSaved feature_scaler.joblib to disk.")

    ensemble, _ = load_ensemble_models(NUM_MODELS)
    evaluate_ensemble(ensemble, X_test_s, y_test.values)

## FULLY IMPLEMENTED BACKTESTER
def backtest_strategy(start_date: str, end_date: str):
    print("\n--- Starting Backtest ---")
    if not os.path.exists(artifact_path('feature_scaler.joblib')) or not os.path.exists(artifact_path('model_0.pth')):
        print("Model/scaler not found. Please run --mode train first.")
        return
        
    scaler = joblib.load(artifact_path('feature_scaler.joblib'))
    ensemble_models, best_features = load_ensemble_models(NUM_MODELS)

    fetch_start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=300)).strftime('%Y-%m-%d')
    all_data = collect_data(start=fetch_start_date, end=end_date)
    
    backtest_data = all_data[(all_data.index.date >= pd.to_datetime(start_date).date()) & 
                             (all_data.index.date <= pd.to_datetime(end_date).date())]
    if backtest_data.empty:
        print("No data available for the specified backtest date range.")
        return

    initial_capital = 100_000
    capital = initial_capital
    portfolio_history = pd.Series(index=pd.to_datetime(backtest_data.index.unique()).sort_values(), dtype=float)
    active_positions = {}
    trade_log = []

    print(f"Running simulation from {start_date} to {end_date}...")
    for date, daily_data in backtest_data.groupby(level=0):
        # 1. Update portfolio value
        current_value = capital
        for symbol, pos in active_positions.items():
            if symbol in daily_data['symbol'].values:
                current_price = daily_data[daily_data['symbol'] == symbol]['close'].iloc[0]
                current_value += pos['shares'] * current_price
            else: # If stock has no data today, use last known price
                current_value += pos['shares'] * pos['entry_price']
        portfolio_history[date] = current_value

        # 2. Check for exits
        positions_to_exit = [s for s, p in active_positions.items() if (date.to_pydatetime().date() - p['entry_date'].date()).days >= FORWARD_DAYS]
        for symbol in positions_to_exit:
            if symbol in daily_data['symbol'].values:
                exit_price = daily_data[daily_data['symbol'] == symbol]['close'].iloc[0]
                pos = active_positions.pop(symbol)
                trade_profit = (exit_price - pos['entry_price']) * pos['shares']
                capital += pos['shares'] * exit_price
                trade_log.append({'pnl': trade_profit})

        # 3. Look for new entries
        for _, row in daily_data.iterrows():
            symbol = row['symbol']
            if symbol in active_positions: continue
            
            confidence = predict_with_ensemble(ensemble_models, scaler, row, best_features)
            
            if confidence > MIN_CONFIDENCE:
                position_size = portfolio_history[date] * MAX_POSITION_SIZE
                shares_to_buy = int(position_size / row['close'])
                
                if shares_to_buy > 0 and capital >= shares_to_buy * row['close']:
                    capital -= shares_to_buy * row['close']
                    active_positions[symbol] = {'shares': shares_to_buy, 'entry_price': row['close'], 'entry_date': date.to_pydatetime()}

    # Final calculations
    portfolio_history.dropna(inplace=True)
    if len(portfolio_history) < 2:
        print("Not enough portfolio history to calculate metrics.")
        return
        
    daily_returns = portfolio_history.pct_change().dropna()
    total_return = (portfolio_history.iloc[-1] / initial_capital) - 1
    annualized_return = daily_returns.mean() * 252
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    running_max = portfolio_history.cummax()
    drawdown = (portfolio_history - running_max) / running_max
    max_drawdown = drawdown.min() if not drawdown.empty else 0
    wins = sum(1 for trade in trade_log if trade['pnl'] > 0)
    win_rate = (wins / len(trade_log)) if trade_log else 0

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
    parser = argparse.ArgumentParser(description='AI Stock Screener with Feature Selection')
    parser.add_argument('--mode', choices=['train', 'backtest'], default='train', help='Operation mode')
    args = parser.parse_args()

    if not FMP_API_KEY or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        warnings.warn("API keys are not set. Configure FMP_API_KEY, ALPACA_API_KEY, and ALPACA_SECRET_KEY in your environment.")

    if args.mode == 'train':
        run_training_and_save()
    elif args.mode == 'backtest':
        backtest_strategy(start_date="2024-01-01", end_date="2025-07-31")
