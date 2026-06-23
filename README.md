# Trading Bot

A Python research workspace for options pricing, neural-network stock screening, backtesting, and Alpaca paper-trading experiments.

The project is organized around a few runnable scripts and the model artifacts they use. It is intended for experimentation and education, not live financial decision-making.

## Project Layout

| File | Purpose |
| --- | --- |
| `options_dashboard.py` | Streamlit dashboard for option-chain analysis, Black-Scholes pricing, Greeks, and volatility-smile charts. |
| `options_pricing.py` | Black-Scholes pricing utilities used by the dashboard. It can also be run from the command line. |
| `stock_screener.py` | Main enhanced stock screener with technical indicators, market-regime features, model scoring, backtesting, and Alpaca paper-trading support. |
| `ai_stock_screener.py` | Experimental workflow that asks OpenAI for stock ideas, filters them, scores them with the ensemble, and can submit Alpaca paper trades. |
| `ensemble_backtester.py` | Neural-network ensemble training and backtesting workflow. |
| `ensemble_trading_bot.py` | Earlier ensemble bot workflow retained as a reference implementation. |
| `model_0.pth` to `model_4.pth` | Saved PyTorch ensemble checkpoints. |
| `stock_trading_model.pth` | Saved PyTorch checkpoint for the enhanced screener. |
| `stock_regressor_model.pth` | Saved PyTorch regression checkpoint. |
| `feature_scaler.joblib` | Saved feature scaler used before model inference. |
| `stock_trading_model.joblib` | Saved scikit-learn model artifact. |
| `.env.example` | Template for local environment variables. |
| `requirements.txt` | Python dependencies. |

## Setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in only the keys needed for the workflow you want to run.

```powershell
Copy-Item .env.example .env
```

Expected environment variables:

| Variable | Used for |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI-assisted ticker idea generation in `ai_stock_screener.py`. |
| `FMP_API_KEY` | Insider-trading data in `stock_screener.py`. |
| `ALPACA_API_KEY` | Alpaca paper-trading access. |
| `ALPACA_SECRET_KEY` | Alpaca paper-trading access. |
| `ALPACA_BASE_URL` | Alpaca API base URL. Defaults to paper trading. |

## Usage

Run the options dashboard:

```powershell
streamlit run options_dashboard.py
```

Train or backtest the ensemble model:

```powershell
python ensemble_backtester.py --mode train
python ensemble_backtester.py --mode backtest
```

Run the enhanced screener:

```powershell
python stock_screener.py --mode screen
```

Run a backtest with the enhanced screener:

```powershell
python stock_screener.py --mode backtest
```

Run the OpenAI-assisted screener:

```powershell
python ai_stock_screener.py --mode screen
```

## Security

Do not commit real API keys. Keep credentials in `.env`, your shell profile, or your deployment environment. If a key is ever committed or shared, rotate it immediately.

## Disclaimer

This repository is for research and educational use only. It is not financial advice, and it should not be used for live trading without independent review, risk controls, and thorough testing.
