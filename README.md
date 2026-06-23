# Trading Bot

A Python research workspace for options pricing, neural-network stock screening, backtesting, and Alpaca paper-trading experiments.

This repo is organized as a small Python package with model artifacts separated from source code. It is intended for experimentation and education, not live financial decision-making.

## Folder Structure

```text
.
├── artifacts/
│   └── models/
│       ├── model_0.pth ... model_4.pth
│       ├── feature_scaler.joblib
│       ├── stock_trading_model.pth
│       ├── stock_trading_model.joblib
│       └── stock_regressor_model.pth
├── trading_bot/
│   ├── options/
│   │   ├── dashboard.py
│   │   └── pricing.py
│   ├── screeners/
│   │   ├── stock_screener.py
│   │   └── ai_stock_screener.py
│   └── training/
│       ├── ensemble_backtester.py
│       └── ensemble_trading_bot.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Main Components

| Path | Purpose |
| --- | --- |
| `trading_bot/options/dashboard.py` | Streamlit dashboard for option-chain analysis, Black-Scholes pricing, Greeks, and volatility-smile charts. |
| `trading_bot/options/pricing.py` | Black-Scholes pricing utilities used by the dashboard. It can also be run from the command line. |
| `trading_bot/screeners/stock_screener.py` | Main enhanced stock screener with technical indicators, market-regime features, model scoring, backtesting, and Alpaca paper-trading support. |
| `trading_bot/screeners/ai_stock_screener.py` | Experimental workflow that asks OpenAI for stock ideas, filters them, scores them with the ensemble, and can submit Alpaca paper trades. |
| `trading_bot/training/ensemble_backtester.py` | Neural-network ensemble training and backtesting workflow. |
| `trading_bot/training/ensemble_trading_bot.py` | Earlier ensemble bot workflow retained as a reference implementation. |
| `artifacts/models/` | Saved model checkpoints and scalers used by the scripts. |

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
| `OPENAI_API_KEY` | OpenAI-assisted ticker idea generation. |
| `FMP_API_KEY` | Insider-trading data in the enhanced screener. |
| `ALPACA_API_KEY` | Alpaca paper-trading access. |
| `ALPACA_SECRET_KEY` | Alpaca paper-trading access. |
| `ALPACA_BASE_URL` | Alpaca API base URL. Defaults to paper trading. |

## Usage

Run the options dashboard:

```powershell
streamlit run trading_bot/options/dashboard.py
```

Run the options pricing CLI:

```powershell
python -m trading_bot.options.pricing
```

Train or backtest the ensemble model:

```powershell
python -m trading_bot.training.ensemble_backtester --mode train
python -m trading_bot.training.ensemble_backtester --mode backtest
```

Run the enhanced screener:

```powershell
python -m trading_bot.screeners.stock_screener --mode screen
python -m trading_bot.screeners.stock_screener --mode backtest
```

Run the OpenAI-assisted screener:

```powershell
python -m trading_bot.screeners.ai_stock_screener --mode screen
```

## Security

Do not commit real API keys. Keep credentials in `.env`, your shell profile, or your deployment environment. If a key is ever committed or shared, rotate it immediately.

## Disclaimer

This repository is for research and educational use only. It is not financial advice, and it should not be used for live trading without independent review, risk controls, and thorough testing.
