# Trading Bot

Python tools for Black-Scholes option analysis, neural-network stock screening, backtesting, and paper-trading experiments.

## What is included

- `app.py` - Streamlit dashboard for Black-Scholes option pricing and Greeks.
- `black_scholes.py` - command-line Black-Scholes option analyzer.
- `trading_algo.py` - neural-network training and backtesting workflow.
- `nn_screener_enhanced.py` - enhanced stock screener with market-regime, insider, and trading integrations.
- `insider_fetcher.py` - OpenAI-assisted screener and Alpaca paper-trading workflow.
- `*.pth`, `*.joblib` - saved model and scaler artifacts used by the scripts.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in any API keys needed for the workflows you plan to run. The scripts read credentials from environment variables:

- `OPENAI_API_KEY`
- `FMP_API_KEY`
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL`

## Usage

Run the Streamlit dashboard:

```powershell
streamlit run app.py
```

Train or backtest the neural network workflow:

```powershell
python trading_algo.py --mode train
python trading_algo.py --mode backtest
```

Run the enhanced screener:

```powershell
python nn_screener_enhanced.py --mode screen
```

## Security note

Do not commit real API keys. Keep secrets in `.env` or your shell environment.

## Disclaimer

This project is for research and educational use only. It is not financial advice.
