import streamlit as st
import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from black_scholes import black_scholes_and_greeks, plot_volatility_smile

# --- Page Configuration ---
st.set_page_config(
    page_title="Black-Scholes Option Pricer",
    page_icon="📈",
    layout="wide"
)

st.title("Black-Scholes Option Pricing & Analysis Dashboard")

# --- Initialize Session State ---
# This helps the app remember that an analysis has been started.
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Inputs")
    ticker_symbol = st.text_input("Stock Ticker Symbol", "AAPL").upper()
    
    if st.button("Analyze Ticker"):
        # Set the flag to True when the button is clicked
        st.session_state.analysis_started = True
        # Clear previous cache if ticker changes
        st.cache_data.clear()

# --- Main App Logic ---
# The app now runs if the analysis_started flag is True
if st.session_state.analysis_started:
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # --- Fetch Data (with caching to improve performance) ---
        @st.cache_data
        def get_ticker_data(_ticker_obj):
            history = _ticker_obj.history(period='1d')
            if history.empty:
                return None, None
            price = history['Close'].iloc[0]
            expirations = _ticker_obj.options
            return price, expirations

        underlying_price, expirations = get_ticker_data(ticker)

        if underlying_price is None or not expirations:
            st.error(f"Could not fetch data for {ticker_symbol}. Please check the ticker.")
            st.session_state.analysis_started = False # Reset on error
        else:
            st.header(f"Analysis for {ticker_symbol}")
            st.metric("Current Stock Price", f"${underlying_price:,.2f}")

            # --- User Selection for Options ---
            with st.sidebar:
                selected_expiry = st.selectbox("Choose an expiration date:", expirations)
                
                @st.cache_data
                def get_option_chain_data(_ticker_obj, expiry):
                    opt_chain = _ticker_obj.option_chain(expiry)
                    return opt_chain.calls, opt_chain.puts

                calls, puts = get_option_chain_data(ticker, selected_expiry)

                option_type_choice = st.radio("Select Option Type:", ('Call', 'Put'))
                
                if calls is not None and not calls.empty:
                    strike_list = calls['strike'].tolist()
                    selected_strike = st.selectbox("Choose a strike price:", strike_list)
                else:
                    st.warning("No call options data available for this expiration.")
                    selected_strike = None


            # --- Display Volatility Smile ---
            if calls is not None and puts is not None:
                st.subheader("Volatility Smile")
                fig = plot_volatility_smile(calls, puts, ticker_symbol, selected_expiry)
                st.pyplot(fig)

            # --- Perform Calculation and Display Results ---
            if selected_strike:
                if option_type_choice == 'Call':
                    selected_option_data = calls[calls['strike'] == selected_strike].iloc[0]
                else:
                    selected_option_data = puts[puts['strike'] == selected_strike].iloc[0]

                market_price = selected_option_data['lastPrice']
                implied_volatility = selected_option_data['impliedVolatility']

                expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d')
                time_to_expiration = (expiry_date - datetime.now()).days / 365.0

                irx = yf.Ticker('^IRX')
                risk_free_rate = irx.history(period='1d')['Close'].iloc[0] / 100.0

                results = black_scholes_and_greeks(
                    underlying_price, selected_strike, time_to_expiration, risk_free_rate, implied_volatility, option_type_choice.lower()
                )
                
                theoretical_price = results['price']

                # --- Display Results in Columns ---
                st.subheader(f"Analysis for {selected_strike} {option_type_choice}")
                
                col1, col2 = st.columns(2)

                with col1:
                    st.info("Valuation Verdict")
                    if theoretical_price > market_price * 1.05:
                        st.success(f"Theoretically Undervalued")
                    elif theoretical_price < market_price * 0.95:
                        st.error(f"Theoretically Overvalued")
                    else:
                        st.warning(f"Theoretically Fairly Priced")
                    
                    st.metric("Model Price", f"${theoretical_price:,.4f}")
                    st.metric("Market Price", f"${market_price:,.4f}")


                with col2:
                    st.info("Option Greeks (Risk Metrics)")
                    greeks_df = pd.DataFrame({
                        'Metric': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
                        'Value': [
                            f"{results['delta']:.4f}",
                            f"{results['gamma']:.4f}",
                            f"{results['vega']:.4f} (per 1% vol change)",
                            f"{results['theta']:.4f} (per day)",
                            f"{results['rho']:.4f} (per 1% rate change)"
                        ]
                    })
                    st.table(greeks_df)

                st.caption("Disclaimer: This is a theoretical valuation based on the Black-Scholes model and is not financial advice.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.analysis_started = False # Reset on error
