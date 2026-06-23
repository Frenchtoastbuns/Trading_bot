import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# N is the cumulative distribution function (CDF) 
# N_prime is the probability density function (PDF) 
N = norm.cdf
N_prime = norm.pdf

def black_scholes_and_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the Black-Scholes option price and its Greeks.
    """
    # Ensure inputs are floats 
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)

    # Handle T=0 case to avoid division by zero and instead using a very small number close to zero
    if T <= 0:
        T = 1e-10 
        
    # Handling Signma = 0 error
    if sigma <= 0:
        sigma = 1e-10

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # --- Shared Calculations for Greeks ---
    gamma = N_prime(d1) / (S * sigma * np.sqrt(T))
    vega = S * N_prime(d1) * np.sqrt(T) * 0.01

    if option_type.lower() == 'call':
        price = S * N(d1) - K * np.exp(-r * T) * N(d2)
        delta = N(d1)
        theta = (-(S * N_prime(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N(d2)) / 365
        rho = K * T * np.exp(-r * T) * N(d2) * 0.01
    elif option_type.lower() == 'put':
        price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
        delta = N(d1) - 1
        theta = (-(S * N_prime(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * N(-d2) * 0.01
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    return {
        'price': price, 'delta': delta, 'gamma': gamma,
        'vega': vega, 'theta': theta, 'rho': rho
    }

def plot_volatility_smile(calls, puts, ticker_symbol, expiry_date):
    """
    Plots the implied volatility against the strike price for calls and puts.
    """
    plt.figure(figsize=(12, 7))
    plt.scatter(calls['strike'], calls['impliedVolatility'] * 100, color='blue', label='Call Options')
    plt.scatter(puts['strike'], puts['impliedVolatility'] * 100, color='red', label='Put Options')
    
    plt.xlabel('Strike Price ($)')
    plt.ylabel('Implied Volatility (%)')
    plt.title(f'Volatility Smile for {ticker_symbol} - Expiration: {expiry_date}')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    # Pause for a moment to allow the plot to render before continuing
    plt.pause(1) 

    # Main
if __name__ == '__main__':
    try:
        # Input ticker symbol
        ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL): ").upper()
        ticker = yf.Ticker(ticker_symbol)

        # retreive the current stock price 
        underlying_price = ticker.history(period='1d')['Close'].iloc[0]
        if not underlying_price:
            print(f"Could not fetch current price for {ticker_symbol}.")
            exit()

        # retrieve available option expiration dates
        expirations = ticker.options
        if not expirations:
            print(f"No option expiration dates found for {ticker_symbol}.")
            exit()
            
        print("\nAvailable expiration dates:")
        for i, date in enumerate(expirations):
            print(f"{i+1}: {date}")
        
        exp_choice = int(input("Choose an expiration date (by number): ")) - 1
        selected_expiry = expirations[exp_choice]

        # retrieve the option chain for the selected date
        opt_chain = ticker.option_chain(selected_expiry)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Plot the Volatility Smile 
        plot_volatility_smile(calls, puts, ticker_symbol, selected_expiry)
        
        # Strategy Suggestions ---
        print("\n" + "="*45)
        print("Suggestions")
        print("="*45)
        print("Directional choice (Call/Put) depends on your market outlook (Bullish/Bearish).")
        print("Here are some key strike prices to consider based on market data:\n")

        # Find At-the-Money strike
        atm_strike = calls.iloc[(calls['strike'] - underlying_price).abs().argmin()]['strike']
        print(f"-> At-the-Money Strike: {atm_strike:.2f}")
        print("   (Closest to the current stock price. A common choice for balanced risk.)\n")
        
        # Find Highest Volume strike
        if not calls.empty and 'volume' in calls.columns and not calls['volume'].isnull().all():
            high_vol_strike = calls.loc[calls['volume'].idxmax()]['strike']
            print(f"-> Highest Volume Strike: {high_vol_strike:.2f}")
            print("   (Most traded strike today. Indicates high liquidity and interest.)\n")

        # Find Highest Open Interest strike
        if not calls.empty and 'openInterest' in calls.columns and not calls['openInterest'].isnull().all():
            high_oi_strike = calls.loc[calls['openInterest'].idxmax()]['strike']
            print(f"-> Highest Open Interest Strike: {high_oi_strike:.2f}")
            print("   (Most open contracts. Can act as a psychological support/resistance level.)")
        print("="*45 + "\n")

        # Display available strike prices 
        pd.set_option('display.max_rows', None) 
        print("\nAvailable Strike Prices (from Call options):")
        print(calls[['strike', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest']])
        
        strike_price = float(input("\nEnter your desired strike price from the list above: "))

        # Get user input for option type 
        option_type_choice = input("Analyze a 'call' or a 'put' option? ").lower()
        if option_type_choice not in ['call', 'put']:
            print("Invalid option type. Exiting.")
            exit()

        #  Select the option data
        if option_type_choice == 'call':
            selected_option = calls[calls['strike'] == strike_price]
        else:
            selected_option = puts[puts['strike'] == strike_price]

        if selected_option.empty:
            print(f"No {option_type_choice} option found for strike price {strike_price}.")
            exit()
            
        selected_option_data = selected_option.iloc[0]
        implied_volatility = selected_option_data['impliedVolatility']
        market_price = selected_option_data['lastPrice']

        expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d')
        time_to_expiration = (expiry_date - datetime.now()).days / 365.0

        # calcualting risk free rate compared to the 13 week treasury yield
        irx = yf.Ticker('^IRX')
        risk_free_rate = irx.history(period='1d')['Close'].iloc[0] / 100.0

        results = black_scholes_and_greeks(
            underlying_price, strike_price, time_to_expiration, risk_free_rate, implied_volatility, option_type_choice
        )
        
        theoretical_price = results['price']
        verdict = ""
        if theoretical_price > market_price * 1.05: # 5% threshold
            verdict = f"Theoretically Undervalued (Model Price: {theoretical_price:.2f} vs Market: {market_price:.2f})"
        elif theoretical_price < market_price * 0.95: # 5% threshold
            verdict = f"Theoretically Overvalued (Model Price: {theoretical_price:.2f} vs Market: {market_price:.2f})"
        else:
            verdict = f"Theoretically Fairly Priced (Model Price: {theoretical_price:.2f} vs Market: {market_price:.2f})"


        print("\n" + "="*60)
        print(f"      RESULTS FOR {ticker_symbol} {strike_price} {option_type_choice.upper()} OPTION")
        print("="*60)
        print(f"--- Data Inputs ---")
        print(f"Underlying Price (S):   {underlying_price:.2f}")
        print(f"Strike Price (K):       {strike_price:.2f}")
        print(f"Expiration Date:        {selected_expiry}")
        print(f"Time to Expiration (T): {time_to_expiration:.4f} years")
        print(f"Risk-Free Rate (r):     {risk_free_rate:.4%}")
        print(f"Implied Volatility (σ): {implied_volatility:.4%}")
        
        print("\n" + "-" * 35)
        print("         Calculated Results")
        print("-" * 35)
        for key, value in results.items():
            print(f"{key.capitalize():<8}: {value:10.4f}")
        print("-" * 35)
        
        print("\n" + "="*60)
        print("Theoretical Valuation")
        print("="*60)
        print(verdict)
        print("="*60)


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the ticker is correct.")

