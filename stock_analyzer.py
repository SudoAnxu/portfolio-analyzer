import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import requests

st.set_page_config(layout="wide", page_title="Portfolio Analysis")

# Function definitions

def xirr(cash_flows, dates):
    if len(cash_flows) != len(dates):
        raise ValueError("Cash flows and dates must have the same length.")
    dates_series = pd.Series(dates)
    dates_diff = (dates_series - dates_series.iloc[0]).dt.days
    def npv_func(rate):
        return sum(cf / (1 + rate)**(d / 365) for cf, d in zip(cash_flows, dates_diff))
    try:
        from scipy.optimize import newton
        return newton(npv_func, 0.1)
    except (ImportError, RuntimeError):
        low, high = -1.0, 1.0
        for _ in range(100):
            mid = (low + high) / 2
            if npv_func(mid) > 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2

def get_historical_forex_yfinance(start_date, end_date):
    forex_rates = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    forex_rates['USD'] = 1.0
    inr_data = yf.download('INR=X', start=start_date, end=end_date, progress=False, auto_adjust=True)
    sgd_data = yf.download('SGD=X', start=start_date, end=end_date, progress=False, auto_adjust=True)
    forex_rates['INR'] = inr_data['Close']
    forex_rates['SGD'] = sgd_data['Close']
    forex_rates.ffill(inplace=True)
    forex_rates.bfill(inplace=True)
    forex_rates = forex_rates.reset_index().rename(columns={'index': 'Date'}).melt(
        id_vars=['Date'], var_name='Currency', value_name='Rate')
    return forex_rates

def get_latest_news(symbols, api_key):
    latest_news = {}
    for symbol in symbols:
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-07-01&to={datetime.now().strftime('%Y-%m-%d')}&token={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            news_data = response.json()
            if news_data:
                articles = [article['headline'] for article in news_data[:5]]
                latest_news[symbol] = articles
            else:
                latest_news[symbol] = ["No recent news found."]
        except requests.exceptions.RequestException as e:
            latest_news[symbol] = [f"Failed to fetch news: {e}"]
    return latest_news

# --- STREAMLIT UI ---
st.title("\U0001F4C8 Portfolio Returns & Value Tracker")

uploaded_files = st.file_uploader(
    "Upload 3 Trade CSVs (one for each year)", type=['csv'], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 3:
    with st.spinner("Processing..."):
        dfs = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            df_filtered = df[df['DataDiscriminator'] == 'Order'].copy()
            dfs.append(df_filtered)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.dropna(how='all', inplace=True)
        combined_df.dropna(axis=1, how='all', inplace=True)

        combined_df['Quantity'] = combined_df['Quantity'].astype(str).str.replace(',', '', regex=True).astype(float)
        combined_df['Date'] = pd.to_datetime(combined_df['Date/Time']).dt.tz_localize(None)
        combined_df.drop(columns=['Date/Time', 'Trades', 'Header', 'DataDiscriminator', 'Code', 'MTM P/L', 'Proceeds'], inplace=True, errors='ignore')
        combined_df.rename(columns={
            'Asset Category': 'Asset_Category',
            'T. Price': 'Transaction_Price',
            'C. Price': 'Closing_Price',
            'Comm/Fee': 'Commission',
            'Realized P/L': 'Realized_PL',
            'Symbol': 'Ticker'
        }, inplace=True)

        st.subheader("\u2705 Adjusting for Stock Splits")
        for symbol in combined_df['Ticker'].unique():
            if symbol.upper() == 'C6L':
                continue  # skip invalid ticker
            try:
                ticker = yf.Ticker(symbol)
                splits = ticker.splits
                if splits is not None and not splits.empty:
                    for split_date, ratio in splits.items():
                        mask = (combined_df['Ticker'] == symbol) & (combined_df['Date'] < pd.Timestamp(split_date.date()))
                        combined_df.loc[mask, 'Quantity'] *= ratio
                        combined_df.loc[mask, 'Transaction_Price'] /= ratio
            except Exception as e:
                st.warning(f"\u26A0\ufe0f Failed to fetch split data for {symbol}: {e}")

        combined_df['Cash_Flow'] = -1 * combined_df['Quantity'] * combined_df['Transaction_Price']

        unique_symbols = combined_df['Ticker'].unique()
        unique_currencies = combined_df['Currency'].unique()
        start_date_for_data = combined_df['Date'].min().date() - timedelta(days=5)
        end_date = datetime.now().date()

        st.subheader("\U0001F4C9 Fetching Historical Prices")
        all_tickers = [sym for sym in unique_symbols if sym.upper() != 'C6L']
        historical_data_all = yf.download(all_tickers, start=start_date_for_data, end=end_date, progress=False, group_by='ticker', auto_adjust=True)

        historical_data = {}
        if isinstance(historical_data_all.columns, pd.MultiIndex):
            for symbol in all_tickers:
                if symbol in historical_data_all.columns.levels[0]:
                    historical_data[symbol] = historical_data_all[symbol].dropna(how='all')
        else:
            if not historical_data_all.empty:
                historical_data[all_tickers[0]] = historical_data_all

        unique_symbols = list(historical_data.keys())
        # --- Display ALL Historical Data (UPDATED ADDITION) ---
        st.subheader("Historical Price Data for All Holdings")
        
        all_historical_dfs = []
        for symbol, df_data in historical_data.items():
            temp_df = df_data.copy()
            temp_df['Ticker'] = symbol
            all_historical_dfs.append(temp_df)
        
        if all_historical_dfs:
            combined_historical_df = pd.concat(all_historical_dfs)
            # Reorder columns to have Ticker first for better readability
            cols = ['Ticker'] + [col for col in combined_historical_df.columns if col != 'Ticker']
            st.dataframe(combined_historical_df[cols])
        else:
            st.info("No historical data could be fetched for any of the holdings.")
        # --- End of UPDATED ADDITION ---
        forex_rates = get_historical_forex_yfinance(start_date_for_data, end_date)

        combined_df['Date_only'] = combined_df['Date'].dt.date
        forex_rates['Date_only'] = forex_rates['Date'].dt.date
        merged_transactions = pd.merge(combined_df, forex_rates, left_on=['Date_only', 'Currency'], right_on=['Date_only', 'Currency'], how='left')

        all_dates = pd.date_range(start=combined_df['Date'].min().date(), end=end_date, freq='D')
        portfolio_value_df = pd.DataFrame(index=all_dates)
        positions = {}

        for date in all_dates:
            daily_transactions = merged_transactions[merged_transactions['Date_only'] == date.date()]
            for _, row in daily_transactions.iterrows():
                symbol = row['Ticker']
                if symbol in historical_data:
                    if symbol not in positions:
                        positions[symbol] = 0
                    positions[symbol] += row['Quantity']

            total_value_usd = 0
            for symbol, quantity in positions.items():
                if quantity != 0:
                    try:
                        adj_close = historical_data[symbol]['Close'].loc[str(date.date())]
                        total_value_usd += quantity * adj_close
                    except KeyError:
                        pass

            for currency in unique_currencies:
                try:
                    rate = forex_rates[(forex_rates['Date_only'] == date.date()) & (forex_rates['Currency'] == currency)]['Rate'].iloc[0]
                    portfolio_value_df.loc[date, currency] = total_value_usd * rate
                except (IndexError, KeyError):
                    if date > all_dates[0]:
                        portfolio_value_df.loc[date, currency] = portfolio_value_df.loc[date - timedelta(days=1), currency]

        st.subheader("\U0001F4C8 Daily Portfolio Value (in each currency)")
        st.dataframe(portfolio_value_df.tail(10))

        st.subheader("\U0001F4CA XIRR Per Holding")
        xirr_results = {}
        for symbol in unique_symbols:
            holding_transactions = combined_df[combined_df['Ticker'] == symbol].copy()
            if not holding_transactions.empty and symbol in historical_data:
                try:
                    latest_price = historical_data[symbol]['Close'].iloc[-1]
                    latest_date = historical_data[symbol].index[-1]
                    final_quantity = holding_transactions['Quantity'].sum()
                    final_cash_flow_row = {
                        'Date': latest_date,
                        'Cash_Flow': final_quantity * latest_price
                    }
                    cash_flows_df = pd.concat([holding_transactions[['Date', 'Cash_Flow']], pd.DataFrame([final_cash_flow_row])], ignore_index=True)
                    xirr_val = xirr(cash_flows_df['Cash_Flow'].tolist(), cash_flows_df['Date'].tolist())
                    xirr_results[symbol] = f"{xirr_val * 100:.2f}%"
                except (ValueError, IndexError):
                    xirr_results[symbol] = "N/A"
        st.dataframe(pd.DataFrame.from_dict(xirr_results, orient='index', columns=['XIRR (%)']))

        st.subheader("\U0001F4F0 Latest News Headlines (Bonus)")
        finnhub_api_key = st.secrets["FINHUB_API"]
        news = get_latest_news(unique_symbols, finnhub_api_key)
        for symbol, articles in news.items():
            st.markdown(f"**{symbol}**")
            for headline in articles:
                st.markdown(f"- {headline}")
else:
    st.info("\U0001F4C2 Please upload **exactly 3 CSV files** (for 2023, 2024, and 2025).")
