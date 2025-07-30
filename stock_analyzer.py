import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import requests
from groq import Groq
import io
import sys
import re

st.set_page_config(layout="wide", page_title="Portfolio Analysis")

# --- Initialize Session State Flag ---
if 'is_data_processed' not in st.session_state:
    st.session_state.is_data_processed = False

# ---------------------- Function Definitions ----------------------
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

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("💡 Portfolio FAQs")
    st.markdown("**Q1:** What is XIRR?\n\nAnnualized return considering irregular cash flows.")
    st.markdown("**Q2:** Why is portfolio value changing?\n\nBecause of price changes, currency rates, and quantity held.")
    st.markdown("**Q3:** Why is some data missing?\n\nMissing ticker or price history in Yahoo Finance.")
    st.markdown("**Q4:** How is cash flow calculated?\n\nCash Flow = -1 × Quantity × Transaction Price")

    st.divider()

    if st.button("🔄 Clear Data"):
        st.session_state.is_data_processed = False
        st.session_state.pop('combined_df', None)
        st.session_state.pop('chat_history', None)
        st.rerun()

    st.divider()
    st.subheader("🤖 Ask AI About Your Portfolio (ReAct Agent)")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if not st.session_state.get("is_data_processed", False):
        st.info("Upload your CSVs to enable the AI chat functionality.")
    else:
        user_input = st.text_input("Enter your question", key="chat_input")

        def python_repl_tool(code, df):
            import io, sys
            old_stdout = sys.stdout
            redirected_output = sys.stdout = io.StringIO()
            try:
                exec_globals = {'df': df, 'pd': pd, 'np': np}
                exec(code, exec_globals)
                output = redirected_output.getvalue().strip()
            except Exception as e:
                output = f"Error: {e}"
            finally:
                sys.stdout = old_stdout
            return output

        if user_input:
            df = st.session_state.combined_df
            columns = df.columns.tolist()

            system_prompt = f"""
You are an AI assistant with access to a Python REPL to help analyze a stock portfolio.
The portfolio data is stored in a pandas DataFrame named `df`.

IMPORTANT: Use only the following column names: {columns}
Note: The 'MTM P/L' column is not available, so you cannot reference it.

Follow these rules strictly:

1. You must always think step-by-step and show your reasoning before coding.
2. Your response must include:
   - Thought: Your analysis plan in plain English.
   - Action: Always write 'python_repl_tool'.
   - Action Input: Python code to be executed using the REPL.
3. The `Action Input` must end with `print()` showing the final insight, like:
   `print("The top-performing ticker is XYZ with a gain of 12.4%.")`
4. After REPL executes your code and Observation is returned, you must output a `Final Answer`.
   This Final Answer should clearly summarize the insight, using the observation you received.

💡 Format Example:
Thought: Explain what you're trying to do.
Action: python_repl_tool
Action Input: <code>
... (Observation inserted here)
Final Answer: <your conclusion based on the observation>

Begin!
"""

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
            full_llm_output = ""
            if user_input:
                # Save user input to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                    })

                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
                full_llm_output = ""

                try:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                    with st.spinner("Thinking..."):
                        response_container = st.empty()

                        for _ in range(5):
                            response_stream = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=messages,
                                temperature=0,
                                stream=True,
                            )
        
                            full_llm_turn = ""
                            for chunk in response_stream:
                                delta = chunk.choices[0].delta.content or ""
                                full_llm_turn += delta
                                response_container.markdown(f"**AI (thinking)**:\n{full_llm_turn}")

                            full_llm_output += f"\n\n{full_llm_turn}"

                            # Parse the action input
                            action_input_match = re.search(r"Action Input:\s*```python\s*(.*?)\s*```", full_llm_turn, re.DOTALL)
                            if not action_input_match:
                                action_input_match = re.search(r"Action Input:\s*(.*)", full_llm_turn, re.DOTALL)

                            if action_input_match:
                                action_input = action_input_match.group(1).strip()
                                observation = python_repl_tool(action_input, df)

                                # Add full turn and observation to message stack
                                messages.append({"role": "assistant", "content": full_llm_turn})
                                messages.append({"role": "user", "content": f"Observation: {observation}"})

                                # Summarize observation
                                summarized_obs = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": f"Summarize the following observation from a Python output into a meaningful insight for a finance user, aligned with User input: {user_input}"
                                        },
                                        {"role": "user", "content": observation}
                                        ],
                                    temperature=0.1
                                )
                                summary_text = summarized_obs.choices[0].message.content.strip()

                                # Save assistant turn to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "thought": re.search(r"Thought:\s*(.*)", full_llm_turn, re.DOTALL).group(1).strip() if 'Thought:' in full_llm_turn else '',
                                    "action": "python_repl_tool",
                                    "action_input": action_input,
                                    "observation": observation,
                                    "final_answer": summary_text
                                })

                                    # Display current turn
                                with st.expander("Thought: "):
                                    st.markdown(f"**🧠 Thought & Action:**\n```markdown\n{full_llm_turn}\n```\n"
                                    f"**📊 Observation:**\n```python\n{observation}\n```")
                                st.success(f"✅ Final Answer:\n\n{summary_text}")
                            else:
                                st.error("❌ Failed to parse LLM action.")
                                break

                            if "Final Answer:" in full_llm_turn:
                                break

                except Exception as e:
                    st.error(f"❌ LLM Error: {e}")

            # Replay the conversation
            if st.session_state.chat_history:
                for chat in reversed(st.session_state.chat_history):
                    with st.chat_message(chat['role']):
                        if chat["role"] == "assistant" and all(k in chat for k in ["thought", "action", "action_input", "observation", "final_answer"]):
                            st.markdown("✅ **Final Answer:**")
                            st.success(chat["final_answer"])
                            with st.expander("🧠 Show Thought Process & Actions"):
                                st.markdown("**🧠 Thought:**")
                                st.markdown(chat["thought"])

                                st.markdown("**🛠️ Action:**")
                                st.code(chat["action"], language="text")

                                st.markdown("**📥 Code Executed:**")
                                st.code(chat["action_input"], language="python")

                                st.markdown("**📊 Observation:**")
                                st.code(chat["observation"])
                        elif chat["role"] == "user":
                            st.markdown(f"**You:** {chat['content']}")




# ---------------------- Main Dashboard Logic ----------------------
st.title("\U0001F4C8 Portfolio Returns & Value Tracker")

if not st.session_state.is_data_processed:
    uploaded_files = st.file_uploader(
        "Upload 3 Trade CSVs (one for each year)", type=['csv'], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) >=1 :
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
                    continue
                try:
                    ticker = yf.Ticker(symbol)
                    splits = ticker.splits
                    if splits is not None and not splits.empty:
                        for split_date, ratio in splits.items():
                            mask = (combined_df['Ticker'] == symbol) & (combined_df['Date'] < pd.Timestamp(split_date.date()))
                            combined_df.loc[mask, 'Quantity'] *= ratio
                            combined_df.loc[mask, 'Transaction_Price'] /= ratio
                except Exception as e:
                    st.warning(f"⚠️ Failed to fetch split data for {symbol}: {e}")
            combined_df['Cash_Flow'] = -1 * combined_df['Quantity'] * combined_df['Transaction_Price']

            st.session_state.combined_df = combined_df.copy()
            st.session_state.is_data_processed = True
            st.rerun()
    else:
        st.info("📂 Please upload a CSV file")

else:
    combined_df = st.session_state.combined_df
    
    unique_symbols = combined_df['Ticker'].unique()
    unique_currencies = combined_df['Currency'].unique()
    start_date_for_data = combined_df['Date'].min().date() - timedelta(days=5)
    end_date = datetime.now().date()
    
    st.subheader("📉 Fetching Historical Prices")
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
    st.subheader("📊 Historical Price Data for All Holdings")
    all_historical_dfs = []
    for symbol, df_data in historical_data.items():
        temp_df = df_data.copy()
        temp_df['Ticker'] = symbol
        all_historical_dfs.append(temp_df)
    if all_historical_dfs:
        combined_historical_df = pd.concat(all_historical_dfs)
        cols = ['Ticker'] + [col for col in combined_historical_df.columns if col != 'Ticker']
        st.dataframe(combined_historical_df[cols])
    else:
        st.info("No historical data could be fetched for any of the holdings.")
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
        for symbol, quantity in positions.items():
            if quantity != 0 and symbol in historical_data:
                try:
                    # Ensure continuous date index and forward-fill missing prices
                    close_series = historical_data[symbol]['Close']
                    full_index = pd.date_range(start=portfolio_value_df.index.min(), end=portfolio_value_df.index.max(), freq='D')
                    close_series = close_series.reindex(full_index).ffill().fillna(0)

                    price = close_series.loc[date]
                    portfolio_value_df.loc[date, symbol] = quantity * price
                except KeyError:
                    pass

    portfolio_value_df['Total Value (USD)'] = portfolio_value_df.sum(axis=1)
    st.subheader("📈 Daily Portfolio Value Over Time")
    st.line_chart(portfolio_value_df['Total Value (USD)'], use_container_width=True)
    st.subheader("🔍 Individual Holding Performance")
    holding_selected = st.selectbox(
        "Select a holding to view time series value:",
        options=[col for col in portfolio_value_df.columns if col != 'Total Value (USD)']
    )
    holding_series_filled = portfolio_value_df[holding_selected].ffill()
    st.line_chart(holding_series_filled, use_container_width=True)
    st.subheader("📅 Daily Portfolio Table (USD)")
    st.dataframe(portfolio_value_df.tail(10))
    st.subheader("📊 XIRR Per Holding")
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
    st.subheader("📰 Latest News Headlines")
    finnhub_api_key = st.secrets["FINHUB_API"]
    news = get_latest_news(unique_symbols, finnhub_api_key)
    for symbol, articles in news.items():
        st.markdown(f"**{symbol}**")
        for headline in articles:
            st.markdown(f"- {headline}")
