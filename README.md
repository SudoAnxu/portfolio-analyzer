
# 📈 Portfolio Analyzer

A Streamlit-based application to analyze equity trading data across multiple years, adjust for stock splits, convert currencies, calculate XIRR, and track portfolio value over time.

---

## 🚀 Features

- 📊 **Portfolio Value Tracking** – Computes daily portfolio value adjusted for splits and currency.
- 🔁 **Stock Split Adjustment** – Automatically adjusts quantity and price based on split history.
- 💵 **Currency Conversion** – Converts holdings between USD, INR, and SGD using historical forex rates.
- 📈 **XIRR Calculation** – Computes XIRR (Extended Internal Rate of Return) per holding.
- 🗞️ **Latest News** – Displays recent headlines for your holdings (via [Finnhub API](https://finnhub.io/)).

---

## 📂 Input Format

Upload exactly **3 CSV trade files**, one for each year (e.g., 2023, 2024, 2025). The CSV must include these columns:

```csv
Trades,Header,DataDiscriminator,Asset Category,Currency,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code
````

Only rows with `DataDiscriminator == 'Order'` are considered.

---

## 🛠️ Installation

```bash
git clone https://github.com/SudoAnxu/portfolio-analyzer.git
cd portfolio-analyzer
pip install -r requirements.txt
```

Ensure `streamlit` and `yfinance` are included in `requirements.txt`.

---

## 🔐 API Key

Add your **Finnhub API Key** in `.streamlit/secrets.toml`:

```toml
[general]
FINHUB_API = "your_finnhub_api_key"
```

---

## ▶️ Running the App

```bash
streamlit run portfolio_analyzer.py
```

Then go to `http://localhost:8501` in your browser.

---

## 📈 Output & Insights

* **Daily Portfolio Value** in all 3 currencies (USD, INR, SGD)
* **Stock-wise XIRR (%)**
* **Historical Prices** for all tickers
* **News Headlines** per symbol (optional)

---

## 🧠 How It Works

1. Cleans and combines uploaded trade files.
2. Adjusts for stock splits via `yfinance`.
3. Fetches historical forex rates.
4. Converts all trades into base currencies.
5. Fetches historical prices to calculate daily portfolio value.
6. Computes XIRR for each holding.
7. Fetches latest news from Finnhub.


---

## 📎 Notes

* Invalid or delisted tickers (e.g. `C6L`) are ignored.
* Script tolerates missing data and performs forward/backward fills where needed.
* You must upload **exactly 3 CSVs**, or the app won’t run.

---

## 📃 License

MIT License. Open-source contributions welcome!

---

## 🙋‍♂️ Questions?

Feel free to open an issue or contact me.




