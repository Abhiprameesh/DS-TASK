# Trader Performance vs Bitcoin Market Sentiment

This mini-project explores the relationship between trader performance and Bitcoin market sentiment using two datasets provided in the assignment:

- **Bitcoin Market Sentiment Dataset** (`fear_greed_index.csv`)
- **Historical Trader Dataset** (`historical_data.csv`)

The goal is to understand how trading PnL and behavior vary across different sentiment regimes (e.g. *Extreme Fear*, *Fear*, *Neutral*, *Greed*).

---

## Project Structure

- `analysis.py`  
  Main script that:
  - Loads both datasets
  - Cleans and pre-processes the data
  - Aggregates trade-level data to daily performance metrics
  - Joins trader performance with the Fear & Greed Index
  - Produces summary tables, correlations, and visualizations

- `fear_greed_index.csv`  
  Daily Bitcoin Fear & Greed Index values with sentiment classification.

- `historical_data.csv`  
  Trade-level historical data from Hyperliquid (account, coin, price, size, side, closed PnL, timestamps, etc.).

- `requirements.txt`  
  Python dependencies to run the analysis.

- `output/` *(created when you run the script)*  
  Contains generated CSVs and plots, e.g.:
  - `daily_performance_with_sentiment.csv`
  - `correlations.csv`
  - `performance_by_sentiment_class.csv`
  - `scatter_pnl_vs_sentiment.png`
  - `boxplot_pnl_by_sentiment_class.png`
  - `bar_win_rate_by_sentiment_class.png`

---

## How the Analysis Works

### 1. Sentiment Data (`fear_greed_index.csv`)

Columns used:

- `date`: Calendar date of the index value (parsed as datetime)
- `value`: Numeric Fear & Greed Index (0–100, lower = fear, higher = greed)
- `classification`: Text label such as `Extreme Fear`, `Fear`, `Neutral`, `Greed`, etc.

In the script, these are renamed to:

- `sentiment_value`
- `sentiment_class`

The dataset is de-duplicated so there is at most one row per day.

### 2. Trader Data (`historical_data.csv`)

Key columns used:

- `Execution Price`
- `Size Tokens`, `Size USD`
- `Side` (BUY / SELL)
- `Timestamp IST` (trade timestamp, parsed to datetime with day-first format)
- `Closed PnL` (realized PnL for the trade)
- `Fee`

The script:

- Parses `Timestamp IST` into a proper datetime, then derives a daily key `trade_date` (midnight-normalized date).
- Coerces numeric columns like `Size USD`, `Closed PnL`, `Execution Price`, `Fee` into numeric types.
- Drops rows where `trade_date` cannot be parsed.

### 3. Daily Performance Metrics

Trade-level data is aggregated **per day** into a daily performance table with:

- `date` (from `trade_date`)
- `total_pnl` (sum of `Closed PnL` per day)
- `total_volume_usd` (sum of `Size USD` per day)
- `num_trades` (number of trades per day)
- `win_rate` (fraction of trades with positive `Closed PnL`)
- `avg_pnl_per_trade` (mean `Closed PnL` per trade)

This summarizes how the trader (or group of traders) performed each day.

### 4. Merging with Sentiment

The daily performance table is then left-joined with the sentiment table on `date`, producing:

- `daily_performance_with_sentiment.csv`

Each row corresponds to a calendar day, with both trading performance metrics and the associated Fear & Greed Index (numeric and classification) for that day.

### 5. Exploratory Data Analysis & Outputs

The script computes and saves:

- **Descriptive statistics** for numeric columns.
- **Correlation matrix** between key variables:
  - `sentiment_value`
  - `total_pnl`
  - `win_rate`
  - `total_volume_usd`
- **Grouped performance by sentiment class**, including:
  - Average and median daily PnL
  - Average daily win rate
  - Average daily volume
  - Number of days in each sentiment class

Visualizations (saved as PNGs):

- **Scatter plot**: `Daily PnL vs Fear & Greed Index`  
  Helps see whether higher greed/fear correlates with better or worse performance.

- **Boxplot**: `Distribution of Daily PnL by Sentiment Regime`  
  Compares PnL distributions across sentiment classes (`Extreme Fear`, `Fear`, `Neutral`, `Greed`, etc.).

- **Bar plot**: `Average Daily Win Rate by Sentiment Regime`  
  Shows how win rate changes as sentiment moves from fear to greed.

These outputs are designed to let you answer questions like:

- Do traders tend to perform better during **extreme fear** vs **extreme greed**?
- Is there a noticeable relationship between the **intensity** of sentiment (index value) and **PnL** or **win rate**?
- Does trading **volume** cluster in certain sentiment regimes?

---

## How to Run the Project

1. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis script**

   ```bash
   python analysis.py
   ```

4. **Inspect the results**

   - Check the `output/` folder for CSV files and plots.
   - Use the tables and figures to describe:
     - How performance varies across sentiment regimes
     - Any correlation patterns between sentiment and trader success

---

---

## Key Insights (Illustrative)

After aggregating daily performance and joining it with the Fear & Greed Index, a few patterns emerge:

- **Insight 1 – PnL by regime**  
  Daily PnL tends to be **[higher / lower]** during `Fear` compared to `Greed`,
  suggesting that **[brief interpretation, e.g. “contrarian opportunities in fearful markets”]**.

- **Insight 2 – Risk profile in extreme regimes**  
  `Extreme Fear` days show **[higher / lower]** variance in daily PnL than `Neutral` days,
  implying a **[riskier / more stable]** environment even if average PnL is **[similar / different]**.

- **Insight 3 – Volume vs performance**  
  Trading volume (USD) is **[concentrated / relatively flat]** in `Greed` regimes, but this does **[not always / often]**
  correspond to better PnL, indicating that **activity != profitability**.

- **Insight 4 – Correlations with sentiment value**  
  The numeric Fear & Greed Index shows a **[weak / moderate / strong]** correlation with **[total PnL / win rate]**
  (see `output/correlations.csv`), suggesting that **[short interpretation]**.

These insights are based on the provided snapshot of Hyperliquid data and should be interpreted as
exploratory rather than causal.

---

## Possible Extensions

To go beyond this baseline, the following extensions could be explored:

### Analysis Extensions

- **Per-account analysis**  
  Compute daily performance metrics per `Account` and study how different traders respond to the
  same sentiment regime.

- **Long vs short decomposition**  
  Split trades by `Side` (BUY vs SELL) and analyze whether long and short strategies behave
  differently across sentiment regimes.

- **Lagged sentiment effects**  
  Use today’s sentiment to explain **next-day** PnL, win rate, or volume
  (e.g., does extreme fear today set up better performance tomorrow?).

- **Risk metrics per regime**  
  Compare not just average PnL but also volatility, drawdowns, and downside risk by sentiment class.

### Modeling Ideas

- **Classification model for daily PnL sign**  
  Build a simple model (e.g. logistic regression or tree-based classifier) to predict whether
  next-day daily PnL is positive based on sentiment features and volume.

- **Position sizing heuristics**  
  Design simple rules that adapt trade size or risk limits based on the current sentiment regime
  (e.g., reduce leverage in `Extreme Greed` or `Extreme Fear`).

These ideas show how this exploratory analysis could evolve into a more production-like
research or trading framework.

