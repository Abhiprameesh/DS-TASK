import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRADES_PATH = os.path.join(DATA_DIR, "historical_data.csv")
FGI_PATH = os.path.join(DATA_DIR, "fear_greed_index.csv")


def load_trades(path: str = TRADES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse timestamp (IST) as datetime; keep date for aggregation
    df["timestamp_ist"] = pd.to_datetime(
        df["Timestamp IST"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    df["trade_date"] = df["timestamp_ist"].dt.date

    # Numeric fields
    numeric_cols = [
        "Execution Price",
        "Size Tokens",
        "Size USD",
        "Closed PnL",
        "Fee",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Standardized categorical fields
    if "Side" in df.columns:
        df["trade_side"] = df["Side"].str.upper().str.strip()
    if "Direction" in df.columns:
        df["position_event"] = df["Direction"].str.strip()

    # Net PnL = closed PnL - fee (approximate)
    df["net_pnl"] = df.get("Closed PnL", 0).fillna(0) - df.get("Fee", 0).fillna(0)

    return df


def load_fgi(path: str = FGI_PATH) -> pd.DataFrame:
    fgi = pd.read_csv(path)

    # Use the provided date column as canonical date
    fgi["date"] = pd.to_datetime(fgi["date"], format="%Y-%m-%d", errors="coerce").dt.date
    fgi["value"] = pd.to_numeric(fgi["value"], errors="coerce")

    # Clean classification strings
    fgi["classification"] = fgi["classification"].astype(str).str.strip()

    return fgi


def aggregate_daily_trader_performance(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per account / coin / trade_date.

    Metrics:
    - total_gross_pnl: sum Closed PnL
    - total_fees
    - total_net_pnl
    - volume_usd: sum Size USD
    - n_trades
    - win_rate: share of trades with positive Closed PnL
    - long/short split via trade_side (BUY/SELL)
    """

    # Flag winning trades (realized PnL > 0)
    trades["is_win"] = trades["Closed PnL"].fillna(0) > 0

    group_keys = ["Account", "Coin", "trade_date"]

    agg = trades.groupby(group_keys).agg(
        total_gross_pnl=("Closed PnL", "sum"),
        total_fees=("Fee", "sum"),
        total_net_pnl=("net_pnl", "sum"),
        volume_usd=("Size USD", "sum"),
        n_trades=("Trade ID", "count"),
        win_trades=("is_win", "sum"),
    ).reset_index()

    agg["win_rate"] = np.where(agg["n_trades"] > 0, agg["win_trades"] / agg["n_trades"], np.nan)

    return agg


def join_with_sentiment(daily_perf: pd.DataFrame, fgi: pd.DataFrame) -> pd.DataFrame:
    # Align column names for join
    daily_perf = daily_perf.copy()
    daily_perf["date"] = daily_perf["trade_date"]

    merged = daily_perf.merge(fgi[["date", "value", "classification"]], on="date", how="left")

    # Simple sentiment regimes from numeric value if needed
    def numeric_to_regime(v):
        if pd.isna(v):
            return "Unknown"
        if v < 25:
            return "Extreme Fear"
        if v < 45:
            return "Fear"
        if v < 55:
            return "Neutral"
        if v < 75:
            return "Greed"
        return "Extreme Greed"

    merged["numeric_regime"] = merged["value"].apply(numeric_to_regime)

    return merged


def basic_eda(merged: pd.DataFrame, output_dir: str = DATA_DIR) -> None:
    """Produce simple summary tables and plots.

    Saves:
    - joined_daily_performance_with_sentiment.csv
    - png plots in the data directory
    """

    out_csv = os.path.join(output_dir, "joined_daily_performance_with_sentiment.csv")
    merged.to_csv(out_csv, index=False)

    # Overall correlation between sentiment value and net pnl
    corr = merged[["total_net_pnl", "value"]].dropna().corr().iloc[0, 1]
    print(f"Correlation between daily net PnL and Fear-Greed value: {corr:.4f}")

    # Set plotting style
    sns.set(style="whitegrid")

    # Boxplot: net PnL by classification
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged, x="classification", y="total_net_pnl")
    plt.title("Daily Net PnL by Fear-Greed Classification")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "pnl_by_sentiment_classification.png")
    plt.savefig(boxplot_path, dpi=150)
    plt.close()

    # Scatter: sentiment value vs net PnL
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged, x="value", y="total_net_pnl", alpha=0.5)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Daily Net PnL vs Fear-Greed Index Value")
    plt.xlabel("Fear-Greed Index Value")
    plt.ylabel("Daily Net PnL (USD)")
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "pnl_vs_fear_greed_value.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    # Bar: average win rate by sentiment regime
    win_rate_by_regime = merged.groupby("classification")["win_rate"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=win_rate_by_regime, x="classification", y="win_rate")
    plt.title("Average Daily Win Rate by Sentiment Classification")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    winrate_path = os.path.join(output_dir, "win_rate_by_sentiment_classification.png")
    plt.savefig(winrate_path, dpi=150)
    plt.close()

    print(f"Saved joined dataset to: {out_csv}")
    print(f"Saved plots to:\n - {boxplot_path}\n - {scatter_path}\n - {winrate_path}")


def main() -> None:
    print("Loading trades and sentiment data...")
    trades = load_trades(TRADES_PATH)
    fgi = load_fgi(FGI_PATH)

    print("Aggregating daily trader performance...")
    daily_perf = aggregate_daily_trader_performance(trades)

    print("Joining with market sentiment...")
    merged = join_with_sentiment(daily_perf, fgi)

    print("Rows in joined dataset:", len(merged))
    print(merged.head())

    print("Running basic exploratory analysis and saving outputs...")
    basic_eda(merged, DATA_DIR)


if __name__ == "__main__":
    main()
