import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = Path(__file__).resolve().parent
FEAR_GREED_PATH = DATA_DIR / "fear_greed_index.csv"
TRADES_PATH = DATA_DIR / "historical_data.csv"
OUTPUT_DIR = DATA_DIR / "output"


def load_fear_greed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure date is a proper datetime and use it as the main time key
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    # Keep only relevant columns
    df = df[["date", "value", "classification"]]
    df = df.rename(columns={"value": "sentiment_value", "classification": "sentiment_class"})
    # In case there are multiple rows per day, keep the latest by timestamp
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse timestamp columns
    if "Timestamp IST" in df.columns:
        df["timestamp_ist"] = pd.to_datetime(
            df["Timestamp IST"], errors="coerce", dayfirst=True
        )
        df["trade_date"] = df["timestamp_ist"].dt.normalize()
    else:
        # Fallback to a generic "Timestamp" column if format ever changes
        ts_col = "Timestamp" if "Timestamp" in df.columns else None
        if ts_col is None:
            raise ValueError("No recognizable timestamp column found in historical_data.csv")
        df["timestamp_ist"] = pd.to_datetime(df[ts_col], errors="coerce")
        df["trade_date"] = df["timestamp_ist"].dt.normalize()

    # Numeric columns
    for col in ["Size USD", "Closed PnL", "Execution Price", "Fee"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter out rows without a trade_date
    df = df.dropna(subset=["trade_date"]).copy()
    return df


def aggregate_daily_performance(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw trade-level data to daily performance metrics per date.

    Metrics include:
    - total_pnl
    - total_volume_usd
    - num_trades
    - win_rate (share of trades with positive closed PnL)
    - avg_pnl_per_trade
    """

    has_pnl = "Closed PnL" in trades.columns

    grouped = trades.groupby("trade_date")

    daily = grouped.agg(
        total_volume_usd=("Size USD", "sum"),
        num_trades=("Size USD", "count"),
    )

    if has_pnl:
        pnl_sum = grouped["Closed PnL"].sum().rename("total_pnl")
        win_rate = grouped.apply(
            lambda g: np.nan if g["Closed PnL"].isna().all() else (g["Closed PnL"] > 0).mean()
        ).rename("win_rate")
        avg_pnl_per_trade = grouped["Closed PnL"].mean().rename("avg_pnl_per_trade")

        daily = daily.join([pnl_sum, win_rate, avg_pnl_per_trade])

    daily = daily.reset_index().rename(columns={"trade_date": "date"})
    return daily


def merge_with_sentiment(daily: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    merged = daily.merge(sentiment, on="date", how="left")
    return merged


def basic_eda(merged: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save merged dataset for inspection
    merged.to_csv(OUTPUT_DIR / "daily_performance_with_sentiment.csv", index=False)

    # Text summary
    print("\n=== Basic Descriptive Stats ===")
    print(merged.describe(include="all"))

    # Correlation between sentiment value and performance metrics
    corr_cols = [c for c in ["sentiment_value", "total_pnl", "win_rate", "total_volume_usd"] if c in merged.columns]
    corr_df = merged[corr_cols].corr()
    print("\n=== Correlation Matrix (key metrics vs sentiment) ===")
    print(corr_df)
    corr_df.to_csv(OUTPUT_DIR / "correlations.csv")

    # Group by sentiment class
    if "sentiment_class" in merged.columns:
        grp = merged.groupby("sentiment_class").agg(
            avg_pnl=("total_pnl", "mean"),
            median_pnl=("total_pnl", "median"),
            avg_win_rate=("win_rate", "mean"),
            avg_volume=("total_volume_usd", "mean"),
            num_days=("date", "count"),
        )
        print("\n=== Performance by Sentiment Regime ===")
        print(grp)
        grp.to_csv(OUTPUT_DIR / "performance_by_sentiment_class.csv")

    # Visuals
    sns.set(style="whitegrid")

    # 1) Sentiment vs PnL scatter
    if {"sentiment_value", "total_pnl"}.issubset(merged.columns):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=merged, x="sentiment_value", y="total_pnl")
        plt.title("Daily PnL vs Fear & Greed Index")
        plt.xlabel("Fear & Greed Index (higher = greed)")
        plt.ylabel("Total Daily PnL")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "scatter_pnl_vs_sentiment.png", dpi=200)
        plt.close()

    # 2) Boxplot of PnL by sentiment class
    if {"sentiment_class", "total_pnl"}.issubset(merged.columns):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=merged, x="sentiment_class", y="total_pnl", order=sorted(merged["sentiment_class"].dropna().unique()))
        plt.title("Distribution of Daily PnL by Sentiment Regime")
        plt.xlabel("Sentiment Class")
        plt.ylabel("Total Daily PnL")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "boxplot_pnl_by_sentiment_class.png", dpi=200)
        plt.close()

    # 3) Barplot of average win rate by sentiment class
    if {"sentiment_class", "win_rate"}.issubset(merged.columns):
        win_rate_by_sentiment = merged.groupby("sentiment_class")["win_rate"].mean().reset_index()
        plt.figure(figsize=(8, 5))
        sns.barplot(data=win_rate_by_sentiment, x="sentiment_class", y="win_rate", order=sorted(win_rate_by_sentiment["sentiment_class"].dropna().unique()))
        plt.title("Average Daily Win Rate by Sentiment Regime")
        plt.xlabel("Sentiment Class")
        plt.ylabel("Average Win Rate")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "bar_win_rate_by_sentiment_class.png", dpi=200)
        plt.close()


def main() -> None:
    print("Loading Fear & Greed Index data from:", FEAR_GREED_PATH)
    sentiment_df = load_fear_greed(FEAR_GREED_PATH)

    print("Loading historical trader data from:", TRADES_PATH)
    trades_df = load_trades(TRADES_PATH)

    print("Aggregating daily performance metrics...")
    daily_perf = aggregate_daily_performance(trades_df)

    print("Merging with market sentiment...")
    merged = merge_with_sentiment(daily_perf, sentiment_df)

    print("Running basic EDA and saving outputs to:", OUTPUT_DIR)
    basic_eda(merged)

    print("\nDone. Key artifacts saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
