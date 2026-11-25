"""
src/data_loader.py

Handles historical daily return ingestion from a single combined local CSV
with automatic daily yfinance fallback. Normalizes data schemas into aligned
daily percentage return series.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import yfinance as yf


# Default Asset Configuration matching the columns in data/monte_carlo_daily_returns.csv
DEFAULT_ASSET_CONFIG: Dict[str, Dict[str, Union[str, float]]] = {
    "Fidelity Nordic A-Dis SEK": {
        "ticker": "0P00000BVI.F",
        "isin": "LU0048588080",
        "weight": 0.3333,
    },
    "SEB Nordenfond A SEK": {
        "ticker": "0P00000L2O.ST",
        "isin": "SE0000984189",
        "weight": 0.2500,
    },
    "BGF US Basic Value A2": {
        "ticker": "0P00000BV4.F",
        "isin": "LU0072461881",
        "weight": 0.2500,
    },
    "BGF Sustainable Energy A2": {
        "ticker": "0P00000CWL.F",
        "isin": "LU0171289902",
        "weight": 0.1667,
    },
}


class DataLoader:
    """
    Data ingestion pipeline loading daily returns from a single combined CSV
    with daily yfinance fetching as a fallback.
    """

    def __init__(
        self,
        asset_config: Optional[Dict[str, Dict]] = None,
        data_dir: Union[str, Path] = "data",
        csv_filename: str = "monte_carlo_daily_returns.csv",
    ):
        self.asset_config = asset_config or DEFAULT_ASSET_CONFIG
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / csv_filename

    def load_combined_daily_csv(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Loads a single multi-asset daily return CSV file.
        Expected format: a 'Date' column plus one column of daily decimal returns per fund.
        """
        target_path = Path(file_path) if file_path else self.csv_path
        if not target_path.exists():
            raise FileNotFoundError(f"Combined daily returns CSV missing: {target_path}")

        df = pd.read_csv(target_path)

        # Locate the date column case-insensitively
        date_cols = [col for col in df.columns if col.strip().lower() == "date"]
        if not date_cols:
            raise KeyError(f"CSV file {target_path.name} must contain a 'Date' column.")

        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        df.index.name = "Date"

        expected_cols = list(self.asset_config.keys())
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise KeyError(
                f"CSV file {target_path.name} is missing expected fund columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        return df[expected_cols].astype(float)

    def fetch_single_asset_yfinance(
        self, ticker: str, start_date: str, end_date: Optional[str] = None
    ) -> Optional[pd.Series]:
        """
        Fallback: fetches daily adjusted close prices from yfinance and converts to daily returns.
        """
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if data.empty or "Close" not in data.columns:
                return None

            prices = data["Close"]
            if isinstance(prices, pd.DataFrame):
                prices = prices.squeeze()

            prices = prices.dropna()
            prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
            prices = prices[~prices.index.duplicated(keep="last")]

            returns = prices.pct_change().dropna()
            return returns if not returns.empty else None
        except Exception:
            return None

    def load_aligned_dataset(
        self, start_date: str = "2023-01-01", end_date: Optional[str] = "2026-01-31"
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Loads daily return series for all configured assets from the single local CSV
        (or yfinance fallback) and aligns them to the requested date window.
        """
        if self.csv_path.exists():
            print(f"Loading daily returns from local CSV: {self.csv_path}")
            df_aligned = self.load_combined_daily_csv(self.csv_path)
        else:
            print(f"Local CSV not found at {self.csv_path}. Fetching daily data from yfinance...")
            series_list: List[pd.Series] = []
            for asset_key, meta in self.asset_config.items():
                ticker = meta.get("ticker")
                series = (
                    self.fetch_single_asset_yfinance(ticker, start_date, end_date)
                    if ticker
                    else None
                )
                if series is None or series.empty:
                    raise RuntimeError(
                        f"Could not load daily data for '{asset_key}' via yfinance and {self.csv_path} is missing."
                    )
                series.name = asset_key
                series_list.append(series)

            df_aligned = pd.concat(series_list, axis=1).sort_index().ffill().dropna()

        # Filter strictly by start_date and end_date bounds
        df_aligned = df_aligned[df_aligned.index >= pd.to_datetime(start_date)]
        if end_date:
            df_aligned = df_aligned[df_aligned.index <= pd.to_datetime(end_date)]

        df_aligned = df_aligned.dropna()

        # Normalize weights so they sum strictly to 1.0
        raw_weights = {k: float(v["weight"]) for k, v in self.asset_config.items()}
        total_weight = sum(raw_weights.values())
        normalized_weights = {k: v / total_weight for k, v in raw_weights.items()}

        print("\nData Loading Complete:")
        print(
            f" - Aligned Window: {df_aligned.index.min().strftime('%Y-%m-%d')} "
            f"to {df_aligned.index.max().strftime('%Y-%m-%d')}"
        )
        print(f" - Total Daily Observations: {len(df_aligned)}")
        print(f" - Normalized Weights: {normalized_weights}\n")

        return df_aligned, normalized_weights
