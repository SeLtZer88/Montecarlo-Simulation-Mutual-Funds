"""
src/data_loader.py

Handles historical price data retrieval via yfinance with automatic local CSV fallbacks.
Normalizes data schemas into standardized monthly percentage return series aligned strictly
from a specified start date to the latest available date.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yfinance as yf


# Default Asset Configuration
DEFAULT_ASSET_CONFIG: Dict[str, Dict[str, Union[str, float]]] = {
    "Fidelity_Nordic": {
        "ticker": "0P00000L2O.ST",  # Fidelity Funds - Nordic Fund A-SEK
        "isin": "SE0000984189",
        "csv_filename": "SE0000984189.csv",
        "weight": 0.3333,
    },
    "Nordenfond": {
        "ticker": "0P00000BVI.F",   # Nordenfond equivalent ticker
        "isin": "LU0048588080",
        "csv_filename": "LU0048588080.csv",
        "weight": 0.2500,
    },
    "BGF_US_Basic_Value": {
        "ticker": "0P00000BV4.F",   # BlackRock Global Funds US Basic Value A2 USD
        "isin": "LU0072461881",
        "csv_filename": "LU0072461881.csv",
        "weight": 0.2500,
    },
    "BGF_Sustainable_Energy": {
        "ticker": "0P00000CWL.F",   # BlackRock Sustainable Energy A2 USD
        "isin": "LU0124384867",
        "csv_filename": "LU0124384867.csv",
        "weight": 0.1667,
    },
}


class DataLoader:
    """
    Data ingestion pipeline with yfinance fetching, local CSV fallback,
    and schema normalization for financial return series.
    """

    def __init__(
        self,
        asset_config: Optional[Dict[str, Dict]] = None,
        data_dir: Union[str, Path] = "data",
    ):
        self.asset_config = asset_config or DEFAULT_ASSET_CONFIG
        self.data_dir = Path(data_dir)

    def load_single_asset_csv(self, file_path: Path) -> pd.Series:
        """
        Loads a single asset return series from a local CSV file.
        Supports columns: ['date', 'close'] OR ['date', 'return_pct'].
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Local fallback file missing: {file_path}")

        df = pd.read_csv(file_path)
        df.columns = [col.strip().lower() for col in df.columns]

        if "date" not in df.columns:
            raise KeyError(f"CSV file {file_path.name} must contain a 'date' column.")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

        if "return_pct" in df.columns:
            returns = df["return_pct"].astype(float)
        elif "close" in df.columns or "adj close" in df.columns:
            price_col = "adj close" if "adj close" in df.columns else "close"
            monthly_prices = df[price_col].astype(float).resample("ME").last()
            returns = monthly_prices.pct_change().dropna()
        else:
            raise KeyError(
                f"CSV file {file_path.name} must contain either 'close' or 'return_pct' column."
            )

        return returns

    def fetch_single_asset_yfinance(
        self, ticker: str, start_date: str, end_date: Optional[str] = None
    ) -> Optional[pd.Series]:
        """
        Fetches adjusted monthly close prices from yfinance and converts to returns.
        """
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1mo",
                progress=False,
                auto_adjust=True,
            )
            if data.empty or "Close" not in data.columns:
                return None

            prices = data["Close"]
            if isinstance(prices, pd.DataFrame):
                prices = prices.squeeze()

            prices = prices.dropna().resample("ME").last()
            returns = prices.pct_change().dropna()
            return returns if not returns.empty else None
        except Exception:
            return None

    def load_asset_series(
        self,
        asset_key: str,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """
        Retrieves return series for a given asset using yfinance, falling back to CSV.
        """
        meta = self.asset_config[asset_key]
        ticker = meta.get("ticker")
        csv_name = meta.get("csv_filename", f"{asset_key}.csv")
        csv_path = self.data_dir / csv_name

        returns = None

        # Attempt 1: Fetch via yfinance
        if ticker:
            returns = self.fetch_single_asset_yfinance(ticker, start_date, end_date)

        # Attempt 2: Fallback to Local CSV
        if returns is None or len(returns) == 0:
            print(f"[{asset_key}] Yahoo Finance fetch unsuccessful/incomplete. Using fallback: {csv_path}")
            returns = self.load_single_asset_csv(csv_path)

        # Filter strictly by start_date and end_date bounds
        returns.index = pd.to_datetime(returns.index)
        returns = returns[returns.index >= pd.to_datetime(start_date)]
        if end_date:
            returns = returns[returns.index <= pd.to_datetime(end_date)]

        returns.name = asset_key
        return returns

    def load_aligned_dataset(
        self, start_date: str = "2010-01-01", end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Loads return series for all configured assets and aligns them to a shared time window via inner join.
        """
        series_list: List[pd.Series] = []
        raw_weights: Dict[str, float] = {}

        for asset_key in self.asset_config:
            series = self.load_asset_series(asset_key, start_date, end_date)
            series_list.append(series)
            raw_weights[asset_key] = float(self.asset_config[asset_key]["weight"])

        # Inner-join alignment to ensure identical dates across all 4 funds
        df_aligned = pd.concat(series_list, axis=1, join="inner").sort_index()

        # Normalize weights so they sum strictly to 1.0
        total_weight = sum(raw_weights.values())
        normalized_weights = {k: v / total_weight for k, v in raw_weights.items()}

        print(f"\nData Loading Complete:")
        print(f" - Aligned Window: {df_aligned.index.min().strftime('%Y-%m')} to {df_aligned.index.max().strftime('%Y-%m')}")
        print(f" - Total Monthly Observations: {len(df_aligned)}")
        print(f" - Normalized Weights: {normalized_weights}\n")

        return df_aligned, normalized_weights
