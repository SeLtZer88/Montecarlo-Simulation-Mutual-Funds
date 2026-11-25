"""
tests/test_data_loader.py
"""

import pandas as pd
import pytest
from src.data_loader import DataLoader


def test_data_loader_single_daily_csv(tmp_path):
    mock_csv = tmp_path / "monte_carlo_daily_returns.csv"
    mock_data = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-02", periods=5, freq="B"),
            "Fund_A": [0.010, -0.005, 0.012, 0.003, -0.002],
            "Fund_B": [0.004, 0.008, -0.006, 0.009, 0.001],
        }
    )
    mock_data.to_csv(mock_csv, index=False)

    custom_config = {
        "Fund_A": {"weight": 0.6},
        "Fund_B": {"weight": 0.4},
    }

    loader = DataLoader(asset_config=custom_config, data_dir=tmp_path)
    df_returns, weights = loader.load_aligned_dataset(
        start_date="2023-01-01", end_date="2023-01-10"
    )

    assert isinstance(df_returns, pd.DataFrame)
    assert len(df_returns) == 5
    assert list(df_returns.columns) == ["Fund_A", "Fund_B"]
    assert pytest.approx(sum(weights.values()), 1e-6) == 1.0
    assert round(df_returns["Fund_A"].iloc[0], 4) == 0.0100


if __name__ == "__main__":
    pytest.main([__file__])
