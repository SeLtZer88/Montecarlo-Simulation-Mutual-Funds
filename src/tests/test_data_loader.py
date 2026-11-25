"""
tests/test_data_loader.py
"""

from pathlib import Path
import pandas as pd
import pytest
from src.data_loader import DataLoader


def test_data_loader_fallback(tmp_path):
    # Create mock CSV
    mock_csv = tmp_path / "LU0072461881.csv"
    mock_data = pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=12, freq="ME"),
            "close": [100, 102, 101, 105, 108, 107, 110, 112, 115, 114, 118, 120],
        }
    )
    mock_data.to_csv(mock_csv, index=False)

    loader = DataLoader(data_dir=tmp_path)
    returns = loader.load_single_asset_csv(mock_csv)

    assert isinstance(returns, pd.Series)
    assert len(returns) == 11  # pct_change drops first row
    assert round(returns.iloc[0], 4) == 0.0200


if __name__ == "__main__":
    pytest.main([__file__])
