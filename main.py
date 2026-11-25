"""
main.py

Runs a correlated daily Monte Carlo portfolio simulation using the aligned
returns loaded by src/data_loader.py (252 trading days/year).
"""

import matplotlib
matplotlib.use("Agg")  # Headless backend for GitHub Codespaces

import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import DataLoader


def run_monte_carlo(
    initial_capital: float = 100_000.0,
    years: int = 5,
    num_simulations: int = 10_000,
    seed: int = 42,
):
    # 1. Load daily returns and normalized weights from data/monte_carlo_daily_returns.csv
    loader = DataLoader(data_dir="data", csv_filename="monte_carlo_daily_returns.csv")
    df_returns, weights_dict = loader.load_aligned_dataset(
        start_date="2023-01-01",
        end_date="2026-01-31",
    )

    asset_names = list(df_returns.columns)
    weight_vec = np.array([weights_dict[name] for name in asset_names])

    # 2. Estimate daily mean returns and covariance matrix
    mean_daily = df_returns.mean().values
    cov_daily = df_returns.cov().values
    cholesky_L = np.linalg.cholesky(cov_daily)

    trading_days = years * 252
    n_assets = len(asset_names)

    # 3. Simulate correlated daily asset paths via Cholesky decomposition
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((trading_days, n_assets, num_simulations))
    # Correlated daily returns: shape (trading_days, n_assets, num_simulations)
    sim_daily_returns = mean_daily[:, None] + np.einsum("ij,tjs->tis", cholesky_L, z)
    sim_daily_returns = np.clip(sim_daily_returns, -0.99, None)

    # Buy-and-hold asset growth paths combined by initial portfolio weights
    asset_growth = np.cumprod(1.0 + sim_daily_returns, axis=0)
    portfolio_paths = initial_capital * np.tensordot(asset_growth, weight_vec, axes=([1], [0]))

    # Prepend Day 0 initial capital
    start_row = np.full((1, num_simulations), initial_capital)
    portfolio_paths = np.vstack([start_row, portfolio_paths])

    # 4. Compute summary statistics
    ending_values = portfolio_paths[-1]
    cagr = (ending_values / initial_capital) ** (1.0 / years) - 1.0

    hist_port_daily = df_returns.values @ weight_vec
    ann_return = hist_port_daily.mean() * 252
    ann_vol = hist_port_daily.std() * np.sqrt(252)

    print("=== Historical Portfolio Stats (Daily Annualized) ===")
    print(f"Annualized Expected Return : {ann_return:.2%}")
    print(f"Annualized Volatility      : {ann_vol:.2%}")
    print(f"\n=== {years}-Year Monte Carlo Results ({num_simulations:,} runs) ===")
    print(f"Initial Capital            : {initial_capital:,.2f}")
    print(f"Mean Ending Value          : {np.mean(ending_values):,.2f}")
    print(f"Median Ending Value (50th) : {np.median(ending_values):,.2f} (CAGR: {np.median(cagr):.2%})")
    print(f"5th Percentile (Downside)  : {np.percentile(ending_values, 5):,.2f} (CAGR: {np.percentile(cagr, 5):.2%})")
    print(f"95th Percentile (Upside)   : {np.percentile(ending_values, 95):,.2f} (CAGR: {np.percentile(cagr, 95):.2%})")
    print(f"Probability of Loss        : {np.mean(ending_values < initial_capital):.2%}")

    # 5. Save fan chart to PNG
    days_axis = np.arange(trading_days + 1) / 252.0
    p5 = np.percentile(portfolio_paths, 5, axis=1)
    p25 = np.percentile(portfolio_paths, 25, axis=1)
    p50 = np.percentile(portfolio_paths, 50, axis=1)
    p75 = np.percentile(portfolio_paths, 75, axis=1)
    p95 = np.percentile(portfolio_paths, 95, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(days_axis, p5, p95, color="#3b82f6", alpha=0.15, label="5th–95th Percentile")
    plt.fill_between(days_axis, p25, p75, color="#3b82f6", alpha=0.30, label="25th–75th Percentile")
    plt.plot(days_axis, p50, color="#1d4ed8", linewidth=2, label="Median Path (50th)")
    plt.axhline(initial_capital, color="gray", linestyle="--", linewidth=1, label="Initial Capital")
    plt.title(f"Monte Carlo Portfolio Simulation ({years} Years, {num_simulations:,} Simulations)")
    plt.xlabel("Years")
    plt.ylabel("Portfolio Value")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("monte_carlo_results.png", dpi=150)
    print("\nSaved simulation chart to: monte_carlo_results.png")


if __name__ == "__main__":
    run_monte_carlo()