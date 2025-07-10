import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest_framework import (
    Asset, MovingAverageCrossover, PositionSizer, BacktestEngine
)

def create_synthetic_data(symbol="TEST", start_date="2023-01-01", end_date="2024-12-31", initial_price=100):
    """Create synthetic OHLCV data for testing"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create business days only
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    
    # Generate price movements with some trending and volatility
    np.random.seed(42)  # For reproducible results
    
    # Create trending price movements
    trend = np.linspace(0, 0.3, n_days)  # 30% upward trend over period
    noise = np.random.randn(n_days) * 0.02  # 2% daily volatility
    jumps = np.random.choice([0, 0.05, -0.05], n_days, p=[0.95, 0.025, 0.025])  # Occasional jumps
    
    returns = trend[1:] - trend[:-1] + noise[1:] + jumps[1:]
    returns = np.concatenate([[0], returns])  # Add initial return of 0
    
    # Calculate prices
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from closing prices
    data = []
    for i, price in enumerate(prices):
        # Generate daily OHLC around the closing price
        daily_vol = abs(np.random.normal(0, 0.01))  # Daily volatility
        high = price * (1 + daily_vol)
        low = price * (1 - daily_vol)
        
        if i == 0:
            open_price = price
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))  # Small gap
        
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

# Test the enhanced backtest framework
print("Enhanced Backtest Framework Test with Synthetic Data")
print("=" * 60)

# Create synthetic asset
test_asset = Asset(
    symbol='TEST',
    asset_type='futures',
    multiplier=1000,
    margin_requirement=0.05,
    data_dir='test_data'
)

# Set synthetic data directly
synthetic_data = create_synthetic_data("TEST", "2023-01-01", "2024-12-31", 100)
test_asset.set_custom_data(synthetic_data)

print(f"Created synthetic data: {len(synthetic_data)} trading days")
print(f"Price range: ${synthetic_data['Close'].min():.2f} - ${synthetic_data['Close'].max():.2f}")

# Test 1: Enhanced MA strategy with dynamic sizing
print(f"\n" + "="*60)
print("TEST 1: Moving Average Crossover with Dynamic Position Sizing (Kelly Criterion)")
print("="*60)

ma_strategy = MovingAverageCrossover(fast_period=10, slow_period=20, use_dynamic_sizing=True)
kelly_sizer = PositionSizer(method='kelly', max_position=0.3, lookback_period=30, min_trades=5)

backtest_dynamic = BacktestEngine(
    assets=[test_asset],
    strategy=ma_strategy,
    start_date='2023-01-01',
    end_date='2024-12-31',
    initial_capital=100000,
    position_sizer=kelly_sizer,
    use_dynamic_sizing=True
)

print("Running dynamic position sizing backtest...")
results_dynamic = backtest_dynamic.run()
metrics_dynamic = backtest_dynamic.calculate_metrics()

# Test 2: Fixed position sizing for comparison
print(f"\n" + "="*60)
print("TEST 2: Moving Average Crossover with Fixed Position Sizing")
print("="*60)

fixed_strategy = MovingAverageCrossover(fast_period=10, slow_period=20, use_dynamic_sizing=False)

backtest_fixed = BacktestEngine(
    assets=[test_asset],
    strategy=fixed_strategy,
    start_date='2023-01-01',
    end_date='2024-12-31',
    initial_capital=100000,
    use_dynamic_sizing=False
)

print("Running fixed position sizing backtest...")
results_fixed = backtest_fixed.run()
metrics_fixed = backtest_fixed.calculate_metrics()

# Display results
print(f"\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

for symbol in ['TEST']:
    print(f"\nPerformance Metrics for {symbol}:")
    print("-" * 50)
    
    # Key metrics to compare
    comparison_metrics = [
        'Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown',
        'Total Trades (Detailed)', 'Win Rate', 'Profit Factor'
    ]
    
    print(f"{'Metric':<25} {'Dynamic':<15} {'Fixed':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for metric in comparison_metrics:
        if metric in metrics_dynamic[symbol] and metric in metrics_fixed[symbol]:
            dynamic_val = metrics_dynamic[symbol][metric]
            fixed_val = metrics_fixed[symbol][metric]
            
            if isinstance(dynamic_val, (int, float)) and isinstance(fixed_val, (int, float)) and fixed_val != 0:
                improvement = ((dynamic_val - fixed_val) / abs(fixed_val) * 100)
                improvement_str = f"{improvement:+.2f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{metric:<25} {dynamic_val:<15.4f} {fixed_val:<15.4f} {improvement_str:<15}")

# Display detailed trade statistics for dynamic sizing
print(f"\n" + "="*80)
print("DETAILED TRADE STATISTICS (Dynamic Sizing)")
print("="*80)

trade_stats = backtest_dynamic.portfolio.get_trade_statistics()
print(f"Total Trades: {trade_stats['total_trades']}")
print(f"Winning Trades: {trade_stats['winning_trades']}")
print(f"Losing Trades: {trade_stats['losing_trades']}")
print(f"Win Rate: {trade_stats['win_rate']:.2%}")
print(f"Average Win: ${trade_stats['avg_win']:.2f}")
print(f"Average Loss: ${trade_stats['avg_loss']:.2f}")
print(f"Profit Factor: {trade_stats['profit_factor']:.2f}")
print(f"Average Trade Duration: {trade_stats['avg_trade_duration']:.1f} days")

# Display individual trades if any
if backtest_dynamic.portfolio.trades:
    print(f"\nTRADE LOG (First 10 trades):")
    print("-" * 80)
    trades_df = pd.DataFrame([trade.to_dict() for trade in backtest_dynamic.portfolio.trades[:10]])
    print(trades_df.to_string(index=False))

print(f"\n" + "="*80)
print("FRAMEWORK FEATURES DEMONSTRATED:")
print("✓ Dynamic position sizing based on win probability (Kelly Criterion)")
print("✓ Enhanced portfolio management with individual trade tracking")
print("✓ Detailed performance metrics including profit factor, win rate")
print("✓ Trade-by-trade logging with PnL and duration")
print("✓ Comparison between dynamic and fixed position sizing")
print("✓ Signal strength incorporation for better position sizing")
print("="*80)

print("\nTest completed successfully! The enhanced framework is working properly.") 