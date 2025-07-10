# SPX Moving Average Crossover Trading Strategy

A comprehensive trend-following trading strategy for the S&P 500 (SPX) using moving average crossovers. This strategy generates buy and sell signals based on the crossover of fast and slow moving averages.

## Features

- **Complete Trading System**: Fetches data, generates signals, backtests performance, and calculates metrics
- **Comprehensive Analysis**: Includes performance metrics like Sharpe ratio, alpha, beta, maximum drawdown
- **Parameter Optimization**: Built-in functionality to optimize moving average parameters
- **Visualization**: Detailed charts showing price action, signals, returns, and drawdowns
- **Multiple Timeframes**: Flexible date ranges for backtesting
- **Risk Management**: Calculates key risk metrics and position sizing

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib yfinance scikit-learn seaborn
```

## Quick Start

### Basic Usage

```python
from spx_ma_crossover_strategy import SPXTrendFollowingStrategy

# Create strategy instance
strategy = SPXTrendFollowingStrategy(fast_ma=20, slow_ma=50, initial_capital=100000)

# Run complete analysis
metrics = strategy.run_full_analysis(start_date='2020-01-01')
```

### Custom Parameters

```python
# Test different moving average periods
strategy = SPXTrendFollowingStrategy(
    fast_ma=10,           # Fast moving average period
    slow_ma=30,           # Slow moving average period  
    initial_capital=50000 # Starting capital
)

# Run analysis for specific date range
metrics = strategy.run_full_analysis(
    start_date='2022-01-01', 
    end_date='2024-01-01'
)
```

### Step-by-Step Analysis

```python
from spx_ma_crossover_strategy import SPXTrendFollowingStrategy

# Initialize strategy
strategy = SPXTrendFollowingStrategy(fast_ma=15, slow_ma=40)

# Step 1: Fetch SPX data
strategy.fetch_spx_data(start_date='2020-01-01')

# Step 2: Calculate moving averages
strategy.calculate_moving_averages()

# Step 3: Generate trading signals
strategy.generate_signals()

# Step 4: Backtest strategy
strategy.backtest_strategy()

# Step 5: Calculate performance metrics
metrics = strategy.calculate_performance_metrics()

# Step 6: Display results and plot charts
strategy.print_performance_summary()
strategy.plot_results()
```

## Parameter Optimization

Find the best moving average parameters for your timeframe:

```python
from spx_ma_crossover_strategy import optimize_ma_parameters

# Optimize parameters
best_params, results_df = optimize_ma_parameters(
    start_date='2020-01-01',
    end_date='2024-01-01',
    fast_range=(5, 25),    # Test fast MA from 5 to 25
    slow_range=(30, 100)   # Test slow MA from 30 to 100
)

print(f"Best parameters: Fast MA = {best_params[0]}, Slow MA = {best_params[1]}")

# View top 5 combinations
print(results_df.nlargest(5, 'Sharpe_Ratio'))
```

## Strategy Logic

### Signal Generation
- **Buy Signal**: Generated when fast MA crosses above slow MA
- **Sell Signal**: Generated when fast MA crosses below slow MA
- **Position**: Long when fast MA > slow MA, flat otherwise

### Entry/Exit Rules
1. **Entry**: Enter long position when fast MA crosses above slow MA
2. **Exit**: Exit position when fast MA crosses below slow MA
3. **No Short Positions**: Strategy only goes long or stays in cash

### Risk Management
- **Maximum Drawdown Tracking**: Monitors peak-to-trough decline
- **Position Sizing**: Uses full capital allocation (can be modified)
- **Signal Strength**: Calculates MA separation as signal confidence

## Performance Metrics

The strategy calculates comprehensive performance metrics:

### Return Metrics
- **Total Return**: Cumulative return over the period
- **Annualized Return**: Compound annual growth rate
- **Strategy vs Market**: Comparison with buy-and-hold

### Risk Metrics
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Beta**: Correlation with market movements
- **Alpha**: Excess return over market

### Trading Metrics
- **Number of Trades**: Total buy signals generated
- **Win Rate**: Percentage of profitable trades
- **Signal Strength**: Distribution of MA separation

## Example Output

```
============================================================
SPX MOVING AVERAGE CROSSOVER STRATEGY PERFORMANCE
============================================================
Strategy Parameters:
  Fast MA: 20 periods
  Slow MA: 50 periods
  Initial Capital: $100,000.00

Performance Metrics:
----------------------------------------
Total Return - Strategy (%)        :      50.87%
Total Return - Market (%)          :     151.42%
Annualized Return - Strategy (%)   :       8.07%
Annualized Return - Market (%)     :      18.99%
Volatility - Strategy (%)          :      12.59%
Volatility - Market (%)            :      20.00%
Sharpe Ratio - Strategy            :      0.482
Sharpe Ratio - Market              :      0.850
Maximum Drawdown (%)               :     -28.89%
Win Rate (%)                       :      55.38%
Number of Trades                   :         13
Beta                               :      0.396
Alpha (%)                          :      -0.66%
Final Portfolio Value ($)          : $150,870.39
Final Benchmark Value ($)          : $251,423.74
----------------------------------------

Strategy Assessment:
Strategy outperformed market by: -100.55%
✗ Strategy has lower risk-adjusted returns (lower Sharpe ratio)
✗ Strategy generated negative alpha: -0.66%
```

## Visualization

The strategy generates six comprehensive charts:

1. **Price & Moving Averages**: SPY price with MA lines and buy/sell signals
2. **Cumulative Returns**: Strategy vs buy-and-hold performance comparison
3. **Portfolio Values**: Dollar value growth over time
4. **Drawdown Chart**: Peak-to-trough decline visualization
5. **Rolling Returns**: 30-day rolling return comparison
6. **Signal Strength**: Distribution of MA separation percentages

## Testing Suite

Run the comprehensive test suite:

```bash
python test_strategy.py
```

This will execute:
- Basic strategy test (20/50 MA)
- Custom parameters test (10/30 MA)
- Shorter timeframe test (2022-2024)
- Strategy comparison (multiple configurations)
- Parameter optimization (optional)

## Strategy Considerations

### Advantages
- **Simple and Robust**: Easy to understand and implement
- **Trend Following**: Captures major market trends
- **Risk Management**: Built-in stop-loss via crossover exits
- **Backtesting**: Comprehensive historical testing capabilities

### Limitations
- **Lagging Indicators**: Moving averages are backward-looking
- **Whipsaws**: Frequent signals in sideways markets
- **Bull Market Dependency**: May underperform in strong uptrends
- **Transaction Costs**: Not included in current implementation

### Improvements
- Add transaction cost modeling
- Implement position sizing rules
- Include volatility-based stops
- Add multi-timeframe analysis
- Incorporate additional filters (volume, momentum)

## Data Source

- **Ticker**: SPY (SPDR S&P 500 ETF) as proxy for SPX
- **Data Provider**: Yahoo Finance via yfinance library
- **Frequency**: Daily data
- **Adjustments**: Split and dividend adjusted prices

## Files Structure

```
├── spx_ma_crossover_strategy.py  # Main strategy implementation
├── test_strategy.py              # Comprehensive test suite
├── README.md                     # This documentation
├── requirements.txt              # Python dependencies (optional)
└── venv/                        # Virtual environment
```

## Contributing

Feel free to contribute improvements:
- Add new technical indicators
- Implement additional risk metrics
- Enhance visualization capabilities
- Add real-time trading integration
- Improve parameter optimization algorithms

## Disclaimer

This strategy is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough testing and consider your risk tolerance before implementing any trading strategy with real capital.

## License

This project is open source. Use at your own risk.