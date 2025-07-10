#!/usr/bin/env python3
"""
Test script for SPX Moving Average Crossover Strategy

This script demonstrates different ways to use the strategy:
1. Basic strategy execution
2. Parameter optimization
3. Custom parameter testing
"""

from spx_ma_crossover_strategy import SPXTrendFollowingStrategy, optimize_ma_parameters
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

def test_basic_strategy():
    """Test the strategy with default parameters"""
    print("="*80)
    print("TESTING BASIC STRATEGY (MA 20/50)")
    print("="*80)
    
    strategy = SPXTrendFollowingStrategy(fast_ma=20, slow_ma=50, initial_capital=100000)
    metrics = strategy.run_full_analysis(start_date='2020-01-01')
    
    return metrics

def test_custom_parameters():
    """Test the strategy with custom parameters"""
    print("="*80)
    print("TESTING CUSTOM PARAMETERS (MA 10/30)")
    print("="*80)
    
    strategy = SPXTrendFollowingStrategy(fast_ma=10, slow_ma=30, initial_capital=100000)
    metrics = strategy.run_full_analysis(start_date='2020-01-01')
    
    return metrics

def test_shorter_timeframe():
    """Test the strategy on a shorter timeframe"""
    print("="*80)
    print("TESTING SHORTER TIMEFRAME (2022-2024)")
    print("="*80)
    
    strategy = SPXTrendFollowingStrategy(fast_ma=15, slow_ma=40, initial_capital=100000)
    metrics = strategy.run_full_analysis(start_date='2022-01-01', end_date='2024-01-01')
    
    return metrics

def test_parameter_optimization():
    """Test parameter optimization"""
    print("="*80)
    print("TESTING PARAMETER OPTIMIZATION")
    print("="*80)
    
    # Run optimization with smaller ranges for faster execution
    best_params, results_df = optimize_ma_parameters(
        start_date='2022-01-01', 
        end_date='2024-01-01',
        fast_range=(5, 21),  # Test MA periods 5, 10, 15, 20
        slow_range=(30, 61)  # Test MA periods 30, 40, 50, 60
    )
    
    print(f"\nOptimization Results Summary:")
    if best_params is not None:
        print(f"Best Parameters: Fast MA = {best_params[0]}, Slow MA = {best_params[1]}")
    else:
        print("No valid parameters found during optimization")
    print(f"\nTop 5 Parameter Combinations by Sharpe Ratio:")
    print(results_df.nlargest(5, 'Sharpe_Ratio')[['Fast_MA', 'Slow_MA', 'Sharpe_Ratio', 'Total_Return', 'Max_Drawdown']])
    
    return best_params, results_df

def compare_strategies():
    """Compare multiple strategy configurations"""
    print("="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    configurations = [
        {'fast_ma': 10, 'slow_ma': 30, 'name': 'Aggressive (10/30)'},
        {'fast_ma': 20, 'slow_ma': 50, 'name': 'Balanced (20/50)'},
        {'fast_ma': 30, 'slow_ma': 70, 'name': 'Conservative (30/70)'}
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nTesting {config['name']} configuration...")
        strategy = SPXTrendFollowingStrategy(
            fast_ma=config['fast_ma'], 
            slow_ma=config['slow_ma'], 
            initial_capital=100000
        )
        
        # Fetch data
        strategy.fetch_spx_data(start_date='2020-01-01')
        strategy.calculate_moving_averages()
        strategy.generate_signals()
        strategy.backtest_strategy()
        metrics = strategy.calculate_performance_metrics()
        
        results.append({
            'Configuration': config['name'],
            'Total Return (%)': metrics['Total Return - Strategy (%)'],
            'Sharpe Ratio': metrics['Sharpe Ratio - Strategy'],
            'Max Drawdown (%)': metrics['Maximum Drawdown (%)'],
            'Number of Trades': metrics['Number of Trades'],
            'Win Rate (%)': metrics['Win Rate (%)']
        })
    
    # Display comparison
    print(f"\n{'Configuration':<20} {'Total Return':<12} {'Sharpe':<8} {'Max DD':<10} {'Trades':<8} {'Win Rate':<10}")
    print("-" * 75)
    for result in results:
        print(f"{result['Configuration']:<20} "
              f"{result['Total Return (%)']:>10.2f}% "
              f"{result['Sharpe Ratio']:>7.3f} "
              f"{result['Max Drawdown (%)']:>9.2f}% "
              f"{result['Number of Trades']:>7} "
              f"{result['Win Rate (%)']:>9.2f}%")
    
    return results

if __name__ == "__main__":
    print("SPX Moving Average Crossover Strategy - Test Suite")
    print("="*80)
    
    # Test 1: Basic strategy
    basic_metrics = test_basic_strategy()
    
    # Test 2: Custom parameters
    custom_metrics = test_custom_parameters()
    
    # Test 3: Shorter timeframe
    short_metrics = test_shorter_timeframe()
    
    # Test 4: Strategy comparison
    comparison_results = compare_strategies()
    
    # Test 5: Parameter optimization (comment out if it takes too long)
    print("\nSkipping parameter optimization for faster execution.")
    print("Uncomment the lines below to run optimization:")
    print("# best_params, opt_results = test_parameter_optimization()")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)