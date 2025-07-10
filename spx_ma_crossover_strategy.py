import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class SPXTrendFollowingStrategy:
    """
    SPX Trend Following Strategy using Moving Average Crossover
    
    This strategy generates buy/sell signals based on the crossover of 
    fast and slow moving averages on the S&P 500 index.
    """
    
    def __init__(self, fast_ma=20, slow_ma=50, initial_capital=100000):
        """
        Initialize the strategy parameters
        
        Parameters:
        fast_ma (int): Period for fast moving average
        slow_ma (int): Period for slow moving average
        initial_capital (float): Starting capital for backtesting
        """
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.portfolio = None
        
    def fetch_spx_data(self, start_date='2020-01-01', end_date=None):
        """
        Fetch SPX data from Yahoo Finance
        
        Parameters:
        start_date (str): Start date for data retrieval
        end_date (str): End date for data retrieval (default: today)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching SPX data from {start_date} to {end_date}")
        
        # Use SPY ETF as proxy for SPX
        ticker = "SPY"
        self.data = yf.download(ticker, start=start_date, end=end_date)
        
        # Clean the data
        self.data = self.data.dropna()
        
        print(f"Downloaded {len(self.data)} trading days of data")
        return self.data
    
    def calculate_moving_averages(self):
        """
        Calculate fast and slow moving averages
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_spx_data() first.")
        
        # Calculate moving averages using closing prices
        self.data[f'MA_{self.fast_ma}'] = self.data['Close'].rolling(window=self.fast_ma).mean()
        self.data[f'MA_{self.slow_ma}'] = self.data['Close'].rolling(window=self.slow_ma).mean()
        
        # Calculate the difference between MAs for signal strength
        self.data['MA_diff'] = self.data[f'MA_{self.fast_ma}'] - self.data[f'MA_{self.slow_ma}']
        self.data['MA_diff_pct'] = (self.data['MA_diff'] / self.data[f'MA_{self.slow_ma}']) * 100
        
        print(f"Calculated {self.fast_ma}-period and {self.slow_ma}-period moving averages")
        
    def generate_signals(self):
        """
        Generate buy/sell signals based on MA crossover
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("Moving averages not calculated. Call calculate_moving_averages() first.")
        
        # Initialize signals DataFrame
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['price'] = self.data['Close']
        self.signals['fast_ma'] = self.data[f'MA_{self.fast_ma}']
        self.signals['slow_ma'] = self.data[f'MA_{self.slow_ma}']
        self.signals['ma_diff'] = self.data['MA_diff']
        
        # Generate signals: 1 for buy, 0 for hold, -1 for sell
        self.signals['signal'] = 0
        
        # Buy signal when fast MA crosses above slow MA
        if len(self.signals) > self.fast_ma:
            self.signals['signal'][self.fast_ma:] = np.where(
                self.signals['fast_ma'][self.fast_ma:] > self.signals['slow_ma'][self.fast_ma:], 1, 0
            )
        
        # Generate trading positions (difference in signals)
        self.signals['positions'] = self.signals['signal'].diff()
        
        # Mark entry and exit points
        self.signals['buy_signal'] = np.where(self.signals['positions'] == 1, 1, 0)
        self.signals['sell_signal'] = np.where(self.signals['positions'] == -1, 1, 0)
        
        # Calculate signal strength based on MA separation
        if 'MA_diff_pct' in self.data.columns:
            self.signals['signal_strength'] = abs(self.data['MA_diff_pct'])
        else:
            self.signals['signal_strength'] = 0
        
        # Remove NaN values
        self.signals = self.signals.dropna()
        
        print(f"Generated signals for {len(self.signals)} trading days")
        print(f"Buy signals: {self.signals['buy_signal'].sum()}")
        print(f"Sell signals: {self.signals['sell_signal'].sum()}")
        
    def backtest_strategy(self):
        """
        Backtest the strategy and calculate performance metrics
        """
        if self.signals is None or len(self.signals) == 0:
            raise ValueError("Signals not generated. Call generate_signals() first.")
        
        # Initialize portfolio
        self.portfolio = pd.DataFrame(index=self.signals.index)
        self.portfolio['price'] = self.signals['price']
        self.portfolio['signal'] = self.signals['signal']
        self.portfolio['positions'] = self.signals['positions']
        
        # Calculate daily returns
        self.portfolio['market_returns'] = self.portfolio['price'].pct_change()
        
        # Calculate strategy returns (only when in position)
        self.portfolio['strategy_returns'] = (
            self.portfolio['market_returns'] * self.portfolio['signal'].shift(1)
        )
        
        # Calculate cumulative returns
        self.portfolio['market_cumulative'] = (1 + self.portfolio['market_returns']).cumprod()
        self.portfolio['strategy_cumulative'] = (1 + self.portfolio['strategy_returns']).cumprod()
        
        # Calculate portfolio value
        self.portfolio['portfolio_value'] = self.initial_capital * self.portfolio['strategy_cumulative']
        self.portfolio['benchmark_value'] = self.initial_capital * self.portfolio['market_cumulative']
        
        # Calculate drawdown
        self.portfolio['running_max'] = self.portfolio['portfolio_value'].expanding().max()
        self.portfolio['drawdown'] = (
            (self.portfolio['portfolio_value'] - self.portfolio['running_max']) / 
            self.portfolio['running_max']
        ) * 100
        
        print("Backtesting completed")
        
    def calculate_performance_metrics(self):
        """
        Calculate comprehensive performance metrics
        """
        if self.portfolio is None:
            raise ValueError("Portfolio not created. Call backtest_strategy() first.")
        
        # Remove NaN values for calculations
        portfolio_clean = self.portfolio.dropna()
        
        # Basic performance metrics
        total_return_strategy = (portfolio_clean['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        total_return_market = (portfolio_clean['benchmark_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Annualized returns (assuming 252 trading days per year)
        trading_days = len(portfolio_clean)
        years = trading_days / 252
        
        annualized_return_strategy = ((portfolio_clean['portfolio_value'].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100
        annualized_return_market = ((portfolio_clean['benchmark_value'].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100
        
        # Volatility (annualized)
        strategy_volatility = portfolio_clean['strategy_returns'].std() * np.sqrt(252) * 100
        market_volatility = portfolio_clean['market_returns'].std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_strategy = (annualized_return_strategy/100 - risk_free_rate) / (strategy_volatility/100)
        sharpe_market = (annualized_return_market/100 - risk_free_rate) / (market_volatility/100)
        
        # Maximum drawdown
        max_drawdown = portfolio_clean['drawdown'].min()
        
        # Win rate
        winning_trades = (portfolio_clean['strategy_returns'] > 0).sum()
        total_trades = (portfolio_clean['strategy_returns'] != 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Number of trades
        num_trades = self.signals['buy_signal'].sum() if self.signals is not None else 0
        
        # Beta calculation
        covariance = np.cov(portfolio_clean['strategy_returns'].dropna(), 
                           portfolio_clean['market_returns'].dropna())[0][1]
        market_variance = np.var(portfolio_clean['market_returns'].dropna())
        beta = covariance / market_variance if market_variance != 0 else 0
        
        # Alpha calculation
        alpha = (annualized_return_strategy/100) - (risk_free_rate + beta * (annualized_return_market/100 - risk_free_rate))
        alpha *= 100  # Convert to percentage
        
        # Create performance summary
        self.performance_metrics = {
            'Total Return - Strategy (%)': round(total_return_strategy, 2),
            'Total Return - Market (%)': round(total_return_market, 2),
            'Annualized Return - Strategy (%)': round(annualized_return_strategy, 2),
            'Annualized Return - Market (%)': round(annualized_return_market, 2),
            'Volatility - Strategy (%)': round(strategy_volatility, 2),
            'Volatility - Market (%)': round(market_volatility, 2),
            'Sharpe Ratio - Strategy': round(sharpe_strategy, 3),
            'Sharpe Ratio - Market': round(sharpe_market, 3),
            'Maximum Drawdown (%)': round(max_drawdown, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Number of Trades': num_trades,
            'Beta': round(beta, 3),
            'Alpha (%)': round(alpha, 2),
            'Final Portfolio Value ($)': round(portfolio_clean['portfolio_value'].iloc[-1], 2),
            'Final Benchmark Value ($)': round(portfolio_clean['benchmark_value'].iloc[-1], 2)
        }
        
        return self.performance_metrics
    
    def plot_results(self, figsize=(15, 12)):
        """
        Create comprehensive visualization of strategy performance
        """
        if self.portfolio is None or len(self.portfolio) == 0:
            raise ValueError("Strategy not backtested. Call backtest_strategy() first.")
        
        if self.signals is None or len(self.signals) == 0:
            raise ValueError("Signals not generated. Call generate_signals() first.")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Price and Moving Averages
        ax1 = axes[0, 0]
        ax1.plot(self.signals.index, self.signals['price'], label='SPY Price', linewidth=1)
        ax1.plot(self.signals.index, self.signals['fast_ma'], label=f'MA {self.fast_ma}', linewidth=1)
        ax1.plot(self.signals.index, self.signals['slow_ma'], label=f'MA {self.slow_ma}', linewidth=1)
        
        # Mark buy and sell signals
        buy_signals = self.signals[self.signals['buy_signal'] == 1]
        sell_signals = self.signals[self.signals['sell_signal'] == 1]
        
        if len(buy_signals) > 0:
            ax1.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', 
                       s=50, label='Buy Signal', zorder=5)
        if len(sell_signals) > 0:
            ax1.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', 
                       s=50, label='Sell Signal', zorder=5)
        
        ax1.set_title('SPY Price with Moving Average Crossover Signals')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Returns Comparison
        ax2 = axes[0, 1]
        portfolio_clean = self.portfolio.dropna()
        if len(portfolio_clean) > 0:
            ax2.plot(portfolio_clean.index, portfolio_clean['strategy_cumulative'], 
                    label='Strategy', linewidth=2)
            ax2.plot(portfolio_clean.index, portfolio_clean['market_cumulative'], 
                    label='Buy & Hold', linewidth=2)
        ax2.set_title('Cumulative Returns Comparison')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Portfolio Values
        ax3 = axes[0, 2]
        if len(portfolio_clean) > 0:
            ax3.plot(portfolio_clean.index, portfolio_clean['portfolio_value'], 
                    label='Strategy Portfolio', linewidth=2)
            ax3.plot(portfolio_clean.index, portfolio_clean['benchmark_value'], 
                    label='Buy & Hold Portfolio', linewidth=2)
        ax3.set_title('Portfolio Value Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Drawdown
        ax4 = axes[1, 0]
        if len(portfolio_clean) > 0:
            ax4.fill_between(portfolio_clean.index, portfolio_clean['drawdown'], 0, 
                            color='red', alpha=0.3)
            ax4.plot(portfolio_clean.index, portfolio_clean['drawdown'], color='red', linewidth=1)
        ax4.set_title('Strategy Drawdown')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Rolling Returns (30-day)
        ax5 = axes[1, 1]
        if len(portfolio_clean) > 30:
            rolling_strategy = portfolio_clean['strategy_returns'].rolling(30).mean() * 100
            rolling_market = portfolio_clean['market_returns'].rolling(30).mean() * 100
            ax5.plot(portfolio_clean.index, rolling_strategy, label='Strategy (30D)', linewidth=1)
            ax5.plot(portfolio_clean.index, rolling_market, label='Market (30D)', linewidth=1)
        ax5.set_title('30-Day Rolling Returns')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Return (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Signal Strength Distribution
        ax6 = axes[1, 2]
        if 'signal_strength' in self.signals.columns:
            signal_strength = self.signals['signal_strength'].dropna()
            if len(signal_strength) > 0:
                ax6.hist(signal_strength, bins=30, alpha=0.7, edgecolor='black')
        ax6.set_title('Distribution of Signal Strength')
        ax6.set_xlabel('MA Separation (%)')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def print_performance_summary(self):
        """
        Print formatted performance summary
        """
        if not hasattr(self, 'performance_metrics'):
            self.calculate_performance_metrics()
        
        print("\n" + "="*60)
        print("SPX MOVING AVERAGE CROSSOVER STRATEGY PERFORMANCE")
        print("="*60)
        print(f"Strategy Parameters:")
        print(f"  Fast MA: {self.fast_ma} periods")
        print(f"  Slow MA: {self.slow_ma} periods")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print("\nPerformance Metrics:")
        print("-"*40)
        
        for metric, value in self.performance_metrics.items():
            if isinstance(value, (int, float)):
                if '$' in metric:
                    print(f"{metric:<35}: ${value:>10,.2f}")
                elif '%' in metric:
                    print(f"{metric:<35}: {value:>10.2f}%")
                else:
                    print(f"{metric:<35}: {value:>10.3f}")
            else:
                print(f"{metric:<35}: {value:>10}")
        
        print("-"*40)
        
        # Strategy assessment
        total_return_diff = (self.performance_metrics['Total Return - Strategy (%)'] - 
                           self.performance_metrics['Total Return - Market (%)'])
        
        print(f"\nStrategy Assessment:")
        print(f"Strategy outperformed market by: {total_return_diff:.2f}%")
        
        if self.performance_metrics['Sharpe Ratio - Strategy'] > self.performance_metrics['Sharpe Ratio - Market']:
            print("✓ Strategy has better risk-adjusted returns (higher Sharpe ratio)")
        else:
            print("✗ Strategy has lower risk-adjusted returns (lower Sharpe ratio)")
            
        if self.performance_metrics['Alpha (%)'] > 0:
            print(f"✓ Strategy generated positive alpha: {self.performance_metrics['Alpha (%)']:.2f}%")
        else:
            print(f"✗ Strategy generated negative alpha: {self.performance_metrics['Alpha (%)']:.2f}%")
    
    def run_full_analysis(self, start_date='2020-01-01', end_date=None):
        """
        Run the complete strategy analysis pipeline
        """
        print("Starting SPX Moving Average Crossover Strategy Analysis")
        print("="*60)
        
        # Run the complete pipeline
        self.fetch_spx_data(start_date, end_date)
        self.calculate_moving_averages()
        self.generate_signals()
        self.backtest_strategy()
        self.calculate_performance_metrics()
        
        # Display results
        self.print_performance_summary()
        self.plot_results()
        
        return self.performance_metrics

def optimize_ma_parameters(start_date='2020-01-01', end_date=None, fast_range=(5, 25), slow_range=(30, 100)):
    """
    Optimize moving average parameters by testing different combinations
    """
    print("Optimizing Moving Average Parameters...")
    print("="*50)
    
    best_sharpe = -999
    best_params = None
    results = []
    
    fast_mas = range(fast_range[0], fast_range[1], 5)
    slow_mas = range(slow_range[0], slow_range[1], 10)
    
    for fast_ma in fast_mas:
        for slow_ma in slow_mas:
            if fast_ma >= slow_ma:  # Skip invalid combinations
                continue
                
            try:
                strategy = SPXTrendFollowingStrategy(fast_ma=fast_ma, slow_ma=slow_ma)
                strategy.fetch_spx_data(start_date, end_date)
                strategy.calculate_moving_averages()
                strategy.generate_signals()
                strategy.backtest_strategy()
                metrics = strategy.calculate_performance_metrics()
                
                sharpe = metrics['Sharpe Ratio - Strategy']
                total_return = metrics['Total Return - Strategy (%)']
                max_dd = metrics['Maximum Drawdown (%)']
                num_trades = metrics['Number of Trades']
                
                results.append({
                    'Fast_MA': fast_ma,
                    'Slow_MA': slow_ma,
                    'Sharpe_Ratio': sharpe,
                    'Total_Return': total_return,
                    'Max_Drawdown': max_dd,
                    'Num_Trades': num_trades
                })
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (fast_ma, slow_ma)
                    
                print(f"MA({fast_ma},{slow_ma}): Sharpe={sharpe:.3f}, Return={total_return:.2f}%, DD={max_dd:.2f}%")
                
            except Exception as e:
                print(f"Error with MA({fast_ma},{slow_ma}): {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\nOptimization Complete!")
    print(f"Best parameters: Fast MA = {best_params[0]}, Slow MA = {best_params[1]}")
    print(f"Best Sharpe Ratio: {best_sharpe:.3f}")
    
    # Plot optimization results
    if len(results_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sharpe ratio heatmap
        pivot_sharpe = results_df.pivot(index='Slow_MA', columns='Fast_MA', values='Sharpe_Ratio')
        sns.heatmap(pivot_sharpe, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0])
        axes[0].set_title('Sharpe Ratio by MA Parameters')
        
        # Total return heatmap
        pivot_return = results_df.pivot(index='Slow_MA', columns='Fast_MA', values='Total_Return')
        sns.heatmap(pivot_return, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[1])
        axes[1].set_title('Total Return (%) by MA Parameters')
        
        plt.tight_layout()
        plt.show()
    
    return best_params, results_df

if __name__ == "__main__":
    # Example usage: Run strategy with default parameters
    strategy = SPXTrendFollowingStrategy(fast_ma=20, slow_ma=50)
    metrics = strategy.run_full_analysis(start_date='2020-01-01')
    
    # Uncomment to run parameter optimization
    # best_params, optimization_results = optimize_ma_parameters()