import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple
import os
import hashlib


class PositionSizer:
    """Class for dynamic position sizing based on win probability and risk management"""
    
    def __init__(self, method: str = 'kelly', max_position: float = 1.0, 
                 lookback_period: int = 50, min_trades: int = 10):
        """
        Initialize position sizer
        
        Parameters:
        -----------
        method : str
            Position sizing method ('kelly', 'fixed_fractional', 'volatility_target')
        max_position : float
            Maximum position size as fraction of capital
        lookback_period : int
            Number of periods to look back for calculating win probability
        min_trades : int
            Minimum number of trades needed before using dynamic sizing
        """
        self.method = method
        self.max_position = max_position
        self.lookback_period = lookback_period
        self.min_trades = min_trades
        
    def calculate_win_probability(self, returns: pd.Series) -> Tuple[float, float, int]:
        """
        Calculate win probability, average win, average loss from historical returns
        
        Parameters:
        -----------
        returns : pd.Series
            Historical returns for the strategy
            
        Returns:
        --------
        Tuple[float, float, int]
            (win_probability, avg_win_loss_ratio, num_trades)
        """
        # Only consider non-zero returns (actual trades)
        trade_returns = returns[returns != 0]
        
        if len(trade_returns) < self.min_trades:
            return 0.5, 1.0, len(trade_returns)  # Default values
        
        # Calculate win probability
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        
        win_prob = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0.5
        
        # Calculate average win/loss ratio
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        return win_prob, avg_win_loss_ratio, len(trade_returns)
    
    def calculate_position_size(self, signal: float, current_data: pd.Series, 
                              historical_returns: pd.Series, current_capital: float,
                              asset: 'Asset') -> float:
        """
        Calculate dynamic position size based on signal and win probability
        
        Parameters:
        -----------
        signal : float
            Raw signal from strategy (-1 to 1)
        current_data : pd.Series
            Current market data row
        historical_returns : pd.Series
            Historical strategy returns for win probability calculation
        current_capital : float
            Current available capital
        asset : Asset
            Asset being traded
            
        Returns:
        --------
        float
            Position size (number of contracts/shares)
        """
        if signal == 0:
            return 0
        
        # Get recent returns for calculation - ensure it's a pandas Series
        historical_returns_series = pd.Series(historical_returns) if not isinstance(historical_returns, pd.Series) else historical_returns
        recent_returns = historical_returns_series[-self.lookback_period:] if len(historical_returns_series) >= self.lookback_period else historical_returns_series
        win_prob, avg_win_loss_ratio, num_trades = self.calculate_win_probability(recent_returns)  # type: ignore
        
        # If insufficient trade history, use conservative fixed sizing
        if num_trades < self.min_trades:
            base_size = 0.1  # Conservative 10% of max position
        else:
            if self.method == 'kelly':
                # Kelly Criterion: f = (bp - q) / b
                # where b = avg_win_loss_ratio, p = win_prob, q = 1-win_prob
                if avg_win_loss_ratio > 0:
                    kelly_fraction = (avg_win_loss_ratio * win_prob - (1 - win_prob)) / avg_win_loss_ratio
                    # Cap Kelly at reasonable levels to avoid over-leveraging
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))
                    base_size = kelly_fraction
                else:
                    base_size = 0.1
                    
            elif self.method == 'fixed_fractional':
                # Simple fixed fractional based on win probability
                base_size = min(0.1 + (win_prob - 0.5) * 0.4, 0.3)  # 10% to 30% based on win prob
                
            elif self.method == 'volatility_target':
                # Size based on volatility to target constant risk
                if len(recent_returns) > 5:
                    vol = recent_returns.std()
                    target_vol = 0.02  # 2% daily volatility target
                    base_size = min(target_vol / (vol + 1e-8), 0.3)  # Cap at 30%
                else:
                    base_size = 0.1
            else:
                base_size = 0.1
        
        # Apply signal direction and strength
        sized_fraction = base_size * abs(signal) * np.sign(signal)
        
        # Cap at maximum position
        sized_fraction = max(-self.max_position, min(sized_fraction, self.max_position))
        
        # Convert fraction to actual position size based on current price and capital
        current_price = current_data['Close']
        available_capital = current_capital * 0.95  # Keep 5% cash buffer
        
        # For futures, consider margin requirements
        if asset.asset_type == 'futures':
            contract_value = current_price * asset.multiplier
            margin_requirement = contract_value * asset.margin_requirement
            max_contracts = available_capital / margin_requirement
            position_size = sized_fraction * max_contracts
        else:
            # For stocks
            max_shares = available_capital / current_price
            position_size = sized_fraction * max_shares
            
        return float(round(float(position_size), 0))  # Round to whole contracts/shares


class Trade:
    """Class representing a single trade"""
    
    def __init__(self, symbol: str, entry_date: datetime, entry_price: float, 
                 quantity: float, side: str):
        self.symbol = symbol
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = quantity  # Can be fractional for partial closes
        self.side = side  # 'long' or 'short'
        self.exit_date = None
        self.exit_price = None
        self.pnl = None
        self.duration = None
        self.is_open = True
        
    def close_trade(self, exit_date: datetime, exit_price: float, asset: 'Asset'):
        """Close the trade and calculate PnL"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.is_open = False
        self.duration = (exit_date - self.entry_date).days
        
        # Calculate PnL based on asset type
        if asset.asset_type == 'futures':
            if self.side == 'long':
                self.pnl = (exit_price - self.entry_price) * self.quantity * asset.multiplier
            else:  # short
                self.pnl = (self.entry_price - exit_price) * self.quantity * asset.multiplier
        else:
            # Stocks
            if self.side == 'long':
                self.pnl = (exit_price - self.entry_price) * self.quantity
            else:
                self.pnl = (self.entry_price - exit_price) * self.quantity
    
    def to_dict(self) -> Dict:
        """Convert trade to dictionary for DataFrame creation"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'duration_days': self.duration,
            'pnl': self.pnl,
            'is_open': self.is_open
        }


class Portfolio:
    """Enhanced class for tracking portfolio performance with detailed trade management"""
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize portfolio
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital for the portfolio
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_cash = initial_capital
        self.positions = {}  # symbol -> {'quantity': float, 'avg_price': float, 'unrealized_pnl': float}
        self.trades = []  # List of Trade objects
        self.equity_curve = []
        self.daily_returns = []
        self.margin_used = 0.0
        
    def get_position_quantity(self, symbol: str) -> float:
        """Get current position quantity for a symbol"""
        return self.positions.get(symbol, {'quantity': 0})['quantity']
    
    def update_position(self, symbol: str, new_quantity: float, price: float, 
                       date: datetime, asset: 'Asset'):
        """
        Update position and record trades
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
        new_quantity : float
            New total position quantity
        price : float
            Current market price
        date : datetime
            Trade date
        asset : Asset
            Asset object for calculations
        """
        current_pos = self.positions.get(symbol, {'quantity': 0, 'avg_price': 0, 'unrealized_pnl': 0})
        current_quantity = current_pos['quantity']
        
        quantity_change = new_quantity - current_quantity
        
        if abs(quantity_change) < 1e-6:  # No significant change
            return
        
        # Handle position changes
        if current_quantity == 0 and new_quantity != 0:
            # Opening new position
            side = 'long' if new_quantity > 0 else 'short'
            trade = Trade(symbol, date, price, abs(new_quantity), side)
            self.trades.append(trade)
            
            # Update position
            self.positions[symbol] = {
                'quantity': new_quantity,
                'avg_price': price,
                'unrealized_pnl': 0
            }
            
        elif current_quantity != 0 and new_quantity == 0:
            # Closing entire position
            side = 'long' if current_quantity > 0 else 'short'
            
            # Find and close open trades
            for trade in reversed(self.trades):
                if (trade.symbol == symbol and trade.is_open and 
                    trade.side == side):
                    trade.close_trade(date, price, asset)
                    self.available_cash += trade.pnl
                    break
            
            # Clear position
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0, 'unrealized_pnl': 0}
            
        elif np.sign(current_quantity) != np.sign(new_quantity):
            # Reversing position (close and open new)
            # First close existing position
            current_side = 'long' if current_quantity > 0 else 'short'
            for trade in reversed(self.trades):
                if (trade.symbol == symbol and trade.is_open and 
                    trade.side == current_side):
                    trade.close_trade(date, price, asset)
                    self.available_cash += trade.pnl
                    break
            
            # Then open new position
            new_side = 'long' if new_quantity > 0 else 'short'
            trade = Trade(symbol, date, price, abs(new_quantity), new_side)
            self.trades.append(trade)
            
            self.positions[symbol] = {
                'quantity': new_quantity,
                'avg_price': price,
                'unrealized_pnl': 0
            }
            
        else:
            # Adding to existing position
            old_value = current_quantity * current_pos['avg_price']
            new_value = quantity_change * price
            total_quantity = current_quantity + quantity_change
            
            if total_quantity != 0:
                avg_price = (old_value + new_value) / total_quantity
            else:
                avg_price = price
                
            self.positions[symbol] = {
                'quantity': total_quantity,
                'avg_price': avg_price,
                'unrealized_pnl': 0
            }
            
            # Record additional trade
            if abs(quantity_change) > 1e-6:
                side = 'long' if quantity_change > 0 else 'short'
                trade = Trade(symbol, date, price, abs(quantity_change), side)
                self.trades.append(trade)
        
        # Update margin usage for futures
        if asset.asset_type == 'futures':
            contract_value = price * asset.multiplier * abs(new_quantity)
            self.margin_used = contract_value * asset.margin_requirement
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float], 
                                assets: Dict[str, 'Asset']) -> float:
        """Calculate current portfolio value including unrealized PnL"""
        total_value = self.available_cash
        
        for symbol, position in self.positions.items():
            if position['quantity'] != 0 and symbol in current_prices:
                current_price = current_prices[symbol]
                asset = assets[symbol]
                
                if asset.asset_type == 'futures':
                    # For futures, PnL is marked-to-market
                    if position['quantity'] > 0:  # long
                        unrealized_pnl = ((current_price - position['avg_price']) * 
                                        position['quantity'] * asset.multiplier)
                    else:  # short
                        unrealized_pnl = ((position['avg_price'] - current_price) * 
                                        abs(position['quantity']) * asset.multiplier)
                else:
                    # For stocks
                    unrealized_pnl = ((current_price - position['avg_price']) * 
                                    position['quantity'])
                
                total_value += unrealized_pnl
                self.positions[symbol]['unrealized_pnl'] = unrealized_pnl
        
        return total_value
    
    def get_trade_statistics(self) -> Dict[str, float]:
        """Calculate detailed trade statistics"""
        closed_trades = [t for t in self.trades if not t.is_open and t.pnl is not None]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_trade_duration': 0,
                'total_pnl': 0
            }
        
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_duration = np.mean([t.duration for t in closed_trades if t.duration is not None])
        total_pnl = sum(t.pnl for t in closed_trades)
        
        return {
            'total_trades': float(total_trades),
            'winning_trades': float(len(winning_trades)),
            'losing_trades': float(len(losing_trades)),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'avg_trade_duration': float(avg_duration),
            'total_pnl': float(total_pnl)
        }
    
    def calculate_returns(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio returns based on positions (backward compatibility)
        
        Parameters:
        -----------
        positions_df : pd.DataFrame
            DataFrame with positions and market data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio returns and metrics
        """
        df = positions_df.copy()
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['strategy_returns'] = df['position'] * df['returns']
        
        # Calculate equity curve
        df['equity_curve'] = (1 + df['strategy_returns']).cumprod() * self.initial_capital
        
        # Calculate drawdowns
        df['peak'] = df['equity_curve'].cummax()
        df['drawdown'] = (df['equity_curve'] - df['peak']) / df['peak']
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1
        
        return df


class Strategy(ABC):
    """Abstract base class for trading strategies with dynamic position sizing support"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with OHLCV and any additional columns
            
        Returns:
        --------
        pd.DataFrame
            Data with added signal column (1 for long, -1 for short, 0 for flat)
        """
        pass


class MovingAverageCrossover(Strategy):
    """Simple moving average crossover strategy with dynamic position sizing"""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50, 
                 use_dynamic_sizing: bool = True):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_dynamic_sizing = use_dynamic_sizing
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate moving averages
        df['fast_ma'] = df['Close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['Close'].rolling(window=self.slow_period).mean()
        
        # Generate base signals
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1
        
        # Add signal strength based on MA spread (for dynamic sizing)
        if self.use_dynamic_sizing:
            # Calculate normalized spread between MAs
            ma_spread = (df['fast_ma'] - df['slow_ma']) / df['Close']
            
            # Normalize spread to 0-1 range for signal strength
            spread_std = ma_spread.rolling(window=50).std()
            signal_strength = np.abs(ma_spread) / (spread_std + 1e-8)
            signal_strength = np.clip(signal_strength, 0, 2)  # Cap at 2x normal
            signal_strength = signal_strength / 2  # Normalize to 0-1
            
            # Apply strength to signals
            df['signal'] = df['signal'] * signal_strength
        
        # Generate positions (signal changes)
        df['position'] = df['signal'].shift(1)
        df['position'].fillna(0, inplace=True)
        
        return df


class VolumeSpikeMomentum(Strategy):
    """Strategy that follows price direction after volume spikes"""
    
    def __init__(self, volume_threshold: float = 2.0, lookback_period: int = 20):
        """
        Initialize the volume spike momentum strategy
        
        Parameters:
        -----------
        volume_threshold : float
            Multiple of average volume that constitutes a spike (e.g., 2.0 = 2x avg volume)
        lookback_period : int
            Period used to calculate the average volume
        """
        self.volume_threshold = volume_threshold
        self.lookback_period = lookback_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate rolling average volume
        df['avg_volume'] = df['Volume'].rolling(window=self.lookback_period).mean()
        
        # Identify volume spikes
        df['volume_spike'] = (df['Volume'] > (df['avg_volume'] * self.volume_threshold)).astype(int)
        
        # Price direction (1 for up, -1 for down, 0 for no change)
        df['price_direction'] = np.sign(df['Close'].diff())
        
        # Signal follows the direction 1 period after a volume spike
        df['signal'] = 0
        
        # Shift the volume spike to align with the next period
        spike_indices = df[df['volume_spike'] == 1].index
        
        # For each spike, get the direction of the next period and set as signal
        for idx in spike_indices:
            try:
                idx_position = df.index.get_loc(idx)
                if isinstance(idx_position, (int, np.integer)) and idx_position + 1 < len(df):
                    next_idx = df.index[idx_position + 1]
                    df.loc[next_idx, 'signal'] = df.loc[next_idx, 'price_direction']
            except (IndexError, KeyError):
                # Handle the case where the spike is at the last data point
                continue
        
        # Generate positions from signals
        df['position'] = df['signal'].shift(1)
        df['position'].fillna(0, inplace=True)
        
        return df


class Asset:
    """Class representing a tradable asset"""
    
    def __init__(self, symbol: str, asset_type: str = 'futures', 
                 multiplier: float = 1.0, margin_requirement: float = 0.1,
                 data_dir: str = 'market_data'):
        """
        Initialize an asset
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        asset_type : str
            Type of asset (futures, stock, etc.)
        multiplier : float
            Contract multiplier for futures
        margin_requirement : float
            Initial margin requirement as a fraction of contract value
        data_dir : str
            Directory to store CSV data files
        """
        self.symbol = symbol
        self.asset_type = asset_type
        self.multiplier = multiplier
        self.margin_requirement = margin_requirement
        self.data = None
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _get_cache_filename(self, start_date: str, end_date: str) -> str:
        """Generate a unique filename for cached data"""
        # Create a hash of symbol and date range for unique filename
        cache_key = f"{self.symbol}_{start_date}_{end_date}"
        filename = f"{cache_key}.csv"
        return os.path.join(self.data_dir, filename)
    
    def fetch_data(self, start_date: str, end_date: str, force_download: bool = False) -> pd.DataFrame:
        """
        Fetch market data for the asset, using cached CSV if available
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        force_download : bool
            If True, forces re-download even if cached data exists
            
        Returns:
        --------
        pd.DataFrame
            Market data with OHLCV columns
        """
        cache_file = self._get_cache_filename(start_date, end_date)
        
        # Try to load from cache first
        if not force_download and os.path.exists(cache_file):
            try:
                print(f"Loading {self.symbol} data from cache: {cache_file}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
                # Check if this is a malformed yfinance cache file (with ticker symbols in data rows)
                # If the first few rows contain the ticker symbol, we need to clean it up
                if len(df) > 0:
                    # Check if any of the first few rows contain the ticker symbol
                    for i in range(min(3, len(df))):
                        if any(str(self.symbol) in str(val) for val in df.iloc[i].values if pd.notna(val)):
                            print(f"Detected malformed cache file. Removing corrupted rows...")
                            # Find first row that contains actual numeric data
                            for j in range(len(df)):
                                try:
                                    # Try to convert the Close value to float
                                    if 'Close' in df.columns:
                                        float(df['Close'].iloc[j])
                                        df = df.iloc[j:]  # Keep from this row onwards
                                        break
                                except (ValueError, TypeError):
                                    continue
                            break
                
                # Ensure OHLCV columns are numeric
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except (TypeError, ValueError) as e:
                            print(f"Warning: Could not convert column {col} to numeric: {e}")
                            # Try to handle multi-level columns
                            if hasattr(df[col], 'iloc'):
                                try:
                                    df[col] = pd.to_numeric(df[col].iloc[:, 0] if len(df[col].shape) > 1 else df[col], errors='coerce')
                                except:
                                    pass
                
                # Drop any rows with NaN values after conversion
                df = df.dropna()
                
                # If the data is empty or too small after cleaning, force re-download
                if len(df) < 10:
                    print(f"Cached data is corrupted or too small ({len(df)} rows). Re-downloading...")
                    raise ValueError("Corrupted cache file")
                
                print(f"Cached data loaded: {len(df)} trading days")
                self.data = df
                return df
            except Exception as e:
                print(f"Error loading cached data: {e}. Downloading fresh data...")
        
        # Download fresh data
        print(f"Downloading {self.symbol} data ({start_date} to {end_date})...")
        try:
            df = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
            
            if df is None or df.empty:
                raise ValueError(f"Received empty dataset for {self.symbol}")
            
            # Handle MultiIndex columns from yfinance
            if hasattr(df.columns, 'levels'):  # MultiIndex columns
                print("Detected MultiIndex columns, flattening...")
                # Flatten MultiIndex columns by taking the first level (the OHLCV names)
                df.columns = df.columns.get_level_values(0)
            
            # Ensure we have the expected columns
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns {missing_columns}")
                # Try alternative column names
                column_mapping = {
                    'Adj Close': 'Close',
                    'Adj_Close': 'Close'
                }
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns and new_name not in df.columns:
                        df[new_name] = df[old_name]
            
            # Ensure OHLCV columns are numeric
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            # Only keep the OHLCV columns we need
            columns_to_keep = [col for col in numeric_columns if col in df.columns]
            if columns_to_keep:
                df = pd.DataFrame(df[columns_to_keep])  # Ensure we return a DataFrame
            
            print(f"Data downloaded: {len(df)} trading days")
            print(f"Columns: {list(df.columns)}")
            
            # Save to cache with proper format
            df.to_csv(cache_file)
            print(f"Data cached to: {cache_file}")
            
            self.data = df
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to download data for {self.symbol}: {str(e)}")

    def set_custom_data(self, data: pd.DataFrame) -> None:
        """
        Set custom market data for the asset
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with OHLCV data, should have columns: Open, High, Low, Close, Volume
            and a DatetimeIndex
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {missing_columns}")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex")
        
        self.data = data
        print(f"Custom data set for {self.symbol}: {len(data)} intervals")


class BacktestEngine:
    """Enhanced backtesting engine with dynamic position sizing"""
    
    def __init__(self, assets: List[Asset], strategy: Strategy, 
                 start_date: str, end_date: str, initial_capital: float = 100000.0,
                 position_sizer: Optional[PositionSizer] = None, 
                 use_dynamic_sizing: bool = True):
        """
        Initialize backtesting engine
        
        Parameters:
        -----------
        assets : List[Asset]
            List of assets to trade
        strategy : Strategy
            Trading strategy to use
        start_date : str
            Start date for backtest in 'YYYY-MM-DD' format
        end_date : str
            End date for backtest in 'YYYY-MM-DD' format
        initial_capital : float
            Initial capital for the backtest
        position_sizer : Optional[PositionSizer]
            Position sizer for dynamic sizing. If None, creates default Kelly sizer
        use_dynamic_sizing : bool
            Whether to use dynamic position sizing
        """
        self.assets = assets
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio = Portfolio(initial_capital)
        self.results = {}
        self.use_dynamic_sizing = use_dynamic_sizing
        
        # Initialize position sizer
        if use_dynamic_sizing:
            self.position_sizer = position_sizer if position_sizer else PositionSizer(method='kelly')
        else:
            self.position_sizer = None
        
    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Run the enhanced backtest with dynamic position sizing
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of results for each asset
        """
        assets_dict = {asset.symbol: asset for asset in self.assets}
        strategy_returns_history = pd.Series(dtype=float)  # Track historical returns for position sizing
        
        for asset in self.assets:
            # Fetch data (or use existing data if already set)
            if asset.data is not None:
                print(f"Using existing data for {asset.symbol}: {len(asset.data)} periods")
                data = asset.data
            else:
                data = asset.fetch_data(self.start_date, self.end_date)
            
            # Generate signals
            signals_df = self.strategy.generate_signals(data)
            
            # Add dynamic position sizing if enabled
            if self.use_dynamic_sizing and self.position_sizer:
                # Initialize position size column
                signals_df['position_size'] = 0.0
                signals_df['actual_position'] = 0.0
                
                # Process each row for dynamic position sizing
                for i, (date, row) in enumerate(signals_df.iterrows()):
                    current_signal = float(row['signal'])
                    current_price = float(row['Close'])
                    current_date = date if isinstance(date, datetime) else datetime.fromisoformat(str(date))
                    
                    if current_signal != 0:
                        # Calculate dynamic position size
                        current_capital = self.portfolio.calculate_portfolio_value(
                            {asset.symbol: current_price}, assets_dict
                        )
                        
                        position_size = self.position_sizer.calculate_position_size(
                            signal=current_signal,
                            current_data=row,
                            historical_returns=strategy_returns_history,
                            current_capital=current_capital,
                            asset=asset
                        )
                        
                        signals_df.loc[date, 'position_size'] = position_size
                        signals_df.loc[date, 'actual_position'] = position_size
                        
                        # Update portfolio position
                        self.portfolio.update_position(
                            symbol=asset.symbol,
                            new_quantity=position_size,
                            price=current_price,
                            date=current_date,
                            asset=asset
                        )
                        
                        # Calculate return for this period and add to history
                        if i > 0:
                            prev_price = float(signals_df['Close'].iloc[i-1])
                            current_return = (current_price - prev_price) / prev_price * np.sign(position_size)
                            strategy_returns_history = pd.concat([strategy_returns_history, pd.Series([current_return], index=[date])])
                    else:
                        # Close position
                        self.portfolio.update_position(
                            symbol=asset.symbol,
                            new_quantity=0,
                            price=current_price,
                            date=current_date,
                            asset=asset
                        )
                        signals_df.loc[date, 'actual_position'] = 0
                
                # Use actual positions for return calculation
                signals_df['position'] = signals_df['actual_position']
            
            # Calculate portfolio performance (backward compatibility)
            results_df = self.portfolio.calculate_returns(signals_df)
            
            # Add enhanced metrics - calculate portfolio value for each row
            portfolio_values = []
            for idx, row in results_df.iterrows():
                current_price = float(row['Close'])
                portfolio_value = self.portfolio.calculate_portfolio_value(
                    {asset.symbol: current_price}, assets_dict
                )
                portfolio_values.append(portfolio_value)
            results_df['portfolio_value'] = portfolio_values
            
            self.results[asset.symbol] = results_df
            
        return self.results
    
    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate enhanced performance metrics for each asset including detailed trade statistics
        
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary of metrics for each asset
        """
        metrics = {}
        
        # Get detailed trade statistics from portfolio
        trade_stats = self.portfolio.get_trade_statistics()
        
        for symbol, df in self.results.items():
            # Calculate performance metrics
            total_return = df['strategy_cumulative_returns'].iloc[-1]
            annual_return = (1 + total_return) ** (252 / len(df)) - 1
            
            # Risk metrics
            daily_std = df['strategy_returns'].std()
            annualized_vol = daily_std * np.sqrt(252)
            sharpe_ratio = annual_return / annualized_vol if annualized_vol != 0 else 0
            max_drawdown = df['drawdown'].min()
            
            # Basic trade metrics (for backward compatibility)
            df['trade'] = df['position'].diff().fillna(0)
            total_trades_basic = (df['trade'] != 0).sum()
            
            # Enhanced metrics from portfolio
            metrics[symbol] = {
                'Total Return': float(total_return),
                'Annual Return': float(annual_return),
                'Annualized Volatility': float(annualized_vol),
                'Sharpe Ratio': float(sharpe_ratio),
                'Max Drawdown': float(max_drawdown),
                'Total Trades (Basic)': float(total_trades_basic),
                
                # Enhanced trade statistics
                'Total Trades (Detailed)': trade_stats['total_trades'],
                'Winning Trades': trade_stats['winning_trades'],
                'Losing Trades': trade_stats['losing_trades'],
                'Win Rate': trade_stats['win_rate'],
                'Average Win': trade_stats['avg_win'],
                'Average Loss': trade_stats['avg_loss'],
                'Profit Factor': trade_stats['profit_factor'],
                'Average Trade Duration (Days)': trade_stats['avg_trade_duration'],
                'Total PnL from Trades': trade_stats['total_pnl'],
                
                # Portfolio value metrics
                'Final Portfolio Value': float(df['portfolio_value'].iloc[-1]) if 'portfolio_value' in df.columns else float(self.portfolio.initial_capital),
                'Total Portfolio Return': float((df['portfolio_value'].iloc[-1] - self.portfolio.initial_capital) / self.portfolio.initial_capital) if 'portfolio_value' in df.columns else 0.0
            }
            
        return metrics
    
    def plot_results(self, symbol: Optional[str] = None):
        """
        Plot comprehensive backtest results including positions
        
        Parameters:
        -----------
        symbol : Optional[str]
            Symbol to plot results for. If None, plots for all assets.
        """
        if symbol is not None:
            if symbol not in self.results:
                raise ValueError(f"No results for symbol {symbol}")
            symbols = [symbol]
        else:
            symbols = list(self.results.keys())
            
        for sym in symbols:
            df = self.results[sym]
            
            # Create a comprehensive figure with 4 subplots
            fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1, 2, 1]})
            
            # Plot 1: Price with Moving Averages and position highlighting
            ax1 = axs[0]
            ax1.plot(df.index, df['Close'], label=sym, color='blue')
            
            # Add moving averages if available
            if 'fast_ma' in df.columns and 'slow_ma' in df.columns:
                ax1.plot(df.index, df['fast_ma'], label=f'Fast MA', color='orange', alpha=0.8)
                ax1.plot(df.index, df['slow_ma'], label=f'Slow MA', color='purple', alpha=0.8)
                
                # If we're using a MovingAverageCrossover strategy, add labels with periods
                if isinstance(self.strategy, MovingAverageCrossover):
                    ax1.plot([], [], label=f'Fast MA ({self.strategy.fast_period} periods)', color='orange')
                    ax1.plot([], [], label=f'Slow MA ({self.strategy.slow_period} periods)', color='purple')
            
            # Highlight background based on position
            for i in range(1, len(df)):
                pos_value = df['position'].iloc[i]
                if hasattr(pos_value, 'iloc'):  # If it's a Series, get the first value
                    pos_value = pos_value.iloc[0]
                    
                if pos_value > 0:  # Long position
                    ax1.axvspan(df.index[i-1], df.index[i], alpha=0.1, color='green')
                elif pos_value < 0:  # Short position
                    ax1.axvspan(df.index[i-1], df.index[i], alpha=0.1, color='red')
            
            ax1.set_title(f'{sym} Price and Strategy Performance', fontsize=14)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(alpha=0.3)
            
            # Plot 2: Positions
            ax2 = axs[1]
            ax2.fill_between(df.index, df['position'], 0, where=df['position'] > 0, color='green', alpha=0.5, label='Long')
            ax2.fill_between(df.index, df['position'], 0, where=df['position'] < 0, color='red', alpha=0.5, label='Short')
            ax2.plot(df.index, df['position'], color='black', linewidth=0.8)
            
            # Add markers for position changes
            entries = df.index[(df['position'].shift(1) == 0) & (df['position'] != 0)]
            exits = df.index[(df['position'].shift(1) != 0) & (df['position'] == 0)]
            
            for entry in entries:
                pos = df.loc[entry, 'position']
                if hasattr(pos, 'iloc'):  # If it's a Series, get the first value
                    pos = pos.iloc[0]
                    
                if pos != 0:  # Only mark non-zero positions
                    color = 'green' if pos > 0 else 'red'
                    marker = '^' if pos > 0 else 'v'
                    ax2.scatter(entry, pos, color=color, s=80, marker=marker, zorder=5)
            
            for exit in exits:
                ax2.scatter(exit, 0, color='black', s=80, marker='o', zorder=5)
            
            ax2.set_ylabel('Position', fontsize=12)
            ax2.legend(loc='upper right')
            ax2.grid(alpha=0.3)
            
            # Plot 3: Strategy Performance
            ax3 = axs[2]
            ax3.plot(df.index, df['cumulative_returns'], label='Buy & Hold', color='gray')
            ax3.plot(df.index, df['strategy_cumulative_returns'], label='Strategy', color='blue')
            ax3.set_ylabel('Cumulative Returns', fontsize=12)
            ax3.legend(loc='upper left')
            ax3.grid(alpha=0.3)
            
            # Plot 4: Drawdowns
            ax4 = axs[3]
            ax4.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.5)
            ax4.set_ylabel('Drawdown', fontsize=12)
            ax4.set_xlabel('Date', fontsize=12)
            ax4.grid(alpha=0.3)
            
            # If we have date index, add vertical lines for month boundaries
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 30:
                # The safer way to identify month changes
                dates = pd.Series(df.index)
                months = dates.dt.month
                month_changes = []
                
                # Find the first day of each month
                for i in range(1, len(months)):
                    if months.iloc[i] != months.iloc[i-1]:
                        month_changes.append(df.index[i])
                
                for month_change in month_changes:
                    for ax in axs:
                        ax.axvline(x=month_change, color='gray', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Create a second plot with daily summary
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 5:
                # Check if we have intraday data
                has_intraday = False
                try:
                    # Check if we have sub-daily frequency by looking at time differences
                    time_diffs = df.index.to_series().diff().dropna()
                    has_intraday = any(td < pd.Timedelta(days=1) for td in time_diffs)
                except (AttributeError, TypeError):
                    # Some datetime indices might not support this operation
                    pass
                    
                if has_intraday:
                    try:
                        # Daily position and price
                        daily_positions = df['position'].resample('D').last()
                        daily_close = df['Close'].resample('D').last()
                        
                        # Calculate daily returns
                        daily_returns = df['strategy_returns'].resample('D').sum()
                        
                        # Create the daily summary plots
                        fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
                        
                        # Plot 1: Daily price
                        ax1 = axs[0]
                        ax1.plot(daily_close.index, daily_close, label=f'{sym} Price', color='blue')
                        ax1.set_title(f'{sym} Daily Summary', fontsize=14)
                        ax1.set_ylabel('Price', fontsize=12)
                        ax1.legend()
                        ax1.grid(alpha=0.3)
                        
                        # Plot 2: Daily positions
                        ax2 = axs[1]
                        
                        # Handle the case where positions might be Series objects
                        colors = []
                        for p in daily_positions:
                            if hasattr(p, 'iloc'):  # If it's a Series, get the first value
                                p = p.iloc[0]
                            colors.append('green' if p > 0 else 'red' if p < 0 else 'gray')
                            
                        ax2.bar(daily_positions.index, daily_positions, color=colors, alpha=0.7)
                        ax2.set_ylabel('End-of-Day Position', fontsize=12)
                        ax2.grid(alpha=0.3)
                        
                        # Plot 3: Daily returns
                        ax3 = axs[2]
                        
                        # Handle the case where returns might be Series objects
                        colors = []
                        daily_return_values = []
                        
                        for r in daily_returns:
                            if hasattr(r, 'iloc'):  # If it's a Series, get the first value
                                r = r.iloc[0]
                            colors.append('green' if r > 0 else 'red')
                            daily_return_values.append(r * 100)  # Convert to percentage
                            
                        ax3.bar(daily_returns.index, daily_return_values, color=colors, alpha=0.7)
                        ax3.set_ylabel('Daily Return (%)', fontsize=12)
                        ax3.set_xlabel('Date', fontsize=12)
                        ax3.grid(alpha=0.3)
                        
                        plt.tight_layout()
                        plt.show()
                    except Exception as e:
                        print(f"Warning: Could not generate daily summary plot: {e}")


# Example usage with TY futures using dynamic position sizing
if __name__ == "__main__":
    # Create asset - using ^TNX as a proxy for 10-year Treasury futures
    # ZN is the 10-year Treasury futures symbol, but we'll use ^TNX for data availability
    ty_futures = Asset(
        symbol='^TNX',  # 10-year Treasury Yield
        asset_type='futures',
        multiplier=1000,  # Each contract represents $1000 times the face value
        margin_requirement=0.05,  # 5% margin requirement
        data_dir='market_data'  # Directory for cached data
    )
    
    # Create strategy with 10d and 20d moving averages and dynamic sizing enabled
    ma_strategy = MovingAverageCrossover(fast_period=10, slow_period=20, use_dynamic_sizing=True)
    
    # Create position sizer using Kelly Criterion
    kelly_sizer = PositionSizer(
        method='kelly',
        max_position=0.3,  # Maximum 30% of capital at risk
        lookback_period=30,  # Look back 30 periods for win probability
        min_trades=5  # Need at least 5 trades before using dynamic sizing
    )
    
    # Create and run enhanced backtest with dynamic position sizing
    backtest = BacktestEngine(
        assets=[ty_futures],
        strategy=ma_strategy,
        start_date='2023-01-01',
        end_date='2024-12-31',
        initial_capital=100000,
        position_sizer=kelly_sizer,
        use_dynamic_sizing=True
    )
    
    results = backtest.run()
    metrics = backtest.calculate_metrics()
    
    # Print enhanced metrics
    for symbol, metric_dict in metrics.items():
        print(f"\nEnhanced Performance Metrics for {symbol} using Dynamic Position Sizing:")
        print("=" * 80)
        
        # Print basic performance metrics
        print("BASIC PERFORMANCE:")
        basic_metrics = ['Total Return', 'Annual Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown']
        for key in basic_metrics:
            if key in metric_dict:
                value = metric_dict[key]
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        print("\nTRADE STATISTICS:")
        trade_metrics = ['Total Trades (Detailed)', 'Winning Trades', 'Losing Trades', 'Win Rate', 
                        'Average Win', 'Average Loss', 'Profit Factor', 'Average Trade Duration (Days)']
        for key in trade_metrics:
            if key in metric_dict:
                value = metric_dict[key]
                print(f"{key}: {value:.4f}" if isinstance(value, float) and 'Trades' not in key else f"{key}: {value}")
        
        print("\nPORTFOLIO METRICS:")
        portfolio_metrics = ['Final Portfolio Value', 'Total Portfolio Return', 'Total PnL from Trades']
        for key in portfolio_metrics:
            if key in metric_dict:
                value = metric_dict[key]
                print(f"{key}: {value:.2f}" if 'Value' in key or 'PnL' in key else f"{key}: {value:.4f}")
    
    # Print trade details
    print(f"\nDETAILED TRADE LOG:")
    print("=" * 80)
    if backtest.portfolio.trades:
        trade_df = pd.DataFrame([trade.to_dict() for trade in backtest.portfolio.trades])
        print(trade_df.to_string(index=False))
    else:
        print("No completed trades to display.")
    
    # Compare with fixed position sizing
    print(f"\n" + "="*80)
    print("COMPARISON: Fixed vs Dynamic Position Sizing")
    print("="*80)
    
    # Run comparison with fixed sizing
    fixed_backtest = BacktestEngine(
        assets=[ty_futures],
        strategy=MovingAverageCrossover(fast_period=10, slow_period=20, use_dynamic_sizing=False),
        start_date='2023-01-01',
        end_date='2024-12-31',
        initial_capital=100000,
        use_dynamic_sizing=False
    )
    
    fixed_results = fixed_backtest.run()
    fixed_metrics = fixed_backtest.calculate_metrics()
    
    # Print comparison
    for symbol in metrics.keys():
        print(f"\nComparison for {symbol}:")
        comparison_metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Total Trades (Detailed)']
        for metric in comparison_metrics:
            if metric in metrics[symbol] and metric in fixed_metrics[symbol]:
                dynamic_val = metrics[symbol][metric]
                fixed_val = fixed_metrics[symbol][metric]
                print(f"{metric}:")
                print(f"  Dynamic Sizing: {dynamic_val:.4f}")
                print(f"  Fixed Sizing:   {fixed_val:.4f}")
                if isinstance(dynamic_val, (int, float)) and isinstance(fixed_val, (int, float)):
                    improvement = ((dynamic_val - fixed_val) / fixed_val * 100) if fixed_val != 0 else 0
                    print(f"  Improvement:    {improvement:+.2f}%")
    
    # Plot results
    print(f"\nGenerating plots...")
    backtest.plot_results() 