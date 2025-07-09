import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns

def analyze_ty_option_skew_leading_effect(skew_ma_window=5):
    """
    Script to analyze if TY option 25 delta vol skew has a leading effect on TY futures
    
    Parameters:
    skew_ma_window (int): Moving average window for smoothing option skew data
    """
    
    # Note: This is a template script. You'll need to replace data sources with actual market data
    # as yfinance doesn't have comprehensive options data for treasury futures
    
    print("Analyzing TY Option 25 Delta Vol Skew Leading Effect on TY Futures")
    print(f"Using {skew_ma_window}-period moving average for skew smoothing")
    print("=" * 60)
    
    # Step 1: Data Collection (Replace with actual data sources)
    # You'll need to connect to options data provider (Bloomberg, Refinitiv, etc.)
    
    def get_ty_futures_data():
        """Get TY futures price data"""
        # Placeholder - replace with actual TY futures data
        # TY futures symbol varies by exchange (e.g., ZN for CME)
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        return pd.DataFrame({'date': dates, 'ty_futures_price': prices})
    
    def get_ty_option_skew_data():
        """Get TY option 25 delta vol skew data"""
        # Placeholder - replace with actual options vol surface data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(123)
        
        # 25 delta put vol - 25 delta call vol (typical skew measure)
        # Make it more jumpy to simulate real intraday skew behavior
        base_skew = np.random.randn(len(dates)) * 2 + 1
        jumpy_noise = np.random.randn(len(dates)) * 1.5  # Additional jumpiness
        skew = base_skew + jumpy_noise
        return pd.DataFrame({'date': dates, 'vol_skew_25d': skew})
    
    # Step 2: Load and prepare data
    ty_futures = get_ty_futures_data()
    option_skew = get_ty_option_skew_data()
    
    # Merge datasets
    data = pd.merge(ty_futures, option_skew, on='date', how='inner')
    data = data.sort_values('date').reset_index(drop=True)
    
    # Apply moving average to option skew to smooth out jumpiness
    data['vol_skew_25d_smooth'] = data['vol_skew_25d'].rolling(window=skew_ma_window, center=True).mean()
    
    # Calculate changes: absolute change for futures, change for smoothed skew
    data['ty_change'] = data['ty_futures_price'].diff()  # Absolute change instead of pct_change
    data['skew_change'] = data['vol_skew_25d_smooth'].diff()
    
    # Remove NaN values (will be more due to moving average)
    data = data.dropna()
    
    print(f"Data loaded: {len(data)} observations from {data['date'].min()} to {data['date'].max()}")
    print(f"Skew smoothing: {skew_ma_window}-period moving average applied")
    print(f"TY futures: Using absolute price change instead of percentage change")
    
    # Step 3: Leading Effect Analysis
    
    def calculate_lead_lag_correlations(data, max_lags=10):
        """Calculate correlations between skew and future returns"""
        correlations = {}
        p_values = {}
        
        for lag in range(0, max_lags + 1):
            if lag == 0:
                # Contemporaneous correlation
                corr, p_val = stats.pearsonr(data['skew_change'], data['ty_change'])
            else:
                # Leading effect: skew change today vs TY change in lag days
                if len(data) > lag:
                    skew_today = data['skew_change'][:-lag]
                    ty_change_future = data['ty_change'].shift(-lag)[:-lag]
                    
                    # Remove NaN values
                    mask = ~(np.isnan(skew_today) | np.isnan(ty_change_future))
                    if mask.sum() > 10:  # Need sufficient data points
                        corr, p_val = stats.pearsonr(skew_today[mask], ty_change_future[mask])
                    else:
                        corr, p_val = np.nan, np.nan
                else:
                    corr, p_val = np.nan, np.nan
            
            correlations[lag] = corr
            p_values[lag] = p_val
        
        return correlations, p_values
    
    # Calculate correlations
    correlations, p_values = calculate_lead_lag_correlations(data, max_lags=10)
    
    # Step 4: Statistical Tests and Results
    print("\nLead-Lag Correlation Analysis:")
    print("-" * 40)
    print(f"{'Lag (days)':<10} {'Correlation':<12} {'P-value':<10} {'Significant'}")
    print("-" * 40)
    
    significant_lags = []
    for lag in sorted(correlations.keys()):
        corr = correlations[lag]
        p_val = p_values[lag]
        significant = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        
        if not np.isnan(corr):
            print(f"{lag:<10} {corr:<12.4f} {p_val:<10.4f} {significant}")
            if p_val < 0.05 and lag > 0:
                significant_lags.append(lag)
    
    # Step 5: Regression Analysis for significant lags
    if significant_lags:
        print(f"\nRegression Analysis for Significant Lags: {significant_lags}")
        print("-" * 50)
        
        for lag in significant_lags[:3]:  # Analyze top 3 significant lags
            # Prepare regression data
            if lag > 0:
                X = data['skew_change'][:-lag].values
                y = data['ty_change'].shift(-lag)[:-lag].values
                
                # Remove NaN values
                mask = ~(np.isnan(X) | np.isnan(y))
                X_clean = X[mask].reshape(-1, 1)
                y_clean = y[mask]
                
                if len(X_clean) > 10:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    model = LinearRegression()
                    model.fit(X_clean, y_clean)
                    y_pred = model.predict(X_clean)
                    r2 = r2_score(y_clean, y_pred)
                    
                    print(f"Lag {lag} days:")
                    print(f"  Coefficient: {model.coef_[0]:.6f}")
                    print(f"  R-squared: {r2:.4f}")
                    print(f"  Interpretation: 1 unit increase in skew change predicts "
                          f"{model.coef_[0]:.6f} point change in TY futures {lag} days later")
                    print()
    
    # Step 6: Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Time series comparison - raw vs smoothed skew
    ax1 = axes[0, 0]
    ax1.plot(data['date'], data['vol_skew_25d'], 'lightblue', label='Raw Skew', alpha=0.6)
    ax1.plot(data['date'], data['vol_skew_25d_smooth'], 'blue', label=f'{skew_ma_window}MA Skew', linewidth=2)
    ax1.set_title('Raw vs Smoothed Option Skew')
    ax1.set_ylabel('Vol Skew 25D')
    ax1.legend()
    
    # Plot 2: Time series of both variables
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    ax2.plot(data['date'], data['skew_change'], 'b-', label='Smoothed Skew Change', alpha=0.7)
    ax2_twin.plot(data['date'], data['ty_change'], 'r-', label='TY Absolute Change', alpha=0.7)
    ax2.set_title('Smoothed Skew Change vs TY Absolute Change')
    ax2.set_ylabel('Skew Change', color='b')
    ax2_twin.set_ylabel('TY Absolute Change', color='r')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # Plot 3: Correlation by lag
    ax3 = axes[0, 2]
    lags = list(correlations.keys())
    corr_values = [correlations[lag] for lag in lags]
    bars = ax3.bar(lags, corr_values, alpha=0.7)
    ax3.set_title('Lead-Lag Correlations')
    ax3.set_xlabel('Lag (days)')
    ax3.set_ylabel('Correlation')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Color significant correlations
    for i, (lag, p_val) in enumerate(p_values.items()):
        if p_val < 0.05:
            bars[i].set_color('red')
        elif p_val < 0.1:
            bars[i].set_color('orange')
    
    # Plot 4: Scatter plot for contemporaneous relationship
    ax4 = axes[1, 0]
    ax4.scatter(data['skew_change'], data['ty_change'], alpha=0.6)
    ax4.set_title('Contemporaneous Relationship')
    ax4.set_xlabel('Smoothed Skew Change')
    ax4.set_ylabel('TY Absolute Change')
    
    # Add trend line
    mask = ~(np.isnan(data['skew_change']) | np.isnan(data['ty_change']))
    if mask.sum() > 1:
        z = np.polyfit(data['skew_change'][mask], data['ty_change'][mask], 1)
        p = np.poly1d(z)
        ax4.plot(data['skew_change'], p(data['skew_change']), "r--", alpha=0.8)
    
    # Plot 5: Rolling correlation
    ax5 = axes[1, 1]
    window = 30
    rolling_corr = data['skew_change'].rolling(window).corr(data['ty_change'])
    ax5.plot(data['date'], rolling_corr)
    ax5.set_title(f'{window}-Day Rolling Correlation')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Rolling Correlation')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 6: Distribution comparison
    ax6 = axes[1, 2]
    ax6.hist(data['skew_change'].dropna(), bins=30, alpha=0.5, label='Skew Change', density=True)
    ax6_twin = ax6.twinx()
    ax6_twin.hist(data['ty_change'].dropna(), bins=30, alpha=0.5, label='TY Change', color='red', density=True)
    ax6.set_title('Distribution of Changes')
    ax6.set_xlabel('Change Value')
    ax6.set_ylabel('Density (Skew)', color='blue')
    ax6_twin.set_ylabel('Density (TY)', color='red')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Step 7: Summary and Conclusions
    print("\nSUMMARY:")
    print("=" * 50)
    
    max_corr_lag = max(correlations.keys(), key=lambda k: abs(correlations[k]) if not np.isnan(correlations[k]) else 0)
    max_corr = correlations[max_corr_lag]
    
    print(f"Strongest correlation found at lag {max_corr_lag} days: {max_corr:.4f}")
    print(f"P-value: {p_values[max_corr_lag]:.4f}")
    
    if significant_lags:
        print(f"Significant leading effects found at lags: {significant_lags} days")
        print("This suggests smoothed TY option skew may have predictive power for TY futures absolute changes.")
    else:
        print("No significant leading effects found.")
        print("Smoothed TY option skew does not appear to have strong predictive power for TY futures changes.")
    
    print(f"\nMETHODOLOGY IMPROVEMENTS APPLIED:")
    print(f"1. Applied {skew_ma_window}-period moving average to option skew data to reduce jumpiness")
    print(f"2. Used absolute price change for TY futures instead of percentage change")
    print(f"3. Enhanced visualization to show raw vs smoothed skew comparison")
    
    print("\nNOTE: This analysis uses simulated data. Replace with actual market data for real analysis.")
    print("Consider additional factors: economic announcements, market volatility, term structure changes.")
    
    return data, correlations, p_values

# Run the analysis
if __name__ == "__main__":
    # You can adjust the moving average window as needed
    data, correlations, p_values = analyze_ty_option_skew_leading_effect(skew_ma_window=5)
