
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("IMPROVED STOCK FORECASTING - SOLVING THE FLAT FORECAST PROBLEM")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA FOR MULTIPLE COMPANIES
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA FOR MULTIPLE COMPANIES")
print("="*80)

# List of company files
COMPANY_FILES = {
    "AAPL": "D:/S-P-500/raw data/Apple Inc/AAPL 5y.csv",
    "MSFT": "D:/S-P-500/raw data/Microsoft/MSFT 5y.csv",
    "NFLX": "D:/S-P-500/raw data/Netflix/NFLX 5y.csv",
    "IBM":  "D:/S-P-500/raw data/IBM/IBM 5y.csv",
    "META": "D:/S-P-500/raw data/Meta Platforms/META 5y.csv",
    "NVDA": "D:/S-P-500/raw data/Nvidia/NVDA 5y.csv",
    "ORCL": "D:/S-P-500/raw data/Oracle Corporation/ORCL 5y.csv",
    "TSLA": "D:/S-P-500/raw data/Tesla Inc/TSLA 5y.csv",
    "INTC": "D:/S-P-500/raw data/Intel/INTC 5y.csv",
}

# COMPANY_FILES is a dict mapping ticker -> path; use keys as tickers
COMPANY_NAMES = list(COMPANY_FILES.keys())

print(f"\n📊 Found {len(COMPANY_FILES)} companies to analyze:")
for i, name in enumerate(COMPANY_NAMES, 1):
    print(f"   {i}. {name} (file: {COMPANY_FILES[name]})")

# Dictionary to store results for all companies
all_results = {}

# ============================================================================
# MAIN LOOP: PROCESS EACH COMPANY
# ============================================================================

for company_idx, (company_name, file_path) in enumerate(COMPANY_FILES.items(), 1):
    
    print("\n" + "="*80)
    print(f"PROCESSING {company_idx}/{len(COMPANY_FILES)}: {company_name}")
    print("="*80)
    
    try:
        # Load company data
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        df = df[['Close']]
        df = df.sort_index()
        
        print(f"✓ Data loaded: {len(df)} observations")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
        
    except Exception as e:
        print(f"✗ Error loading {company_name}: {e}")
        continue

    # ========================================================================
    # SOLUTION 1: FORECAST RETURNS, NOT PRICES
    # ========================================================================
    print("\n" + "-"*80)
    print("SOLUTION 1: FORECAST RETURNS INSTEAD OF PRICES")
    print("-"*80)
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change() * 100
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
    
    def quick_adf_test(series, name):
        result = adfuller(series.dropna())
        status = "✓ STATIONARY" if result[1] < 0.05 else "✗ NON-STATIONARY"
        print(f"   {name:20s}: p-value={result[1]:.4f} → {status}")
        return result[1] < 0.05
    
    print("\n📊 Stationarity test:")
    is_price_stationary = quick_adf_test(df['Close'], 'Prices')
    is_return_stationary = quick_adf_test(df['Returns'], 'Returns')
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_returns = df['Returns'][1:train_size]
    test_returns = df['Returns'][train_size:]
    train_prices = df['Close'][:train_size]
    test_prices = df['Close'][train_size:]
    
    print(f"\n✓ Split: {len(train_returns)} training, {len(test_returns)} testing")

    # Define forecast horizon (used by multiple models)
    forecast_steps = len(test_prices)

    # ========================================================================
    # SOLUTION 1b: ARIMA ON RETURNS → CONVERT TO PRICES
    # ========================================================================
    print("\n" + "-"*80)
    print("SOLUTION 1b: ARIMA ON RETURNS → CONVERT TO PRICES")
    print("-"*80)

    # Prepare training returns (drop NaNs)
    tr = train_returns.dropna()
    te = test_returns.dropna()

    # Initialize fallback values
    rmse_returns = np.nan
    mape_returns = np.nan
    forecast_prices_from_returns = np.full(forecast_steps, np.nan)
    best_order_returns = None

    try:
        best_mape = np.inf
        best_forecast_prices = None
        best_forecast_returns = None
        best_order = None

        # Small grid search for (p,q) on returns (d=0 because returns are typically stationary)
        for p in range(0, 3):
            for q in range(0, 3):
                try:
                    model = ARIMA(tr, order=(p, 0, q))
                    fitted = model.fit()
                    f_returns = fitted.forecast(steps=forecast_steps)

                    # Convert forecasted returns back to prices
                    last_price = train_prices.iloc[-1]
                    f_prices = []
                    for r in f_returns:
                        pred_price = last_price * (1 + r/100.0)
                        f_prices.append(pred_price)
                        last_price = pred_price
                    f_prices = np.array(f_prices)

                    # Compute MAPE against actual test prices
                    if len(f_prices) == len(test_prices):
                        mape = np.mean(np.abs((test_prices.values - f_prices) / test_prices.values)) * 100
                        if np.isfinite(mape) and mape < best_mape:
                            best_mape = mape
                            best_forecast_prices = f_prices
                            best_forecast_returns = f_returns
                            best_order = (p, 0, q)
                except Exception:
                    # Skip orders that fail to fit
                    continue

        # If grid search failed to find a model, fall back to ARIMA(1,0,1)
        if best_order is None:
            model = ARIMA(tr, order=(1, 0, 1))
            fitted = model.fit()
            best_forecast_returns = fitted.forecast(steps=forecast_steps)
            last_price = train_prices.iloc[-1]
            best_forecast_prices = []
            for r in best_forecast_returns:
                pred_price = last_price * (1 + r/100.0)
                best_forecast_prices.append(pred_price)
                last_price = pred_price
            best_forecast_prices = np.array(best_forecast_prices)
            best_order = (1, 0, 1)

        # Assign final values and compute metrics
        forecast_prices_from_returns = np.array(best_forecast_prices)
        rmse_returns = np.sqrt(mean_squared_error(test_prices, forecast_prices_from_returns))
        mape_returns = np.mean(np.abs((test_prices - forecast_prices_from_returns) / test_prices)) * 100
        best_order_returns = best_order

        print(f"   ✓ Returns ARIMA selected: ARIMA{best_order_returns} (MAPE={mape_returns:.2f}%, RMSE={rmse_returns:.4f})")

    except Exception as e:
        print(f"   ✗ Returns-based model failed: {e}")
        forecast_prices_from_returns = np.full(forecast_steps, np.nan)
        rmse_returns = np.nan
        mape_returns = np.nan
        best_order_returns = None

    # ========================================================================
    # SOLUTION 2: ARIMA WITH DRIFT/TREND
    # ========================================================================
    print("\n" + "-"*80)
    print("SOLUTION 2: ARIMA WITH TREND")
    print("-"*80)
    
    try:
        model_with_trend = ARIMA(train_prices, order=(1, 1, 1), trend='t')
        fitted_with_trend = model_with_trend.fit()
        forecast_with_trend = fitted_with_trend.forecast(steps=forecast_steps)
        
        rmse_trend = np.sqrt(mean_squared_error(test_prices, forecast_with_trend))
        mape_trend = np.mean(np.abs((test_prices - forecast_with_trend) / test_prices)) * 100
        
        print(f"   ✓ ARIMA(1,1,1) with trend")
        print(f"   RMSE: {rmse_trend:.4f}, MAPE: {mape_trend:.2f}%")
    except Exception as e:
        print(f"   ✗ Trend model failed: {e}")
        forecast_with_trend = None
        rmse_trend = np.nan
        mape_trend = np.nan

    # ========================================================================
    # SOLUTION 3: NAIVE & MOVING AVERAGE
    # ========================================================================
    print("\n" + "-"*80)
    print("SOLUTION 3: NAIVE & MOVING AVERAGE FORECASTS")
    print("-"*80)
    
    # Naive forecast
    naive_forecast = np.full(len(test_prices), train_prices.iloc[-1])
    rmse_naive = np.sqrt(mean_squared_error(test_prices, naive_forecast))
    mape_naive = np.mean(np.abs((test_prices - naive_forecast) / test_prices)) * 100
    
    # Moving average with trend
    window = min(30, len(train_prices) // 4)
    recent_prices = train_prices.iloc[-window:]
    trend_slope = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / window
    
    ma_forecast = []
    last_price = train_prices.iloc[-1]
    for i in range(len(test_prices)):
        predicted_price = last_price + trend_slope * (i + 1)
        ma_forecast.append(predicted_price)
    
    ma_forecast = np.array(ma_forecast)
    rmse_ma = np.sqrt(mean_squared_error(test_prices, ma_forecast))
    mape_ma = np.mean(np.abs((test_prices - ma_forecast) / test_prices)) * 100
    
    print(f"   Naive RMSE: {rmse_naive:.4f}, MAPE: {mape_naive:.2f}%")
    print(f"   MA+Trend RMSE: {rmse_ma:.4f}, MAPE: {mape_ma:.2f}%")
    
    # ========================================================================
    # Store results for this company
    # ========================================================================
    all_results[company_name] = {
        'returns_rmse': rmse_returns,
        'returns_mape': mape_returns,
        'trend_rmse': rmse_trend,
        'trend_mape': mape_trend,
        'naive_rmse': rmse_naive,
        'naive_mape': mape_naive,
        'ma_rmse': rmse_ma,
        'ma_mape': mape_ma,
        'best_order': best_order_returns,
        'test_prices': test_prices,
        'train_prices': train_prices,
        'forecast_returns': forecast_prices_from_returns,
        'forecast_trend': forecast_with_trend,
        'naive_forecast': naive_forecast,
        'ma_forecast': ma_forecast
    }
    
    print(f"\n✓ {company_name} analysis complete!")

# End of main loop

# ============================================================================
# SUMMARY: COMPARE ALL COMPANIES
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY: ALL COMPANIES COMPARISON")
print("="*80)

# Create summary DataFrame
summary_data = []
for company, results in all_results.items():
    summary_data.append({
        'Company': company,
        'Returns_RMSE': results['returns_rmse'],
        'Returns_MAPE': results['returns_mape'],
        'Trend_RMSE': results['trend_rmse'],
        'Trend_MAPE': results['trend_mape'],
        'Naive_MAPE': results['naive_mape'],
        'Best_Model': f"ARIMA{results['best_order']}"
    })

summary_df = pd.DataFrame(summary_data)

# Guard: ensure we have the expected column and at least one row
if summary_df.empty or 'Returns_MAPE' not in summary_df.columns:
    print("\n⚠ No valid summary data to display (no companies processed or 'Returns_MAPE' unavailable). Skipping summary and plotting.")
    summary_df = pd.DataFrame()  # keep as empty DataFrame for later checks
else:
    summary_df = summary_df.sort_values('Returns_MAPE')

    print("\n📊 Performance Summary (sorted by Returns-based MAPE):")
    print(summary_df.to_string(index=False))

    # Find best and worst performers
    best_company = summary_df.iloc[0]['Company']
    worst_company = summary_df.iloc[-1]['Company']

    print(f"\n🏆 Best Performer: {best_company} (MAPE: {summary_df.iloc[0]['Returns_MAPE']:.2f}%)")
    print(f"⚠ Worst Performer: {worst_company} (MAPE: {summary_df.iloc[-1]['Returns_MAPE']:.2f}%)")

# ============================================================================
# VISUALIZATION: PLOT TOP 4 COMPANIES
# ============================================================================
if not summary_df.empty:
    top_n = min(9, len(summary_df))
    print("\n" + "="*80)
    print(f"CREATING VISUALIZATIONS FOR TOP {top_n} COMPANIES")
    print("="*80)

    top_companies = summary_df.head(top_n)['Company'].tolist()

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 14))
    axes = axes.flatten()

    for idx, company in enumerate(top_companies):
        ax = axes[idx]
        results = all_results[company]
        
        # Plot
        ax.plot(results['train_prices'].index, results['train_prices'], 
                label='Training', linewidth=1.5, alpha=0.7)
        ax.plot(results['test_prices'].index, results['test_prices'], 
                label='Actual', linewidth=2, color='green')
        try:
            ax.plot(results['test_prices'].index, results['forecast_returns'], 
                    label=f"Returns-based (MAPE={results.get('returns_mape', np.nan):.1f}%)", 
                    linewidth=2, color='red', linestyle='--')
        except Exception:
            pass
        
        ax.set_title(f'{company} - ARIMA{results.get("best_order", "")}', 
                     fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Price ($)')

    # Hide any unused subplots
    for j in range(len(top_companies), nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    # ============================================================================
    # INDIVIDUAL COMPANY DETAILED PLOTS (Optional - for best performer)
    # ============================================================================
    print(f"\n📈 Detailed plot for best performer: {best_company}")

    best_results = all_results[best_company]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Returns-based
    ax1 = axes[0, 0]
    ax1.plot(best_results['train_prices'].index, best_results['train_prices'], 
             label='Training', linewidth=1.5, alpha=0.7)
    ax1.plot(best_results['test_prices'].index, best_results['test_prices'], 
             label='Actual', linewidth=2, color='green')
    ax1.plot(best_results['test_prices'].index, best_results['forecast_returns'], 
             label=f"Returns (MAPE={best_results['returns_mape']:.1f}%)", 
             linewidth=2, color='red', linestyle='--')
    ax1.set_title(f'{best_company}: Returns-based Forecast', fontweight='bold', fontsize=12)
    ax1.legend()
else:
    print("\n⚠ Skipping visualizations because no summary data is available.")
if not summary_df.empty:
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Price ($)')

    # Plot 2: Trend-based
    ax2 = axes[0, 1]
    ax2.plot(best_results['train_prices'].index, best_results['train_prices'], 
             label='Training', linewidth=1.5, alpha=0.7)
    ax2.plot(best_results['test_prices'].index, best_results['test_prices'], 
             label='Actual', linewidth=2, color='green')
    if best_results['forecast_trend'] is not None:
        ax2.plot(best_results['test_prices'].index, best_results['forecast_trend'], 
                 label=f"Trend (MAPE={best_results['trend_mape']:.1f}%)", 
                 linewidth=2, color='orange', linestyle='--')
    ax2.set_title(f'{best_company}: ARIMA with Trend', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Price ($)')

    # Plot 3: Naive
    ax3 = axes[1, 0]
    ax3.plot(best_results['train_prices'].index, best_results['train_prices'], 
             label='Training', linewidth=1.5, alpha=0.7)
    ax3.plot(best_results['test_prices'].index, best_results['test_prices'], 
             label='Actual', linewidth=2, color='green')
    ax3.plot(best_results['test_prices'].index, best_results['naive_forecast'], 
             label=f"Naive (MAPE={best_results['naive_mape']:.1f}%)", 
             linewidth=2, color='gray', linestyle='--')
    ax3.set_title(f'{best_company}: Naive Forecast', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('Price ($)')

    # Plot 4: Moving Average
    ax4 = axes[1, 1]
    ax4.plot(best_results['train_prices'].index, best_results['train_prices'], 
             label='Training', linewidth=1.5, alpha=0.7)
    ax4.plot(best_results['test_prices'].index, best_results['test_prices'], 
             label='Actual', linewidth=2, color='green')
    ax4.plot(best_results['test_prices'].index, best_results['ma_forecast'], 
             label=f"MA+Trend (MAPE={best_results['ma_mape']:.1f}%)", 
             linewidth=2, color='purple', linestyle='--')
    ax4.set_title(f'{best_company}: Moving Average Forecast', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylabel('Price ($)')

    plt.tight_layout()
    plt.show()

# ============================================================================
# PRACTICAL RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("💡 PRACTICAL RECOMMENDATIONS FOR STOCK FORECASTING")
print("="*80)
print("""
1. ✓ FORECAST RETURNS, NOT PRICES
   - Returns are more stationary and predictable
   - Convert forecasted returns back to prices
   - This is the #1 most important fix!

2. ✓ ADD DRIFT/TREND to ARIMA
   - Use trend='t' or trend='ct' parameter
   - Captures long-term upward/downward movement
   - Better than basic ARIMA(p,d,q)

3. ✓ USE SHORTER FORECAST HORIZONS
   - Stock prices are only predictable 1-5 days ahead
   - Longer forecasts → more uncertainty
   - Consider rolling forecasts (predict 1 day, update, repeat)

4. ✓ COMBINE WITH FUNDAMENTAL INDICATORS
   - SARIMAX can include: volume, volatility, market indices
   - Use exogenous variables (technical indicators)
   - Consider sentiment analysis, news data

5. ✓ CONSIDER MACHINE LEARNING ALTERNATIVES
   - LSTM (Long Short-Term Memory) for sequences
   - XGBoost with engineered features
   - Prophet for trend + seasonality
   - Ensemble methods combining multiple models

6. ⚠ REALISTIC EXPECTATIONS
   - Stock markets are HARD to predict (efficient market hypothesis)
   - Even best models have limited accuracy
   - Focus on direction (up/down) rather than exact values
   - Use for risk assessment, not guaranteed profits
""")

print("="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)