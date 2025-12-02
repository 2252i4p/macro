import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================
BASE_DIR = '/Users/lee/Library/CloudStorage/GoogleDrive-leehb823@gmail.com/내 드라이브/지금 작업중/TFT 논문/거시경제 지표'
PRICE_DIR = os.path.join(BASE_DIR, '가격 데이터')
OUTPUT_FILE = os.path.join(BASE_DIR, '무제 폴더/event_study_results_last.xlsx')

# Timezone offsets
KST_OFFSET = timedelta(hours=9)
UTC3_OFFSET = timedelta(hours=3)

# Analysis Horizons
HORIZONS = {
    '30m': timedelta(minutes=30),
    '12h': timedelta(hours=12),
    '1d': timedelta(days=1),
    '1w': timedelta(weeks=1)
}

# Sample Period
START_DATE = datetime(2013, 1, 1)
END_DATE = datetime(2025, 9, 30)

# ==========================================
# Data Loading Functions
# ==========================================

def load_macro_data():
    """Loads all macro indicator Excel files."""
    all_files = glob.glob(os.path.join(BASE_DIR, '*.xlsx'))
    macro_data = []
    
    print("Loading Macro Data...")
    for f in all_files:
        if '~$' in f: continue # Skip temp files
        if 'event_study_results.xlsx' in f: continue # Skip output file
        
        filename = os.path.basename(f)
        indicator_name = filename.replace('.xlsx', '')
        
        try:
            df = pd.read_excel(f, header=None)
            # Assign columns manually
            # Expected: 일자, 시간, 실제, 예측, 이전
            if len(df.columns) >= 5:
                df.columns = ['일자', '시간', '실제', '예측', '이전'] + list(df.columns[5:])
            else:
                print(f"Skipping {filename}: Not enough columns. Found {len(df.columns)}")
                continue
                
            # Create Datetime (KST)
            # Handle '일자' (Date)
            if pd.api.types.is_numeric_dtype(df['일자']):
                # Excel serial date
                # 41870 is around 2014. 
                # Excel base date is usually 1899-12-30
                df['Date_Temp'] = pd.to_datetime(df['일자'], unit='D', origin='1899-12-30')
            else:
                df['Date_Temp'] = pd.to_datetime(df['일자'], errors='coerce')
            
            # Handle '시간' (Time)
            # Could be string "21:30  " or datetime.time object
            def parse_time(t):
                if pd.isna(t): return None
                if isinstance(t, str):
                    t = t.strip()
                    try:
                        return datetime.strptime(t, '%H:%M:%S').time()
                    except:
                        try:
                            return datetime.strptime(t, '%H:%M').time()
                        except:
                            return None
                if isinstance(t, datetime):
                    return t.time()
                return t # Assume it's already time object
                
            df['Time_Temp'] = df['시간'].apply(parse_time)
            
            # Combine
            df = df.dropna(subset=['Date_Temp', 'Time_Temp'])
            df['Datetime'] = df.apply(lambda r: datetime.combine(r['Date_Temp'].date(), r['Time_Temp']), axis=1)
            
            # Filter by Sample Period
            df = df[(df['Datetime'] >= START_DATE) & (df['Datetime'] <= END_DATE)]
            
            # Filter: 2013 Jan Interest Rate (Redundant if START_DATE handles it, but kept for safety)
            if '금리' in indicator_name:
                df = df[~((df['Datetime'].dt.year == 2013) & (df['Datetime'].dt.month == 1))]
            
            df['Indicator'] = indicator_name
            
            # Clean numeric columns
            for col in ['실제', '예측', '이전']:
                # Remove non-numeric chars (like 'K', '%') if string
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Calculate Surprise
            df['Surprise'] = df['실제'] - df['예측']
            
            macro_data.append(df[['Datetime', 'Indicator', '실제', '예측', '이전', 'Surprise']])
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    if not macro_data:
        return pd.DataFrame()
        
    return pd.concat(macro_data, ignore_index=True)

def load_btc():
    """Loads BTC-USD 1-min data."""
    print("Loading BTC Data...")
    path = os.path.join(PRICE_DIR, 'btcusd_1-min_data.csv')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
        
    # Columns: Timestamp, Open, High, Low, Close, Volume
    # Timestamp is Unix timestamp
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s') + KST_OFFSET # Unix is UTC, add 9h for KST
    df.set_index('Datetime', inplace=True)
    return df[['Open', 'Close']]

def load_sp_futures():
    """Loads S&P 500 Futures 30-min data."""
    print("Loading S&P Futures Data...")
    path = os.path.join(PRICE_DIR, 'USA500IDXUSD_M30.csv')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
        
    # Columns mismatch: Header has 6, Data has 7.
    # Data: Datetime, Open, High, Low, Close, Vol1, Vol2
    try:
        df = pd.read_csv(path, sep='\t', header=0, names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Vol1', 'Vol2'])
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
        
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    return df[['Open', 'Close']]

def load_gold():
    """Loads Gold 15-min data."""
    print("Loading Gold Data...")
    path = os.path.join(PRICE_DIR, '금', 'XAU_15m_data.csv')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
        
    # Columns: Date;Open;High;Low;Close;Volume
    # Separator is semicolon
    try:
        df = pd.read_csv(path, sep=';')
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

    # Parse format: YYYY.MM.DD HH:MM
    df['Datetime'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    
    # Convert UTC+3 to KST (UTC+9) -> Add 6 hours
    df['Datetime'] = df['Datetime'] + timedelta(hours=6)
    
    df.set_index('Datetime', inplace=True)
    return df[['Open', 'Close']]

def load_brent():
    """Loads Brent Oil 30-min data."""
    print("Loading Brent Oil Data...")
    path = os.path.join(PRICE_DIR, 'BRENTCMDUSD_M30.csv')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
        
    # Columns mismatch: Header has 6, Data has 7.
    try:
        df = pd.read_csv(path, sep='\t', header=0, names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Vol1', 'Vol2'])
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    return df[['Open', 'Close']]

def get_price_at_time(price_df, target_time, method='open'):
    """
    Finds the price at or immediately after target_time.
    Returns (price, actual_time)
    """
    try:
        # Use searchsorted to find the index position
        idx = price_df.index.searchsorted(target_time)
        
        if idx >= len(price_df):
            return None, None
            
        actual_time = price_df.index[idx]
        
        # If the gap is too large (e.g., > 2 days for short horizons), maybe ignore?
        # For now, we take the next available candle as requested.
        
        if method == 'open':
            return price_df.iloc[idx]['Open'], actual_time
        else:
            return price_df.iloc[idx]['Close'], actual_time
            
    except KeyError:
        return None, None

def calculate_event_returns(macro_df, price_df, asset_name):
    """Calculates returns for each event for a specific asset."""
    results = []
    
    print(f"Calculating returns for {asset_name}...")
    
    for idx, row in macro_df.iterrows():
        event_time = row['Datetime']
        
        # Base Price: Open price at or after event time
        base_price, base_time = get_price_at_time(price_df, event_time, 'open')
        
        if base_price is None:
            # print(f"No base price for {event_time}") # Too noisy
            continue
            
        event_res = row.to_dict()
        event_res['Asset'] = asset_name
        event_res['Base_Time'] = base_time
        event_res['Base_Price'] = base_price
        
        for horizon_name, delta in HORIZONS.items():
            target_time = base_time + delta
            
            # Target Price: Open price at or after target_time
            # Note: Usually returns are Close-to-Close or Open-to-Open. 
            # User said: "발표시간 Open 각 타임스케일 지난 후 Open값을 수익률 산정에 사용한다."
            # So Open-to-Open.
            
            target_price, actual_target_time = get_price_at_time(price_df, target_time, 'open')
            
            if target_price is not None:
                # Log Return or Simple Return? Usually Log Return for stats, but Simple is easier to interpret.
                # User asked for "수익률". Let's use Simple Return: (P_t - P_0) / P_0
                ret = (target_price - base_price) / base_price
                event_res[f'Return_{horizon_name}'] = ret
            else:
                event_res[f'Return_{horizon_name}'] = np.nan
                
        results.append(event_res)
    
    return pd.DataFrame(results)

def calculate_baseline_stats(price_df, horizon_name, delta):
    """
    Calculates baseline statistics for the entire period.
    Resamples price data to the horizon frequency and calculates returns.
    """
    try:
        # Determine resampling rule
        if horizon_name == '30m': rule = '30min'
        elif horizon_name == '12h': rule = '12h'
        elif horizon_name == '1d': rule = 'D'
        elif horizon_name == '1w': rule = 'W'
        else: return None
        
        # Filter price_df by Sample Period for Baseline Calculation
        # We want the baseline to reflect the "usual" behavior during the sample period.
        mask = (price_df.index >= START_DATE) & (price_df.index <= END_DATE)
        period_price_df = price_df.loc[mask]
        
        if period_price_df.empty:
            return None

        # Resample Close price
        resampled = period_price_df['Close'].resample(rule).last().dropna()
        
        returns = resampled.pct_change().dropna()
        
        return {
            'Baseline_Mean': returns.mean(),
            'Baseline_Std': returns.std(),
            'Baseline_Abs_Mean': returns.abs().mean(),
            'Baseline_Abs_Std': returns.abs().std()
        }
    except Exception as e:
        print(f"Error calc baseline for {horizon_name}: {e}")
        return None

def calculate_statistics(df, assets_dict):
    """Calculates descriptive statistics with baselines."""
    stats_list = []
    
    # Pre-calculate baselines for each asset and horizon
    baselines = {}
    for asset_name, price_df in assets_dict.items():
        baselines[asset_name] = {}
        for h_name, h_delta in HORIZONS.items():
            baselines[asset_name][h_name] = calculate_baseline_stats(price_df, h_name, h_delta)
    
    # Group by Asset, Indicator
    grouped = df.groupby(['Asset', 'Indicator'])
    
    for (asset, indicator), group in grouped:
        for horizon in HORIZONS.keys():
            col = f'Return_{horizon}'
            if col not in group.columns: continue
            
            clean_data = group[col].dropna()
            if len(clean_data) < 2: continue
            
            # Baseline stats
            base_stats = baselines.get(asset, {}).get(horizon)
            if not base_stats:
                base_stats = {'Baseline_Mean': np.nan, 'Baseline_Std': np.nan, 'Baseline_Abs_Mean': np.nan, 'Baseline_Abs_Std': np.nan}
            
            # Common fields
            common = {
                'Asset': asset,
                'Indicator': indicator,
                'Horizon': horizon,
                **base_stats
            }
            
            # Helper to create row
            def create_row(condition, data):
                return {
                    **common,
                    'Condition': condition,
                    'N': len(data),
                    'Mean': data.mean(),
                    'Std': data.std(),
                    'Skewness': data.skew(),
                    'Kurtosis': data.kurtosis(),
                    'Event_Abs_Mean': data.abs().mean(),
                    'Event_Abs_Std': data.abs().std()
                }

            # 1. Overall
            stats_list.append(create_row('All', clean_data))
            
            # 2. Surprise Existence
            has_forecast = group.dropna(subset=['실제', '예측'])
            is_match = np.isclose(has_forecast['실제'], has_forecast['예측'], atol=1e-8)
            
            no_surprise_data = has_forecast[is_match][col].dropna()
            surprise_data = has_forecast[~is_match][col].dropna()
            
            if len(no_surprise_data) >=1:
                stats_list.append(create_row('No Surprise', no_surprise_data))
                
            if len(surprise_data) >= 1:
                stats_list.append(create_row('Surprise Exist', surprise_data))
                
            # 3. Surprise Direction
            pos_surprise = has_forecast[has_forecast['실제'] > has_forecast['예측']][col].dropna()
            neg_surprise = has_forecast[has_forecast['실제'] < has_forecast['예측']][col].dropna()
            
            if len(pos_surprise) >= 1:
                stats_list.append(create_row('Surprise Positive (>0)', pos_surprise))
                
            if len(neg_surprise) >= 1:
                stats_list.append(create_row('Surprise Negative (<0)', neg_surprise))

    return pd.DataFrame(stats_list)

def run_regressions(df):
    """Runs regression analysis."""
    reg_results = []
    
    grouped = df.groupby(['Asset', 'Indicator'])
    
    def extract_stats(model):
        """Safely extracts Beta, P-Value, R-Squared."""
        try:
            if len(model.params) < 2:
                return 0.0, 0.0, 0.0
            return model.params[1], model.pvalues[1], model.rsquared
        except:
            return 0.0, 0.0, 0.0

    for (asset, indicator), group in grouped:
        # Filter for valid surprise and returns
        valid_data = group.dropna(subset=['Surprise'])
        
        for horizon in HORIZONS.keys():
            col = f'Return_{horizon}'
            data = valid_data.dropna(subset=[col])
            
            if len(data) < 5: continue
            
            y = data[col]
            
            # Models
            # 1. Abs(Surprise)
            try:
                X_abs = sm.add_constant(data['Surprise'].abs())
                model_abs = sm.OLS(y, X_abs).fit()
                beta, pval, r2 = extract_stats(model_abs)
            except:
                beta, pval, r2 = 0.0, 0.0, 0.0
                
            reg_results.append({
                'Asset': asset, 'Indicator': indicator, 'Horizon': horizon, 'Model': 'Abs(Surprise)',
                'Beta': beta, 'P-Value': pval, 'R-Squared': r2
            })
            
            # 2. Raw Surprise
            try:
                X_raw = sm.add_constant(data['Surprise'])
                model_raw = sm.OLS(y, X_raw).fit()
                beta, pval, r2 = extract_stats(model_raw)
            except:
                beta, pval, r2 = 0.0, 0.0, 0.0
                
            reg_results.append({
                'Asset': asset, 'Indicator': indicator, 'Horizon': horizon, 'Model': 'Raw Surprise',
                'Beta': beta, 'P-Value': pval, 'R-Squared': r2
            })
            
            # 3. Positive Surprise Only
            pos_data = data[data['Surprise'] > 0]
            if len(pos_data) > 5:
                try:
                    X_pos = sm.add_constant(pos_data['Surprise'].abs()) # Abs is same as raw here
                    model_pos = sm.OLS(pos_data[col], X_pos).fit()
                    beta, pval, r2 = extract_stats(model_pos)
                except:
                    beta, pval, r2 = 0.0, 0.0, 0.0
                    
                reg_results.append({
                    'Asset': asset, 'Indicator': indicator, 'Horizon': horizon, 'Model': 'Positive Surprise Only',
                    'Beta': beta, 'P-Value': pval, 'R-Squared': r2
                })
                
            # 4. Negative Surprise Only
            neg_data = data[data['Surprise'] < 0]
            if len(neg_data) > 5:
                try:
                    X_neg = sm.add_constant(neg_data['Surprise'].abs()) # Use Abs magnitude
                    model_neg = sm.OLS(neg_data[col], X_neg).fit()
                    beta, pval, r2 = extract_stats(model_neg)
                except:
                    beta, pval, r2 = 0.0, 0.0, 0.0
                    
                reg_results.append({
                    'Asset': asset, 'Indicator': indicator, 'Horizon': horizon, 'Model': 'Negative Surprise Only',
                    'Beta': beta, 'P-Value': pval, 'R-Squared': r2
                })
                
    return pd.DataFrame(reg_results)

# ==========================================
# Main Execution
# ==========================================

def main():
    # 1. Load Data
    macro_df = load_macro_data()
    if macro_df.empty:
        print("No macro data found.")
        return

    assets = {}
    
    btc = load_btc()
    if btc is not None: assets['BTC-USD'] = btc
    
    sp = load_sp_futures()
    if sp is not None: assets['S&P 500 Futures'] = sp
    
    gold = load_gold()
    if gold is not None: assets['Gold'] = gold
    
    brent = load_brent()
    if brent is not None: assets['Brent Oil'] = brent
    
    if not assets:
        print("No asset data found.")
        return

    # 2. Calculate Returns
    all_event_returns = []
    for name, price_df in assets.items():
        res = calculate_event_returns(macro_df, price_df, name)
        all_event_returns.append(res)
        
    full_df = pd.concat(all_event_returns, ignore_index=True)
    
    # 3. Statistics
    print("Calculating Statistics...")
    stats_df = calculate_statistics(full_df, assets)
    
    # 4. Regressions
    print("Running Regressions...")
    reg_df = run_regressions(full_df)
    
    # 5. Save Output
    print(f"Saving results to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        reg_df.to_excel(writer, sheet_name='Regressions', index=False)
        full_df.to_excel(writer, sheet_name='Raw_Data', index=False)
        
    print("Done!")

if __name__ == "__main__":
    main()
