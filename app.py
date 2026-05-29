from flask import Flask, Response
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import warnings
import time
import threading
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)
latest_csv = "Loading... (training models)"

# ==================== All your original functions ====================
# (copy them exactly as before – they remain unchanged)
def get_live_data(ticker="AAPL", period="7d", interval="1m"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.columns = df.columns.droplevel(1)
    return df[['Open','High','Low','Close','Volume']].dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close  = (df['Low']  - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_features(df):
    df = df.copy()
    df['rsi'] = compute_rsi(df['Close'], 14)
    df['ema_fast'] = df['Close'].ewm(span=9).mean()
    df['ema_slow'] = df['Close'].ewm(span=21).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['mom'] = df['Close'].pct_change(5)
    df['atr'] = compute_atr(df, 14)
    rolling_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['Close'].rolling(20).mean() + 2 * rolling_std
    df['bb_lower'] = df['Close'].rolling(20).mean() - 2 * rolling_std
    df['bb_pos'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['my_signal'] = (df['rsi'] - 50) * df['vol_ratio'] * df['bb_pos']
    return df.dropna()

def create_multi_horizon_dataset(df, horizons, atr_mult=1.5):
    features = ['rsi','macd','mom','atr','bb_pos','vol_ratio','my_signal']
    rows = []
    max_h = max(horizons)
    n = len(df)
    for i in range(n - max_h):
        base_price = df['Close'].iloc[i]
        base_atr   = df['atr'].iloc[i]
        lower_bar  = base_price - atr_mult * base_atr
        upper_bar  = base_price + atr_mult * base_atr
        for h in horizons:
            fut = i + h
            if fut >= n:
                continue
            target_down = 1 if df['Close'].iloc[fut] < base_price else 0
            touch = 2
            for j in range(i+1, i+h+1):
                if df['Low'].iloc[j] <= lower_bar:
                    touch = 0
                    break
                if df['High'].iloc[j] >= upper_bar:
                    touch = 1
                    break
            min_low  = df['Low'].iloc[i+1:i+h+1].min()
            max_high = df['High'].iloc[i+1:i+h+1].max()
            row = {**df[features].iloc[i].to_dict(),
                   'horizon': h,
                   'target_down': target_down,
                   'first_touch': touch,
                   'min_low': min_low,
                   'max_high': max_high}
            rows.append(row)
    return pd.DataFrame(rows)

def train_models(data):
    feat = ['rsi','macd','mom','atr','bb_pos','vol_ratio','my_signal','horizon']
    X = data[feat]
    y_down = data['target_down']
    X_train, X_test, y_train, y_test = train_test_split(X, y_down, test_size=0.2, random_state=42)
    scaler_down = StandardScaler()
    X_train_s = scaler_down.fit_transform(X_train)
    cal_down = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        cv=3, method='isotonic')
    cal_down.fit(X_train_s, y_train)
    y_touch = data['first_touch']
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_touch, test_size=0.2, random_state=42)
    scaler_touch = StandardScaler()
    X_train_ts = scaler_touch.fit_transform(X_train_t)
    rf_touch = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf_touch.fit(X_train_ts, y_train_t)
    y_min = data['min_low']
    X_train_min, X_test_min, y_train_min, y_test_min = train_test_split(X, y_min, test_size=0.2, random_state=42)
    scaler_min = StandardScaler()
    X_train_min_s = scaler_min.fit_transform(X_train_min)
    rf_min = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf_min.fit(X_train_min_s, y_train_min)
    y_max = data['max_high']
    X_train_max, X_test_max, y_train_max, y_test_max = train_test_split(X, y_max, test_size=0.2, random_state=42)
    scaler_max = StandardScaler()
    X_train_max_s = scaler_max.fit_transform(X_train_max)
    rf_max = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf_max.fit(X_train_max_s, y_train_max)
    return ((cal_down, scaler_down), (rf_touch, scaler_touch),
            (rf_min, scaler_min), (rf_max, scaler_max))

def predict_for_horizon(latest_df, horizon, models):
    (model_down, scaler_down), (model_touch, scaler_touch), \
    (model_min, scaler_min), (model_max, scaler_max) = models
    features = ['rsi','macd','mom','atr','bb_pos','vol_ratio','my_signal']
    X_raw = latest_df[features].iloc[-1:].copy()
    X_raw['horizon'] = horizon
    X_s_down  = scaler_down.transform(X_raw)
    X_s_touch = scaler_touch.transform(X_raw)
    X_s_min   = scaler_min.transform(X_raw)
    X_s_max   = scaler_max.transform(X_raw)
    price = latest_df['Close'].iloc[-1]
    proba_down = model_down.predict_proba(X_s_down)[0]
    p_down, p_up = proba_down[0], proba_down[1]
    touch_proba = model_touch.predict_proba(X_s_touch)[0]
    p_first_down, p_first_up, p_neither = touch_proba[0], touch_proba[1], touch_proba[2]
    any_hit = p_first_down + p_first_up
    cond_down = p_first_down / any_hit if any_hit > 0 else 0.5
    cond_up   = p_first_up / any_hit if any_hit > 0 else 0.5
    exp_min = model_min.predict(X_s_min)[0]
    exp_max = model_max.predict(X_s_max)[0]
    direction = "DOWN" if cond_down > cond_up else "UP"
    extreme = exp_min if direction == "DOWN" else exp_max
    return {
        'horizon': horizon,
        'p_down': p_down,
        'p_up': p_up,
        'cond_down': cond_down,
        'cond_up': cond_up,
        'direction': direction,
        'extreme': extreme
    }

def minutes_until_close(now_utc=None):
    if now_utc is None:
        now_utc = datetime.utcnow()
    utc_hour = now_utc.hour
    utc_min  = now_utc.minute
    close_utc_minutes = 20 * 60
    current_utc_minutes = utc_hour * 60 + utc_min
    return max(0, close_utc_minutes - current_utc_minutes)

def generate_forecast_csv(df_live, models):
    fixed_horizons = [1, 5, 15, 120, 480, 1440]
    mins_to_close = minutes_until_close()
    results = []
    for h in fixed_horizons:
        res = predict_for_horizon(df_live, h, models)
        results.append(res)
    if mins_to_close > 0:
        res_close = predict_for_horizon(df_live, mins_to_close, models)
        res_close['horizon'] = f"to close ({mins_to_close}m)"
        results.append(res_close)
    df = pd.DataFrame(results)
    df['p_down']    = (df['p_down'] * 100).round(1).astype(str) + '%'
    df['p_up']      = (df['p_up'] * 100).round(1).astype(str) + '%'
    df['cond_down'] = (df['cond_down'] * 100).round(1).astype(str) + '%'
    df['cond_up']   = (df['cond_up'] * 100).round(1).astype(str) + '%'
    df['extreme']   = df['extreme'].round(2)
    df.rename(columns={
        'horizon': 'Horizon',
        'p_down': 'P(Down)',
        'p_up': 'P(Up)',
        'cond_down': 'Cond Dn 1st',
        'cond_up': 'Cond Up 1st',
        'direction': 'Direction',
        'extreme': 'Likely Extreme'
    }, inplace=True)
    return df.to_csv(index=False)

# ==================== Background loop ====================
def forecast_loop():
    global latest_csv
    print("Training models...")
    df_hist = get_live_data("AAPL", period="7d", interval="1m")
    df_hist = compute_features(df_hist)
    fixed_horizons = [1, 5, 15, 120, 480, 1440]
    data = create_multi_horizon_dataset(df_hist, horizons=fixed_horizons)
    models = train_models(data)
    print("Models ready. Updating every 15 seconds...")
    while True:
        try:
            df_live = get_live_data("AAPL", period="3d", interval="1m")
            df_live = compute_features(df_live)
            if not df_live.empty:
                latest_csv = generate_forecast_csv(df_live, models)
            time.sleep(15)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(15)

# ==================== Flask route ====================
@app.route('/forecast.csv')
def serve_csv():
    return Response(latest_csv, mimetype='text/csv',
                    headers={"Cache-Control": "no-cache"})

# Start background thread when the app starts (not on first request, for Render)
@app.before_first_request
def start_background():
    threading.Thread(target=forecast_loop, daemon=True).start()

# Render requires the host to be 0.0.0.0 and uses the PORT env variable
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)