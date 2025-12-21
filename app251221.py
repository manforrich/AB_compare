import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
import datetime
import pandas as pd
import pytz
import numpy as np

# 1. 設定網頁標題
st.set_page_config(page_title="全方位股票分析系統 (Pro+ADX版)", layout="wide")

# --- 側邊欄：模式選擇 ---
st.sidebar.title("🚀 功能選單")
app_mode = st.sidebar.selectbox("選擇功能", ["📊 單一個股分析", "🔍 策略選股器"])

# ========================================================
#   共用函數區
# ========================================================
def get_stock_data(ticker, mode="預設區間", period="1y", start=None, end=None):
    try:
        if mode == "預設區間":
            hist = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        else:
            hist = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        if hist.empty: 
            return None, "Yahoo Finance 回傳空資料，請檢查代碼或日期範圍。"

        # 處理 MultiIndex
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
            
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in hist.columns for col in required_cols):
            return None, f"資料欄位缺失，抓到的欄位: {list(hist.columns)}"

        return hist, None
    except Exception as e:
        return None, str(e)

# --- MDD 計算函數 ---
def calculate_mdd(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return max_drawdown * 100

# --- MACD 計算函數 ---
def calculate_macd(df, fast=12, slow=26, signal=9):
    data = df.copy()
    data['EMA_Fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
    data['EMA_Slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
    data['MACD'] = data['EMA_Fast'] - data['EMA_Slow']
    data['Signal_Line'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
    return data

# --- RSI 計算函數 ---
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# --- [新增] ADX 計算函數 (純 Pandas 實作) ---
def calculate_adx(df, period=14):
    data = df.copy()
    
    # 1. 計算 +DM, -DM, TR
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
    data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    data['UpMove'] = data['High'] - data['High'].shift(1)
    data['DownMove'] = data['Low'].shift(1) - data['Low']
    
    data['+DM'] = np.where((data['UpMove'] > data['DownMove']) & (data['UpMove'] > 0), data['UpMove'], 0)
    data['-DM'] = np.where((data['DownMove'] > data['UpMove']) & (data['DownMove'] > 0), data['DownMove'], 0)
    
    # 2. 平滑處理 (Wilder's Smoothing) -> 近似於 EMA(alpha=1/period)
    # 為了效能，這裡使用 EWM 近似
    alpha = 1 / period
    data['TR_smooth'] = data['TR'].ewm(alpha=alpha, adjust=False).mean()
    data['+DM_smooth'] = data['+DM'].ewm(alpha=alpha, adjust=False).mean()
    data['-DM_smooth'] = data['-DM'].ewm(alpha=alpha, adjust=False).mean()
    
    # 3. 計算 DI
    data['+DI'] = 100 * (data['+DM_smooth'] / data['TR_smooth'])
    data['-DI'] = 100 * (data['-DM_smooth'] / data['TR_smooth'])
    
    # 4. 計算 DX 與 ADX
    data['DX'] = 100 * abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])
    data['ADX'] = data['DX'].ewm(alpha=alpha, adjust=False).mean()
    
    return data['ADX'].fillna(0)

# --- 預估成交量計算 ---
def calculate_volume_analysis(df):
    last_date = df.index[-1]
    current_vol = df['Volume'].iloc[-1]
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    
    tw_tz = pytz.timezone('Asia/Taipei')
    now = datetime.datetime.now(tw_tz)
    
    is_today = last_date.date() == now.date()
    est_volume = current_vol
    vol_status = "收盤確認"
    
    if is_today:
        start_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
        end_time = now.replace(hour=13, minute=30, second=0, microsecond=0)
        if start_time < now < end_time:
            delta = now - start_time
            mins = delta.total_seconds() / 60
            if mins > 0:
                est_volume = current_vol * (270 / mins)
                vol_status = "盤中預估 ⏳"
        elif now < start_time:
             vol_status = "尚未開盤 💤"
    
    vol_ratio = est_volume / vol_ma5 if vol_ma5 > 0 else 0
    return est_volume, vol_ma5, vol_ratio, vol_status

# --- 均線扣抵 ---
def render_deduction_analysis(df, ma_days=20):
    st.markdown(f"### 🔮 MA{ma_days} 均線扣抵與未來預測")
    if len(df) < ma_days: return
    
    deduction_price = df['Close'].iloc[-ma_days]
    current_ma = df['Close'].rolling(ma_days).mean().iloc[-1]
    last_close = df['Close'].iloc[-1]
    
    col1, col2 = st.columns(2)
    col1.metric("目前 MA", f"{current_ma:.2f}")
    col2.metric(f"扣抵值", f"{deduction_price:.2f}", 
                delta="壓力" if deduction_price > last_close else "支撐", delta_color="inverse")
    
    sim_price = st.slider("預估明日收盤價", float(last_close*0.9), float(last_close*1.1), float(last_close))
    new_ma = current_ma + (sim_price - deduction_price) / ma_days
    trend = "翻揚 📈" if new_ma > current_ma else "下彎 📉"
    st.info(f"預測明日 MA: {new_ma:.2f} ({trend})")

# --- [ADX整合版] 策略回測函數 ---
def run_backtest_v2(df, short_window, long_window, initial_capital, 
                    stop_loss_pct=0.05, take_profit_pct=0.15, 
                    use_trend_filter=True, use_rsi_filter=True, 
                    use_adx_filter=True, adx_threshold=25):
    
    data = df.copy()
    data['S_MA'] = data['Close'].rolling(short_window).mean()
    data['L_MA'] = data['Close'].rolling(long_window).mean()
    
    if use_trend_filter:
        data['Trend_MA'] = data['Close'].rolling(60).mean() # 季線
    if use_rsi_filter:
        data['RSI'] = calculate_rsi(data)
    if use_adx_filter:
        data['ADX'] = calculate_adx(data) # 計算 ADX

    cash = initial_capital
    holdings = 0
    asset_history = []
    trade_log = []
    entry_price = 0
    in_position = False
    
    # 避免 Look-ahead bias
    start_idx = max(long_window, 60 if use_trend_filter else 0, 20)
    
    for i in range(start_idx, len(data)):
        date = data.index[i]
        price = data['Close'].iloc[i]
        
        # 1. 進場訊號檢查
        s_ma = data['S_MA'].iloc[i]
        l_ma = data['L_MA'].iloc[i]
        prev_s = data['S_MA'].iloc[i-1]
        prev_l = data['L_MA'].iloc[i-1]
        
        is_golden_cross = (prev_s < prev_l) and (s_ma > l_ma)
        
        # 濾網狀態
        trend_ok = True
        if use_trend_filter and price < data['Trend_MA'].iloc[i]: trend_ok = False
            
        rsi_ok = True
        if use_rsi_filter and data['RSI'].iloc[i] > 75: rsi_ok = False
            
        adx_ok = True
        if use_adx_filter:
            # ADX < 閾值 (例如25) 代表盤整，不進場
            if data['ADX'].iloc[i] < adx_threshold: adx_ok = False

        # --- 買進 ---
        if not in_position:
            if is_golden_cross and trend_ok and rsi_ok and adx_ok:
                holdings = cash / price
                cash = 0
                entry_price = price
                in_position = True
                trade_log.append({"日期": date, "動作": "買進", "價格": price, "原因": "金叉+濾網通過", "資產": holdings*price})
        
        # --- 賣出 ---
        elif in_position:
            action = None
            reason = ""
            
            # 停損
            if price <= entry_price * (1 - stop_loss_pct):
                action = "賣出"
                reason = f"停損 ({stop_loss_pct*100}%)"
            # 停利
            elif price >= entry_price * (1 + take_profit_pct):
                action = "賣出"
                reason = f"停利 ({take_profit_pct*100}%)"
            # 死叉
            elif (prev_s > prev_l) and (s_ma < l_ma):
                action = "賣出"
                reason = "死亡交叉"
                
            if action:
                cash = holdings * price
                holdings = 0
                in_position = False
                trade_log.append({"日期": date, "動作": action, "價格": price, "原因": reason, "資產": cash})

        asset_history.append(cash + (holdings * price))
        
    # 補齊長度
    asset_history = [initial_capital]*(len(data)-len(asset_history)) + asset_history
    data['Total_Asset'] = asset_history
    return data, pd.DataFrame(trade_log)

# ========================================================
#   Main UI
# ========================================================
if app_mode == "📊 單一個股分析":
    st.title("📊 個股策略分析 (含 ADX 濾網)")
    
    st.sidebar.header("1. 數據設定")
    ticker = st.sidebar.text_input("股票代碼", "2330.TW")
    period = st.sidebar.selectbox("資料區間", ["1y", "2y", "3y", "5y", "10y"], index=2)
    
    st.sidebar.header("2. 策略參數")
    initial_capital = st.sidebar.number_input("初始本金", 1000000)
    s_ma = st.sidebar.number_input("短均線", 5)
    l_ma = st.sidebar.number_input("長均線", 20)
    
    st.sidebar.divider()
    st.sidebar.subheader("🛡️ 進階濾網 (Filters)")
    
    use_trend = st.sidebar.checkbox("✅ 季線 (60MA) 趨勢濾網", True, help="股價 > 季線才買")
    use_rsi = st.sidebar.checkbox("✅ RSI 過熱濾網 (>75不買)", True)
    
    # [新增] ADX 選項
    use_adx = st.sidebar.checkbox("✅ ADX 動能濾網 (避開盤整)", True)
    adx_val = st.sidebar.number_input("ADX 門檻值", value=25.0, step=1.0, help="ADX > 此數值才視為有趨勢，通常設 20 或 25")
    
    st.sidebar.divider()
    sl_pct = st.sidebar.number_input("停損 %", 5.0) / 100
    tp_pct = st.sidebar.number_input("停利 %", 15.0) / 100
    
    btn_run = st.sidebar.button("🚀 執行分析與回測")

    if btn_run and ticker:
        with st.spinner("計算中..."):
            df, msg = get_stock_data(ticker, period=period)
            if df is None:
                st.error(msg)
            else:
                # 計算指標
                df = calculate_macd(df)
                df['ADX'] = calculate_adx(df) # 顯示用
                
                # --- 1. 顯示基本資訊 ---
                st.subheader(f"{ticker} 走勢概覽")
                col1, col2, col3, col4 = st.columns(4)
                est_vol, _, vol_ratio, vol_status = calculate_volume_analysis(df)
                
                curr_price = df['Close'].iloc[-1]
                curr_adx = df['ADX'].iloc[-1]
                
                col1.metric("現價", f"{curr_price:.2f}")
                col2.metric(f"預估量 ({vol_status})", f"{int(est_vol):,}", f"量比: {vol_ratio:.1f}x")
                
                # ADX 狀態顯示
                adx_status = "趨勢強勁 🔥" if curr_adx > 25 else "盤整/無趨勢 💤"
                col3.metric("ADX 動能", f"{curr_adx:.2f}", adx_status)
                col4.metric("RSI (14)", f"{calculate_rsi(df).iloc[-1]:.2f}")

                # --- 2. 畫圖 ---
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
                
                # K線 + MA
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(s_ma).mean(), name=f'MA{s_ma}', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(l_ma).mean(), name=f'MA{l_ma}', line=dict(color='blue')), row=1, col=1)
                
                # ADX 指標圖 
                fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX', line=dict(color='purple')), row=2, col=1)
                fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=adx_val, y1=adx_val, line=dict(color="red", dash="dash"), row=2, col=1)
                
                # MACD
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD柱'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='DIF'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='DEM'), row=3, col=1)
                
                fig.update_layout(height=800, xaxis_rangeslider_visible=False, title="技術指標與 ADX 趨勢圖")
                st.plotly_chart(fig, use_container_width=True)
                
                # --- 3. 執行回測 ---
                st.divider()
                st.subheader("💰 策略回測結果")
                
                res, logs = run_backtest_v2(df, s_ma, l_ma, initial_capital, sl_pct, tp_pct, use_trend, use_rsi, use_adx, adx_val)
                buy_hold = (initial_capital / df['Close'].iloc[0]) * df['Close']
                
                # 計算績效
                ret_strat = (res['Total_Asset'].iloc[-1] - initial_capital) / initial_capital * 100
                ret_bh = (buy_hold.iloc[-1] - initial_capital) / initial_capital * 100
                
                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("策略總報酬", f"{ret_strat:.1f}%", f"{ret_strat - ret_bh:.1f}% (vs 大盤)")
                c_res2.metric("交易次數", f"{len(logs[logs['動作']=='賣出'])} 次")
                
                # 勝率
                win_rate = 0
                if not logs.empty:
                    sells = logs[logs['動作'] == '賣出']
                    buys = logs[logs['動作'] == '買進']
                    if not sells.empty:
                        wins = sum([1 for i in range(len(sells)) if sells.iloc[i]['價格'] > buys.iloc[i]['價格']])
                        win_rate = (wins / len(sells)) * 100
                c_res3.metric("勝率", f"{win_rate:.1f}%")

                # 畫資金曲線
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=res.index, y=res['Total_Asset'], name='策略資金', line=dict(color='gold', width=2)))
                fig_bt.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold, name='買進持有', line=dict(color='gray', dash='dot')))
                st.plotly_chart(fig_bt, use_container_width=True)
                
                with st.expander("交易明細"):
                    st.dataframe(logs)

elif app_mode == "🔍 策略選股器":
    st.title("🔍 強勢股掃描 (金叉 + ADX)")
    st.info("掃描條件：黃金交叉 + 股價在季線上 + ADX > 25 (趨勢強)")
    
    tickers_input = st.text_area("股票清單", "2330, 2317, 2454, 2603, 2609, 2615, 3037, 3035")
    if st.button("開始掃描"):
        tickers = [t.strip()+".TW" for t in tickers_input.split(",") if t.strip()]
        results = []
        bar = st.progress(0)
        
        for i, t in enumerate(tickers):
            bar.progress((i+1)/len(tickers))
            try:
                df = yf.download(t, period="6mo", progress=False)
                if df.empty: continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                # 計算簡易指標
                df['S'] = df['Close'].rolling(5).mean()
                df['L'] = df['Close'].rolling(20).mean()
                df['季線'] = df['Close'].rolling(60).mean()
                df['ADX'] = calculate_adx(df)
                
                curr = df.iloc[-1]
                prev = df.iloc[-2]
                
                # 條件
                gc = (prev['S'] < prev['L'] and curr['S'] > curr['L'])
                trend_ok = curr['Close'] > curr['季線']
                adx_ok = curr['ADX'] > 25
                
                if gc and trend_ok and adx_ok:
                    results.append({"代碼": t, "現價": curr['Close'], "ADX": f"{curr['ADX']:.1f}", "訊號": "強勢金叉 🔥"})
            except: pass
            
        bar.empty()
        if results: st.dataframe(pd.DataFrame(results))
        else: st.warning("無符合強勢條件的股票")
