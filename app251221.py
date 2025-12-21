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
st.set_page_config(page_title="全方位股票分析系統 (Pro版)", layout="wide")

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

        # 處理 MultiIndex (yfinance 新版可能的格式)
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

# --- [新增] RSI 計算函數 ---
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# --- 預估成交量計算函數 ---
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
            elapsed_minutes = delta.total_seconds() / 60
            if elapsed_minutes > 0:
                est_volume = current_vol * (270 / elapsed_minutes)
                vol_status = "盤中預估 ⏳"
        elif now < start_time:
             vol_status = "尚未開盤 💤"
    
    vol_ratio = est_volume / vol_ma5 if vol_ma5 > 0 else 0
    return est_volume, vol_ma5, vol_ratio, vol_status

# --- 均線扣抵與預測函數 ---
def render_deduction_analysis(df, ma_days=20):
    st.markdown(f"### 🔮 MA{ma_days} 均線扣抵與未來預測")
    
    if len(df) < ma_days:
        st.warning("資料不足，無法計算扣抵值")
        return

    deduction_price = df['Close'].iloc[-ma_days]
    deduction_date = df.index[-ma_days].strftime('%Y-%m-%d')
    current_ma = df['Close'].rolling(ma_days).mean().iloc[-1]
    last_close = df['Close'].iloc[-1]
    
    col1, col2 = st.columns(2)
    col1.metric("目前 MA 數值", f"{current_ma:.2f}")
    col2.metric(f"扣抵值 ({deduction_date})", f"{deduction_price:.2f}", 
                delta="壓力 (高於現價)" if deduction_price > last_close else "支撐 (低於現價)",
                delta_color="inverse") 
    
    st.write("#### 🎛️ 明日股價模擬器")
    sim_min = float(last_close * 0.9)
    sim_max = float(last_close * 1.1)
    sim_price = st.slider("預估明日收盤價", min_value=sim_min, max_value=sim_max, value=float(last_close), step=0.5)
    
    new_ma = current_ma + (sim_price - deduction_price) / ma_days
    trend = "翻揚 📈" if new_ma > current_ma else "下彎 📉"
    if abs(new_ma - current_ma) < 0.01: trend = "持平 ➖"
    
    c_res1, c_res2, c_res3 = st.columns(3)
    c_res1.metric("模擬明日股價", f"{sim_price:.2f}")
    c_res2.metric("預測明日 MA", f"{new_ma:.2f}", f"{new_ma - current_ma:.2f}")
    c_res3.info(f"均線趨勢：**{trend}**")

# --- [大幅優化] 策略回測函數 ---
def run_backtest_optimized(df, short_window, long_window, initial_capital, 
                           stop_loss_pct=0.05, take_profit_pct=0.15, 
                           use_trend_filter=True, trend_ma_days=60,
                           use_rsi_filter=True):
    
    data = df.copy()
    # 確保基本指標存在
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    
    # 計算額外濾網指標
    if use_trend_filter:
        data['Trend_MA'] = data['Close'].rolling(window=trend_ma_days).mean()
    
    if use_rsi_filter:
        data['RSI'] = calculate_rsi(data)

    # 初始化變數
    cash = initial_capital
    holdings = 0
    asset_history = []
    trade_log = []
    entry_price = 0
    
    # 狀態標記
    in_position = False
    
    # 為了避免 look-ahead bias，我們逐行遍歷
    start_idx = max(long_window, trend_ma_days if use_trend_filter else 0, 20)
    
    for i in range(start_idx, len(data)):
        curr_date = data.index[i]
        curr_price = data['Close'].iloc[i]
        
        # 當日均線數值
        s_ma = data['Short_MA'].iloc[i]
        l_ma = data['Long_MA'].iloc[i]
        prev_s_ma = data['Short_MA'].iloc[i-1]
        prev_l_ma = data['Long_MA'].iloc[i-1]
        
        # 濾網條件檢查
        trend_ok = True
        if use_trend_filter:
            trend_ma = data['Trend_MA'].iloc[i]
            if curr_price < trend_ma: # 股價在季線下，不做多
                trend_ok = False
                
        rsi_ok = True
        if use_rsi_filter:
            curr_rsi = data['RSI'].iloc[i]
            if curr_rsi > 75: # RSI 過熱，不追高
                rsi_ok = False

        # --- 進場邏輯 (黃金交叉 + 濾網) ---
        is_golden_cross = (prev_s_ma < prev_l_ma) and (s_ma > l_ma)
        
        if not in_position:
            if is_golden_cross and trend_ok and rsi_ok:
                holdings = cash / curr_price
                cash = 0
                entry_price = curr_price
                in_position = True
                trade_log.append({
                    "日期": curr_date.strftime('%Y-%m-%d'),
                    "動作": "買進",
                    "價格": curr_price,
                    "原因": "金叉確認",
                    "資產": holdings * curr_price
                })
        
        # --- 出場邏輯 (停損 / 停利 / 死亡交叉) ---
        elif in_position:
            action = None
            reason = ""
            
            # 1. 停損
            if curr_price <= entry_price * (1 - stop_loss_pct):
                action = "賣出"
                reason = f"觸發停損 ({stop_loss_pct*100}%)"
            
            # 2. 停利
            elif curr_price >= entry_price * (1 + take_profit_pct):
                action = "賣出"
                reason = f"觸發停利 ({take_profit_pct*100}%)"
            
            # 3. 死亡交叉
            elif (prev_s_ma > prev_l_ma) and (s_ma < l_ma):
                action = "賣出"
                reason = "死亡交叉"
            
            if action:
                cash = holdings * curr_price
                holdings = 0
                in_position = False
                trade_log.append({
                    "日期": curr_date.strftime('%Y-%m-%d'),
                    "動作": action,
                    "價格": curr_price,
                    "原因": reason,
                    "資產": cash
                })

        # 紀錄當日資產
        current_asset_value = cash + (holdings * curr_price)
        asset_history.append(current_asset_value)
    
    # 補齊前面的空白資料
    pad_len = len(data) - len(asset_history)
    asset_history = [initial_capital] * pad_len + asset_history
    
    data['Total_Asset'] = asset_history
    trade_df = pd.DataFrame(trade_log)
    
    return data, trade_df

# ========================================================
#   模式 A: 單一個股分析
# ========================================================
if app_mode == "📊 單一個股分析":
    st.title("📊 單一個股分析儀表板 (Pro)")
    
    st.sidebar.header("數據設定")
    input_ticker = st.sidebar.text_input("輸入股票代碼", value="2330.TW")
    stock_id = input_ticker if not (input_ticker.isdigit() and len(input_ticker) == 4) else input_ticker + ".TW"

    time_mode = st.sidebar.radio("時間模式", ["預設區間", "自訂日期"])
    start_date, end_date, selected_period = None, None, None
    
    if time_mode == "預設區間":
        selected_period = st.sidebar.selectbox("範圍", ["1y", "2y", "3y", "5y", "10y"], index=2)
    else:
        default_start = datetime.date(2020, 1, 1)
        start_date = st.sidebar.date_input("開始", default_start)
        end_date = st.sidebar.date_input("結束", datetime.date.today())

    st.sidebar.subheader("圖表指標")
    ma_days = st.sidebar.multiselect("均線 (MA)", [5, 10, 20, 60, 120, 240], default=[20, 60])
    show_macd = st.sidebar.checkbox("顯示 MACD", True) 

    st.sidebar.divider()
    st.sidebar.subheader("💰 策略回測參數")
    initial_capital = st.sidebar.number_input("初始本金", value=1000000)
    
    st.sidebar.markdown("**均線設定 (觸發信號)**")
    c1, c2 = st.sidebar.columns(2)
    s_ma_param = c1.number_input("短均線", value=5)
    l_ma_param = c2.number_input("長均線", value=20)
    
    st.sidebar.markdown("**🛡️ 風險控制與濾網**")
    use_trend = st.sidebar.checkbox("啟用季線(60MA) 趨勢濾網", value=True, help="只在股價 > 60MA 時才做多")
    use_rsi = st.sidebar.checkbox("啟用 RSI 過熱濾網", value=True, help="RSI > 75 時不追高")
    
    col_sl, col_tp = st.sidebar.columns(2)
    sl_pct = col_sl.number_input("停損 %", value=5.0, step=0.5) / 100
    tp_pct = col_tp.number_input("停利 %", value=15.0, step=0.5) / 100
    
    run_backtest_btn = st.sidebar.button("🚀 執行回測")

    if stock_id:
        with st.spinner('資料下載中...'):
            df, error_msg = get_stock_data(stock_id, time_mode, period=selected_period, start=start_date, end=end_date)
        
        if df is not None and not df.empty:
            df = calculate_macd(df)
            
            # Drawdown 計算
            roll_max = df['Close'].cummax()
            df['Drawdown'] = (df['Close'] - roll_max) / roll_max

            est_vol, vol_ma5, vol_ratio, vol_status = calculate_volume_analysis(df)

            # 1. 基本資訊
            st.subheader(f"{stock_id} 走勢與量能分析")
            c1, c2, c3, c4, c5 = st.columns(5)
            
            close = df['Close'].iloc[-1]
            change = close - df['Close'].iloc[-2]
            pct = (change / df['Close'].iloc[-2]) * 100
            hist_mdd = calculate_mdd(df['Close'])
            
            c1.metric("當前股價", f"{close:.2f}", f"{change:.2f} ({pct:.2f}%)")
            
            ratio_color = "normal"
            if vol_ratio >= 1.5: ratio_color = "inverse"
            elif vol_ratio <= 0.7: ratio_color = "off"
            
            c2.metric(f"預估量 ({vol_status})", f"{int(est_vol):,}", f"量比: {vol_ratio:.1f}x", delta_color=ratio_color)
            c3.metric("5日均量", f"{int(vol_ma5):,}")
            c4.metric("區間最高", f"{df['High'].max():.2f}")
            c5.metric("歷史 MDD", f"{hist_mdd:.2f}%")

            # 2. 技術分析主圖
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)

            # Row 1: K Line
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
            colors = ['orange', 'blue', 'purple', 'black', 'red', 'green']
            for i, d in enumerate(sorted(ma_days)):
                ma = df['Close'].rolling(d).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma, mode='lines', name=f"MA{d}", line=dict(width=1.5, color=colors[i%len(colors)])), row=1, col=1)

            # Row 2: Volume
            vol_color = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_color, name="量"), row=2, col=1)
            
            # Row 3: MACD
            if show_macd:
                hist_color = ['red' if h < 0 else 'green' for h in df['MACD_Hist']]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=hist_color, name='MACD'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange'), name='快線'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='blue'), name='慢線'), row=3, col=1)

            fig.update_layout(height=800, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # === 均線扣抵分析 ===
            st.divider()
            if ma_days:
                target_ma = st.selectbox("🎯 選擇扣抵均線", ma_days, index=0)
                render_deduction_analysis(df, ma_days=target_ma)

            # --- 回測結果區 ---
            if run_backtest_btn:
                st.divider()
                st.subheader("💰 策略績效報告")
                
                # 執行優化回測
                res_opt, log_opt = run_backtest_optimized(
                    df, s_ma_param, l_ma_param, initial_capital, 
                    stop_loss_pct=sl_pct, 
                    take_profit_pct=tp_pct,
                    use_trend_filter=use_trend,
                    use_rsi_filter=use_rsi
                )
                
                # 計算買進持有 (Benchmark)
                buy_hold = (initial_capital / df['Close'].iloc[0]) * df['Close']
                
                # 績效計算 helper
                def get_perf_metrics(series):
                    total_ret = ((series.iloc[-1] - initial_capital) / initial_capital) * 100
                    mdd = calculate_mdd(series)
                    return total_ret, mdd

                p_opt, m_opt = get_perf_metrics(res_opt['Total_Asset'])
                p_bh, m_bh = get_perf_metrics(buy_hold)

                # 勝率計算
                win_rate = 0
                total_trades = 0
                if not log_opt.empty:
                    sells = log_opt[log_opt['動作'] == '賣出']
                    buys = log_opt[log_opt['動作'] == '買進']
                    # 簡單配對計算 (假設先進先出，且每次清倉)
                    profit_trades = 0
                    total_trades = len(sells)
                    
                    if total_trades > 0:
                        # 這裡做一個簡單的獲利判斷
                        # 由於我們邏輯是清倉，我們可以用 log 裡的 '資產' 來回推
                        # 或是檢查 '原因' 裡是否為停利，或是賣出價格 > 買入價格
                        # 為了準確，我們用價格比較
                        # 注意：這裡假設 buys 和 sells 是成對的，實際回測邏輯是一買一賣
                        for i in range(len(sells)):
                            sell_price = sells.iloc[i]['價格']
                            buy_price = buys.iloc[i]['價格']
                            if sell_price > buy_price:
                                profit_trades += 1
                        win_rate = (profit_trades / total_trades) * 100

                # 顯示指標
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("策略總報酬", f"{p_opt:.1f}%", delta=f"{p_opt-p_bh:.1f}% (vs 大盤)")
                col_b.metric("最大回撤 (MDD)", f"{m_opt:.1f}%")
                col_c.metric("交易次數", f"{total_trades} 次")
                col_d.metric("交易勝率", f"{win_rate:.1f}%")

                # 圖表 1: 資產成長
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=res_opt.index, y=res_opt['Total_Asset'], name='優化策略', line=dict(color='gold', width=2)))
                fig_bt.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold, name='買進持有', line=dict(color='gray', dash='dot')))
                fig_bt.update_layout(title="📈 資產成長曲線", height=400, hovermode="x unified")
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # 交易明細
                with st.expander("📜 查看詳細交易紀錄", expanded=True):
                    if not log_opt.empty:
                        # 格式化表格
                        st.dataframe(log_opt.style.format({"價格": "{:.2f}", "資產": "{:.0f}"}))
                    else:
                        st.info("此區間無觸發交易訊號 (可能是濾網太嚴格或無趨勢)")

        else:
            st.error(f"❌ 無法讀取數據: {error_msg}")

# ========================================================
#   模式 B: 策略選股器 (簡易版)
# ========================================================
elif app_mode == "🔍 策略選股器":
    st.title("🔍 均線策略選股器")
    st.info("此功能掃描「黃金交叉」狀態，您可以再進入個股分析查看詳細濾網回測。")
    
    c1, c2 = st.columns(2)
    s_ma = c1.number_input("短均線", value=5)
    l_ma = c2.number_input("長均線", value=20)
    user_tickers = st.text_area("觀察清單 (逗號分隔)", "2330, 2317, 2454, 2308, 2603, 2609, 2615, 0050")
    
    if st.button("🚀 開始掃描"):
        tickers = [t.strip()+".TW" if not t.strip().endswith(".TW") else t.strip() for t in user_tickers.split(",") if t.strip()]
        results = []
        bar = st.progress(0)
        
        for i, t in enumerate(tickers):
            bar.progress((i+1)/len(tickers))
            try:
                df = yf.download(t, period="6mo", auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                if not df.empty and len(df) > l_ma:
                    df['S'] = df['Close'].rolling(s_ma).mean()
                    df['L'] = df['Close'].rolling(l_ma).mean()
                    
                    curr = df.iloc[-1]
                    prev = df.iloc[-2]
                    
                    # 判斷金叉
                    gc = (prev['S'] < prev['L'] and curr['S'] > curr['L'])
                    # 判斷多頭排列
                    bull = (curr['Close'] > curr['S'] > curr['L'])
                    
                    if gc or bull:
                        results.append({
                            "代碼": t, 
                            "現價": f"{curr['Close']:.2f}", 
                            "訊號": "黃金交叉 🚀" if gc else "多頭排列 📈"
                        })
            except: continue
        
        bar.empty()
        if results: 
            st.dataframe(pd.DataFrame(results))
        else: 
            st.warning("無符合條件股票")
