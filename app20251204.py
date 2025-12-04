import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
import datetime
import pandas as pd

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

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
            
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in hist.columns for col in required_cols):
            return None, f"資料欄位缺失，抓到的欄位: {list(hist.columns)}"

        return hist, None
    except Exception as e:
        return None, str(e)

def get_google_news(query):
    try:
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        feed = feedparser.parse(rss_url)
        return feed.entries
    except: return []

# --- MDD 計算函數 (數值) ---
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

# --- 回測函數 ---
def run_backtest(df, short_window, long_window, initial_capital, use_macd_filter=False):
    data = df.copy()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    
    if 'MACD' not in data.columns:
        data = calculate_macd(data)

    data['Signal'] = 0
    data.iloc[long_window:, data.columns.get_loc('Signal')] = 0 
    
    if not use_macd_filter:
        mask = data['Short_MA'] > data['Long_MA']
        data.loc[mask, 'Signal'] = 1
    else:
        signals = []
        status = 0
        for i in range(len(data)):
            s_ma = data['Short_MA'].iloc[i]
            l_ma = data['Long_MA'].iloc[i]
            macd_val = data['MACD'].iloc[i]
            
            if pd.isna(s_ma) or pd.isna(l_ma) or pd.isna(macd_val):
                signals.append(0)
                continue
            
            if status == 0:
                if (s_ma > l_ma) and (macd_val > 0): status = 1
            elif status == 1:
                if (s_ma < l_ma) and (macd_val < 0): status = 0
            signals.append(status)
        data['Signal'] = signals

    data['Position'] = data['Signal'].diff()
    cash = initial_capital
    holdings = 0
    asset_history = []
    trade_log = [] 
    
    for i in range(len(data)):
        price = data['Close'].iloc[i]
        date = data.index[i]
        
        if pd.isna(price):
            asset_history.append(asset_history[-1] if asset_history else initial_capital)
            continue

        position_change = data['Position'].iloc[i]
        
        if position_change == 1 and cash > 0:
            holdings = cash / price
            cash = 0
            trade_log.append({"日期": date.strftime('%Y-%m-%d'), "動作": "買進", "價格": price, "資產": holdings*price})
        elif position_change == -1 and holdings > 0:
            cash = holdings * price
            holdings = 0
            trade_log.append({"日期": date.strftime('%Y-%m-%d'), "動作": "賣出", "價格": price, "資產": cash})
            
        asset_history.append(cash + (holdings * price))
        
    data['Total_Asset'] = asset_history
    trade_df = pd.DataFrame(trade_log)
    return data, trade_df

# ========================================================
#   模式 A: 單一個股分析
# ========================================================
if app_mode == "📊 單一個股分析":
    st.title("📊 單一個股分析儀表板")
    
    st.sidebar.header("數據設定")
    input_ticker = st.sidebar.text_input("輸入股票代碼", value="2330.TW")
    if input_ticker.isdigit() and len(input_ticker) == 4:
        stock_id = input_ticker + ".TW"
    else:
        stock_id = input_ticker

    time_mode = st.sidebar.radio("時間模式", ["預設區間", "自訂日期"])
    start_date, end_date, selected_period = None, None, None
    
    if time_mode == "預設區間":
        selected_period = st.sidebar.selectbox("範圍", ["1y", "3y", "5y", "10y", "20y", "max"], index=2)
        if selected_period == "max":
            st.sidebar.info("💡 選擇 'max' 會抓取所有歷史資料。")
    else:
        default_start = datetime.date(1980, 1, 1)
        start_date = st.sidebar.date_input("開始", default_start)
        end_date = st.sidebar.date_input("結束", datetime.date.today())

    st.sidebar.subheader("圖表指標")
    ma_days = st.sidebar.multiselect("均線 (MA)", [5, 20, 60, 120, 240], default=[20, 60])
    show_signals = st.sidebar.checkbox("顯示買賣訊號", value=True)
    show_bb = st.sidebar.checkbox("布林通道", False)
    show_vp = st.sidebar.checkbox("籌碼密集區", True)
    show_macd = st.sidebar.checkbox("MACD", True) 

    st.sidebar.divider()
    st.sidebar.subheader("💰 回測參數")
    initial_capital = st.sidebar.number_input("初始本金", value=1000000)
    
    st.sidebar.markdown("**策略 A (純均線)**")
    s1_short = st.sidebar.number_input("A 短均線", value=5, key="s1_s")
    s1_long = st.sidebar.number_input("A 長均線", value=20, key="s1_l")
    
    st.sidebar.divider()
    st.sidebar.markdown("**策略 B (均線+MACD)**")
    use_macd_b = st.sidebar.checkbox("✅ 啟用 MACD 濾網", value=True)
    s2_short = st.sidebar.number_input("B 短均線", value=5, key="s2_s")
    s2_long = st.sidebar.number_input("B 長均線", value=20, key="s2_l")
    
    run_backtest_btn = st.sidebar.button("🚀 執行回測")

    if stock_id:
        with st.spinner('資料下載中...'):
            df, error_msg = get_stock_data(stock_id, time_mode, period=selected_period, start=start_date, end=end_date)
        
        if df is not None and not df.empty:
            df = calculate_macd(df)
            
            # Drawdown 計算 (For Main Chart)
            roll_max = df['Close'].cummax()
            df['Drawdown'] = (df['Close'] - roll_max) / roll_max

            # 1. 基本資訊
            st.subheader(f"{stock_id} 走勢分析")
            c1, c2, c3, c4 = st.columns(4)
            close = df['Close'].iloc[-1]
            change = close - df['Close'].iloc[-2]
            pct = (change / df['Close'].iloc[-2]) * 100
            hist_mdd = calculate_mdd(df['Close'])
            
            c1.metric("當前股價", f"{close:.2f}", f"{change:.2f} ({pct:.2f}%)")
            c2.metric("區間最高", f"{df['High'].max():.2f}")
            c3.metric("區間最低", f"{df['Low'].min():.2f}")
            c4.metric("歷史 MDD (買持)", f"{hist_mdd:.2f}%")

            # 2. 技術分析主圖
            fig = make_subplots(
                rows=4, cols=1, 
                shared_xaxes=True, 
                row_heights=[0.5, 0.1, 0.15, 0.25], 
                vertical_spacing=0.03
            )

            # Row 1: K Line
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
            colors = ['orange', 'blue', 'purple', 'black']
            for i, d in enumerate(sorted(ma_days)):
                ma = df['Close'].rolling(d).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma, mode='lines', name=f"MA{d}", line=dict(width=1.5, color=colors[i%4])), row=1, col=1)

            # Row 2: Volume
            vol_color = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_color, name="量"), row=2, col=1)
            
            # Row 3: MACD
            if show_macd:
                hist_color = ['red' if h < 0 else 'green' for h in df['MACD_Hist']]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=hist_color, name='MACD'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange'), name='快線'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='blue'), name='慢線'), row=3, col=1)

            # Row 4: Main Chart Drawdown (Benchmark)
            fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], fill='tozeroy', mode='lines', line=dict(color='gray', width=1), name='買進持有回撤'), row=4, col=1)

            fig.update_layout(height=900, xaxis_rangeslider_visible=False)
            fig.update_yaxes(title_text="回撤 %", tickformat=".0%", row=4, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # --- 回測結果區 ---
            if run_backtest_btn:
                st.divider()
                st.subheader("💰 策略績效與風險分析")
                
                # 計算回測
                res1, log1 = run_backtest(df, s1_short, s1_long, initial_capital, False)
                res2, log2 = run_backtest(df, s2_short, s2_long, initial_capital, use_macd_b)
                buy_hold = (initial_capital / df['Close'].iloc[0]) * df['Close']
                
                # 計算回撤序列 (Series)
                def get_dd_series(series):
                    return (series - series.cummax()) / series.cummax()

                dd_A = get_dd_series(res1['Total_Asset'])
                dd_B = get_dd_series(res2['Total_Asset'])
                dd_BH = get_dd_series(buy_hold) # 買進持有回撤

                # 績效指標
                def get_perf(series):
                    ret = ((series.iloc[-1] - initial_capital) / initial_capital) * 100
                    mdd = calculate_mdd(series)
                    return ret, mdd

                p1, m1 = get_perf(res1['Total_Asset'])
                p2, m2 = get_perf(res2['Total_Asset'])
                pb, mb = get_perf(buy_hold)

                # 顯示指標
                col_a, col_b, col_c = st.columns(3)
                col_a.info(f"策略 A (純均線)"); col_a.metric("報酬率", f"{p1:.1f}%", f"MDD: {m1:.1f}%")
                col_b.info(f"策略 B (均線+MACD)"); col_b.metric("報酬率", f"{p2:.1f}%", f"MDD: {m2:.1f}%")
                col_c.warning(f"買進持有 (基準)"); col_c.metric("報酬率", f"{pb:.1f}%", f"MDD: {mb:.1f}%")

                # 圖表 1: 資產成長
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=res1.index, y=res1['Total_Asset'], name='策略A 資產', line=dict(color='gold')))
                fig_bt.add_trace(go.Scatter(x=res2.index, y=res2['Total_Asset'], name='策略B 資產', line=dict(color='cyan')))
                fig_bt.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold, name='買進持有', line=dict(color='gray', dash='dot')))
                fig_bt.update_layout(title="📈 資產成長曲線", height=400, hovermode="x unified")
                st.plotly_chart(fig_bt, use_container_width=True)

                # 圖表 2: 水下圖比較 (New!) - 比較三個策略的歷史回撤
                fig_dd_ts = go.Figure()
                fig_dd_ts.add_trace(go.Scatter(x=dd_BH.index, y=dd_BH, fill='tozeroy', line=dict(color='gray', width=1), name='買進持有 (基準)'))
                fig_dd_ts.add_trace(go.Scatter(x=dd_A.index, y=dd_A, line=dict(color='gold', width=1.5), name='策略A 回撤'))
                fig_dd_ts.add_trace(go.Scatter(x=dd_B.index, y=dd_B, line=dict(color='cyan', width=1.5), name='策略B 回撤'))
                
                fig_dd_ts.update_layout(
                    title="🌊 水下圖 (Underwater Plot) - 歷史回撤比較",
                    yaxis_title="回撤幅度 %",
                    height=350,
                    hovermode="x unified"
                )
                fig_dd_ts.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_dd_ts, use_container_width=True)

                # 圖表 3: 風險分布圖
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=dd_A, name='策略 A', nbinsx=100, opacity=0.6, marker_color='gold'))
                fig_dist.add_trace(go.Histogram(x=dd_B, name='策略 B', nbinsx=100, opacity=0.6, marker_color='cyan'))
                fig_dist.add_trace(go.Histogram(x=dd_BH, name='買進持有', nbinsx=100, opacity=0.4, marker_color='gray'))
                
                fig_dist.update_layout(
                    title="📊 回撤機率分布 (Risk Distribution)",
                    xaxis_title="回撤幅度 %",
                    yaxis_title="發生天數",
                    barmode='overlay',
                    height=350
                )
                fig_dist.update_xaxes(tickformat=".0%")
                st.plotly_chart(fig_dist, use_container_width=True)
                
                c_log1, c_log2 = st.columns(2)
                with c_log1:
                    with st.expander(f"📜 策略 A 交易明細"):
                        if not log1.empty: st.dataframe(log1)
                        else: st.write("無交易")
                with c_log2:
                    with st.expander(f"📜 策略 B 交易明細"):
                        if not log2.empty: st.dataframe(log2)
                        else: st.write("無交易")

        else:
            st.error(f"❌ 無法讀取數據: {error_msg}")

elif app_mode == "🔍 策略選股器":
    st.title("🔍 均線策略選股器")
    c1, c2 = st.columns(2)
    s_ma = c1.number_input("短均線", value=5)
    l_ma = c2.number_input("長均線", value=20)
    user_tickers = st.text_area("觀察清單", "2330, 2317, 2454, 2308, 0050")
    if st.button("🚀 開始掃描"):
        tickers = [t.strip()+".TW" for t in user_tickers.split(",") if t.strip()]
        results = []
        bar = st.progress(0)
        for i, t in enumerate(tickers):
            bar.progress((i+1)/len(tickers))
            try:
                df = yf.download(t, period="3mo", auto_adjust=True, progress=False)
                if not df.empty and len(df) > l_ma:
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df['S'] = df['Close'].rolling(s_ma).mean()
                    df['L'] = df['Close'].rolling(l_ma).mean()
                    curr, prev = df.iloc[-1], df.iloc[-2]
                    if (prev['S'] < prev['L'] and curr['S'] > curr['L']) or (curr['Close'] > curr['S'] > curr['L']):
                        results.append({"代碼": t, "現價": curr['Close'], "訊號": "多頭/金叉"})
            except: continue
        bar.empty()
        if results: st.dataframe(pd.DataFrame(results))
        else: st.warning("無符合條件股票")
