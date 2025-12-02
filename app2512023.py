import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
import datetime
import pandas as pd

# 1. 設定網頁標題
st.set_page_config(page_title="全方位股票分析系統", layout="wide")

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
            return None, "Yahoo Finance 回傳空資料，請檢查代碼。"

        # 處理 MultiIndex
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

# --- 升級版回測函數 (含 MACD 濾網邏輯) ---
def run_backtest(df, short_window, long_window, initial_capital, use_macd_filter=False):
    data = df.copy()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    
    # 確保有 MACD 數據
    if 'MACD' not in data.columns:
        data = calculate_macd(data)

    data['Signal'] = 0
    # 避免前幾天 MA 為空值時產生訊號
    data.iloc[long_window:, data.columns.get_loc('Signal')] = 0 
    
    # ---------------------------------------------------------
    # 策略邏輯分支
    # ---------------------------------------------------------
    if not use_macd_filter:
        # === 原始策略：純均線交叉 ===
        mask = data['Short_MA'] > data['Long_MA']
        data.loc[mask, 'Signal'] = 1
    else:
        # === 進階策略：均線 + MACD 濾網 ===
        signals = []
        status = 0 # 0: 空手, 1: 持有
        
        for i in range(len(data)):
            s_ma = data['Short_MA'].iloc[i]
            l_ma = data['Long_MA'].iloc[i]
            macd_val = data['MACD'].iloc[i]
            
            if pd.isna(s_ma) or pd.isna(l_ma) or pd.isna(macd_val):
                signals.append(0)
                continue
            
            if status == 0:
                # 進場判斷：均線金叉 且 MACD 為正
                if (s_ma > l_ma) and (macd_val > 0):
                    status = 1
            elif status == 1:
                # 出場判斷：均線死叉 且 MACD 為負
                if (s_ma < l_ma) and (macd_val < 0):
                    status = 0
            
            signals.append(status)
        
        data['Signal'] = signals

    # ---------------------------------------------------------
    # 計算部位變化與資金
    # ---------------------------------------------------------
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
        
        # 買入訊號
        if position_change == 1 and cash > 0:
            holdings = cash / price
            cash = 0
            trade_log.append({
                "日期": date.strftime('%Y-%m-%d'),
                "動作": "買進 (Buy)",
                "價格": round(price, 2),
                "股數": round(holdings, 4),
                "資金餘額": 0,
                "總資產": round(holdings * price, 0),
                "備註": "均線金叉 + MACD>0" if use_macd_filter else "均線金叉"
            })
            
        # 賣出訊號
        elif position_change == -1 and holdings > 0:
            cash = holdings * price
            holdings = 0
            trade_log.append({
                "日期": date.strftime('%Y-%m-%d'),
                "動作": "賣出 (Sell)",
                "價格": round(price, 2),
                "股數": 0,
                "資金餘額": round(cash, 0),
                "總資產": round(cash, 0),
                "備註": "均線死叉 + MACD<0" if use_macd_filter else "均線死叉"
            })
            
        current_asset = cash + (holdings * price)
        asset_history.append(current_asset)
        
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
        selected_period = st.sidebar.selectbox("範圍", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
    else:
        default_start = datetime.date.today() - datetime.timedelta(days=365)
        start_date = st.sidebar.date_input("開始", default_start)
        end_date = st.sidebar.date_input("結束", datetime.date.today())

    st.sidebar.subheader("圖表指標")
    ma_days = st.sidebar.multiselect("均線 (MA)", [5, 10, 20, 60, 120], default=[5, 20])
    show_signals = st.sidebar.checkbox("顯示買賣訊號 (MA交叉)", value=True)
    show_bb = st.sidebar.checkbox("布林通道", False)
    show_vp = st.sidebar.checkbox("籌碼密集區", True)
    show_macd = st.sidebar.checkbox("MACD", True) 

    st.sidebar.divider()
    st.sidebar.subheader("💰 雙策略回測比較")
    initial_capital = st.sidebar.number_input("初始本金", value=100000)
    
    # 策略 A 設定
    st.sidebar.markdown("**策略 A (純均線策略)**")
    s1_short = st.sidebar.number_input("A 短均線", value=5, key="s1_s")
    s1_long = st.sidebar.number_input("A 長均線", value=20, key="s1_l")
    
    st.sidebar.divider()
    
    # 策略 B 設定
    st.sidebar.markdown("**策略 B (進階策略)**")
    use_macd_b = st.sidebar.checkbox("✅ 啟用 MACD 趨勢濾網", value=True, help="勾選後，買入需MACD>0，賣出需MACD<0")
    s2_short = st.sidebar.number_input("B 短均線", value=5, key="s2_s")
    s2_long = st.sidebar.number_input("B 長均線", value=20, key="s2_l")
    
    run_backtest_btn = st.sidebar.button("🚀 執行雙策略回測")

    if stock_id:
        df, error_msg = get_stock_data(stock_id, time_mode, period=selected_period, start=start_date, end=end_date)
        
        if df is not None and not df.empty:
            df = calculate_macd(df)

            # 1. 股價資訊
            c1, c2, c3, c4 = st.columns(4)
            close = df['Close'].iloc[-1]
            change = close - df['Close'].iloc[-2]
            pct = (change / df['Close'].iloc[-2]) * 100
            c1.metric("股價", f"{close:.2f}", f"{change:.2f} ({pct:.2f}%)")
            c2.metric("最高", f"{df['High'].max():.2f}")
            c3.metric("最低", f"{df['Low'].min():.2f}")
            c4.metric("成交量", f"{int(df['Volume'].iloc[-1]):,}")

            # 2. 繪圖
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                row_heights=[0.6, 0.15, 0.25], 
                vertical_spacing=0.03
            )

            # Row 1: K 線
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
            
            colors = ['orange', 'blue', 'purple', 'black']
            sorted_ma_days = sorted(ma_days)
            for i, d in enumerate(sorted_ma_days):
                ma = df['Close'].rolling(d).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma, mode='lines', name=f"MA{d}", line=dict(width=1.5, color=colors[i%4])), row=1, col=1)

            if show_signals and len(sorted_ma_days) >= 2:
                s_window = sorted_ma_days[0]
                l_window = sorted_ma_days[1]
                temp_s = df['Close'].rolling(s_window).mean()
                temp_l = df['Close'].rolling(l_window).mean()
                buy_cond = (temp_s.shift(1) < temp_l.shift(1)) & (temp_s > temp_l)
                sell_cond = (temp_s.shift(1) > temp_l.shift(1)) & (temp_s < temp_l)
                
                buy_points = df.loc[buy_cond]
                sell_points = df.loc[sell_cond]
                
                if not buy_points.empty:
                    fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#ff2b2b'), name='均線買訊'), row=1, col=1)
                if not sell_points.empty:
                    fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points['High'] * 1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#00cc00'), name='均線賣訊'), row=1, col=1)

            if show_bb:
                mid = df['Close'].rolling(20).mean()
                std = df['Close'].rolling(20).std()
                fig.add_trace(go.Scatter(x=df.index, y=mid+2*std, line=dict(color='rgba(0,100,255,0.3)'), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=mid-2*std, line=dict(color='rgba(0,100,255,0.3)'), fill='tonexty', fillcolor='rgba(0,100,255,0.1)', name='布林'), row=1, col=1)

            if show_vp:
                fig.add_trace(go.Histogram(y=df['Close'], x=df['Volume'], histfunc='sum', orientation='h', nbinsy=50, name="籌碼", xaxis='x4', yaxis='y', marker=dict(color='rgba(31,119,180,0.3)'), hoverinfo='none'))
                fig.update_layout(xaxis4=dict(overlaying='x', side='top', showgrid=False, visible=False, range=[df['Volume'].max()*3, 0]))

            # Row 2: 量
            vol_color = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_color, name="量"), row=2, col=1)
            
            # Row 3: MACD
            if show_macd:
                hist_color = ['red' if h < 0 else 'green' for h in df['MACD_Hist']]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=hist_color, name='MACD 柱狀'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange', width=1.5), name='MACD 快線'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='blue', width=1.5), name='Signal 慢線'), row=3, col=1)

            fig.update_layout(height=700, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02))
            fig.update_xaxes(type='date', row=1, col=1)
            fig.update_xaxes(type='date', row=2, col=1)
            fig.update_xaxes(type='date', row=3, col=1)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            with st.expander("📰 相關新聞 (點擊展開)"):
                for item in get_google_news(stock_id)[:6]:
                    st.markdown(f"- [{item.title}]({item.link}) ({item.published})")

            # --- 回測結果 ---
            if run_backtest_btn:
                st.divider()
                st.subheader("💰 策略績效比較")
                
                # 策略 A & B 計算
                res1, log1 = run_backtest(df, s1_short, s1_long, initial_capital, use_macd_filter=False)
                res2, log2 = run_backtest(df, s2_short, s2_long, initial_capital, use_macd_filter=use_macd_b)
                
                buy_hold_series = (initial_capital / df['Close'].iloc[0]) * df['Close']
                
                pct1 = ((res1['Total_Asset'].iloc[-1] - initial_capital) / initial_capital) * 100
                pct2 = ((res2['Total_Asset'].iloc[-1] - initial_capital) / initial_capital) * 100
                pct_bh = ((buy_hold_series.iloc[-1] - initial_capital) / initial_capital) * 100

                mdd1 = calculate_mdd(res1['Total_Asset'])
                mdd2 = calculate_mdd(res2['Total_Asset'])
                mdd_bh = calculate_mdd(buy_hold_series)

                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.info(f"策略 A: 純均線 ({s1_short}/{s1_long})")
                    st.metric("總報酬率", f"{pct1:.2f}%", f"{int(res1['Total_Asset'].iloc[-1] - initial_capital):,}")
                    st.metric("📉 MDD", f"{mdd1:.2f}%")
                
                with col_b:
                    filter_text = " + MACD" if use_macd_b else ""
                    st.info(f"策略 B: 均線{filter_text} ({s2_short}/{s2_long})")
                    st.metric("總報酬率", f"{pct2:.2f}%", f"{int(res2['Total_Asset'].iloc[-1] - initial_capital):,}")
                    st.metric("📉 MDD", f"{mdd2:.2f}%")

                with col_c:
                    st.warning("基準 (買進持有)")
                    st.metric("總報酬率", f"{pct_bh:.2f}%", f"{int(buy_hold_series.iloc[-1] - initial_capital):,}")
                    st.metric("📉 MDD", f"{mdd_bh:.2f}%")

                fig_bt = go.Figure()
                
                # --- 策略 A 繪圖 ---
                fig_bt.add_trace(go.Scatter(x=res1.index, y=res1['Total_Asset'], mode='lines', name=f'策略 A 資產', line=dict(color='gold', width=2)))
                # A 買賣點 (實心三角形)
                buy_A = res1[res1['Position'] == 1]
                sell_A = res1[res1['Position'] == -1]
                fig_bt.add_trace(go.Scatter(x=buy_A.index, y=buy_A['Total_Asset'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='red'), name='A 買進'))
                fig_bt.add_trace(go.Scatter(x=sell_A.index, y=sell_A['Total_Asset'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='green'), name='A 賣出'))

                # --- 策略 B 繪圖 ---
                fig_bt.add_trace(go.Scatter(x=res2.index, y=res2['Total_Asset'], mode='lines', name=f'策略 B 資產', line=dict(color='cyan', width=2, dash='dot')))
                # B 買賣點 (空心三角形，方便重疊時辨識)
                buy_B = res2[res2['Position'] == 1]
                sell_B = res2[res2['Position'] == -1]
                fig_bt.add_trace(go.Scatter(x=buy_B.index, y=buy_B['Total_Asset'], mode='markers', marker=dict(symbol='triangle-up-open', size=12, color='red', line_width=2), name='B 買進'))
                fig_bt.add_trace(go.Scatter(x=sell_B.index, y=sell_B['Total_Asset'], mode='markers', marker=dict(symbol='triangle-down-open', size=12, color='green', line_width=2), name='B 賣出'))
                
                fig_bt.update_layout(height=450, hovermode="x unified", title="💰 資金成長與買賣點位")
                st.plotly_chart(fig_bt, use_container_width=True)

                c_log1, c_log2 = st.columns(2)
                with c_log1:
                    with st.expander(f"📜 策略 A 交易明細 ({len(log1)} 筆)"):
                        if not log1.empty:
                            st.dataframe(log1, use_container_width=True)
                        else: st.write("無交易紀錄")
                with c_log2:
                    with st.expander(f"📜 策略 B 交易明細 ({len(log2)} 筆)"):
                        if not log2.empty:
                            st.dataframe(log2, use_container_width=True)
                        else: st.write("無交易紀錄")

        else:
            st.error(f"❌ 無法讀取數據: {error_msg}")

# ========================================================
#   模式 B: 策略選股器
# ========================================================
elif app_mode == "🔍 策略選股器":
    st.title("🔍 均線策略選股器")
    c1, c2 = st.columns(2)
    s_ma = c1.number_input("短均線", value=5)
    l_ma = c2.number_input("長均線", value=20)
    
    default_tickers = "2330, 2317, 2454, 2308, 2303, 2603, 2609, 2615, 2881, 2882, 0050, 0056, 00878, 3231, 2382, 6669"
    user_tickers = st.text_area("觀察清單", default_tickers)
    
    if st.button("🚀 開始掃描"):
        tickers = [t.strip()+".TW" for t in user_tickers.split(",") if t.strip()]
        results = []
        bar = st.progress(0)
        
        for i, t in enumerate(tickers):
            bar.progress((i+1)/len(tickers))
            try:
                df = yf.download(t, period="3mo", auto_adjust=True, progress=False)
                if not df.empty and len(df) > l_ma:
                    if isinstance(df.columns, pd.MultiIndex): 
                        df.columns = df.columns.get_level_values(0)
                        
                    df['S'] = df['Close'].rolling(s_ma).mean()
                    df['L'] = df['Close'].rolling(l_ma).mean()
                    curr, prev = df.iloc[-1], df.iloc[-2]
                    
                    is_gold = (prev['S'] < prev['L']) and (curr['S'] > curr['L'])
                    is_bull = (curr['Close'] > curr['S']) and (curr['S'] > curr['L'])
                    
                    if is_gold or is_bull:
                        results.append({
                            "代碼": t.replace(".TW",""), 
                            "現價": f"{curr['Close']:.2f}",
                            "訊號": "✨ 黃金交叉" if is_gold else "🔥 多頭排列"
                        })
            except: continue
            
        bar.empty()
        if results:
            st.success(f"找到 {len(results)} 檔")
            st.dataframe(pd.DataFrame(results).style.applymap(lambda v: 'background-color: #d4edda' if '黃金' in v else '#fff3cd', subset=['訊號']), use_container_width=True)
        else: st.warning("無符合條件股票")
