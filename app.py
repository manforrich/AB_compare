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
#  共用函數區
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

def run_backtest(df, short_window, long_window, initial_capital):
    data = df.copy()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    data['Signal'] = 0
    data.iloc[long_window:, data.columns.get_loc('Signal')] = 0 
    
    mask = data['Short_MA'] > data['Long_MA']
    data.loc[mask, 'Signal'] = 1
    data['Position'] = data['Signal'].diff()
    
    cash = initial_capital
    holdings = 0
    asset_history = []
    
    for i in range(len(data)):
        price = data['Close'].iloc[i]
        if pd.isna(price):
            asset_history.append(asset_history[-1] if asset_history else initial_capital)
            continue

        position_change = data['Position'].iloc[i]
        
        if position_change == 1 and cash > 0:
            holdings = cash / price
            cash = 0
        elif position_change == -1 and holdings > 0:
            cash = holdings * price
            holdings = 0
        current_asset = cash + (holdings * price)
        asset_history.append(current_asset)
        
    data['Total_Asset'] = asset_history
    return data

# ========================================================
#  模式 A: 單一個股分析
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
    show_gaps = st.sidebar.checkbox("跳空缺口", True)

    st.sidebar.divider()
    st.sidebar.subheader("💰 雙策略回測比較")
    initial_capital = st.sidebar.number_input("初始本金", value=100000)
    
    st.sidebar.markdown("**策略 A (預設)**")
    s1_short = st.sidebar.number_input("A 短均線", value=5, key="s1_s")
    s1_long = st.sidebar.number_input("A 長均線", value=20, key="s1_l")
    
    st.sidebar.markdown("**策略 B (對照組)**")
    s2_short = st.sidebar.number_input("B 短均線", value=5, key="s2_s")
    s2_long = st.sidebar.number_input("B 長均線", value=30, key="s2_l")
    
    run_backtest_btn = st.sidebar.button("🚀 執行雙策略回測")

    if stock_id:
        df, error_msg = get_stock_data(stock_id, time_mode, period=selected_period, start=start_date, end=end_date)
        
        if df is not None and not df.empty:
            # 1. 股價資訊
            c1, c2, c3, c4 = st.columns(4)
            close = df['Close'].iloc[-1]
            change = close - df['Close'].iloc[-2]
            pct = (change / df['Close'].iloc[-2]) * 100
            c1.metric("股價", f"{close:.2f}", f"{change:.2f} ({pct:.2f}%)")
            c2.metric("最高", f"{df['High'].max():.2f}")
            c3.metric("最低", f"{df['Low'].min():.2f}")
            c4.metric("成交量", f"{int(df['Volume'].iloc[-1]):,}")

            # 2. 繪圖 (主畫面 K 線)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
            
            colors = ['orange', 'blue', 'purple', 'black']
            sorted_ma_days = sorted(ma_days)
            for i, d in enumerate(sorted_ma_days):
                ma = df['Close'].rolling(d).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma, mode='lines', name=f"MA{d}", line=dict(width=1.5, color=colors[i%4])), row=1, col=1)

            # 主圖表的買賣訊號
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
                    fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#ff2b2b'), name='主圖買訊'), row=1, col=1)
                if not sell_points.empty:
                    fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points['High'] * 1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#00cc00'), name='主圖賣訊'), row=1, col=1)

            if show_bb:
                mid = df['Close'].rolling(20).mean()
                std = df['Close'].rolling(20).std()
                fig.add_trace(go.Scatter(x=df.index, y=mid+2*std, line=dict(color='rgba(0,100,255,0.3)'), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=mid-2*std, line=dict(color='rgba(0,100,255,0.3)'), fill='tonexty', fillcolor='rgba(0,100,255,0.1)', name='布林'), row=1, col=1)

            if show_vp:
                fig.add_trace(go.Histogram(y=df['Close'], x=df['Volume'], histfunc='sum', orientation='h', nbinsy=50, name="籌碼", xaxis='x3', yaxis='y', marker=dict(color='rgba(31,119,180,0.3)'), hoverinfo='none'))
                fig.update_layout(xaxis3=dict(overlaying='x', side='top', showgrid=False, visible=False, range=[df['Volume'].max()*3, 0]))

            vol_color = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_color, name="量"), row=2, col=1)
            
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02))
            fig.update_xaxes(type='date', row=1, col=1)
            fig.update_xaxes(type='date', row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            with st.expander("📰 相關新聞 (點擊展開)"):
                for item in get_google_news(stock_id)[:6]:
                    st.markdown(f"- [{item.title}]({item.link}) ({item.published})")

            # --- 回測結果 (新增訊號點) ---
            if run_backtest_btn:
                st.divider()
                st.subheader("💰 策略績效大對決")
                
                res1 = run_backtest(df, s1_short, s1_long, initial_capital)
                res2 = run_backtest(df, s2_short, s2_long, initial_capital)
                
                final1 = res1['Total_Asset'].iloc[-1]
                pct1 = ((final1 - initial_capital) / initial_capital) * 100
                final2 = res2['Total_Asset'].iloc[-1]
                pct2 = ((final2 - initial_capital) / initial_capital) * 100
                
                buy_hold_shares = initial_capital / df['Close'].iloc[0]
                final_bh = buy_hold_shares * df['Close'].iloc[-1]
                pct_bh = ((final_bh - initial_capital) / initial_capital) * 100

                col_a, col_b, col_c = st.columns(3)
                col_a.metric(f"策略 A ({s1_short}/{s1_long})", f"{pct1:.2f}%", f"{int(final1):,}")
                col_b.metric(f"策略 B ({s2_short}/{s2_long})", f"{pct2:.2f}%", f"{int(final2):,}")
                col_c.metric("基準 (買進持有)", f"{pct_bh:.2f}%", f"{int(final_bh):,}")

                # --- 畫回測圖表 ---
                fig_bt = go.Figure()
                
                # 1. 策略 A (金線 + 實心三角)
                fig_bt.add_trace(go.Scatter(x=res1.index, y=res1['Total_Asset'], mode='lines', name=f'策略 A 資產', line=dict(color='gold', width=2)))
                # 抓出買賣點
                buy_A = res1[res1['Position'] == 1]
                sell_A = res1[res1['Position'] == -1]
                fig_bt.add_trace(go.Scatter(x=buy_A.index, y=buy_A['Total_Asset'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'), name='A 買進'))
                fig_bt.add_trace(go.Scatter(x=sell_A.index, y=sell_A['Total_Asset'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='green'), name='A 賣出'))

                # 2. 策略 B (青線 + 空心圓)
                fig_bt.add_trace(go.Scatter(x=res2.index, y=res2['Total_Asset'], mode='lines', name=f'策略 B 資產', line=dict(color='cyan', width=2, dash='dot')))
                # 抓出買賣點
                buy_B = res2[res2['Position'] == 1]
                sell_B = res2[res2['Position'] == -1]
                # 使用 circle-open (空心圓) 來區隔
                fig_bt.add_trace(go.Scatter(x=buy_B.index, y=buy_B['Total_Asset'], mode='markers', marker=dict(symbol='circle-open', size=8, color='red', line_width=2), name='B 買進'))
                fig_bt.add_trace(go.Scatter(x=sell_B.index, y=sell_B['Total_Asset'], mode='markers', marker=dict(symbol='circle-open', size=8, color='green', line_width=2), name='B 賣出'))

                fig_bt.update_layout(height=400, hovermode="x unified", title="資金成長比較 (含買賣點)")
                st.plotly_chart(fig_bt, use_container_width=True)

        else:
            st.error(f"❌ 無法讀取數據: {error_msg}")

# ========================================================
#  模式 B: 策略選股器
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
