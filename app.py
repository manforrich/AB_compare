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
#  共用函數區 (核心修復：強制格式化)
# ========================================================
def get_stock_data(ticker, mode="預設區間", period="1y", start=None, end=None):
    try:
        # 使用 yf.download 抓取
        if mode == "預設區間":
            hist = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        else:
            hist = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        if hist.empty: 
            return None, "Yahoo Finance 回傳空資料，請檢查代碼。"

        # --- 關鍵修復：處理 MultiIndex ---
        # 如果欄位是多層的 (例如: ('Close', '2330.TW'))，我們只保留第一層 ('Close')
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
            
        # 確保所有必要的欄位都在
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
    
    # 確保是數值型態 (防呆)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    
    data['Signal'] = 0
    # 避免前面的 NaN 造成問題
    data.iloc[long_window:, data.columns.get_loc('Signal')] = 0 
    
    mask = data['Short_MA'] > data['Long_MA']
    data.loc[mask, 'Signal'] = 1
    
    data['Position'] = data['Signal'].diff()
    
    cash = initial_capital
    holdings = 0
    asset_history = []
    
    for i in range(len(data)):
        price = data['Close'].iloc[i]
        # 如果價格是 NaN (例如停牌)，則跳過或延用上一次資產
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

            # 2. 繪圖
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
            
            colors = ['orange', 'blue', 'purple', 'black']
            for i, d in enumerate(ma_days):
                ma = df['Close'].rolling(d).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma, mode='lines', name=f"MA{d}", line=dict(width=1.5, color=colors[i%4])), row=1, col=1)

            if show_bb:
                mid = df['Close'].rolling(20).mean()
                std = df['Close'].rolling(20).std()
                fig.add_trace(go.Scatter(x=df.index, y=mid+2*std, line=dict(color='rgba(0,100,255,0.3)'), showlegend=False), row=1, col=1)
                fig.add_trace
