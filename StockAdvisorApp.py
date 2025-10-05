import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
import random
from typing import List, Tuple, Dict
import time

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== DATABASE FUNCTIONS ====================

def init_database():
    """Initialize SQLite database for portfolio tracking"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Portfolios table
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios
                 (portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  portfolio_name TEXT,
                  initial_investment REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(user_id))''')
    
    # Holdings table
    c.execute('''CREATE TABLE IF NOT EXISTS holdings
                 (holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  portfolio_id INTEGER,
                  ticker TEXT,
                  shares REAL,
                  weight REAL,
                  purchase_price REAL,
                  purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id))''')
    
    conn.commit()
    conn.close()

def create_user(username: str) -> int:
    """Create a new user"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
        user_id = c.lastrowid
    except sqlite3.IntegrityError:
        c.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        user_id = c.fetchone()[0]
    conn.close()
    return user_id

def get_user_portfolios(user_id: int) -> pd.DataFrame:
    """Get all portfolios for a user"""
    conn = sqlite3.connect('portfolio.db')
    query = """SELECT p.portfolio_id, p.portfolio_name, p.initial_investment, 
                      COUNT(h.holding_id) as num_holdings
               FROM portfolios p
               LEFT JOIN holdings h ON p.portfolio_id = h.portfolio_id
               WHERE p.user_id = ?
               GROUP BY p.portfolio_id"""
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df

def create_portfolio(user_id: int, portfolio_name: str, initial_investment: float) -> int:
    """Create a new portfolio"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute("INSERT INTO portfolios (user_id, portfolio_name, initial_investment) VALUES (?, ?, ?)",
              (user_id, portfolio_name, initial_investment))
    conn.commit()
    portfolio_id = c.lastrowid
    conn.close()
    return portfolio_id

def add_holding(portfolio_id: int, ticker: str, shares: float, weight: float, purchase_price: float):
    """Add a stock holding to portfolio"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute("""INSERT INTO holdings (portfolio_id, ticker, shares, weight, purchase_price)
                 VALUES (?, ?, ?, ?, ?)""",
              (portfolio_id, ticker, shares, weight, purchase_price))
    conn.commit()
    conn.close()

def get_portfolio_holdings(portfolio_id: int) -> pd.DataFrame:
    """Get all holdings for a portfolio"""
    conn = sqlite3.connect('portfolio.db')
    df = pd.read_sql_query("""SELECT * FROM holdings WHERE portfolio_id = ?""", 
                           conn, params=(portfolio_id,))
    conn.close()
    return df

# ==================== STOCK DATA FUNCTIONS ====================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Fetch historical stock data"""
    try:
        data = yf.download(tickers, period=period, group_by='ticker', progress=False)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def get_realtime_price(ticker: str) -> Dict:
    """Get real-time stock price and info"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'current_price': info.get('currentPrice', 0),
            'previous_close': info.get('previousClose', 0),
            'change': info.get('currentPrice', 0) - info.get('previousClose', 0),
            'change_percent': ((info.get('currentPrice', 0) - info.get('previousClose', 0)) / 
                              info.get('previousClose', 1)) * 100,
            'volume': info.get('volume', 0),
            'market_cap': info.get('marketCap', 0),
            'name': info.get('longName', ticker)
        }
    except Exception as e:
        st.error(f"Error fetching real-time data for {ticker}: {e}")
        return {}

# ==================== PORTFOLIO ANALYSIS FUNCTIONS ====================

def calculate_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentage daily returns"""
    returns_df = prices_df.pct_change() * 100
    returns_df.fillna(0, inplace=True)
    return returns_df

def generate_portfolio_weights(n: int) -> np.ndarray:
    """Generate random portfolio weights that sum to 1"""
    weights = np.random.random(n)
    weights = weights / np.sum(weights)
    return weights

def calculate_portfolio_metrics(returns_df: pd.DataFrame, weights: np.ndarray, 
                                rf_rate: float = 0.03) -> Dict:
    """Calculate portfolio metrics including Sharpe ratio"""
    # Expected portfolio return (annualized)
    expected_return = np.sum(weights * returns_df.mean()) * 252
    
    # Portfolio volatility (risk)
    covariance = returns_df.cov() * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
    
    # Sharpe ratio
    sharpe_ratio = (expected_return - rf_rate) / volatility if volatility > 0 else 0
    
    return {
        'expected_return': expected_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio
    }

def run_monte_carlo_simulation(returns_df: pd.DataFrame, n_simulations: int = 5000, 
                               rf_rate: float = 0.03) -> pd.DataFrame:
    """Run Monte Carlo simulation for portfolio optimization"""
    n_assets = len(returns_df.columns)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_simulations):
        # Generate random weights
        weights = generate_portfolio_weights(n_assets)
        
        # Calculate metrics
        metrics = calculate_portfolio_metrics(returns_df, weights, rf_rate)
        
        results.append({
            'weights': weights,
            'expected_return': metrics['expected_return'],
            'volatility': metrics['volatility'],
            'sharpe_ratio': metrics['sharpe_ratio']
        })
        
        # Update progress
        if i % 100 == 0:
            progress_bar.progress((i + 1) / n_simulations)
            status_text.text(f"Running simulation {i + 1}/{n_simulations}")
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# ==================== VISUALIZATION FUNCTIONS ====================

def plot_stock_prices(data: pd.DataFrame, tickers: List[str], title: str):
    """Plot stock prices using Plotly"""
    fig = go.Figure()
    
    for ticker in tickers:
        if len(tickers) == 1:
            prices = data['Adj Close']
        else:
            prices = data[ticker]['Adj Close']
        fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name=ticker))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        plot_bgcolor='white'
    )
    
    return fig

def plot_efficient_frontier(sim_results: pd.DataFrame):
    """Plot efficient frontier from Monte Carlo simulation"""
    fig = px.scatter(
        sim_results,
        x='volatility',
        y='expected_return',
        color='sharpe_ratio',
        size='sharpe_ratio',
        hover_data=['sharpe_ratio'],
        title='Efficient Frontier - Portfolio Optimization',
        labels={
            'volatility': 'Volatility (Risk)',
            'expected_return': 'Expected Return',
            'sharpe_ratio': 'Sharpe Ratio'
        }
    )
    
    # Highlight optimal portfolio
    optimal_idx = sim_results['sharpe_ratio'].idxmax()
    optimal_portfolio = sim_results.iloc[optimal_idx]
    
    fig.add_trace(go.Scatter(
        x=[optimal_portfolio['volatility']],
        y=[optimal_portfolio['expected_return']],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(size=20, color='red', symbol='star')
    ))
    
    fig.update_layout(plot_bgcolor='white')
    return fig

def plot_correlation_heatmap(returns_df: pd.DataFrame):
    """Plot correlation heatmap"""
    corr = returns_df.corr()
    fig = px.imshow(
        corr,
        text_auto='.2f',
        aspect='auto',
        title='Stock Returns Correlation Matrix',
        color_continuous_scale='RdBu_r'
    )
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    # Initialize database
    init_database()
    
    st.markdown('<h1 class="main-header">📈 Portfolio Optimizer Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Dashboard", "📊 Stock Screener", "💼 My Portfolios", 
         "🎯 Portfolio Optimizer", "📈 Real-Time Tracker"]
    )
    
    # User selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("User Settings")
    username = st.sidebar.text_input("Username", value="demo_user")
    if st.sidebar.button("Login/Register"):
        user_id = create_user(username)
        st.session_state['user_id'] = user_id
        st.sidebar.success(f"Logged in as {username}")
    
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = create_user(username)
    
    # Route to different pages
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Stock Screener":
        show_stock_screener()
    elif page == "💼 My Portfolios":
        show_portfolios()
    elif page == "🎯 Portfolio Optimizer":
        show_optimizer()
    elif page == "📈 Real-Time Tracker":
        show_realtime_tracker()

def show_dashboard():
    """Display main dashboard"""
    st.header("Welcome to Portfolio Optimizer Pro")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Users", "1,234", "+5%")
    with col2:
        st.metric("Active Portfolios", "567", "+12%")
    with col3:
        st.metric("Stocks Tracked", "1,000+", "+50")
    
    st.markdown("---")
    
    st.subheader("Quick Start Guide")
    st.write("""
    1. **Stock Screener**: Search and analyze individual stocks
    2. **My Portfolios**: Create and manage your portfolios
    3. **Portfolio Optimizer**: Run Monte Carlo simulations to find optimal asset allocation
    4. **Real-Time Tracker**: Monitor live stock prices and portfolio performance
    """)
    
    # Sample market overview
    st.subheader("Market Overview")
    major_indices = ['SPY', 'QQQ', 'DIA', 'IWM']
    
    cols = st.columns(len(major_indices))
    for i, ticker in enumerate(major_indices):
        with cols[i]:
            data = get_realtime_price(ticker)
            if data:
                st.metric(
                    ticker,
                    f"${data.get('current_price', 0):.2f}",
                    f"{data.get('change_percent', 0):.2f}%"
                )

def show_stock_screener():
    """Stock screening and analysis page"""
    st.header("📊 Stock Screener & Analysis")
    
    # Stock input
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        period = st.selectbox("Time Period", 
                             ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                             index=3)
    
    with col2:
        if st.button("Analyze Stock", type="primary"):
            with st.spinner(f"Fetching data for {ticker}..."):
                # Get real-time data
                realtime_data = get_realtime_price(ticker)
                
                if realtime_data:
                    st.subheader(f"{realtime_data.get('name', ticker)}")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${realtime_data.get('current_price', 0):.2f}",
                                 f"{realtime_data.get('change_percent', 0):.2f}%")
                    with col2:
                        st.metric("Previous Close", f"${realtime_data.get('previous_close', 0):.2f}")
                    with col3:
                        st.metric("Volume", f"{realtime_data.get('volume', 0):,}")
                    with col4:
                        market_cap = realtime_data.get('market_cap', 0)
                        st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                    
                    # Get historical data
                    data = get_stock_data([ticker], period=period)
                    
                    if not data.empty:
                        # Price chart
                        st.subheader("Price History")
                        fig = plot_stock_prices(data, [ticker], f"{ticker} Stock Price")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate returns
                        if len(data) > 1:
                            returns = data['Adj Close'].pct_change() * 100
                            returns = returns.dropna()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Daily Returns Distribution")
                                fig = px.histogram(returns, nbins=50, 
                                                 title="Daily Returns Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.subheader("Statistics")
                                stats = {
                                    "Mean Daily Return": f"{returns.mean():.2f}%",
                                    "Std Deviation": f"{returns.std():.2f}%",
                                    "Min Return": f"{returns.min():.2f}%",
                                    "Max Return": f"{returns.max():.2f}%",
                                    "Annualized Return": f"{returns.mean() * 252:.2f}%",
                                    "Annualized Volatility": f"{returns.std() * np.sqrt(252):.2f}%"
                                }
                                for key, value in stats.items():
                                    st.metric(key, value)

def show_portfolios():
    """Portfolio management page"""
    st.header("💼 My Portfolios")
    
    user_id = st.session_state.get('user_id')
    
    # Create new portfolio
    with st.expander("➕ Create New Portfolio"):
        with st.form("new_portfolio_form"):
            portfolio_name = st.text_input("Portfolio Name")
            initial_investment = st.number_input("Initial Investment ($)", 
                                                min_value=1000.0, value=100000.0, step=1000.0)
            
            if st.form_submit_button("Create Portfolio"):
                if portfolio_name:
                    portfolio_id = create_portfolio(user_id, portfolio_name, initial_investment)
                    st.success(f"Portfolio '{portfolio_name}' created successfully!")
                    st.rerun()
    
    # Display existing portfolios
    portfolios = get_user_portfolios(user_id)
    
    if not portfolios.empty:
        st.subheader("Your Portfolios")
        
        for idx, portfolio in portfolios.iterrows():
            with st.expander(f"📁 {portfolio['portfolio_name']} - ${portfolio['initial_investment']:,.2f}"):
                portfolio_id = portfolio['portfolio_id']
                
                # Add new holding
                with st.form(f"add_holding_{portfolio_id}"):
                    st.subheader("Add New Holding")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        ticker = st.text_input("Ticker").upper()
                    with col2:
                        shares = st.number_input("Shares", min_value=0.0, value=1.0)
                    with col3:
                        weight = st.number_input("Weight", min_value=0.0, max_value=1.0, value=0.1)
                    
                    if st.form_submit_button("Add Holding"):
                        if ticker:
                            price_data = get_realtime_price(ticker)
                            if price_data:
                                add_holding(portfolio_id, ticker, shares, weight, 
                                          price_data['current_price'])
                                st.success(f"Added {ticker} to portfolio")
                                st.rerun()
                
                # Display holdings
                holdings = get_portfolio_holdings(portfolio_id)
                
                if not holdings.empty:
                    st.subheader("Current Holdings")
                    
                    # Get current prices
                    holdings_display = holdings.copy()
                    current_values = []
                    
                    for _, holding in holdings.iterrows():
                        price_data = get_realtime_price(holding['ticker'])
                        current_price = price_data.get('current_price', 0)
                        current_value = holding['shares'] * current_price
                        current_values.append(current_value)
                    
                    holdings_display['current_price'] = [get_realtime_price(t).get('current_price', 0) 
                                                        for t in holdings['ticker']]
                    holdings_display['current_value'] = current_values
                    holdings_display['gain_loss'] = holdings_display['current_value'] - (
                        holdings_display['shares'] * holdings_display['purchase_price'])
                    holdings_display['gain_loss_pct'] = (
                        holdings_display['gain_loss'] / (holdings_display['shares'] * 
                        holdings_display['purchase_price']) * 100)
                    
                    st.dataframe(holdings_display[['ticker', 'shares', 'purchase_price', 
                                                   'current_price', 'current_value', 
                                                   'gain_loss', 'gain_loss_pct']], 
                               use_container_width=True)
                    
                    # Portfolio summary
                    total_value = sum(current_values)
                    total_invested = sum(holdings['shares'] * holdings['purchase_price'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Value", f"${total_value:,.2f}")
                    with col2:
                        st.metric("Total Invested", f"${total_invested:,.2f}")
                    with col3:
                        gain_loss = total_value - total_invested
                        gain_loss_pct = (gain_loss / total_invested * 100) if total_invested > 0 else 0
                        st.metric("Gain/Loss", f"${gain_loss:,.2f}", f"{gain_loss_pct:.2f}%")
    else:
        st.info("No portfolios yet. Create one to get started!")

def show_optimizer():
    """Portfolio optimization page"""
    st.header("🎯 Portfolio Optimizer")
    
    st.write("""
    Use Monte Carlo simulation to find the optimal portfolio allocation that maximizes 
    the Sharpe ratio (risk-adjusted return).
    """)
    
    # Stock selection
    st.subheader("1. Select Stocks")
    default_tickers = "AAPL,MSFT,GOOGL,AMZN,TSLA,JPM,V,JNJ,WMT,PG"
    tickers_input = st.text_area("Enter tickers (comma-separated)", default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox("Historical Period", ["1y", "2y", "3y", "5y"], index=1)
    with col2:
        n_simulations = st.number_input("Number of Simulations", 
                                       min_value=100, max_value=50000, value=5000, step=100)
    with col3:
        rf_rate = st.number_input("Risk-Free Rate", min_value=0.0, max_value=0.1, 
                                 value=0.03, step=0.01, format="%.3f")
    
    if st.button("Run Optimization", type="primary"):
        with st.spinner("Fetching stock data..."):
            # Get historical data
            data = get_stock_data(tickers, period=period)
            
            if not data.empty:
                # Extract adjusted close prices
                if len(tickers) == 1:
                    prices_df = pd.DataFrame(data['Adj Close'])
                    prices_df.columns = tickers
                else:
                    prices_df = pd.DataFrame({ticker: data[ticker]['Adj Close'] 
                                            for ticker in tickers})
                
                prices_df = prices_df.dropna()
                
                # Calculate returns
                returns_df = calculate_daily_returns(prices_df)
                returns_df = returns_df.iloc[1:]  # Remove first row of zeros
                
                # Display correlation
                st.subheader("Stock Correlation Analysis")
                fig = plot_correlation_heatmap(returns_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Run Monte Carlo simulation
                st.subheader("Running Monte Carlo Simulation")
                sim_results = run_monte_carlo_simulation(returns_df, n_simulations, rf_rate)
                
                # Find optimal portfolio
                optimal_idx = sim_results['sharpe_ratio'].idxmax()
                optimal_portfolio = sim_results.iloc[optimal_idx]
                
                # Display results
                st.success("✅ Optimization Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Return", 
                             f"{optimal_portfolio['expected_return']*100:.2f}%")
                with col2:
                    st.metric("Volatility (Risk)", 
                             f"{optimal_portfolio['volatility']*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", 
                             f"{optimal_portfolio['sharpe_ratio']:.3f}")
                
                # Optimal weights
                st.subheader("Optimal Portfolio Allocation")
                weights_df = pd.DataFrame({
                    'Ticker': tickers,
                    'Weight (%)': optimal_portfolio['weights'] * 100
                }).sort_values('Weight (%)', ascending=False)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.dataframe(weights_df, use_container_width=True)
                
                with col2:
                    fig = px.pie(weights_df, values='Weight (%)', names='Ticker',
                               title='Portfolio Allocation')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Efficient frontier
                st.subheader("Efficient Frontier")
                fig = plot_efficient_frontier(sim_results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Save portfolio option
                st.subheader("Save Optimized Portfolio")
                with st.form("save_portfolio"):
                    portfolio_name = st.text_input("Portfolio Name", 
                                                  value=f"Optimized_{datetime.now().strftime('%Y%m%d')}")
                    initial_investment = st.number_input("Investment Amount ($)", 
                                                        min_value=1000.0, value=100000.0)
                    
                    if st.form_submit_button("Save Portfolio"):
                        user_id = st.session_state.get('user_id')
                        portfolio_id = create_portfolio(user_id, portfolio_name, initial_investment)
                        
                        # Add holdings based on optimal weights
                        for ticker, weight in zip(tickers, optimal_portfolio['weights']):
                            price_data = get_realtime_price(ticker)
                            allocation = initial_investment * weight
                            shares = allocation / price_data['current_price']
                            add_holding(portfolio_id, ticker, shares, weight, 
                                      price_data['current_price'])
                        
                        st.success(f"Portfolio '{portfolio_name}' saved successfully!")

def show_realtime_tracker():
    """Real-time stock price tracker"""
    st.header("📈 Real-Time Stock Tracker")
    
    # Stock input
    watchlist_input = st.text_input("Enter tickers to track (comma-separated)", 
                                    "AAPL,MSFT,GOOGL,AMZN,TSLA")
    tickers = [t.strip().upper() for t in watchlist_input.split(',') if t.strip()]
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (every 30 seconds)", value=False)
    refresh_button = st.button("🔄 Refresh Now")
    
    # Placeholder for data
    data_placeholder = st.empty()
    
    # Refresh logic
    if auto_refresh or refresh_button:
        with data_placeholder.container():
            st.subheader(f"Live Prices - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Create columns for stock cards
            n_cols = 3
            cols = st.columns(n_cols)
            
            for idx, ticker in enumerate(tickers):
                with cols[idx % n_cols]:
                    data = get_realtime_price(ticker)
                    
                    if data:
                        st.markdown(f"### {ticker}")
                        st.markdown(f"**{data.get('name', ticker)}**")
                        
                        price_color = "green" if data.get('change', 0) >= 0 else "red"
                        st.markdown(f"<h2 style='color: {price_color};'>${data.get('current_price', 0):.2f}</h2>", 
                                  unsafe_allow_html=True)
                        
                        change = data.get('change', 0)
                        change_pct = data.get('change_percent', 0)
                        st.markdown(f"**Change:** {change:+.2f} ({change_pct:+.2f}%)")
                        st.markdown(f"**Volume:** {data.get('volume', 0):,}")
                        st.markdown("---")
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()

if __name__ == "__main__":
    main()