# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import ta

# Streamlit configuration
st.set_page_config(page_title="Investment Recommendation", page_icon="📈", layout="wide")

# Load mutual fund data
@st.cache_data
def load_mutual_fund_data():
    return pd.read_csv("comprehensive_mutual_funds_data.csv")

# Load gold price data
@st.cache_data
def load_gold_data():
    gold_data = pd.read_csv("Gold_Prices.csv", parse_dates=['Date'], index_col='Date')
    gold_data['Return_1Y'] = gold_data['Close'].pct_change(252) * 100
    gold_data['Return_3Y'] = gold_data['Close'].pct_change(756) * 100
    gold_data['Return_5Y'] = gold_data['Close'].pct_change(1260) * 100
    gold_data.dropna(inplace=True)
    return gold_data[['Return_1Y', 'Return_3Y', 'Return_5Y']]

# Load Nifty data
@st.cache_data
def load_nifty_data():
    nifty_data = pd.read_csv("NSEI .csv", parse_dates=['Date'], index_col='Date')
    nifty_data['Log_Close'] = np.log(nifty_data['Close'])

    for window in [50, 100, 200]:
        nifty_data[f'MA_{window}'] = nifty_data['Close'].rolling(window=window).mean()

    nifty_data['RSI'] = ta.momentum.rsi(nifty_data['Close'])
    nifty_data['Momentum'] = nifty_data['Close'] - nifty_data['Close'].shift(10)

    # Log returns for prediction
    nifty_data['Return_1Y'] = nifty_data['Log_Close'].diff(252) * 100
    nifty_data['Return_3Y'] = nifty_data['Log_Close'].diff(756) * 100
    nifty_data['Return_5Y'] = nifty_data['Log_Close'].diff(1260) * 100

    nifty_data.dropna(inplace=True)
    return nifty_data

# Random Forest prediction for Mutual Funds
@st.cache_data
def predict_mf(data, timeframe, risk_level):
    df = data[data['risk_level'] == risk_level].dropna(subset=[timeframe])
    if df.empty:
        return None

    X = df[['expense_ratio', 'fund_size_cr', 'fund_age_yr']]
    y = df[timeframe]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['Predicted Returns'] = model.predict(X)

    top_funds = df.sort_values(by=['Predicted Returns', 'expense_ratio'], ascending=[False, True]).head(3)
    return top_funds[['scheme_name', 'Predicted Returns', 'sharpe', 'expense_ratio', 'fund_size_cr']]

# Random Forest prediction for Nifty
@st.cache_data
def predict_nifty(data, timeframe):
    X = data[['MA_50', 'MA_100', 'MA_200', 'RSI', 'Momentum']]
    y = data[timeframe]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X)
    avg_pred = np.mean(predictions)

    return avg_pred

# Load datasets
mf_data = load_mutual_fund_data()
gold_data = load_gold_data()
nifty_data = load_nifty_data()

# Streamlit UI
st.title("📊 Investment Recommendation Dashboard")
st.markdown("Compare **Mutual Funds vs. Gold vs. Nifty** based on predicted returns.")

# Sidebar inputs
st.sidebar.header("🔧 Configure Your Preferences")
timeframe_choice = st.sidebar.selectbox("📆 Investment Timeframe", ["1 Year", "3 Years", "5 Years"])
risk_level_choice = st.sidebar.slider("⚖️ Risk Level", 1, 6, 3)

# Mapping selections
mf_map = {"1 Year": "returns_1yr", "3 Years": "returns_3yr", "5 Years": "returns_5yr"}
gold_nifty_map = {"1 Year": "Return_1Y", "3 Years": "Return_3Y", "5 Years": "Return_5Y"}

# Predict button
if st.sidebar.button("🚀 Predict"):
    with st.spinner("Calculating..."):
        # Mutual Funds
        top_mfs = predict_mf(mf_data, mf_map[timeframe_choice], risk_level_choice)
        mf_avg_return = top_mfs['Predicted Returns'].mean() if top_mfs is not None else 0

        # Gold
        gold_avg_return = gold_data[gold_nifty_map[timeframe_choice]].mean()

        # Nifty
        nifty_pred_return = predict_nifty(nifty_data, gold_nifty_map[timeframe_choice])

        # Display Mutual Funds
        st.markdown("### 🏆 Top 3 Mutual Funds")
        if top_mfs is not None:
            st.dataframe(top_mfs.style.format({"Predicted Returns": "{:.2f}"}))
        else:
            st.warning("No mutual funds match your criteria.")

        # Performance Comparison
        comp_df = pd.DataFrame({
            "Asset Class": ["Top Mutual Funds (Avg)", "Gold", "Nifty"],
            "Average Predicted Return (%)": [mf_avg_return, gold_avg_return, nifty_pred_return]
        })
        st.markdown("### 📈 Performance Comparison")
        st.dataframe(comp_df.style.format({"Average Predicted Return (%)": "{:.2f}"}))

        # Personalized Recommendation
        best_asset = comp_df.sort_values(by="Average Predicted Return (%)", ascending=False).iloc[0]
        recommendation = f"The asset class with the highest predicted return is **{best_asset['Asset Class']}** with an average predicted return of **{best_asset['Average Predicted Return (%)']:.2f}%**. Consider allocating more resources here."

        st.markdown("### 🎯 Personalized Recommendation")
        st.success(recommendation)
