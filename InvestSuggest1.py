# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set up the Streamlit page configuration
st.set_page_config(page_title="Investment Recommendation", page_icon="📈", layout="wide")

# Load preprocessed mutual fund data
@st.cache_data
def load_mutual_fund_data():
    file_path = "comprehensive_mutual_funds_data.csv"
    return pd.read_csv(file_path)

# Load preprocessed gold price data
@st.cache_data
def load_gold_data():
    file_path = "Gold_Prices.csv"
    gold_data = pd.read_csv(file_path)

    # Convert Date column to datetime format and set index
    gold_data['Date'] = pd.to_datetime(gold_data['Date'])
    gold_data.set_index('Date', inplace=True)

    # Compute returns for different timeframes
    gold_data['Return_1Y'] = gold_data['Close'].pct_change(periods=252) * 100
    gold_data['Return_3Y'] = gold_data['Close'].pct_change(periods=252 * 3) * 100
    gold_data['Return_5Y'] = gold_data['Close'].pct_change(periods=252 * 5) * 100

    # Drop NaN values caused by rolling calculations
    gold_data.dropna(inplace=True)

    return gold_data[['Return_1Y', 'Return_3Y', 'Return_5Y']]

# Train a model for mutual fund predictions
@st.cache_data
def train_mutual_fund_model(data, timeframe, risk_level):
    filtered_data = data[data['risk_level'] == risk_level].dropna(subset=[timeframe])

    if filtered_data.empty:
        return None  # No funds match the criteria

    X = filtered_data[['expense_ratio', 'fund_size_cr', 'fund_age_yr']]
    y = filtered_data[timeframe]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    filtered_data['Predicted Returns'] = model.predict(X)
    top_funds = filtered_data.sort_values(by=['Predicted Returns', 'expense_ratio'], ascending=[False, True]).head(3)

    return top_funds[['scheme_name', 'Predicted Returns', 'sharpe', 'expense_ratio', 'fund_size_cr']]

# Load data
mutual_fund_data = load_mutual_fund_data()
gold_data = load_gold_data()

# Streamlit UI
st.title("📊 Investment Recommendation Dashboard")
st.markdown("Compare **Mutual Funds vs. Gold** and get personalized investment suggestions based on past performance and risk level.")

# Sidebar for user inputs
st.sidebar.header("🔧 Configure Your Investment Preferences")
investment_timeframe = st.sidebar.selectbox("📆 Select Investment Timeframe", options=["1 Year", "3 Years", "5 Years"])
risk_level = st.sidebar.slider("⚖️ Select Risk Level", min_value=1, max_value=6, value=3)

# Mapping timeframes to dataset columns
timeframe_map = {
    "1 Year": "returns_1yr",
    "3 Years": "returns_3yr",
    "5 Years": "returns_5yr"
}
selected_timeframe_mutual_funds = timeframe_map[investment_timeframe]
selected_timeframe_gold = f"Return_{investment_timeframe.replace(' ', '')[0]}Y"

# Predict button
if st.sidebar.button("🚀 Predict"):
    with st.spinner("Generating recommendations..."):
        # Get Top 3 Mutual Funds
        top_mutual_funds = train_mutual_fund_model(mutual_fund_data, selected_timeframe_mutual_funds, risk_level)

        # Get Gold Returns
        gold_avg_return = gold_data[selected_timeframe_gold].mean()

        # Display Mutual Fund Recommendations
        if top_mutual_funds is not None:
            st.markdown("### 🏆 Top 3 Mutual Fund Recommendations")
            st.dataframe(top_mutual_funds.style.format({"Predicted Returns": "{:.2f}"}))
        else:
            st.warning("No mutual funds found matching your criteria. Try adjusting your risk level or timeframe.")

        # Performance Comparison Table
        comparison_df = pd.DataFrame({
            "Asset Class": ["Top Mutual Funds (Avg)", "Gold"],
            "Average Return (%)": [top_mutual_funds["Predicted Returns"].mean() if top_mutual_funds is not None else 0, gold_avg_return]
        })

        st.markdown("### 📈 Performance Comparison: Mutual Funds vs. Gold")
        st.dataframe(comparison_df.style.format({"Average Return (%)": "{:.2f}"}))

        # Investment Recommendation Logic
        st.markdown("### 🎯 Personalized Investment Recommendation")

        if gold_avg_return > comparison_df["Average Return (%)"][0]:
            recommendation = "Gold has shown **stronger performance** than Mutual Funds in this timeframe. Consider allocating part of your portfolio to Gold."
        elif comparison_df["Average Return (%)"][0] > gold_avg_return:
            recommendation = "Mutual Funds have outperformed Gold in this timeframe. Investing in **top mutual funds** may be the best option."
        else:
            recommendation = "Both Gold and Mutual Funds have performed similarly. A **diversified approach** (e.g., 50% Gold, 50% Mutual Funds) may be beneficial."

        st.success(recommendation)
