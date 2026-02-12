import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Product Demand Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL & DATA ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("demand_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("ProductDemand.csv")

try:
    model = load_model()
    df = load_data()
except:
    st.error("Make sure demand_model.pkl and ProductDemand.csv are in the same folder.")
    st.stop()

# ---------------- SIDEBAR INPUT SECTION ---------------- #
st.sidebar.header("🧾 Enter Product Details")

store_id = st.sidebar.number_input(
    "Store ID",
    value=int(df["Store ID"].iloc[0])
)

total_price = st.sidebar.number_input(
    "Total Price",
    value=float(df["Total Price"].mean()),
    format="%.2f"
)

base_price = st.sidebar.number_input(
    "Base Price",
    value=float(df["Base Price"].mean()),
    format="%.2f"
)

predict_btn = st.sidebar.button("🚀 Predict Demand")

# ---------------- MAIN TITLE ---------------- #
st.title("📊 Product Demand Analytics Dashboard")
st.markdown("### Machine Learning Based Demand Forecasting")
st.markdown("---")

# ---------------- DATA VISUALIZATION ---------------- #
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Units Sold Distribution")

    fig1 = px.histogram(
        df,
        x="Units Sold",
        nbins=30,
        template="plotly_dark"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("💰 Price vs Units Sold")

    fig2 = px.scatter(
        df,
        x="Total Price",
        y="Units Sold",
        opacity=0.6,
        template="plotly_dark"
       
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------------- PREDICTION SECTION ---------------- #
if predict_btn:
    with st.spinner("Analyzing market trends and predicting demand..."):
        time.sleep(1.5)

        features = np.array([[store_id, total_price, base_price]])
        prediction = model.predict(features)
        predicted_units = int(prediction[0])

    st.markdown("""
                <div style='
                background-color:#111827;
                padding:25px;
                border-radius:15px;
                border:1px solid #1f2937;
                margin-bottom:20px;
                '>
                <h2 style='color:#F9FAFB;'>🎯 Prediction Result</h2>
                </div>
                """, unsafe_allow_html=True)


    colA, colB, colC = st.columns(3)

    colA.metric("Predicted Units Sold", predicted_units)
    colB.metric("Price Difference", round(base_price - total_price, 2))

    demand_level = (
        "High" if predicted_units > df["Units Sold"].quantile(0.66)
        else "Medium" if predicted_units > df["Units Sold"].quantile(0.33)
        else "Low"
    )

    colC.metric("Demand Level", demand_level)

    st.success("Prediction completed successfully!")

st.markdown("---")

# ---------------- DATASET EXPLORER ---------------- #
st.markdown("## 📊 Dataset Explorer")

col1, col2, col3 = st.columns(3)

# Store Filter
store_list = sorted(df["Store ID"].unique())
selected_store = col1.selectbox("Filter by Store ID", ["All"] + list(store_list))

# Units Sold Range Filter
min_units = int(df["Units Sold"].min())
max_units = int(df["Units Sold"].max())

selected_range = col2.slider(
    "Units Sold Range",
    min_value=min_units,
    max_value=max_units,
    value=(min_units, max_units)
)

# Price Filter
max_price = float(df["Total Price"].max())
selected_price = col3.slider(
    "Maximum Total Price",
    min_value=0.0,
    max_value=max_price,
    value=max_price
)

# Apply Filters
filtered_df = df.copy()

if selected_store != "All":
    filtered_df = filtered_df[filtered_df["Store ID"] == selected_store]

filtered_df = filtered_df[
    (filtered_df["Units Sold"] >= selected_range[0]) &
    (filtered_df["Units Sold"] <= selected_range[1]) &
    (filtered_df["Total Price"] <= selected_price)
]

st.write(f"Showing {len(filtered_df)} records")

st.dataframe(filtered_df, use_container_width=True, height=400)

st.download_button(
    label="📥 Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_product_demand.csv",
    mime="text/csv"
)

# ---------------- FULL DATA VIEW ---------------- #
with st.expander("📂 View Full Dataset"):
    st.write(f"Total Records: {len(df)}")
    st.dataframe(df, use_container_width=True, height=400)
# ---------------- CUSTOM DARK SaaS STYLE ---------------- #
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}

h1, h2, h3, h4 {
    color: #F9FAFB;
    font-weight: 600;
}

.stMetric {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1f2937;
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}

div[data-testid="stDataFrame"] {
    background-color: #111827;
    border-radius: 10px;
    padding: 10px;
}

.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    font-weight: 500;
}

.stButton>button:hover {
    background-color: #1D4ED8;
}

.stDownloadButton>button {
    background-color: #059669;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}

.stDownloadButton>button:hover {
    background-color: #047857;
}

hr {
    border: 1px solid #1f2937;
}

</style>
""", unsafe_allow_html=True)
