import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import datetime

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="SmartCart AI",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SmartCart AI: E-Commerce Intelligence System")
st.markdown("---")

# ==========================================================
# STEP 1: LOAD DATA & MODELS (CACHED)
# ==========================================================
# We use @st.cache_data so we don't reload these massive files every time you click a button
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/clean_data.csv', parse_dates=['InvoiceDate'])
    return df

@st.cache_resource
def load_models():
    # Load the "Brain"
    risk_model = joblib.load('models/abandonment_model.pkl')
    
    # Load the "Recommendation Engine"
    rec_matrix = joblib.load('models/recommendation_engine.pkl') # User Similarity
    item_matrix = joblib.load('models/user_item_matrix.pkl')     # User-Item
    prod_lookup = joblib.load('models/product_lookup.pkl')       # Product Names
    return risk_model, rec_matrix, item_matrix, prod_lookup

# Load everything
with st.spinner("Loading AI Models and Data..."):
    df = load_data()
    rf_model, sim_df, user_item_matrix, product_lookup = load_models()

# ==========================================================
# SIDEBAR: USER SELECTION
# ==========================================================
st.sidebar.header("User Selection")

# We can only select users that exist in our Recommendation Matrix (Top 1000 users)
available_users = user_item_matrix.index.tolist()
selected_user_id = st.sidebar.selectbox("Select a Customer ID to Analyze", available_users)

# Filter data for the selected user
user_data = df[df['CustomerID'] == selected_user_id]

# ==========================================================
# STEP 2: CALCULATE FEATURES FOR RISK MODEL
# ==========================================================
# We need to calculate Recency/Monetary on the fly to feed the model
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Aggregate
user_metrics = user_data.agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Quantity': 'sum',
    'Total_Spend': 'sum'
}).to_dict()

# Create feature list matching training order: ['Recency', 'Monetary', 'Avg_Basket_Size']
recency = user_metrics['InvoiceDate']
monetary = user_metrics['Total_Spend']
freq = user_metrics['InvoiceNo']
avg_basket = user_metrics['Quantity'] / freq

features_for_prediction = [[recency, monetary, avg_basket]]

# ==========================================================
# STEP 3: PREDICT ABANDONMENT RISK
# ==========================================================
prediction = rf_model.predict(features_for_prediction)[0]
probability = rf_model.predict_proba(features_for_prediction)[0][1] # Probability of Class 1 (Risk)

# ==========================================================
# MAIN DASHBOARD LAYOUT
# ==========================================================

# --- Row 1: KPIs ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Customer ID", selected_user_id)

with col2:
    # Color coding the risk
    risk_label = "🔴 HIGH RISK" if prediction == 1 else "🟢 LOYAL"
    delta_color = "inverse" # Optional styling
    st.metric("Churn Prediction", risk_label, f"{probability*100:.1f}% Confidence")

with col3:
    st.metric("Total Lifetime Value", f"${monetary:,.2f}")

st.markdown("---")

# --- Row 2: Timeline & Recommendations ---
col_left, col_right = st.columns([2, 1])

# --- LEFT COLUMN: BEHAVIOR TIMELINE ---
with col_left:
    st.subheader("🕒 User Behavior Timeline")
    
    # Prepare data for Plotly
    timeline_data = user_data.groupby(['InvoiceNo', 'InvoiceDate']).agg(
        Items=('Quantity', 'sum'),
        Spend=('Total_Spend', 'sum')
    ).reset_index()
    
    fig = px.scatter(
        timeline_data,
        x='InvoiceDate',
        y='Items',
        size='Spend',
        color='Spend',
        title="Shopping Sessions (Bubble Size = Money Spent)",
        color_continuous_scale='Blues'
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# --- RIGHT COLUMN: RECOMMENDATIONS ---
with col_right:
    st.subheader("🎁 Recommended Products")
    
    # --- RECOMMENDATION LOGIC (Reused from test script) ---
    try:
        # 1. Get similar users
        user_scores = sim_df[selected_user_id]
        similar_users = user_scores.sort_values(ascending=False).drop(selected_user_id)
        neighbors = similar_users.head(3).index.tolist()
        
        # 2. Get items
        neighbors_purchases = user_item_matrix.loc[neighbors]
        suggested_items = neighbors_purchases.sum(axis=0)
        
        # 3. Filter out owned items
        user_items = user_item_matrix.loc[selected_user_id]
        suggested_items = suggested_items[user_items == 0]
        
        # 4. Sort
        top_recs = suggested_items.sort_values(ascending=False).head(5)
        
        # Display
        for code, score in top_recs.items():
            name = product_lookup.get(str(code), "Unknown Product")
            st.success(f"**{name}**\n*Popularity Score: {int(score)}")
            
    except Exception as e:
        st.error(f"Could not generate recommendations: {e}")
