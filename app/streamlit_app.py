# app.py
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('data/raw/online_retail.csv')
    # Clean data here (dropnas, etc.)
    return df

df = load_data()

# 2. Sidebar for User Interaction
st.title("🛒 SmartCart AI Dashboard")
selected_user_id = st.sidebar.selectbox("Select a Customer to Analyze", df['CustomerID'].unique())

# 3. Analyze Behavior
user_data = df[df['CustomerID'] == selected_user_id]
st.subheader("User Behavior Profile")
total_spend = user_data['Quantity'].sum() # Simplified logic
st.metric("Total Items Interacted With", len(user_data))

# 4. Predict Abandonment (Mockup logic for example)
# In real version, load your trained model here
risk_score = 0.75 
if risk_score > 0.7:
    st.error(f"⚠️ High Abandonment Risk: {risk_score*100}%")
else:
    st.success(f"✅ Low Risk: {risk_score*100}%")

# 5. Recommend Products (Simple logic: Popular items)
# In real version, use cosine_similarity on the user-item matrix
st.subheader("Recommended Products")
recommendations = ['Product A', 'Product B', 'Product C'] 
for item in recommendations:
    st.write(f"- 🛍️ {item}")