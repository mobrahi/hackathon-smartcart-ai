import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# ==========================================================
# STEP 1: LOAD DATA
# ==========================================================
print("Loading data...")
df = pd.read_csv('data/processed/clean_data.csv', parse_dates=['InvoiceDate'])

# ==========================================================
# STEP 2: CLEAN & PREPARE (CRITICAL)
# ==========================================================
print("Preparing data...")

# 1. Clean StockCode: Ensure they are strings and remove whitespace
# This prevents "85123A" and "85123A " from being counted as different products
df['StockCode'] = df['StockCode'].astype(str).str.strip()

# 2. Filter out non-product items (like POSTAGE, BANK CHARGES, DOTCOM POSTAGE)
# These aren't real products we want to recommend
bad_codes = ['POST', 'DOT', 'C2', 'M', 'BANK CHARGES', 'PADS', 'CRUK']
df = df[~df['StockCode'].isin(bad_codes)]

# ==========================================================
# STEP 3: FILTER USERS (The "Hackathon Speed" Hack)
# ==========================================================
# Calculating similarity for 4,000+ users can be slow and memory-heavy.
# Let's filter to the TOP 1,000 users by Total Spend to keep the demo fast.
print("Filtering to top 1000 active users for performance...")

top_users = df.groupby('CustomerID')['Total_Spend'].sum().sort_values(ascending=False).head(1000).index
df_filtered = df[df['CustomerID'].isin(top_users)]

print(f"Data reduced to {df_filtered.shape[0]} rows for {len(top_users)} users.")

# ==========================================================
# STEP 4: CREATE THE USER-ITEM MATRIX
# ==========================================================
print("Building User-Item Matrix...")

# Rows = CustomerID, Columns = StockCode, Values = Quantity
user_item_matrix = df_filtered.pivot_table(
    index='CustomerID', 
    columns='StockCode', 
    values='Quantity', 
    aggfunc='sum', 
    fill_value=0
)

# ==========================================================
# STEP 5: CALCULATE SIMILARITY
# ==========================================================
print("Calculating User Similarity (this might take 30-60 seconds)...")

# This creates a matrix where Row A, Col B = "How similar is User A to User B?"
user_similarity = cosine_similarity(user_item_matrix)

# Convert to DataFrame so we can use UserIDs as index/columns later
user_sim_df = pd.DataFrame(
    user_similarity, 
    index=user_item_matrix.index, 
    columns=user_item_matrix.index
)

# ==========================================================
# STEP 6: CREATE A PRODUCT LOOKUP DICTIONARY
# ==========================================================
# Will need this to convert '85123A' back to 'White T-Light Holder'
product_descriptions = df_filtered.drop_duplicates('StockCode').set_index('StockCode')['Description'].to_dict()

# ==========================================================
# STEP 7: SAVE EVERYTHING
# ==========================================================
print("Saving models...")

os.makedirs('../models', exist_ok=True)

# 1. The Similarity Matrix (The "Brain")
joblib.dump(user_sim_df, '../models/recommendation_engine.pkl')

# 2. The User-Item Matrix (The "Map" - who bought what)
joblib.dump(user_item_matrix, '../models/user_item_matrix.pkl')

# 3. The Product Names (The "Translator")
joblib.dump(product_descriptions, '../models/product_lookup.pkl')

print("Recommendation Engine built successfully!")
print("Files saved:")
print(" - recommendation_engine.pkl")
print(" - user_item_matrix.pkl")
print(" - product_lookup.pkl")