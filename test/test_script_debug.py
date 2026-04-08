import pandas as pd
import joblib
import numpy as np

# 1. Load Models
print("Loading models...")
try:
    sim_df = joblib.load('/Users/mf/code_stuff/hackathon-smartcart-ai/hackathon-smartcart-ai/models/recommendation_engine.pkl')
    matrix = joblib.load('/Users/mf/code_stuff/hackathon-smartcart-ai/hackathon-smartcart-ai/models/user_item_matrix.pkl')
    lookup = joblib.load('/Users/mf/code_stuff/hackathon-smartcart-ai/hackathon-smartcart-ai/models/product_lookup.pkl')
    print("Models loaded!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING FILES: {e}")
    exit()

# 2. DEBUGGING: Check Data Types
print("\n--- DEBUGGING INFO ---")
print(f"Type of sim_df: {type(sim_df)}")
print(f"Type of matrix: {type(matrix)}")

# Check if they are actually DataFrames
if isinstance(sim_df, pd.DataFrame):
    print(f"sim_df shape: {sim_df.shape}")
    print(f"sim_df columns sample: {sim_df.columns[:5].tolist()}") 
else:
    print("ERROR: sim_df is NOT a DataFrame! It is likely a Numpy Array.")
    print("This happens if we forgot to wrap the similarity result in pd.DataFrame()")

if isinstance(matrix, pd.DataFrame):
    print(f"matrix shape: {matrix.shape}")
else:
    print("ERROR: matrix is NOT a DataFrame!")

# 3. The Logic (Safe Mode)
try:
    # Pick a user
    test_user_id = sim_df.index[0]
    print(f"\nTest User ID: {test_user_id}")

    # Get scores
    # FIX: If sim_df is a numpy array (not DataFrame), we can't use .sort_values()
    if isinstance(sim_df, pd.DataFrame):
        user_scores = sim_df[test_user_id]
        similar_users = user_scores.sort_values(ascending=False).drop(test_user_id)
        top_3_neighbors = similar_users.head(3).index.tolist()
    else:
        print("Skipping logic because sim_df is not a DataFrame.")
        top_3_neighbors = []

    if top_3_neighbors:
        print(f"Neighbors: {top_3_neighbors}")
        
        # Neighbors purchases
        neighbors_purchases = matrix.loc[top_3_neighbors]
        suggested_items = neighbors_purchases.sum(axis=0)
        
        # Remove owned items
        test_user_items = matrix.loc[test_user_id]
        suggested_items = suggested_items[test_user_items == 0]
        
        # Show results
        top_5 = suggested_items.sort_values(ascending=False).head(5)
        
        print("\n--- Recommendations ---")
        for code, score in top_5.items():
            name = lookup.get(str(code), "Unknown")
            print(f"{name} (Score: {score})")

except AttributeError as e:
    print(f"\n❌ AttributeError caught: {e}")
    print("This usually means you tried to use a Pandas method (.sort_values, .loc) on a non-Pandas object.")
except Exception as e:
    print(f"\n❌ Unexpected Error: {e}")
