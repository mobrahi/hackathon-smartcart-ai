import pandas as pd
import joblib
import os

# ==========================================================
# 1. LOAD THE MODELS (Just like Member 2 will do)
# ==========================================================
print("Loading saved models...")
try:
    # Load the Similarity Matrix (The DataFrame of users)
    sim_df = joblib.load('/Users/mf/code_stuff/hackathon-smartcart-ai/hackathon-smartcart-ai/models/recommendation_engine.pkl') 
    
    # Load the User-Item Matrix (Who bought what)
    matrix = joblib.load('/Users/mf/code_stuff/hackathon-smartcart-ai/hackathon-smartcart-ai/models/user_item_matrix.pkl')
    
    # Load the Product Names
    lookup = joblib.load('/Users/mf/code_stuff/hackathon-smartcart-ai/hackathon-smartcart-ai/models/product_lookup.pkl')
    
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# ==========================================================
# 2. PICK A TEST USER
# ==========================================================
# We need a user that exists inside our filtered top-1000 list.
# Let's grab the first user ID from the index.
test_user_id = sim_df.index[0]

print(f"\nAnalyzing User: {test_user_id}")

# ==========================================================
# 3. FIND SIMILAR USERS (The Logic)
# ==========================================================
# Get the similarity scores for our test user
user_scores = sim_df[test_user_id]

# Sort users by similarity (Highest score first)
# drop() removes the user themselves (similarity is always 1.0 with yourself)
similar_users = user_scores.sort_values(ascending=False).drop(test_user_id)

# Take the top 3 most similar users
top_3_neighbors = similar_users.head(3).index.tolist()
print(f"Top 3 similar users: {top_3_neighbors}")

# ==========================================================
# 4. GENERATE RECOMMENDATIONS
# ==========================================================
# 1. Get items bought by the neighbors
neighbors_purchases = matrix.loc[top_3_neighbors]

# 2. Sum up the quantities (Items bought multiple times by neighbors are more popular)
suggested_items = neighbors_purchases.sum(axis=0)

# 3. Remove items the TEST USER already bought
# (Don't recommend something they already own!)
test_user_items = matrix.loc[test_user_id]
suggested_items = suggested_items[test_user_items == 0]

# 4. Sort by highest quantity/score
suggested_items = suggested_items.sort_values(ascending=False).head(5)

# ==========================================================
# 5. DISPLAY RESULTS
# ==========================================================
print("\n--- Top 5 Recommended Products ---")

if suggested_items.sum() == 0:
    print("No recommendations found (User might have bought everything!)")
else:
    for stock_code, score in suggested_items.items():
        # Get the actual product name from our lookup dictionary
        product_name = lookup.get(stock_code, "Unknown Product")
        print(f"📦 {product_name} (Code: {stock_code}) | Popularity Score: {int(score)}")