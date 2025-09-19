"""
Target variable creation functionality
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_classification_target(df):
    """
    Create binary classification target from review scores
    Includes analysis and class balance check
    """
    
    if df is None:
        print("[ERROR] No dataset provided.")
        return None
    
    print("\n=== CLASSIFICATION TARGET CREATION ===")
    
    # Filter to orders with reviews only
    initial_orders = len(df)
    df_reviews = df[df['review_score'].notna()].copy()
    orders_with_reviews = len(df_reviews)
    
    print("\n[1] Initial data overview")
    print(f" -> Total orders: {initial_orders:,}")
    print(f" -> Orders with reviews: {orders_with_reviews:,} ({orders_with_reviews/initial_orders*100:.1f}%)")
    
    if orders_with_reviews == 0:
        print("[ERROR] No orders with review scores found.")
        return None
    
    # Review score distribution
    print("\n[2] Review score distribution")
    score_counts = df_reviews['review_score'].value_counts().sort_index()
    score_percentages = (score_counts / orders_with_reviews * 100).round(2)
    for score in range(1, 6):
        count = score_counts.get(score, 0)
        percent = score_percentages.get(score, 0)
        bar = "▇" * int(percent / 2)
        print(f" -> Score {score}: {count:>7,} ({percent:>5.1f}%) {bar}")
    
    # Conservative classification strategy
    print("\n[3] Applying classification strategy")
    print(" -> Logic: 1–2 = Negative (0), 4–5 = Positive (1), 3 = Neutral (excluded)")
    
    conditions = [
        df_reviews['review_score'].isin([1, 2]),  # Negative
        df_reviews['review_score'].isin([4, 5])   # Positive
    ]
    choices = [0, 1]
    df_reviews['sentiment_binary'] = np.select(conditions, choices, default=np.nan)
    
    df_classification = df_reviews.dropna(subset=['sentiment_binary']).copy()
    df_classification['sentiment_binary'] = df_classification['sentiment_binary'].astype(int)
    
    # Target distribution
    final_samples = len(df_classification)
    target_dist = df_classification['sentiment_binary'].value_counts().sort_index()
    target_percentages = (target_dist / final_samples * 100).round(2)
    excluded_neutral = orders_with_reviews - final_samples
    
    print("\n[4] Target distribution")
    print(f" -> Negative (0): {target_dist[0]:>7,} ({target_percentages[0]:>5.1f}%)")
    print(f" -> Positive (1): {target_dist[1]:>7,} ({target_percentages[1]:>5.1f}%)")
    print(f" -> Neutral excluded: {excluded_neutral:>7,} ({excluded_neutral/orders_with_reviews*100:>5.1f}%)")
    print(f" -> Final samples: {final_samples:,}")
    
    # Class balance check
    print("\n[5] Class balance analysis")
    class_ratio = target_percentages[1] / target_percentages[0]
    print(f" -> Positive:Negative ratio = {class_ratio:.2f}:1")
    
    imbalance_gap = abs(target_percentages[0] - target_percentages[1])
    if imbalance_gap > 30:
        print(" -> Severe class imbalance detected")
    elif imbalance_gap > 15:
        print(" -> Moderate class imbalance detected")
    else:
        print(" -> Classes are balanced")
    
    print("\n=== TARGET VARIABLE CREATION COMPLETE ===")
    print(f" -> Final dataset size: {final_samples:,} samples")
    
    return df_classification
