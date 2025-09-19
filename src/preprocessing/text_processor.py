"""
Portuguese text preprocessing functionality
"""
import pandas as pd
import numpy as np
import re
from collections import Counter

def preprocess_portuguese_text(df):
    """
    Comprehensive Portuguese text preprocessing pipeline
    """
    
    if df is None:
        print("[ERROR] No dataset provided.")
        return None
    
    print("\n=== PORTUGUESE TEXT PREPROCESSING ===")
    
    df_text = df.copy()
    total_samples = len(df_text)
    
    # [1] Text availability analysis
    print("\n[1] Text availability")
    has_title = df_text['review_comment_title'].notna().sum()
    has_message = df_text['review_comment_message'].notna().sum()
    has_any_text = df_text[['review_comment_title', 'review_comment_message']].notna().any(axis=1).sum()
    
    print(f" -> Total samples: {total_samples:,}")
    print(f" -> With title:   {has_title:,} ({has_title/total_samples*100:.1f}%)")
    print(f" -> With message: {has_message:,} ({has_message/total_samples*100:.1f}%)")
    print(f" -> With any text:{has_any_text:,} ({has_any_text/total_samples*100:.1f}%)")
    
    # [2] Combine title and message
    print("\n[2] Combining title and message")
    def combine_review_text(row):
        title = str(row['review_comment_title']) if pd.notna(row['review_comment_title']) else ""
        message = str(row['review_comment_message']) if pd.notna(row['review_comment_message']) else ""
        combined = " ".join([t for t in [title, message] if t and t != "nan"]).strip()
        return combined if combined else None
    
    df_text['review_text_combined'] = df_text.apply(combine_review_text, axis=1)
    
    before_filter = len(df_text)
    df_text = df_text[df_text['review_text_combined'].notna()].copy()
    after_filter = len(df_text)
    print(f" -> Before filter: {before_filter:,}")
    print(f" -> After filter:  {after_filter:,}")
    print(f" -> Retention:     {after_filter/before_filter*100:.1f}%")
    
    if after_filter == 0:
        print("[ERROR] No samples with text available.")
        return df
    
    # [3] Clean Portuguese text
    print("\n[3] Cleaning text")
    def clean_portuguese_text(text):
        if pd.isna(text) or text == "":
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URLs
        text = re.sub(r'\S+@\S+', '', text)  # Emails
        text = re.sub(r'<.*?>', '', text)    # HTML
        text = re.sub(r'[^\w\s\-àáâãäèéêëìíîïòóôõöùúûüç]', ' ', text)  # Special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Extra spaces
        text = ' '.join([w for w in text.split() if len(w) >= 2])  # Remove very short words
        return text
    
    df_text['review_text_cleaned'] = df_text['review_text_combined'].apply(clean_portuguese_text)
    
    before_empty = len(df_text)
    df_text = df_text[df_text['review_text_cleaned'].str.len() > 0].copy()
    after_empty = len(df_text)
    print(f" -> Before empty removal: {before_empty:,}")
    print(f" -> After empty removal:  {after_empty:,}")
    
    # [4] Extract text features
    print("\n[4] Extracting basic text features")
    df_text['text_length'] = df_text['review_text_cleaned'].str.len()
    df_text['word_count'] = df_text['review_text_cleaned'].str.split().str.len()
    df_text['avg_word_length'] = df_text['review_text_cleaned'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    print(f" -> Features extracted for {len(df_text):,} samples")
    
    print("\n=== TEXT PREPROCESSING COMPLETE ===")
    return df_text
