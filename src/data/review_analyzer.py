"""
Review data analysis functionality
"""
import pandas as pd
import matplotlib.pyplot as plt

def analyze_review_data(datasets):
    """
    Deep dive into review data - the classification challenge
    """
    
    if 'order_reviews_dataset' not in datasets:
        print("[ERROR] Review dataset not found.")
        return None
    
    reviews_df = datasets['order_reviews_dataset']
    
    print("\n=== REVIEW DATA ANALYSIS (CLASSIFICATION TARGET) ===")
    
    # Basic review statistics
    total_reviews = len(reviews_df)
    print(f"Total reviews: {total_reviews:,}")
    
    # Review score distribution
    print("\nReview score distribution:")
    score_dist = reviews_df['review_score'].value_counts().sort_index()
    score_percent = (score_dist / total_reviews * 100).round(2)
    
    for score in range(1, 6):
        count = score_dist.get(score, 0)
        percent = score_percent.get(score, 0)
        stars = "★" * score
        print(f" -> {stars:<5} Score {score}: {count:>6,} reviews ({percent:>5.1f}%)")
    
    # Text review availability
    print("\nText review availability:")
    has_title = reviews_df['review_comment_title'].notna().sum()
    has_message = reviews_df['review_comment_message'].notna().sum()
    has_any_text = reviews_df[['review_comment_title', 'review_comment_message']].notna().any(axis=1).sum()
    
    print(f" -> With title:   {has_title:,} ({has_title/total_reviews*100:.1f}%)")
    print(f" -> With message: {has_message:,} ({has_message/total_reviews*100:.1f}%)")
    print(f" -> With any text:{has_any_text:,} ({has_any_text/total_reviews*100:.1f}%)")
    
    # Sample review texts
    print("\nSample review texts (Portuguese):")
    text_reviews = reviews_df[reviews_df['review_comment_message'].notna()]
    if len(text_reviews) > 0:
        for i, (_, row) in enumerate(text_reviews.head(3).iterrows()):
            score = row['review_score']
            text = row['review_comment_message']
            if len(str(text)) > 100:
                text = str(text)[:100] + "..."
            print(f" -> Score {score}: \"{text}\"")
    
    return {
        'total_reviews': total_reviews,
        'score_distribution': score_dist.to_dict(),
        'score_percentages': score_percent.to_dict(),
        'text_availability': {
            'has_title': has_title,
            'has_message': has_message,
            'has_any_text': has_any_text
        }
    }
