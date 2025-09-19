"""
BONUS CHALLENGE: Extract structured reasons for sentiment
"""
import pandas as pd
import numpy as np
from collections import Counter

def extract_sentiment_reasons(df, model_results=None):
    """
    BONUS CHALLENGE: Extract structured reasons for positive/negative sentiment
    """
    
    if df is None:
        print("No dataset provided")
        return {
            'reason_categories': {},
            'reason_mentions': {},
            'coverage_percentage': 0.0,
            'enhanced_dataframe': pd.DataFrame()
        }
    
    print("BONUS CHALLENGE: SENTIMENT REASON EXTRACTION")
    print("="*70)
    print("Goal: Extract structured reasons that make reviews positive or negative")
    
    # Filter to samples with text reviews
    text_df = df[df['review_text_cleaned'].notna()].copy()
    print(f"\nText samples available: {len(text_df):,}")
    
    if len(text_df) == 0:
        print("No text samples available for reason extraction")
        return {
            'reason_categories': {},
            'reason_mentions': {},
            'coverage_percentage': 0.0,
            'enhanced_dataframe': pd.DataFrame()
        }
    
    # Define reason categories and keywords
    print("\nDEFINING REASON CATEGORIES:")
    
    reason_categories = {
        'delivery_time': {
            'positive': [
                'rápido', 'rapidez', 'rápida', 'pontual', 'prazo',
                'chegou rápido', 'entrega rápida', 'no prazo'
            ],
            'negative': [
                'demorou', 'atrasou', 'atraso', 'demora', 'lento',
                'não chegou', 'muito tempo'
            ]
        },
        'product_quality': {
            'positive': [
                'qualidade', 'excelente', 'perfeito', 'ótimo', 'resistente',
                'bem feito', 'como descrito', 'original'
            ],
            'negative': [
                'defeito', 'quebrado', 'ruim', 'péssimo', 'frágil',
                'mal feito', 'diferente', 'falsificado'
            ]
        },
        'seller_service': {
            'positive': [
                'atendimento', 'educado', 'prestativo', 'ajudou',
                'bem embalado', 'recomendo loja'
            ],
            'negative': [
                'não respondeu', 'mal educado', 'mal atendimento',
                'mal embalado', 'loja ruim'
            ]
        },
        'price_value': {
            'positive': [
                'barato', 'bom preço', 'vale pena', 'custo benefício',
                'promoção', 'preço justo'
            ],
            'negative': [
                'caro', 'preço alto', 'muito caro', 'não vale',
                'frete caro', 'abusivo'
            ]
        }
    }
    
    print(f"Defined {len(reason_categories)} reason categories:")
    for category in reason_categories.keys():
        pos_count = len(reason_categories[category]['positive'])
        neg_count = len(reason_categories[category]['negative'])
        print(f"  {category}: {pos_count} positive + {neg_count} negative keywords")
    
    # Extract reasons for each review
    print("\nEXTRACTING REASONS FROM REVIEW TEXTS:")
    
    def extract_reasons_from_text(text, sentiment):
        """Extract specific reasons from review text"""
        if pd.isna(text) or text == "":
            return {}
        
        text_lower = str(text).lower()
        detected_reasons = {}
        
        for category, keywords in reason_categories.items():
            pos_matches = sum(1 for keyword in keywords['positive'] if keyword in text_lower)
            neg_matches = sum(1 for keyword in keywords['negative'] if keyword in text_lower)
            
            if pos_matches > 0 or neg_matches > 0:
                if pos_matches > neg_matches:
                    category_sentiment = 'positive'
                    strength = pos_matches
                elif neg_matches > pos_matches:
                    category_sentiment = 'negative'
                    strength = neg_matches
                else:
                    category_sentiment = 'positive' if sentiment == 1 else 'negative'
                    strength = max(pos_matches, neg_matches)
                
                detected_reasons[category] = {
                    'sentiment': category_sentiment,
                    'strength': strength,
                    'positive_matches': pos_matches,
                    'negative_matches': neg_matches
                }
        
        return detected_reasons
    
    # Apply reason extraction
    print("Processing review texts...")
    
    text_df['extracted_reasons'] = text_df.apply(
        lambda row: extract_reasons_from_text(row['review_text_cleaned'], row['sentiment_binary']),
        axis=1
    )
    
    # Create structured reason features
    for category in reason_categories.keys():
        text_df[f'mentions_{category}'] = text_df['extracted_reasons'].apply(
            lambda x: 1 if category in x else 0
        )
        text_df[f'{category}_sentiment'] = text_df['extracted_reasons'].apply(
            lambda x: x[category]['sentiment'] if category in x else None
        )
        text_df[f'{category}_strength'] = text_df['extracted_reasons'].apply(
            lambda x: x[category]['strength'] if category in x else 0
        )
    
    print("Reason extraction complete!")
    
    # Analyze reason patterns
    print("\nREASON PATTERN ANALYSIS:")
    
    reason_mentions = {}
    for category in reason_categories.keys():
        mentions = text_df[f'mentions_{category}'].sum()
        percentage = mentions / len(text_df) * 100
        reason_mentions[category] = {
            'count': mentions,
            'percentage': percentage
        }
        print(f"{category}: {mentions:,} mentions ({percentage:.1f}%)")
    
    # Generate example extractions
    print("\nEXAMPLE REASON EXTRACTIONS:")
    
    examples_found = 0
    for idx, row in text_df.head(100).iterrows():
        if examples_found >= 3:
            break
            
        reasons = row['extracted_reasons']
        if len(reasons) > 0:
            sentiment_label = "Positive" if row['sentiment_binary'] == 1 else "Negative"
            text_sample = row['review_text_cleaned'][:100] + "..." if len(row['review_text_cleaned']) > 100 else row['review_text_cleaned']
            
            print(f"\nExample {examples_found + 1} ({sentiment_label}):")
            print(f"Text: \"{text_sample}\"")
            print("Reasons detected:")
            
            for category, details in reasons.items():
                print(f"  - {category}: {details['sentiment']} (strength: {details['strength']})")
            
            examples_found += 1
    
    # Summary
    total_reviews_with_reasons = text_df[[f'mentions_{cat}' for cat in reason_categories.keys()]].any(axis=1).sum()
    coverage_percentage = total_reviews_with_reasons / len(text_df) * 100
    
    print(f"\nSENTIMENT REASON EXTRACTION COMPLETE!")
    print(f"Total reviews analyzed: {len(text_df):,}")
    print(f"Reviews with detected reasons: {total_reviews_with_reasons:,} ({coverage_percentage:.1f}%)")
    print(f"Reason categories: {len(reason_categories)}")
    
    # Prepare final results
    reason_extraction_results = {
        'reason_categories': reason_categories,
        'reason_mentions': reason_mentions,
        'coverage_percentage': coverage_percentage,
        'enhanced_dataframe': text_df
    }
    
    print("\nBONUS CHALLENGE IMPLEMENTED!")
    print("Structured insights into WHY customers give positive/negative reviews!")
    
    return reason_extraction_results
