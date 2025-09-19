from src.data import load_olist_data, assess_data_quality_local, DataValidator, analyze_review_data, create_comprehensive_master_dataset
from src.preprocessing import create_classification_target, preprocess_portuguese_text, engineer_comprehensive_features
from src.modeling import build_classification_models, extract_sentiment_reasons

def main():
    print("Brazilian E-commerce Review Classification Pipeline")
    print("=" * 60)
    
    # Data ingestion phase
    datasets = load_olist_data('data')
    if datasets is None:
        return
    
    quality_report = assess_data_quality_local(datasets)
    
    validator = DataValidator(datasets)
    validation_results = validator.validate_all_datasets()
    print(validator.get_validation_summary())
    
    review_analysis = analyze_review_data(datasets)
    master_df = create_comprehensive_master_dataset(datasets)
    
    if master_df is None:
        return
    
    # Preprocessing phase
    print("\nSTARTING PREPROCESSING PHASE...")
    
    df_with_target = create_classification_target(master_df)
    if df_with_target is None:
        return
    
    df_processed = preprocess_portuguese_text(df_with_target)
    if df_processed is None:
        return
    
    df_engineered = engineer_comprehensive_features(df_processed)
    if df_engineered is None:
        return
    
    # Modeling phase
    print("\nSTARTING MODELING PHASE...")
    
    model_results = build_classification_models(df_engineered)
    if model_results is None:
        return
    
    # Bonus challenge
    print("\nSTARTING BONUS CHALLENGE...")
    reason_results = extract_sentiment_reasons(df_engineered, model_results)
    
    print(f"\nPIPELINE COMPLETE!")
    print(f"Best model: {model_results['best_model_name']}")
    print(f"Final dataset: {df_engineered.shape}")
    
    if reason_results:
        print(f"Bonus challenge: {reason_results['coverage_percentage']:.1f}% reason coverage")

if __name__ == "__main__":
    main()