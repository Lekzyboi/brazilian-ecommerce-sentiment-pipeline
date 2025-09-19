"""
Data quality assessment functionality
"""
import pandas as pd
import numpy as np

def assess_data_quality_local(datasets):
    """
    Data quality assessment for all datasets
    """
    
    if datasets is None:
        print("[ERROR] No datasets available for assessment.")
        return None
    
    print("\n=== COMPREHENSIVE DATA QUALITY ASSESSMENT ===")
    
    quality_summary = {}
    
    for name, df in datasets.items():
        print(f"\n--- {name.upper().replace('_', ' ')} ---")
        
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Missing values
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / total_rows * 100).round(2)
        
        # Duplicates
        duplicates = df.duplicated().sum()
        
        # Data types
        dtypes_summary = df.dtypes.value_counts().to_dict()
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        # Store in summary
        quality_summary[name] = {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'missing_values': missing_values[missing_values > 0].to_dict(),
            'missing_percentages': missing_percent[missing_percent > 0].to_dict(),
            'duplicates': duplicates,
            'data_types': dtypes_summary,
            'memory_mb': memory_mb
        }
        
        # Print concise overview
        print(f"Rows × Cols: {total_rows:,} × {total_cols}")
        print(f"Memory usage: {memory_mb:.2f} MB")
        print(f"Duplicate rows: {duplicates:,}")
        
        # Missing values (only if present)
        if len(missing_values[missing_values > 0]) > 0:
            print("Missing values:")
            for col, missing in missing_values[missing_values > 0].items():
                percent = missing_percent[col]
                print(f"   • {col}: {missing:,} ({percent:.1f}%)")
        else:
            print("Missing values: None")
        
        # Data types
        print(f"Column data types: {dtypes_summary}")
    
    return quality_summary
