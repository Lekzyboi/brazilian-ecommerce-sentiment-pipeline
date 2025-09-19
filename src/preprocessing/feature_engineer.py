"""
Comprehensive feature engineering functionality
"""
import pandas as pd
import numpy as np

def engineer_comprehensive_features(df):
    """
    Feature engineering including datetime, numerical, and categorical features
    """
    
    if df is None:
        print("[ERROR] No dataset provided.")
        return None
    
    print("\n=== FEATURE ENGINEERING STARTED ===")
    
    df_features = df.copy()
    initial_features = len(df_features.columns)
    print(f"Initial feature count: {initial_features}")
    
    # 1. DATETIME FEATURE ENGINEERING
    print("\n[1] Datetime features")
    datetime_columns = [
        'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date', 'review_creation_date'
    ]
    
    for col in datetime_columns:
        if col in df_features.columns:
            df_features[col] = pd.to_datetime(df_features[col], errors='coerce')
            print(f" -> Converted {col} to datetime")
    
    if 'order_purchase_timestamp' in df_features.columns:
        print(" -> Extracting purchase timing features")
        df_features['purchase_year'] = df_features['order_purchase_timestamp'].dt.year
        df_features['purchase_month'] = df_features['order_purchase_timestamp'].dt.month
        df_features['purchase_hour'] = df_features['order_purchase_timestamp'].dt.hour
        df_features['purchase_dayofweek'] = df_features['order_purchase_timestamp'].dt.dayofweek
        df_features['purchase_is_weekend'] = df_features['purchase_dayofweek'].isin([5, 6]).astype(int)
        df_features['purchase_business_hours'] = (
            (df_features['purchase_hour'] >= 9) & (df_features['purchase_hour'] <= 18)
        ).astype(int)
    
    if all(c in df_features.columns for c in ['order_purchase_timestamp', 'order_delivered_customer_date']):
        df_features['delivery_time_days'] = (
            df_features['order_delivered_customer_date'] - df_features['order_purchase_timestamp']
        ).dt.days
        df_features['delivery_speed'] = pd.cut(
            df_features['delivery_time_days'],
            bins=[-float('inf'), 3, 7, 15, 30, float('inf')],
            labels=['Same_Week', 'Fast', 'Normal', 'Slow', 'Very_Slow']
        )
    
    if all(c in df_features.columns for c in ['order_estimated_delivery_date', 'order_delivered_customer_date']):
        df_features['delivery_delay_days'] = (
            df_features['order_delivered_customer_date'] - df_features['order_estimated_delivery_date']
        ).dt.days
        df_features['delivered_early'] = (df_features['delivery_delay_days'] < 0).astype(int)
        df_features['delivered_late'] = (df_features['delivery_delay_days'] > 0).astype(int)
    
    # 2. PRICE AND VALUE
    print("\n[2] Price & value features")
    if 'total_price' in df_features.columns and 'total_freight' in df_features.columns:
        df_features['total_order_cost'] = df_features['total_price'] + df_features['total_freight']
        df_features['freight_percentage'] = (
            df_features['total_freight'] / (df_features['total_price'] + 0.01) * 100
        )
        df_features['price_tier'] = pd.qcut(
            df_features['total_price'].fillna(0), q=5, 
            labels=['Budget', 'Low', 'Mid', 'High', 'Premium'], duplicates='drop'
        )
        df_features['expensive_shipping'] = (df_features['freight_percentage'] > 30).astype(int)
        df_features['free_shipping'] = (df_features['total_freight'] == 0).astype(int)
        print(" -> Created total cost, freight %, tiers, and shipping flags")
    
    # 3. ORDER COMPLEXITY
    print("\n[3] Order complexity features")
    if 'total_items' in df_features.columns:
        df_features['is_single_item'] = (df_features['total_items'] == 1).astype(int)
        df_features['is_bulk_order'] = (df_features['total_items'] > 5).astype(int)
        
        complexity_factors = []
        if 'payment_methods_count' in df_features.columns:
            complexity_factors.append(df_features['payment_methods_count'] > 1)
        if 'uses_installments' in df_features.columns:
            complexity_factors.append(df_features['uses_installments'] == 1)
        complexity_factors.append(df_features['total_items'] > 3)
        
        if complexity_factors:
            df_features['order_complexity_score'] = sum(complexity_factors)
        print(" -> Added single/bulk flags and complexity score")
    
    # 4. GEOGRAPHIC
    print("\n[4] Geographic features")
    if 'customer_state' in df_features.columns:
        southeast = ['SP', 'RJ', 'MG', 'ES']
        south = ['RS', 'SC', 'PR']
        northeast = ['BA', 'PE', 'CE', 'PB', 'RN', 'AL', 'SE', 'MA', 'PI']
        df_features['customer_southeast'] = df_features['customer_state'].isin(southeast).astype(int)
        df_features['customer_south'] = df_features['customer_state'].isin(south).astype(int)
        df_features['customer_northeast'] = df_features['customer_state'].isin(northeast).astype(int)
        print(" -> Added regional indicators")
    
    if all(c in df_features.columns for c in ['customer_state', 'seller_state']):
        df_features['same_state_delivery'] = (
            df_features['customer_state'] == df_features['seller_state']
        ).astype(int)
        print(" -> Added same-state delivery flag")
    
    # 5. SELLER PERFORMANCE
    print("\n[5] Seller performance features")
    if 'primary_seller_id' in df_features.columns:
        if 'order_purchase_timestamp' in df_features.columns:
            df_features = df_features.sort_values('order_purchase_timestamp')
            df_features['seller_order_volume'] = df_features.groupby('primary_seller_id').cumcount()
        else:
            print(" [Warning] Using total seller volume (risk of leakage)")
            seller_counts = df_features['primary_seller_id'].value_counts()
            df_features['seller_order_volume'] = df_features['primary_seller_id'].map(seller_counts)
        
        df_features['seller_experience'] = pd.cut(
            df_features['seller_order_volume'].fillna(0),
            bins=[-1, 4, 19, 99, float('inf')],
            labels=['New', 'Developing', 'Experienced', 'Power_Seller']
        )
        df_features['is_power_seller'] = (df_features['seller_order_volume'] > 99).astype(int)
        print(" -> Added seller volume, experience, and power seller flag")
    
    # 6. CUSTOMER BEHAVIOR
    print("\n[6] Customer behavior features")
    if 'customer_unique_id' in df_features.columns:
        if 'order_purchase_timestamp' in df_features.columns:
            df_features = df_features.sort_values('order_purchase_timestamp')
            df_features['customer_order_count'] = df_features.groupby('customer_unique_id').cumcount()
        else:
            print(" [Warning] Using total customer orders (risk of leakage)")
            customer_counts = df_features['customer_unique_id'].value_counts()
            df_features['customer_order_count'] = df_features['customer_unique_id'].map(customer_counts)
        
        df_features['customer_type'] = pd.cut(
            df_features['customer_order_count'].fillna(0),
            bins=[-1, 0, 2, 9, float('inf')],
            labels=['First_Time', 'Occasional', 'Regular', 'Loyal']
        )
        df_features['is_repeat_customer'] = (df_features['customer_order_count'] > 0).astype(int)
        print(" -> Added customer order count, loyalty categories, and repeat flag")
    
    # 7. MISSING VALUES
    print("\n[7] Missing value handling")
    missing_summary = df_features.isnull().sum()
    cols_with_missing = missing_summary[missing_summary > 0]
    
    if len(cols_with_missing) > 0:
        print(f" -> Found {len(cols_with_missing)} columns with missing values")
        categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_features[col].isnull().sum() > 0:
                if df_features[col].dtype.name == 'category':
                    if 'Unknown' not in df_features[col].cat.categories:
                        df_features[col] = df_features[col].cat.add_categories(['Unknown'])
                df_features[col] = df_features[col].fillna('Unknown')
                print(f"    Filled categorical {col} with 'Unknown'")
        
        numerical_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_features[col].isnull().sum() > 0:
                if any(k in col.lower() for k in ['count', 'volume', 'score']):
                    df_features[col] = df_features[col].fillna(0)
                    print(f"    Filled {col} with 0")
                else:
                    fill_value = df_features[col].median()
                    df_features[col] = df_features[col].fillna(fill_value)
                    print(f"    Filled {col} with median ({fill_value:.2f})")
    else:
        print(" -> No missing values detected")
    
    # 8. FEATURE QUALITY SUMMARY
    print("\n[8] Feature quality summary")
    new_features = len(df_features.columns) - initial_features
    print(f" -> Created {new_features} new features")
    print(f" -> Total features: {len(df_features.columns)}")
    print(f" -> Remaining missing values: {df_features.isnull().sum().sum()}")
    print(f" -> Memory usage: {df_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df_features
