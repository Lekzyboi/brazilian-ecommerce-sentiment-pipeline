"""
Data merging functionality to create master dataset
"""
import pandas as pd

def create_comprehensive_master_dataset(datasets):
    """
    Create a comprehensive master dataset by joining all tables
    This is the foundation for the classification model
    """
    
    if datasets is None:
        print("[ERROR] No datasets available for merging.")
        return None
    
    print("\n=== MASTER DATASET CREATION STARTED ===")
    
    # Step 1: Orders
    print("Step 1: Using orders dataset as the central hub...")
    if 'orders_dataset' not in datasets:
        print("[ERROR] Orders dataset not found. Cannot proceed.")
        return None
    
    master_df = datasets['orders_dataset'].copy()
    print(f" -> Orders dataset shape: {master_df.shape}")
    print(f" -> Unique orders: {master_df['order_id'].nunique():,}")
    
    # Step 2: Order items
    print("\nStep 2: Merging order items...")
    if 'order_items_dataset' in datasets:
        order_items = datasets['order_items_dataset'].copy()
        print(f" -> Raw order items: {order_items.shape[0]:,}")
        
        items_aggregated = order_items.groupby('order_id').agg({
            'order_item_id': 'count',
            'product_id': lambda x: '|'.join(x.astype(str)),
            'seller_id': lambda x: '|'.join(x.astype(str)),
            'price': ['sum', 'mean', 'std'],
            'freight_value': ['sum', 'mean']
        }).reset_index()
        
        items_aggregated.columns = [
            'order_id', 'total_items', 'product_ids', 'seller_ids',
            'total_price', 'avg_item_price', 'price_std',
            'total_freight', 'avg_freight'
        ]
        
        items_aggregated['is_multi_item'] = (items_aggregated['total_items'] > 1).astype(int)
        items_aggregated['price_std'] = items_aggregated['price_std'].fillna(0)
        items_aggregated['total_order_value'] = items_aggregated['total_price'] + items_aggregated['total_freight']
        items_aggregated['freight_ratio'] = items_aggregated['total_freight'] / (items_aggregated['total_price'] + 0.01)
        items_aggregated['primary_product_id'] = items_aggregated['product_ids'].str.split('|').str[0]
        items_aggregated['primary_seller_id'] = items_aggregated['seller_ids'].str.split('|').str[0]
        
        master_df = master_df.merge(items_aggregated, on='order_id', how='left')
        print(f" -> After merge: {master_df.shape}")
    
    # Step 3: Customers
    print("\nStep 3: Merging customer information...")
    if 'customers_dataset' in datasets:
        customers = datasets['customers_dataset'].copy()
        print(f" -> Total customers: {customers.shape[0]:,}")
        master_df = master_df.merge(customers, on='customer_id', how='left')
        print(f" -> After merge: {master_df.shape}")
    
    # Step 4: Products
    print("\nStep 4: Merging product details...")
    if 'products_dataset' in datasets and 'primary_product_id' in master_df.columns:
        products = datasets['products_dataset'].copy()
        products_renamed = products.rename(columns={'product_id': 'primary_product_id'})
        master_df = master_df.merge(products_renamed, on='primary_product_id', how='left')
        print(f" -> After merge: {master_df.shape}")
    
    # Step 5: Sellers
    print("\nStep 5: Merging seller details...")
    if 'sellers_dataset' in datasets and 'primary_seller_id' in master_df.columns:
        sellers = datasets['sellers_dataset'].copy()
        sellers_renamed = sellers.rename(columns={
            'seller_id': 'primary_seller_id',
            'seller_zip_code_prefix': 'seller_zip'
        })
        master_df = master_df.merge(sellers_renamed, on='primary_seller_id', how='left')
        print(f" -> After merge: {master_df.shape}")
    
    # Step 6: Payments
    print("\nStep 6: Merging payment information...")
    if 'order_payments_dataset' in datasets:
        payments = datasets['order_payments_dataset'].copy()
        payments_agg = payments.groupby('order_id').agg({
            'payment_sequential': 'count',
            'payment_type': lambda x: '|'.join(x),
            'payment_installments': ['sum', 'max'],
            'payment_value': 'sum'
        }).reset_index()
        
        payments_agg.columns = [
            'order_id', 'payment_methods_count', 'payment_types',
            'total_installments', 'max_installments', 'total_payment_value'
        ]
        
        payments_agg['uses_multiple_payments'] = (payments_agg['payment_methods_count'] > 1).astype(int)
        payments_agg['uses_installments'] = (payments_agg['max_installments'] > 1).astype(int)
        payments_agg['primary_payment_type'] = payments_agg['payment_types'].str.split('|').str[0]
        
        master_df = master_df.merge(payments_agg, on='order_id', how='left')
        print(f" -> After merge: {master_df.shape}")
    
    # Step 7: Reviews (target data)
    print("\nStep 7: Merging reviews (target variable)...")
    if 'order_reviews_dataset' in datasets:
        reviews = datasets['order_reviews_dataset'].copy()
        master_df = master_df.merge(reviews, on='order_id', how='left')
        print(f" -> After merge: {master_df.shape}")
    
    # Step 8: Category translations
    print("\nStep 8: Merging category translations...")
    if 'product_category_name_translation' in datasets:
        categories = datasets['product_category_name_translation'].copy()
        master_df = master_df.merge(categories, on='product_category_name', how='left')
        print(f" -> After merge: {master_df.shape}")
    
    # Final summary
    print("\n=== MASTER DATASET SUMMARY ===")
    print(f"Final shape: {master_df.shape}")
    print(f"Memory usage: {master_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Orders with reviews: {master_df['review_score'].notna().sum():,}")
    print(f"Orders with review text: {master_df['review_comment_message'].notna().sum():,}")
    
    print("\nColumn categories:")
    print(f" - ID columns: {len([col for col in master_df.columns if 'id' in col.lower()])}")
    print(f" - Date columns: {len([col for col in master_df.columns if any(word in col.lower() for word in ['timestamp', 'date'])])}")
    print(f" - Price-related columns: {len([col for col in master_df.columns if any(word in col.lower() for word in ['price', 'value', 'freight'])])}")
    print(f" - Review-related columns: {len([col for col in master_df.columns if 'review' in col.lower()])}")
    print(f" - Product-related columns: {len([col for col in master_df.columns if 'product' in col.lower()])}")
    
    return master_df
