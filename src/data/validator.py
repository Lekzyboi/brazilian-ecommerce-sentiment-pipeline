"""
Data validation functionality
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data integrity"""
    
    REQUIRED_COLUMNS = {
        'orders_dataset': ['order_id', 'customer_id'],
        'order_reviews_dataset': ['order_id', 'review_score'],
        'order_items_dataset': ['order_id', 'product_id'],
        'customers_dataset': ['customer_id'],
        'products_dataset': ['product_id']
    }
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.datasets = datasets
        self.validation_results = {}
    
    def validate_all_datasets(self) -> Dict[str, Dict]:
        """Run all validation checks"""
        logger.info("=== Starting data validation ===")
        
        for name, df in self.datasets.items():
            logger.info(f"Validating: {name}")
            self.validation_results[name] = self._validate_dataset(df, name)
        
        self._validate_relationships()
        return self.validation_results
    
    def _validate_dataset(self, df: pd.DataFrame, name: str) -> Dict:
        """Validate a single dataset"""
        return {
            'schema_validation': self._validate_schema(df, name),
            'data_validation': self._validate_data_integrity(df, name),
            'business_rules': self._validate_business_rules(df, name)
        }
    
    def _validate_schema(self, df: pd.DataFrame, name: str) -> Dict:
        """Validate dataset schema"""
        required_cols = self.REQUIRED_COLUMNS.get(name, [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        return {
            'required_columns_present': len(missing_cols) == 0,
            'missing_columns': missing_cols,
            'total_columns': len(df.columns),
            'expected_columns': len(required_cols)
        }
    
    def _validate_data_integrity(self, df: pd.DataFrame, name: str) -> Dict:
        """Validate data integrity"""
        return {
            'has_data': len(df) > 0,
            'no_all_null_rows': not df.isnull().all(axis=1).any(),
            'id_columns_not_null': self._check_id_columns_not_null(df, name)
        }
    
    def _validate_business_rules(self, df: pd.DataFrame, name: str) -> Dict:
        """Validate business-specific rules"""
        rules = {}
        if name == 'order_reviews_dataset':
            rules.update(self._validate_review_rules(df))
        elif name == 'orders_dataset':
            rules.update(self._validate_order_rules(df))
        elif name == 'order_items_dataset':
            rules.update(self._validate_item_rules(df))
        return rules
    
    def _validate_review_rules(self, df: pd.DataFrame) -> Dict:
        """Validate review-specific business rules"""
        rules = {}
        if 'review_score' in df.columns:
            rules['valid_review_scores'] = df['review_score'].between(1, 5).all()
            rules['review_score_range'] = f"{df['review_score'].min()}–{df['review_score'].max()}"
        return rules
    
    def _validate_order_rules(self, df: pd.DataFrame) -> Dict:
        """Validate order-specific business rules"""
        rules = {}
        datetime_cols = ['order_purchase_timestamp', 'order_delivered_customer_date']
        for col in datetime_cols:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    rules[f'{col}_parseable'] = True
                except Exception:
                    rules[f'{col}_parseable'] = False
        return rules
    
    def _validate_item_rules(self, df: pd.DataFrame) -> Dict:
        """Validate item-specific business rules"""
        rules = {}
        if 'price' in df.columns:
            rules['positive_prices'] = (df['price'] >= 0).all()
        if 'freight_value' in df.columns:
            rules['non_negative_freight'] = (df['freight_value'] >= 0).all()
        return rules
    
    def _check_id_columns_not_null(self, df: pd.DataFrame, name: str) -> Dict:
        """Check that ID columns are not null"""
        required_cols = self.REQUIRED_COLUMNS.get(name, [])
        id_col_results = {}
        for col in required_cols:
            if col in df.columns:
                id_col_results[col] = df[col].notna().all()
        return id_col_results
    
    def _validate_relationships(self) -> None:
        """Validate relationships between datasets"""
        logger.info("Validating inter-dataset relationships")
        
        self.validation_results['relationships'] = {
            'orders_customers': self._validate_foreign_key(
                'orders_dataset', 'customer_id', 'customers_dataset', 'customer_id'
            ),
            'reviews_orders': self._validate_foreign_key(
                'order_reviews_dataset', 'order_id', 'orders_dataset', 'order_id'
            ),
            'items_orders': self._validate_foreign_key(
                'order_items_dataset', 'order_id', 'orders_dataset', 'order_id'
            )
        }
    
    def _validate_foreign_key(self, child_table: str, child_key: str, 
                             parent_table: str, parent_key: str) -> Dict:
        """Validate foreign key relationship"""
        if child_table not in self.datasets or parent_table not in self.datasets:
            return {'status': 'tables_missing'}
        
        child_df = self.datasets[child_table]
        parent_df = self.datasets[parent_table]
        
        if child_key not in child_df.columns or parent_key not in parent_df.columns:
            return {'status': 'columns_missing'}
        
        child_ids = set(child_df[child_key].dropna())
        parent_ids = set(parent_df[parent_key].dropna())
        
        orphaned_records = child_ids - parent_ids
        
        return {
            'status': 'valid' if len(orphaned_records) == 0 else 'orphaned_records',
            'orphaned_count': len(orphaned_records),
            'total_child_records': len(child_ids),
            'integrity_percentage': (
                (1 - len(orphaned_records) / len(child_ids)) * 100 if child_ids else 100
            )
        }
    
    def get_validation_summary(self) -> str:
        """Generate validation summary report"""
        if not self.validation_results:
            return "No validation results available."
        
        summary_lines = ["\n=== DATA VALIDATION SUMMARY ===\n"]
        
        for dataset_name, results in self.validation_results.items():
            if dataset_name == 'relationships':
                continue
            
            summary_lines.append(f"{dataset_name.upper()}:")
            
            schema = results['schema_validation']
            schema_status = "✓" if schema['required_columns_present'] else "✗"
            summary_lines.append(f"  Schema check: {schema_status}")
            
            integrity = results['data_validation']
            data_status = "✓" if integrity['has_data'] else "✗"
            summary_lines.append(f"  Has data: {data_status}")
            
            summary_lines.append("")
        
        if 'relationships' in self.validation_results:
            summary_lines.append("RELATIONSHIP VALIDATION:")
            for rel_name, rel_result in self.validation_results['relationships'].items():
                if rel_result.get('status') == 'valid':
                    summary_lines.append(f"  {rel_name}: ✓")
                else:
                    summary_lines.append(f"  {rel_name}: ✗ ({rel_result.get('status')})")
        
        return "\n".join(summary_lines)
