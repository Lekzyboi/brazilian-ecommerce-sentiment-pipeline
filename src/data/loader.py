"""
Data loading functionality for Brazilian E-commerce dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class OlistDataLoader:
    """Handles loading and initial processing of Olist dataset files"""
    
    EXPECTED_FILES = [
        'olist_orders_dataset.csv',
        'olist_order_items_dataset.csv', 
        'olist_order_reviews_dataset.csv',
        'olist_order_payments_dataset.csv',
        'olist_customers_dataset.csv',
        'olist_products_dataset.csv',
        'olist_sellers_dataset.csv',
        'olist_geolocation_dataset.csv',
        'product_category_name_translation.csv'
    ]
    
    def __init__(self, data_directory: str = 'data'):
        """
        Initialize the data loader
        
        Args:
            data_directory: Path to directory containing CSV files
        """
        self.data_path = Path(data_directory)
        self.datasets = {}
        self.loading_summary = []
        
    def load_all_datasets(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load all Olist dataset files from local directory
        
        Returns:
            Dictionary containing all datasets or None if loading fails
        """
        logger.info("Starting dataset loading...")
        
        if not self._validate_directory():
            return None
            
        self._load_individual_files()
        return self._finalize_loading()
    
    def _validate_directory(self) -> bool:
        """Validate that the data directory exists"""
        if not self.data_path.exists():
            logger.error(f"Directory not found: {self.data_path.absolute()}")
            logger.info("Please ensure the Olist dataset CSV files are placed in the data folder.")
            return False
        return True
    
    def _load_individual_files(self) -> None:
        """Load each CSV file individually"""
        logger.info(f"Looking for {len(self.EXPECTED_FILES)} dataset files...")
        
        for file in self.EXPECTED_FILES:
            file_path = self.data_path / file
            table_name = self._get_table_name(file)
            
            if file_path.exists():
                self._load_single_file(file_path, file, table_name)
            else:
                self._record_missing_file(file, table_name)
    
    def _load_single_file(self, file_path: Path, file: str, table_name: str) -> None:
        """Load a single CSV file"""
        try:
            df = pd.read_csv(file_path)
            self.datasets[table_name] = df
            
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            
            self.loading_summary.append({
                'file': file,
                'table_name': table_name,
                'shape': df.shape,
                'memory_mb': memory_mb,
                'status': 'SUCCESS'
            })
            
            logger.info(f"Loaded {table_name} | Shape: {df.shape} | Memory: {memory_mb:.2f} MB")
            
        except Exception as e:
            self._record_error(file, table_name, str(e))
    
    def _record_missing_file(self, file: str, table_name: str) -> None:
        """Record missing file in summary"""
        self.loading_summary.append({
            'file': file,
            'table_name': 'NOT_FOUND',
            'shape': (0, 0),
            'memory_mb': 0,
            'status': 'FILE_NOT_FOUND'
        })
        logger.warning(f"Missing file: {file}")
    
    def _record_error(self, file: str, table_name: str, error: str) -> None:
        """Record loading error in summary"""
        self.loading_summary.append({
            'file': file,
            'table_name': 'ERROR',
            'shape': (0, 0),
            'memory_mb': 0,
            'status': f'ERROR: {error}'
        })
        logger.error(f"Error loading {file}: {error}")
    
    def _finalize_loading(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Finalize loading process and return results"""
        successful_loads = len([s for s in self.loading_summary if s['status'] == 'SUCCESS'])
        total_memory = sum([s['memory_mb'] for s in self.loading_summary if s['status'] == 'SUCCESS'])
        
        logger.info(f"Files loaded successfully: {successful_loads}/{len(self.EXPECTED_FILES)}")
        logger.info(f"Total memory usage: {total_memory:.2f} MB")
        
        if successful_loads == 0:
            logger.error("No datasets were loaded.")
            return None
        
        if successful_loads < len(self.EXPECTED_FILES):
            logger.warning(f"{len(self.EXPECTED_FILES) - successful_loads} files missing. Analysis may be incomplete.")
        
        return self.datasets
    
    @staticmethod
    def _get_table_name(filename: str) -> str:
        """Convert filename to clean table name"""
        return filename.replace('.csv', '').replace('olist_', '')
    
    def get_loading_summary(self) -> List[Dict[str, object]]:
        """Get detailed loading summary"""
        return self.loading_summary
    
    def get_dataset_overview(self) -> Dict[str, Dict[str, object]]:
        """Get overview of all loaded datasets"""
        if not self.datasets:
            return {}
            
        overview = {}
        for name, df in self.datasets.items():
            overview[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'sample_data': df.head(2).to_dict('records') if len(df) > 0 else []
            }
        return overview


def load_olist_data(data_directory: str = 'data') -> Optional[Dict[str, pd.DataFrame]]:
    """
    Convenience function to load Olist datasets
    
    Args:
        data_directory: Path to directory containing CSV files
        
    Returns:
        Dictionary containing all datasets or None if loading fails
    """
    loader = OlistDataLoader(data_directory)
    return loader.load_all_datasets()
