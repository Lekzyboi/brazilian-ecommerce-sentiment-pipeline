"""
Data ingestion module for Brazilian E-commerce classification
"""

from .loader import OlistDataLoader, load_olist_data
from .quality import assess_data_quality_local
from .validator import DataValidator
from .review_analyzer import analyze_review_data
from .merger import create_comprehensive_master_dataset

__all__ = [
    'OlistDataLoader', 
    'load_olist_data', 
    'assess_data_quality_local', 
    'DataValidator',
    'analyze_review_data',
    'create_comprehensive_master_dataset'
]