"""
Preprocessing module for Brazilian E-commerce classification
"""
from .target_creator import create_classification_target
from .text_processor import preprocess_portuguese_text
from .feature_engineer import engineer_comprehensive_features


__all__ = [
    'create_classification_target',
    'preprocess_portuguese_text',
    'engineer_comprehensive_features'
]