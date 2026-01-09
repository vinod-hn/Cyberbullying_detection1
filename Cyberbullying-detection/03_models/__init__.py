# 03_models - Cyberbullying Detection Models
"""
Pre-trained models for cyberbullying detection.

NO TRAINING REQUIRED - All models are pre-trained and ready for inference.

Available Models:
    - BERT (bert-base-uncased): 99.88% F1 Score
    - mBERT (bert-base-multilingual-cased): 99.57% F1 Score  
    - IndicBERT (google/muril-base-cased): 99.76% F1 Score
    - Baseline (TF-IDF + SVM/LogReg/NB): ~95% F1 Score

Quick Start:
    from model_loader import CyberbullyingDetector
    
    detector = CyberbullyingDetector(model_type='bert')
    result = detector.predict("You're such a loser!")
    print(result['prediction'], result['confidence'])
"""

from .model_loader import (
    CyberbullyingDetector,
    get_best_model,
    list_available_models
)

__all__ = [
    'CyberbullyingDetector',
    'get_best_model', 
    'list_available_models'
]
