"""
Model Loader - Unified interface for loading pre-trained cyberbullying detection models.

This module provides easy access to all trained models:
- Baseline models (Naive Bayes, SVM, Logistic Regression with TF-IDF)
- Transformer models (BERT, mBERT, IndicBERT)

NO TRAINING REQUIRED - All models are pre-trained and ready for inference.

Usage:
    from model_loader import CyberbullyingDetector
    
    # Load best model (BERT by default)
    detector = CyberbullyingDetector()
    result = detector.predict("You're such a loser!")
    
    # Load specific model
    detector = CyberbullyingDetector(model_type='mbert')
    detector = CyberbullyingDetector(model_type='baseline')
"""

import os
import json
import pickle
import logging
import re
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Get project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / '03_models' / 'saved_models'

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Transformer models disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Transformer models disabled.")

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

try:
    from scipy.special import softmax
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class CyberbullyingDetector:
    """
    Unified cyberbullying detection interface.
    
    Loads pre-trained models and provides inference methods.
    
    Model Types:
        - 'bert': BERT base uncased (best performance: 99.88% F1)
        - 'mbert': Multilingual BERT (good for code-mixed text: 99.57% F1)
        - 'indicbert': IndicBERT/MuRIL (optimized for Indian languages: 99.76% F1)
        - 'baseline': TF-IDF + Best baseline model (SVM/LogReg/NB)
    
    Example:
        >>> detector = CyberbullyingDetector(model_type='bert')
        >>> result = detector.predict("You're such a loser!")
        >>> print(result['prediction'], result['confidence'])
    """
    
    # Model performance from training
    MODEL_PERFORMANCE = {
        'bert': {'accuracy': 0.9988, 'f1': 0.9988},
        'mbert': {'accuracy': 0.9957, 'f1': 0.9957},
        'indicbert': {'accuracy': 0.9976, 'f1': 0.9976},
        'baseline': {'accuracy': 0.95, 'f1': 0.95}  # Approximate
    }
    
    AVAILABLE_MODELS = ['bert', 'mbert', 'indicbert', 'baseline']
    
    def __init__(
        self,
        model_type: str = 'bert',
        models_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the detector with a specific model.
        
        Args:
            model_type: Type of model to load ('bert', 'mbert', 'indicbert', 'baseline')
            models_dir: Path to saved models directory
            device: Device for inference ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_type = model_type.lower()
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        
        # Set device
        if device:
            self.device = device
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.label_encoder = None
        
        # Load the model
        self._load_model()
        
        logger.info(f"CyberbullyingDetector initialized with {model_type} model on {self.device}")
    
    def _load_label_encoder(self):
        """Load the label encoder."""
        le_path = self.models_dir / 'label_encoder.pkl'
        if le_path.exists():
            with open(le_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info("Label encoder loaded")
        else:
            # Create default label encoder
            logger.warning("Label encoder not found, using defaults")
            self.classes_ = ['Cyberbullying', 'Not Cyberbullying']
    
    def _load_model(self):
        """Load the specified model."""
        self._load_label_encoder()
        
        if self.model_type == 'baseline':
            self._load_baseline_model()
        elif self.model_type in ['bert', 'mbert', 'indicbert']:
            self._load_transformer_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Available: {self.AVAILABLE_MODELS}")
    
    def _load_baseline_model(self):
        """Load baseline TF-IDF + classifier model."""
        # Load vectorizer
        vectorizer_path = self.models_dir / 'tfidf_vectorizer.pkl'
        if vectorizer_path.exists():
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("TF-IDF vectorizer loaded")
        else:
            raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}")
        
        # Load model
        model_path = self.models_dir / 'best_baseline_model.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Baseline model loaded")
        else:
            raise FileNotFoundError(f"Baseline model not found at {model_path}")
    
    def _load_transformer_model(self):
        """Load transformer model (BERT/mBERT/IndicBERT)."""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "PyTorch and transformers are required for transformer models. "
                "Install with: pip install torch transformers"
            )
        
        model_folder = f'transformer_{self.model_type}'
        model_path = self.models_dir / model_folder
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Transformer model loaded from {model_path}")
        
        # Load test metrics if available
        metrics_path = model_path / 'test_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.test_metrics = json.load(f)
            logger.info(f"Model metrics: {self.test_metrics}")
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Clean and normalize text for model input.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text string
        """
        import pandas as pd
        
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Convert emojis to text
        if EMOJI_AVAILABLE:
            text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_probabilities: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict if text is cyberbullying.
        
        Args:
            text: Input text or list of texts
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results:
                - text: Original text
                - prediction: Predicted class label
                - confidence: Confidence score
                - probabilities: Class probabilities (if requested)
                - is_cyberbullying: Boolean flag
        """
        # Handle single vs batch
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        # Preprocess
        processed_texts = [self.preprocess_text(t) for t in texts]
        
        # Get predictions
        if self.model_type == 'baseline':
            results = self._predict_baseline(processed_texts, return_probabilities)
        else:
            results = self._predict_transformer(processed_texts, return_probabilities)
        
        # Add original texts and boolean flag
        for i, result in enumerate(results):
            result['text'] = texts[i]
            result['is_cyberbullying'] = result['prediction'] != 'Not Cyberbullying'
        
        return results[0] if single_input else results
    
    def _predict_baseline(
        self,
        texts: List[str],
        return_probabilities: bool
    ) -> List[Dict[str, Any]]:
        """Predict using baseline model."""
        # Vectorize
        X = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.model.predict(X)
        
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'prediction': self.label_encoder.inverse_transform([pred])[0] if self.label_encoder else str(pred),
                'confidence': 0.9  # Baseline doesn't have proba for all models
            }
            
            # Get probabilities if available
            if return_probabilities and hasattr(self.model, 'predict_proba'):
                try:
                    probs = self.model.predict_proba(X[i:i+1])[0]
                    result['confidence'] = float(max(probs))
                    if self.label_encoder:
                        result['probabilities'] = {
                            cls: float(p) for cls, p in zip(self.label_encoder.classes_, probs)
                        }
                except:
                    pass
            
            results.append(result)
        
        return results
    
    def _predict_transformer(
        self,
        texts: List[str],
        return_probabilities: bool
    ) -> List[Dict[str, Any]]:
        """Predict using transformer model."""
        results = []
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits.cpu().numpy()
        
        # Apply softmax
        if SCIPY_AVAILABLE:
            probs = softmax(logits, axis=1)
        else:
            # Manual softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        for i in range(len(texts)):
            pred_idx = probs[i].argmax()
            
            result = {
                'prediction': self.label_encoder.inverse_transform([pred_idx])[0] if self.label_encoder else str(pred_idx),
                'confidence': float(probs[i].max())
            }
            
            if return_probabilities and self.label_encoder:
                result['probabilities'] = {
                    cls: float(p) for cls, p in zip(self.label_encoder.classes_, probs[i])
                }
            
            results.append(result)
        
        return results
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Predict on a batch of texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = self.predict(batch)
            all_results.extend(results if isinstance(results, list) else [results])
        
        return all_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_type': self.model_type,
            'device': self.device,
            'models_dir': str(self.models_dir),
            'performance': self.MODEL_PERFORMANCE.get(self.model_type, {}),
            'classes': list(self.label_encoder.classes_) if self.label_encoder else None
        }
        
        if hasattr(self, 'test_metrics'):
            info['test_metrics'] = self.test_metrics
        
        return info


def get_best_model() -> CyberbullyingDetector:
    """Get the best performing model (BERT)."""
    return CyberbullyingDetector(model_type='bert')


def list_available_models() -> Dict[str, Dict[str, float]]:
    """List all available models with their performance metrics."""
    return CyberbullyingDetector.MODEL_PERFORMANCE.copy()


# Quick test
if __name__ == '__main__':
    print("=" * 60)
    print("Cyberbullying Detector - Model Loader Test")
    print("=" * 60)
    
    # List available models
    print("\n📦 Available Models:")
    for model, perf in list_available_models().items():
        print(f"   • {model}: F1={perf['f1']:.4f}, Accuracy={perf['accuracy']:.4f}")
    
    # Test with best model
    print("\n🔄 Loading best model (BERT)...")
    try:
        detector = get_best_model()
        print(f"✅ Model loaded on {detector.device}")
        
        # Test predictions
        test_texts = [
            "You're such a loser, nobody likes you!",
            "Great job on the presentation today!",
            "I hate you so much!",
            "Thanks for helping me 😊"
        ]
        
        print("\n🔮 Test Predictions:\n")
        for text in test_texts:
            result = detector.predict(text)
            emoji_icon = "🚨" if result['is_cyberbullying'] else "✅"
            print(f"{emoji_icon} \"{text}\"")
            print(f"   → {result['prediction']} (Confidence: {result['confidence']:.2%})\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Trying baseline model instead...")
        try:
            detector = CyberbullyingDetector(model_type='baseline')
            print(f"✅ Baseline model loaded")
        except Exception as e2:
            print(f"❌ Baseline also failed: {e2}")
