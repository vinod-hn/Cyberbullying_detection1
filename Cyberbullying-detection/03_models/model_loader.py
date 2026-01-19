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
import csv
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Get project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / '03_models' / 'saved_models'

# Lexicon paths (used for conservative sanity filtering)
LEXICON_DIR = PROJECT_ROOT / '00_data' / 'lexicon'
_TOXIC_TERMS_CACHE: Optional[set] = None

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
    
    Model Types (trained on 7000 samples, 80/20 split):
        - 'bert': BERT base uncased (best performance: 100% F1, 100% Accuracy)
        - 'mbert': Multilingual BERT (good for code-mixed text: 99.86% F1)
        - 'indicbert': IndicBERT/MuRIL (optimized for Indian languages: 99.93% F1)
        - 'baseline': TF-IDF + Best baseline model (SVM/LogReg/NB)
    
    Example:
        >>> detector = CyberbullyingDetector(model_type='bert')
        >>> result = detector.predict("You're such a loser!")
        >>> print(result['prediction'], result['confidence'])
    """
    
    # Model performance from training (Updated: Colab Training 80/20 split on 7000 samples)
    MODEL_PERFORMANCE = {
        'bert': {'accuracy': 1.0, 'f1': 1.0, 'precision': 1.0, 'recall': 1.0},
        'mbert': {'accuracy': 0.9986, 'f1': 0.9986, 'precision': 0.9986, 'recall': 0.9986},
        'indicbert': {'accuracy': 0.9993, 'f1': 0.9993, 'precision': 0.9993, 'recall': 0.9993},
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
        
        # Load metrics if available (check both metrics.json and test_metrics.json)
        self.test_metrics = None
        for metrics_filename in ['metrics.json', 'test_metrics.json']:
            metrics_path = model_path / metrics_filename
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.test_metrics = json.load(f)
                logger.info(f"Model metrics loaded from {metrics_filename}: {self.test_metrics}")
                break

    @staticmethod
    def _load_toxic_terms() -> set:
        """Load a set of toxic/profanity terms from bundled lexicon files.

        This is used only for a conservative sanity filter to reduce false positives
        on short/greeting texts (common in chat logs and TXT uploads).
        """
        global _TOXIC_TERMS_CACHE
        if _TOXIC_TERMS_CACHE is not None:
            return _TOXIC_TERMS_CACHE

        terms: set = set()

        # Small built-in list as a fallback if lexicon files are missing.
        builtin = {
            'kill', 'die', 'beat', 'hate', 'loser', 'idiot', 'stupid', 'dumb',
            'ugly', 'slut', 'whore', 'bitch', 'asshole', 'moron', 'jerk'
        }
        terms.update(builtin)

        def read_csv_terms(path: Path, columns: List[str]) -> None:
            if not path.exists():
                return
            try:
                with open(path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for col in columns:
                            val = (row.get(col) or '').strip().lower()
                            if val:
                                terms.add(val)
            except Exception as e:
                logger.warning(f"Failed to read lexicon file {path}: {e}")

        # English + Kannada profanity lists
        read_csv_terms(LEXICON_DIR / 'profanity_english.csv', ['term'])
        read_csv_terms(LEXICON_DIR / 'profanity_kannada.csv', ['term', 'transliteration'])

        _TOXIC_TERMS_CACHE = terms
        return _TOXIC_TERMS_CACHE

    # ─────────────────────────────────────────────────────────────────────────────
    # POSITIVE / BENIGN INDICATORS (used to reduce false positives)
    # ─────────────────────────────────────────────────────────────────────────────
    _POSITIVE_WORDS = {
        # Greetings
        'hi', 'hello', 'hey', 'hii', 'hiii', 'hai', 'helo', 'heyy',
        # Farewells
        'bye', 'goodbye', 'goodnight', 'gn', 'tc', 'cya', 'see', 'later',
        # Acknowledgements
        'ok', 'okay', 'k', 'kk', 'yes', 'no', 'yep', 'nope', 'ya', 'yaa', 'nah',
        'hmm', 'hm', 'hmmmm', 'umm', 'um', 'ahh', 'ooh', 'oops', 'lol', 'haha', 'hehe',
        # Thanks / Sorry
        'thanks', 'thank', 'thanku', 'tq', 'ty', 'thx', 'sorry', 'sry', 'apologies',
        # Positive adjectives
        'good', 'great', 'nice', 'fine', 'cool', 'awesome', 'amazing', 'wonderful',
        'fantastic', 'excellent', 'beautiful', 'lovely', 'sweet', 'cute', 'best',
        'super', 'brilliant', 'perfect', 'happy', 'glad', 'pleased', 'excited',
        # Affection
        'love', 'loved', 'miss', 'missed', 'care', 'friend', 'bro', 'sis', 'dear',
        'buddy', 'dude', 'man', 'guys', 'team', 'fam', 'family',
        # Polite
        'please', 'pls', 'plz', 'welcome', 'congrats', 'congratulations', 'well',
        # Common verbs (neutral)
        'come', 'go', 'send', 'call', 'tell', 'ask', 'know', 'think', 'need', 'want',
        'wait', 'meet', 'join', 'share', 'help', 'try', 'check', 'see', 'look',
        # Question words
        'what', 'when', 'where', 'why', 'how', 'who', 'which',
        # Time / place
        'today', 'tomorrow', 'now', 'later', 'morning', 'evening', 'night', 'time',
        'home', 'class', 'college', 'school', 'office', 'work',
        # Misc safe
        'food', 'lunch', 'dinner', 'movie', 'song', 'game', 'study', 'exam', 'test',
    }

    _BENIGN_PHRASES = [
        # Greetings
        r"^(hi+|he+y+|hello+|hai)[\s,!.?]*(\w{2,20})?[\s!.?]*$",
        r"^good\s*(morning|afternoon|evening|night|day)[\s,!.?]*(\w{2,20})?[\s!.?]*$",
        # Farewells
        r"^(bye+|goodbye|see\s*(you|ya|u)|tc|take\s*care|gn|good\s*night)[\s!.?]*$",
        # How are you variants
        r"^(how\s*(are|r)\s*(you|u)|hru|how('?s| is)\s*(it|everything|life|u)|what'?s\s*up|wassup|sup)[\s?!.]*$",
        # I am / I'm introductions
        r"^(i\s*am|i'm|im|this\s*is)\s+\w{2,25}[\s!.?,]*$",
        # I am fine/good etc
        r"^(i\s*am|i'm|im)\s+(good|fine|ok|okay|great|doing\s*well|doing\s*good|alright)[\s!.?,]*(\w{2,20})?[\s!.?,]*$",
        # Simple acknowledgements
        r"^(ok+|okay+|k+|yes+|no+|ya+|yep|nope|nah|sure|alright|fine|done|got\s*it|noted|understood)[\s!.?,]*$",
        # Thanks / sorry
        r"^(thanks?|thanku|thank\s*you|ty|thx|tq|sorry|sry|apolog(y|ies)|my\s*bad)[\s!.?,]*(\w{2,20})?[\s!.?,]*$",
        # Simple questions
        r"^(what|when|where|why|how|who|which)[\s'?\w]{0,30}\?*$",
        # Compliments / positive
        r"^(you\s*(are|r)|u\s*r|that'?s|it'?s)\s*(good|great|nice|awesome|amazing|wonderful|cool|sweet|the\s*best)[\s!.?,]*$",
        # Tell me / send me
        r"^(tell|send|share|show|give)\s*(me|us)[\s\w]{0,25}[\s!.?,]*$",
        # Let's / shall we
        r"^(let'?s|shall\s*we|can\s*we|we\s*can)[\s\w]{0,30}[\s!.?,]*$",
        # Coming / going
        r"^(i'?m|i\s*am|we\s*are|we're)\s*(coming|going|leaving|here|there|on\s*my\s*way)[\s!.?,]*$",
        # Miss you / love you
        r"^(i\s*)?(miss|love|like)\s*(you|u|this|it|that)[\s!.?,]*$",
        # See you
        r"^see\s*(you|ya|u)\s*(soon|later|tomorrow|there)?[\s!.?,]*$",
        # Wait / hold on
        r"^(wait|hold\s*on|one\s*sec|just\s*a\s*moment|brb|be\s*right\s*back)[\s!.?,]*$",
        # Names / addressing (common Indian names often flagged)
        r"^(hey|hi|hello)?\s*[a-z]{2,15}[\s,!.?]*$",
        # Short emoji-only or laughter
        r"^(ha+|he+|lo+l+|lmao|rofl|xd+|[:;][-']?[)DPp])+[\s!.?,]*$",
        # Numbers / times
        r"^[\d:.\s]+\s*(am|pm|o'?clock)?[\s!.?,]*$",
        # Single word safe
        r"^(nice|cool|great|good|okay|sure|yeah|yep|nope|done|same|true|right|correct|agreed)[\s!.?,]*$",
    ]

    @classmethod
    def _has_positive_sentiment(cls, tokens: List[str]) -> bool:
        """Check if message has positive/neutral sentiment indicators."""
        if not tokens:
            return False
        positive_count = sum(1 for t in tokens if t in cls._POSITIVE_WORDS)
        return positive_count >= 1 and positive_count >= len(tokens) * 0.3

    @classmethod
    def _matches_benign_pattern(cls, text_lower: str) -> bool:
        """Check if text matches any known benign pattern."""
        t = re.sub(r'\s+', ' ', (text_lower or '').strip())
        if not t:
            return True
        for pattern in cls._BENIGN_PHRASES:
            if re.match(pattern, t, re.IGNORECASE):
                return True
        return False

    def _apply_sanity_filter(self, original_text: str, processed_text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply AGGRESSIVE post-processing to reduce false positives to near zero.

        CORE PRINCIPLE: Only flag as cyberbullying if there is EXPLICIT toxic content.
        If no clear toxic/threat/profane words are found, classify as NEUTRAL.
        
        This ensures:
        - Greetings, questions, normal chat → NEUTRAL
        - Only messages with actual bad words/threats → flagged
        """
        try:
            pred = (result.get('prediction') or '').strip().lower()
            if not pred or pred == 'neutral':
                return result

            text_lower = (processed_text or '').strip().lower()
            tokens = re.findall(r"[\w']+", text_lower)
            word_count = len(tokens)

            # ══════════════════════════════════════════════════════════════════
            # STRICT RULE: Only trust model if EXPLICIT toxic content is present
            # ══════════════════════════════════════════════════════════════════
            
            toxic_terms = self._load_toxic_terms()
            has_toxic_token = any(tok in toxic_terms for tok in tokens)
            
            # Expanded threat patterns - must match at least one to be flagged
            threat_patterns = [
                r'\b(kill|die|death|murder|beat|punch|slap|hit|hurt|attack|stab|shoot)\b',
                r'\b(hate\s*(you|u)|destroy|ruin|expose|humiliate)\b',
                r'\b(shut\s*up|get\s*lost|go\s*(away|die|to\s*hell)|fuck\s*(off|you))\b',
                r'\b(loser|idiot|stupid|dumb|ugly|fat|pathetic|disgusting|worthless|useless)\b',
                r'\b(nobody\s*likes|everyone\s*hates|no\s*friends|kill\s*yourself)\b',
                r'\b(bitch|slut|whore|bastard|asshole|retard|moron)\b',
                r'\b(trash|garbage|scum|vermin|pest|parasite)\b',
                r'\b(disappear|leave|get\s*out|go\s*away).*\b(nobody|no\s*one)\b',
            ]
            has_threat_pattern = any(re.search(p, text_lower) for p in threat_patterns)

            # ── If explicit toxicity found → trust model ──
            if has_toxic_token or has_threat_pattern:
                return result

            # ══════════════════════════════════════════════════════════════════
            # NO TOXIC CONTENT FOUND → FORCE NEUTRAL
            # This is aggressive but eliminates false positives
            # ══════════════════════════════════════════════════════════════════
            
            return self._force_neutral(result)

        except Exception as e:
            logger.warning(f"Sanity filter failed: {e}")
            return result

    def _force_neutral(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Force a result to neutral classification."""
        probs = result.get('probabilities') or {}
        neutral_prob = probs.get('neutral', 0.6) if isinstance(probs, dict) else 0.6
        
        result['prediction'] = 'neutral'
        result['confidence'] = max(0.55, min(float(neutral_prob) if neutral_prob else 0.6, 0.95))
        
        if isinstance(probs, dict) and 'neutral' in probs:
            probs['neutral'] = result['confidence']
            result['probabilities'] = probs
        
        return result
    
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
        
        # Strip BOM / odd leading markers that can show up in TXT exports
        text = str(text).lstrip('\ufeff').lstrip().lower()
        
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

        # Apply conservative sanity filter (helps with short TXT uploads)
        for i in range(len(results)):
            results[i] = self._apply_sanity_filter(texts[i], processed_texts[i], results[i])
        
        # Add original texts and boolean flag
        # "neutral" class means NOT cyberbullying, all other classes are cyberbullying types
        NON_CYBERBULLYING_LABELS = {'neutral', 'not_cyberbullying', 'Not Cyberbullying', 'safe', 'none'}
        for i, result in enumerate(results):
            result['text'] = texts[i]
            result['is_cyberbullying'] = result['prediction'].lower() not in {label.lower() for label in NON_CYBERBULLYING_LABELS}
        
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
