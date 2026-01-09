# Quick Start Guide - Cyberbullying Detection

## 🎉 Models Are Pre-Trained!

**You don't need to train any models.** All models have been trained on Google Colab and are ready for inference.

### Trained Models Performance

| Model | Accuracy | F1 Score | Best For |
|-------|----------|----------|----------|
| **BERT** | 99.88% | 99.88% | Best overall performance |
| **IndicBERT** | 99.76% | 99.76% | Indian languages (Kannada) |
| **mBERT** | 99.57% | 99.57% | Code-mixed text |
| **Baseline (TF-IDF)** | ~95% | ~95% | Fast inference, no GPU |

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd Cyberbullying-detection
pip install -r requirements.txt
```

### Step 2: Test the Model

```python
import sys
sys.path.insert(0, 'path/to/Cyberbullying-detection')

from model_loader import CyberbullyingDetector

# Load best model (BERT)
detector = CyberbullyingDetector(model_type='bert')

# Make prediction
result = detector.predict("You're such a loser!")
print(result)
# {'text': "You're such a loser!", 'prediction': 'Cyberbullying', 
#  'confidence': 0.98, 'is_cyberbullying': True}
```

### Step 3: Run the API

```bash
cd 06_api
python api.py
```

Then open http://localhost:8000/docs for interactive API documentation.

---

## 📁 Project Structure

```
Cyberbullying-detection/
├── 00_data/                    # Dataset files
│   ├── processed/              # Train/val/test splits (ready to use)
│   ├── raw/                    # Original datasets
│   └── lexicon/                # Slang & profanity lists
│
├── 01_preprocessing/           # Text preprocessing modules
│   ├── text_normalizer.py      # Text cleaning
│   ├── emoji_handler.py        # Emoji processing
│   └── slang_expander.py       # Slang expansion
│
├── 02_feature_extraction/      # Feature extractors
│   └── transformer_embedder.py # BERT embeddings
│
├── 03_models/                  # 🔥 MAIN MODULE
│   ├── model_loader.py         # ⭐ Unified inference interface
│   ├── saved_models/           # ⭐ Pre-trained models
│   │   ├── transformer_bert/   # BERT model (best)
│   │   ├── transformer_mbert/  # mBERT model
│   │   ├── transformer_indicbert/  # IndicBERT model
│   │   ├── best_baseline_model.pkl # Baseline model
│   │   ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│   │   └── label_encoder.pkl       # Label encoder
│   └── baseline/               # Baseline model code (reference only)
│
├── 06_api/                     # REST API
│   └── api.py                  # FastAPI application
│
└── requirements.txt            # Dependencies
```

---

## 💻 Usage Examples

### Python - Single Prediction

```python
from model_loader import CyberbullyingDetector

# Choose your model
detector = CyberbullyingDetector(model_type='bert')  # Best accuracy
# detector = CyberbullyingDetector(model_type='baseline')  # Fastest

# Predict
result = detector.predict("Thanks for helping me!")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Is Cyberbullying: {result['is_cyberbullying']}")
```

### Python - Batch Prediction

```python
texts = [
    "You're such an idiot!",
    "Great job on the project!",
    "I hate you so much",
    "Thanks for your help 😊"
]

results = detector.predict_batch(texts, batch_size=32)
for r in results:
    status = "🚨" if r['is_cyberbullying'] else "✅"
    print(f"{status} {r['text'][:40]}... → {r['prediction']}")
```

### API - cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "You are so stupid!", "model_type": "bert"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello friend!", "You idiot!"], "model_type": "bert"}'

# List models
curl "http://localhost:8000/models"
```

---

## 📊 Dataset Overview

- **Total Samples**: ~11,000
- **Languages**: English, Kannada, Code-mixed
- **Classes**: Cyberbullying vs Not Cyberbullying (binary)
- **Split**: 70% train, 15% validation, 15% test

### Data Sources
1. English cyberbullying dataset
2. Kannada cyberbullying dataset
3. Kannada-English code-mixed dataset
4. Emoji-based cyberbullying dataset
5. Profanity/bad words dataset

---

## 🔧 Configuration

### Using Different Models

```python
# Best accuracy (requires GPU for fast inference)
detector = CyberbullyingDetector(model_type='bert')

# Best for Kannada/Indian languages
detector = CyberbullyingDetector(model_type='indicbert')

# Best for code-mixed (Kannada + English)
detector = CyberbullyingDetector(model_type='mbert')

# Fastest (no GPU needed)
detector = CyberbullyingDetector(model_type='baseline')
```

### Force CPU (for transformer models)

```python
detector = CyberbullyingDetector(model_type='bert', device='cpu')
```

---

## 🛠️ Troubleshooting

### "Model not found" Error
Make sure the `saved_models` folder contains the trained models:
```
03_models/saved_models/
├── transformer_bert/
├── transformer_mbert/
├── transformer_indicbert/
├── best_baseline_model.pkl
├── tfidf_vectorizer.pkl
└── label_encoder.pkl
```

### "CUDA out of memory" Error
Use CPU or reduce batch size:
```python
detector = CyberbullyingDetector(model_type='bert', device='cpu')
results = detector.predict_batch(texts, batch_size=8)
```

### "transformers not installed" Error
```bash
pip install torch transformers
```

---

## 📝 Next Steps

1. **Deploy API**: Use Docker or cloud services (AWS/GCP/Azure)
2. **Add Dashboard**: Use the `08_dashboard` module for visualization
3. **Integrate Database**: Use `07_database` for logging predictions
4. **Add Severity Scoring**: Extend with `05_severity_scoring` module

---

## 📚 Additional Resources

- [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) - Training details and metrics
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)
- [03_models/model_loader.py](03_models/model_loader.py) - Model loader source code
