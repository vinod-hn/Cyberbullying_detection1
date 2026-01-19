# Cyberbullying Detection Project – Detailed Technical Report

## 1. Project Overview

This project implements an end‑to‑end cyberbullying detection system for social‑media style text, with a focus on English, Kannada, and Kannada–English code‑mixed content.

Key goals:
- Detect cyberbullying across multiple languages and scripts.
- Support fine‑grained bullying categories and a binary "cyberbullying vs not" view.
- Provide robust preprocessing for noisy, informal, emoji‑rich social text.
- Offer both traditional ML and transformer‑based models, already trained and saved.
- Expose the models through a production‑style FastAPI backend, backed by a database and designed for dashboard integration.

At the current state of the project:
- Data ingestion, preprocessing and feature‑engineering pipelines are implemented.
- Multiple model families are designed; baseline and transformer models are trained and saved.
- A unified model loader (CyberbullyingDetector) is available for easy inference.
- A FastAPI service with authentication, logging, statistics and feedback endpoints is wired up.
- A normalized SQLite schema is defined for messages, predictions, users, alerts, audit logs and feedback.

---

## 2. Data: Sources, Processing and Organization

**Repository data layout** (high‑level):
- 00_data/raw: Original source datasets.
- 00_data/processed: Cleaned and split data ready for model training and evaluation.
- 00_data/anonymized: Anonymized train/val/test splits and mapping logs.
- 00_data/lexicon: Lexicons for slang, profanity and emoji semantics.
- 00_data/metadata: Annotation guidelines, label distributions and quality reports.

### 2.1 Raw Data and Sources

From TRAINING_SUMMARY.md and 00_data/raw, the project consolidates:
- English cyberbullying messages (~2,000 rows).
- Kannada messages (~4,000 rows).
- Code‑mixed Kannada–English (~2,000 rows).
- Emoji‑based cyberbullying dataset (~1,120 rows).
- Bad‑words/profanity dataset (~2,000 rows).

Total raw samples reported: **11,120**, from 5 different sources.

There are **11 fine‑grained classes**, such as insult, neutral, harassment, threat, exclusion, aggression, toxicity, stalking, sexual_harassment, hate and cyberstalking, with a significant class imbalance (roughly 10.88:1 between majority and minority classes). Class imbalance is explicitly accounted for during model training using class weights and specialized algorithms.

### 2.2 Processed and Anonymized Data

The raw datasets are cleaned, merged, labeled and split into:
- Processed Samples: **10,968** retained records (~98.6% of raw), after discarding unusable entries.
- Splits (processed):
  - Train: 7,677 samples (~70%).
  - Validation: 1,645 samples (~15%).
  - Test: 1,646 samples (~15%).

Additionally, 00_data/anonymized stores anonymized versions of the splits and an id_mapping_log.txt, documenting how user‑identifiable information has been replaced with anonymous IDs for privacy.

### 2.3 Metadata and Lexicons

Metadata under 00_data/metadata documents:
- Annotation guidelines and labeling definitions.
- Data quality and inter‑annotator agreement statistics.
- Label distribution (JSON), summarizing class frequencies.

Lexicon resources under 00_data/lexicon include:
- profanity_english.csv and profanity_kannada.csv: profanity lists in both languages.
- english_slang.csv and kannada_slang.csv: slang and informal terms.
- code_mix_patterns.txt: patterns specific to code‑mixed writing.
- emoji_semantics.json: semantic and sentiment metadata for emojis, including cyberbullying associations and detection weights.

These lexicons are consumed by preprocessing and feature extraction components.

---

## 3. Text Preprocessing Pipeline (01_preprocessing)

Preprocessing is implemented as modular Python classes that can be composed into a pipeline. It is tailored for noisy, code‑mixed Kannada–English social media text.

Core modules include:
- text_normalizer.py – TextNormalizer
- emoji_handler.py – EmojiHandler
- slang_expander.py – SlangExpander
- transliterator.py – transliteration support (e.g., romanized Kannada to script).
- code_mix_processor.py – code‑mix‑specific handling and normalization.
- conversation_threader.py – threading of conversation‑level context.

### 3.1 Text Normalization (TextNormalizer)

TextNormalizer performs a multi‑step cleaning pipeline:
- Unicode normalization (NFKC) and removal of zero‑width characters.
- URL removal (http/https, www, ftp patterns).
- Email removal.
- Dataset‑specific hashtag ID removal (e.g. #5a76 style tokens).
- Contraction expansion (don’t → do not, you’re → you are, etc.).
- Character elongation reduction (noooooo → noo) with configurable max_char_repeat.
- Punctuation repetition reduction (!!! → !, ??? → ?).
- Whitespace normalization (tabs/newlines → single spaces) and stripping.
- Lowercasing (while still respecting Kannada script content).

The class can be configured via a config dict (or defaults) to toggle steps such as removing URLs, handling contractions, or preserving certain patterns. Helper methods are also exposed for simpler operations (clean_text, remove_noise, etc.).

### 3.2 Emoji Processing (EmojiHandler)

EmojiHandler provides deep support for emojis as first‑class signals:
- Detection of emojis using broad Unicode ranges for emoticons, pictographs, symbols, flags, etc.
- Extraction of emojis, including support for Zero Width Joiner (ZWJ) sequences and skin tone modifiers.
- Frequency statistics (total count, unique count, per‑emoji frequency).
- Sentiment analysis derived from emoji_semantics.json, assigning sentiment scores and typical contexts.
- Cyberbullying association and severity mapping (e.g. critical/high/medium/low/positive) from detection_weights.
- Categorization by types: offensive, aggressive, threatening, mocking, body‑shaming, sexual‑harassment emojis.

This enables features such as:
- is_emoji_heavy, average emoji sentiment, presence of “bad” emojis, and severity signals that can interact with text‑only models.

### 3.3 Slang and Abbreviation Expansion (SlangExpander)

SlangExpander normalizes informal, abbreviated and code‑mixed expressions by expanding them to more standard equivalents, while being careful with insult/profanity terms:
- Kannada slang: maps common romanized phrases (e.g. machaa, maga, tumba, yaake, illa) to English equivalents or normalized semantics.
- English slang: handles social‑media slang such as dude, bro, lol, lmao, sus, cringe, lit, goat, etc.
- Internet abbreviations: u → you, ur → your, rn → right now, idk → I do not know, asap → as soon as possible, etc.
- Context‑aware expansion: certain insult terms are intentionally preserved (e.g. thotha, singri, moorkha) because they are predictive for bullying.

Together with TextNormalizer, SlangExpander reduces the vocabulary noise and improves downstream modeling.

### 3.4 Code‑Mix and Conversation Handling

Additional modules (code_mix_processor.py, transliterator.py, conversation_threader.py) are designed for:
- Identifying language segments and script usage (Kannada vs Latin).
- Normalizing romanized Kannada forms into consistent tokens or script.
- Building conversation threads so that models can reason about sequences of messages and escalation.

Unit tests in 01_preprocessing/tests validate key behaviors for code‑mix handling, emoji handling, normalization and transliteration.

---

## 4. Feature Extraction (02_feature_extraction)

Feature extraction builds multiple complementary views of each message, beyond raw text, to help models detect cyberbullying.

Main modules include:
- transformer_embedder.py – TransformerEmbedder (BERT/mBERT/IndicBERT embeddings).
- linguistic_features.py – LinguisticFeatures (lexical, syntactic, semantic, readability, code‑mix features).
- behavioral_features.py – BehavioralFeatures (user‑level patterns over time and targets).
- Additional modules (emoji_features.py, contextual_features.py, sarcasm_detector.py, text_embedder.py) support richer features and specialized signals.

### 4.1 Transformer‑Based Embeddings (TransformerEmbedder)

TransformerEmbedder wraps Hugging Face transformer models and provides:
- Support for multiple configurations via MODEL_CONFIGS:
  - bert-base-uncased (English).
  - bert-base-multilingual-cased (mBERT, good for code‑mixed text).
  - ai4bharat/indic-bert (Indic languages, including Kannada).
  - xlm-roberta-base and distilbert-base-multilingual-cased.
- Automatic device selection (CUDA if available, else CPU) with optional override.
- Tokenization, batching and pooling strategies:
  - CLS token pooling.
  - Mean or max pooling over token embeddings.
- Methods for:
  - encode / extract: batch embeddings of shape (n_samples, hidden_size).
  - encode_with_attention: embeddings plus averaged attention maps and tokens.
  - get_token_embeddings: token‑level embeddings for analysis/visualization.
  - similarity / batch_similarity: cosine similarities between texts.

This component underlies transformer‑based classifiers and can be reused for experiments (e.g. similarity search, ablations).

### 4.2 Linguistic Features (LinguisticFeatures)

LinguisticFeatures computes a detailed handcrafted feature set, optionally leveraging NLTK and TextBlob:

Lexical features:
- Character count, word count, unique word count.
- Average word length.
- Vocabulary richness (unique/total).
- Stop‑word ratio and uppercase character ratio.

Syntactic features:
- POS tag counts (nouns, verbs, adjectives, pronouns) via NLTK, when available.
- Number of sentences and average sentence length.
- Punctuation‑based emphasis: exclamation marks, question marks, ellipses, all‑caps words, repeated characters.

Semantic features:
- Sentiment polarity and subjectivity (TextBlob, when installed).
- Flags for positive, negative, neutral polarity ranges.
- Aggressive word counts based on curated word lists (English and Kannada), plus list of detected aggressive terms.

Readability and complexity:
- Heuristic syllable counts and average syllables per word.
- Adapted Flesch reading‑ease scores (0–100).
- A custom complexity score combining long‑word ratio, vocabulary richness, presence of Kannada script and text length.

Code‑mix specific aspects (in extended methods):
- Ratio of Kannada script vs Latin characters.
- Detection of romanized Kannada patterns (pronouns, intensifiers, insults, etc.).

### 4.3 Behavioral and User‑Level Features (BehavioralFeatures)

BehavioralFeatures analyzes user behavior over time to capture patterns that are indicative of harassment campaigns or repeat offending.

Key responsibilities:
- Build per‑user profiles from message histories (build_user_profile):
  - Total messages and text examples.
  - Temporal patterns: peak activity hour, night activity flag, average and minimum gap between messages, burst behavior.
  - Content patterns: average message length, aggregate aggressive word counts, emoji usage, capitalization ratio, aggressive‑user flags.
  - Target patterns: how many unique targets, focused targeting score, primary target.
  - Severity patterns over time: average and maximum severity scores, trend (escalating/de‑escalating/stable), high‑risk flags.

- Frequency features (extract_frequency_features):
  - messages_per_day, burst behaviors, high‑volume user flag.

- Persistence features (extract_persistence_features):
  - repeat‑offender indication.
  - whether a user consistently targets a single individual.
  - whether a user has a history of high‑severity incidents.

- Harassment campaign features (extract_harassment_campaign_features):
  - Detection of coordinated attacks (multiple users, many messages in short time, repeated content).
  - A campaign_score summarizing how strongly the pattern resembles a harassment campaign.

These features are meant to augment message‑level classifiers with user and conversation‑level risk scoring.

---

## 5. Model Families and Training (03_models and 14_scripts)

The project supports three major model families:
- Baseline ML models using TF‑IDF features.
- Context‑aware deep learning models (LSTM/Transformer architectures).
- Fine‑tuned transformer sequence‑classification models (BERT, mBERT, IndicBERT).

### 5.1 Baseline Models

According to TRAINING_SUMMARY.md and 14_scripts/train_models.py, the baseline suite includes:
- Naive Bayes (ComplementNB) with TF‑IDF (1–3 grams, up to 15,000 features).
- SVM (LinearSVC with probability calibration) + TF‑IDF.
- Logistic Regression + TF‑IDF.

Characteristics:
- Designed to handle class imbalance using class weights.
- Support for hyper‑parameter tuning via command‑line flags.
- First training run (Naive Bayes) achieved:
  - Training accuracy ~99.79%.
  - Validation accuracy ~99.82%.
  - Consistent cross‑validation F1 (~99.58% ±0.29).
- Models are serialized to 03_models/saved_models/baseline/ alongside the TF‑IDF vectorizer and label encoder.

### 5.2 Context‑Aware Deep Models

The training script also supports context‑aware models implemented under 03_models/context_aware (trainer imported as ContextModelTrainer):
- BiLSTM‑based encoders with attention.
- Transformer‑style models with multi‑head attention and positional encodings.
- Hierarchical models (word‑, message‑, conversation‑level) for thread‑level classification and severity prediction.

Key training configuration (with defaults in train_models.py):
- Vocab size, embedding dim (random or pre‑trained GloVe), hidden dimension, dropout.
- Class imbalance handling via class weights.
- Gradient clipping and early stopping (patience).
- Train/val/test CSVs read from 00_data/processed.

These models require PyTorch; the script checks availability, device (CPU/GPU) and logs environment details.

### 5.3 Fine‑Tuned Transformer Classifiers

The main deployed models are fine‑tuned transformer classifiers saved under 03_models/saved_models/:
- transformer_bert – BERT base uncased, fine‑tuned for cyberbullying detection.
- transformer_mbert – Multilingual BERT, tuned for code‑mixed text.
- transformer_indicbert – IndicBERT or MuRIL, tuned for Indian languages.

Reported performance on a 7,000‑sample 80/20 split:
- BERT: **100% accuracy and F1** on the held‑out test set.
- IndicBERT: ~99.93% accuracy/F1.
- mBERT: ~99.86% accuracy/F1.

These results are stored in metrics.json or test_metrics.json inside each model folder, and the model_loader reads them when available for reference.

### 5.4 Unified Training Script

14_scripts/train_models.py provides a unified CLI training interface:
- Supports modes: baseline, context, both.
- Select specific models (nb, svm, tfidf, lstm, bilstm, transformer, all).
- Options for hyper‑parameter tuning, epochs, batch size, and embedding type.
- Uses centralized logging to 17_logs/training_YYYYMMDD_HHMMSS.log.

Examples (from TRAINING_SUMMARY.md):
- Train all baseline models:
  - python train_models.py --type baseline --model all
- Train context‑aware models:
  - python train_models.py --type context --model all --epochs 10

---

## 6. Unified Inference Interface (CyberbullyingDetector)

03_models/model_loader.py defines CyberbullyingDetector, a single entry point for loading and running pre‑trained models.

Capabilities:
- Supports model_type in {bert, mbert, indicbert, baseline}.
- Automatically locates saved models under 03_models/saved_models.
- For baseline models:
  - Loads TF‑IDF vectorizer and best_baseline_model.pkl.
  - Uses label_encoder.pkl for class names.
- For transformer models:
  - Uses AutoTokenizer and AutoModelForSequenceClassification from saved model folders.
  - Moves the model to the configured device (CPU/GPU).
  - Optionally loads test metrics from JSON files.

Preprocessing prior to prediction:
- Lowercasing text.
- Emoji demojization (to text) when emoji library is available.
- Removal of URLs, mentions and hashtags.
- Whitespace normalization.

Prediction API:
- predict(text_or_list, return_probabilities=True) → result(s) dictionaries with:
  - text: original text.
  - prediction: predicted label (mapped through label encoder when available).
  - confidence: max class probability.
  - probabilities: full distribution over labels (for transformer / proba‑enabled baselines).
  - is_cyberbullying: boolean, derived by checking whether the predicted label is one of neutral/not_cyberbullying/safe/none vs all other bullying‑related labels.

- predict_batch(texts, batch_size) for efficient batched inference.
- get_model_info() for debugging and introspection (model type, device, expected performance, label set).

This abstraction is used directly in the Quick Start examples and by the API layer.

---

## 7. API Layer and Service Architecture (06_api)

The backend service is implemented using FastAPI.

### 7.1 Application Configuration (app_config.py)

Settings are defined via a pydantic BaseSettings subclass (Settings) with environment‑driven configuration:
- App name, version, debug flag.
- Host, port and reload behavior.
- CORS configuration (origins, methods, headers).
- Authentication toggles and JWT secrets/expiry.
- Default model (bert) and model cache behavior.
- Logging level and request/response logging options.
- Paths for project_root, models_dir and logs_dir.

Environment variables prefixed with CYBERBULLYING_ and an optional .env file are used to override defaults.

### 7.2 Main Application (main.py)

06_api/main.py wires up the FastAPI app with:
- Lifespan management for startup and shutdown using an async context manager:
  - Logs startup information and docs URLs.
  - Initializes the database (init_db/get_db_info in db_helper) and logs DB path.
  - On shutdown, clears the model cache (models_loader.clear_model_cache).

- Application metadata:
  - Title, description and version including model performance summaries.
  - Custom docs URLs (/docs, /redoc, /openapi.json).

- Middleware configuration:
  - CORS via CORSMiddleware.
  - Optional LoggingMiddleware and AuthMiddleware based on config flags.
  - Centralized exception handling setup.

- Route registration:
  - api_router from 06_api/routes/__init__.py mounts sub‑routers for:
    - Health checks (/health).
    - Authentication (/auth/... endpoints for login, token management).
    - Single prediction (/predict) and batch prediction (/predict/batch).
    - Conversation‑level analysis (/predict/conversation).
    - Statistics (/stats) for monitoring usage and label distributions.
    - Feedback (/feedback) for user‑submitted corrections.

- Convenience endpoints:
  - Root (/) summarizing available endpoints and health/docs links.
  - /favicon.ico with a tiny in‑memory PNG to avoid 404s.

The API is runnable via uvicorn directly or by using run_api.py and the Quick Start command.

---

## 8. Database Schema and Persistence (07_database)

The project uses SQLite (local.db at repository root) with a schema defined in 07_database/schema.sql and mirrored via SQLAlchemy models in models.py.

Main tables:

1. messages
   - Stores each original message text and metadata.
   - Fields: id, message_id (external ID), text, source, language, metadata (JSON text), created_at.
   - Indexed on created_at and source.

2. predictions
   - Stores model prediction results linked to messages.
   - Fields: id, prediction_id, message_id (FK → messages.id), model_type, predicted_label, confidence, is_cyberbullying, probabilities (JSON), inference_time_ms, created_at.
   - Indexed by label, model_type, is_cyberbullying, created_at.

3. users
   - User profiles and dashboard authentication.
   - Fields: id, user_id, username, email, password_hash, display_name, role, is_active, risk_score, total_messages, flagged_messages, timestamps.
   - Indexed by username and risk_score.

4. alerts
   - High‑severity incidents that require manual review.
   - Fields: id, alert_id, prediction_id (FK), user_id (FK), severity, status, reason, resolver info, timestamps.
   - Indexed by status, severity, created_at.

5. audit_logs
   - Auditing of API usage and admin actions.
   - Fields: id, log_id, user_id (FK), ip_address, user_agent, action, resource_type, resource_id, request_data (JSON), response_status, created_at.
   - Indexed by action, created_at, and resource (type, id).

6. feedback
   - User feedback on predictions for human‑in‑the‑loop improvement.
   - Fields: id, feedback_id, prediction_id, is_correct, correct_label, comments, created_at.
   - Indexed by prediction_id and is_correct.

This schema supports:
- Full traceability from a prediction back to its source message and user.
- Analytics on labels, models, users and alerts.
- Feedback‑driven model refinement.

---

## 9. Dashboard and Front‑End Integration (08_dashboard)

Folder 08_dashboard provides the scaffold for an operator dashboard:
- assets, components, css, js, pages subfolders indicate a web front‑end for:
  - Visualizing prediction statistics and label distributions.
  - Listing recent cyberbullying incidents and alerts.
  - Inspecting user risk profiles and behavioral patterns.
  - Reviewing and resolving alerts and feedback.

The actual implementation details of the dashboard components are not fully captured in this report, but the folder structure shows that the project is designed for end‑to‑end monitoring and human review.

---

## 10. Deployment, Logging and Utilities

Deployment‑related pieces:
- run_api.py: helper entry for starting the FastAPI service with the correct app module.
- 10_deployment/scripts: shell/PowerShell scripts to support packaging and deployment (e.g., Dockerization, cloud deployment, or service management; specific scripts can be filled or extended as needed).

Logging and utilities:
- 17_logs: central location for training and runtime logs; training scripts append timestamped logs with full configuration and summary.
- 18_utils: reserved for generic helper utilities (currently stubbed with .gitkeep).
- 16_config: for configuration files and environment‑specific overrides.

Quick Start instructions in QUICK_START.md summarize:
- Installation of dependencies via requirements.txt.
- Direct Python usage of CyberbullyingDetector.
- Uvicorn command to run the API and access the interactive docs.

---

## 11. Testing and Quality Assurance

Testing is distributed across multiple modules:
- 01_preprocessing/tests: unit tests for text normalization, emoji handling, code‑mix and transliteration logic.
- 02_feature_extraction/tests: tests for embedding and feature extractors.
- 06_api/tests, 07_database/tests, 08_dashboard/tests, 09_privacy_security/tests, 15_tests: placeholders and/or concrete tests for API routes, DB operations, UI functionality and privacy/security checks.
- test_setup.py at the root: shared testing configuration and fixtures.

This structure allows incremental validation as modules are extended and refactored.

---

## 12. Current Status and Future Work

### 12.1 What Has Been Achieved

- Consolidated multi‑source, multi‑lingual datasets for cyberbullying detection, with strong metadata and anonymization.
- Robust preprocessing tailored to noisy, emoji‑rich, code‑mixed social text (normalization, slang handling, emoji semantics, code‑mix support).
- Rich feature extraction combining transformer embeddings, linguistic features and user‑level behavioral signals.
- Multiple model families implemented, with:
  - Baseline Naive Bayes already trained and validated with near‑perfect metrics.
  - Fine‑tuned transformer classifiers (BERT, mBERT, IndicBERT) trained externally and saved with exceptional performance (up to 100% test F1).
- A unified inference API (CyberbullyingDetector) that hides model‑specific details and exposes a simple Python interface.
- A FastAPI backend exposing prediction, statistics, feedback and auth endpoints, with logging, middleware and CORS configured.
- A normalized relational schema and database helpers to persist messages, predictions, alerts, users, audit logs and feedback.
- Project‑wide logging and a training CLI that makes experimentation reproducible.

### 12.2 Recommended Next Steps

Based on the existing code and documentation:
- Complete full training runs for all baseline models (SVM, TF‑IDF+LogReg), if not already done locally, and compare against transformer models.
- Run and log test‑set evaluations for all deployed models, ensuring no data leakage and verifying generalization on unseen data.
- Harden the API for production:
  - Enable authentication and role‑based access control for sensitive endpoints.
  - Add rate limiting, detailed request logging and monitoring.
- Finalize the dashboard implementation under 08_dashboard to support:
  - Real‑time monitoring of bullying incidents and trends.
  - Efficient triage of alerts and user risk profiles.
- Integrate user feedback loop (07_database.feedback and /feedback endpoint) into model retraining workflows.
- Extend privacy and security checks under 09_privacy_security and add documentation in 12_documentation for compliance considerations.

---

## 13. How to Use This System

For day‑to‑day usage:
- Install dependencies: pip install -r requirements.txt.
- Use CyberbullyingDetector from 03_models/model_loader.py in Python scripts or notebooks for offline analysis.
- Start the FastAPI service (e.g. via uvicorn 06_api.main:app ...) and interact with /docs for online predictions.
- Optionally connect the API to the SQLite database and dashboard for full end‑to‑end operation.

This report summarizes the architecture, pipeline and current capabilities of the project so you can confidently present, extend and deploy the cyberbullying detection system.
