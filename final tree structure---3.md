cyberbullying-detection/
│
├── README.md                                    # Updated: Local setup instructions
├── LICENSE                                      # UNCHANGED
├── .gitignore                                   # Added: .venv/, local.db, logs/
├── requirements.txt                             # Added: fastapi, uvicorn[standard]
├── setup.py                                     # UNCHANGED
│
├── .venv/                                       # NEW: Local virtual env (git ignored)
│   ├── bin/ (Linux/Mac) or Scripts/ (Windows)
│   ├── Lib/ or lib/
│   └── pyvenv.cfg
│
├── local.db                                     # NEW: SQLite DB (git ignored)
├── api.log                                      # NEW: FastAPI logs (git ignored)
│
├── 📁 00_data/                                  # UNCHANGED (2.1GB datasets)
│   ├── 📁 raw/
│   │   ├── kannada-english.csv                 # 5000+ code-mixed rows
│   │   ├── english.csv                         # 3000+ English rows
│   │   ├── kannada.csv                         # 2000+ Kannada rows
│   │   ├── emoji_cyberbullying_dataset.csv     # Emoji-enriched
│   │   └── bad_words.csv                       # Profanity dictionary
│   │
│   ├── 📁 processed/
│   │   ├── train_data.csv                      # 70% split (8000 rows)
│   │   ├── val_data.csv                        # 15% split (1700 rows)
│   │   ├── test_data.csv                       # 15% split (1700 rows)
│   │   ├── train_augmented.csv                 # SMOTE augmented
│   │   └── class_distribution.json             # Label stats
│   │
│   ├── 📁 lexicon/
│   │   ├── kannada_slang.csv                   # 1200+ Kannada slang terms
│   │   ├── english_slang.csv                   # 2500+ English slang
│   │   ├── profanity_kannada.csv               # 450+ Kannada profanity
│   │   ├── profanity_english.csv               # 1800+ English profanity
│   │   ├── emoji_semantics.json                # 500+ emoji mappings
│   │   └── code_mix_patterns.txt               # Romanized Kannada patterns
│   │
│   ├── 📁 anonymized/
│   │   ├── anonymized_train.csv
│   │   ├── anonymized_val.csv
│   │   ├── anonymized_test.csv
│   │   ├── id_mapping_log.txt                  # Encrypted PII mapping
│   │   └── anonymization_report.md
│   │
│   └── 📁 metadata/
│       ├── annotation_guidelines.md
│       ├── label_distribution.json
│       ├── inter_annotator_agreement.txt       # Kappa=0.82
│       └── data_quality_report.md
│
├── 📁 01_preprocessing/                         # UNCHANGED
│   ├── __init__.py
│   ├── text_normalizer.py
│   ├── emoji_handler.py
│   ├── transliterator.py
│   ├── code_mix_processor.py
│   ├── slang_expander.py
│   ├── conversation_threader.py
│   ├── 📁 tests/
│   │   ├── test_normalizer.py
│   │   ├── test_emoji_handler.py
│   │   ├── test_transliterator.py
│   │   └── test_code_mix.py
│   └── preprocessing_config.json
│
├── 📁 02_feature_extraction/                    # UNCHANGED
│   ├── __init__.py
│   ├── text_embedder.py
│   ├── transformer_embedder.py
│   ├── contextual_features.py
│   ├── linguistic_features.py
│   ├── behavioral_features.py
│   ├── emoji_features.py
│   ├── sarcasm_detector.py
│   ├── 📁 embeddings/
│   │   ├── tfidf_vectorizer.pkl               # 2.3MB
│   │   ├── word2vec_model.bin                 # 45MB
│   │   └── fasttext_model.bin                 # 78MB
│   ├── 📁 tests/
│   │   ├── test_embedder.py
│   │   ├── test_contextual_features.py
│   │   └── test_linguistic_features.py
│   └── feature_config.json
│
├── 📁 03_models/                                # UNCHANGED (450MB models)
│   ├── __init__.py
│   ├── 📁 baseline/
│   │   ├── tfidf_classifier.py
│   │   ├── naive_bayes_classifier.py
│   │   ├── svm_classifier.py
│   │   └── train_baseline.py
│   ├── 📁 transformer/
│   │   ├── bert_message_classifier.py
│   │   ├── mbert_classifier.py
│   │   ├── indicbert_classifier.py
│   │   └── train_transformer.py
│   ├── 📁 context_aware/
│   │   ├── conversation_transformer.py
│   │   ├── lstm_context_model.py
│   │   ├── attention_mechanism.py
│   │   └── train_context_model.py
│   ├── 📁 user_behavior/
│   │   ├── user_risk_model.py
│   │   ├── temporal_patterns.py
│   │   └── campaign_detection.py
│   ├── 📁 ensemble/
│   │   ├── ensemble_model.py
│   │   ├── multi_task_learning.py
│   │   └── weight_calibrator.py
│   └── 📁 saved_models/
│       ├── baseline_tfidf/
│       │   ├── model.pkl                      # 2.3MB
│       │   └── metadata.json
│       ├── mbert_finetuned/
│       │   ├── pytorch_model.bin              # 420MB
│       │   ├── config.json
│       │   └── tokenizer_config.json
│       ├── conversation_model/
│       │   └── checkpoint.pt                  # 28MB
│       └── ensemble_model/
│           └── weights.pkl                    # 1.8MB
│
├── 📁 04_evaluation/                            # UNCHANGED
│   ├── __init__.py
│   ├── metrics.py
│   ├── cross_validator.py
│   ├── threshold_optimizer.py
│   ├── ablation_study.py
│   ├── confusion_matrix_analyzer.py
│   ├── severity_calibration.py
│   ├── 📁 results/
│   │   ├── baseline_results.json
│   │   ├── transformer_results.json
│   │   ├── context_model_results.json
│   │   ├── ensemble_results.json
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   ├── pr_curves.png
│   │   ├── ablation_study.md
│   │   └── comparative_analysis.md
│   └── 📁 tests/
│       ├── test_metrics.py
│       └── test_validator.py
│
├── 📁 05_severity_scoring/                      # RENAMED (was 06)
│   ├── __init__.py
│   ├── severity_classifier.py
│   ├── confidence_scorer.py
│   ├── risk_aggregator.py
│   ├── escalation_detector.py
│   ├── 📁 severity_models/
│   │   ├── severity_classifier.pkl
│   │   └── calibration_params.json
│   └── 📁 tests/
│       └── test_severity_scoring.py
│
├── 📁 06_api/                                   # RENAMED (was 07) - FastAPI core
│   ├── __init__.py
│   ├── main.py                                 # uvicorn 06_api.main:app
│   ├── app_config.py                           # Modified: local_config=True
│   ├── models_loader.py
│   ├── 📁 routes/
│   │   ├── __init__.py
│   │   ├── predict.py                         # POST /predict
│   │   ├── batch_predict.py                   # POST /batch-predict
│   │   ├── conversation_predict.py            # POST /conversation-predict
│   │   ├── statistics.py                      # GET /stats (for graphs)
│   │   ├── health.py                          # GET /health
│   │   ├── feedback.py                        # POST /feedback
│   │   └── auth.py                            # Local JWT
│   ├── 📁 middleware/
│   │   ├── __init__.py
│   │   ├── auth_middleware.py
│   │   ├── logging_middleware.py              # Logs to 17_logs/
│   │   └── error_handler.py
│   ├── 📁 schemas/
│   │   ├── __init__.py
│   │   ├── request_schemas.py
│   │   └── response_schemas.py
│   ├── 📁 tests/
│   │   ├── test_api_routes.py
│   │   ├── test_authentication.py
│   │   └── test_performance.py
│   └── requirements_api.txt
│
├── 📁 07_database/                              # RENAMED (was 08) - SQLite focus
│   ├── __init__.py
│   ├── db_config.py                            # sqlite:///./local.db
│   ├── models.py
│   ├── schema.sql
│   ├── 📁 repositories/
│   │   ├── __init__.py
│   │   ├── message_repository.py
│   │   ├── prediction_repository.py
│   │   ├── user_repository.py
│   │   ├── alert_repository.py
│   │   └── audit_log_repository.py
│   └── 📁 tests/
│       └── test_database.py
│   # REMOVED: migrations/ folder (SQLite no need)
│
├── 📁 08_dashboard/                             # RENAMED (was 09) + ENHANCED GRAPHS
│   ├── index.html
│   ├── login.html
│   ├── 📁 pages/
│   │   ├── dashboard.html                     # Main dashboard with graphs
│   │   ├── conversation_viewer.html
│   │   ├── analytics.html                     # Detailed analytics graphs
│   │   ├── reports.html
│   │   ├── user_profiles.html
│   │   ├── intervention_log.html
│   │   └── settings.html
│   ├── 📁 css/
│   │   ├── main.css
│   │   ├── dashboard.css
│   │   ├── graphs.css                         # NEW: Graph styling
│   │   ├── responsive.css
│   │   └── theme.css
│   ├── 📁 js/
│   │   ├── main.js
│   │   ├── api_client.js                      # localhost:8000 calls
│   │   ├── authentication.js
│   │   ├── dashboard.js
│   │   ├── 📁 graphs/                          # NEW: Simple graph components
│   │   │   ├── pie_chart.js                   # Bullying type distribution
│   │   │   ├── bar_chart.js                   # Severity level counts
│   │   │   ├── line_chart.js                  # Trends over time
│   │   │   ├── donut_chart.js                 # Language distribution
│   │   │   └── stats_cards.js                 # Summary stat cards
│   │   ├── chart_config.js                    # NEW: Chart.js configuration
│   │   ├── export_reports.js
│   │   └── utils.js
│   ├── 📁 assets/
│   │   ├── 📁 images/
│   │   ├── 📁 fonts/
│   │   └── animations.css
│   ├── 📁 components/                          # NEW: Reusable UI components
│   │   ├── graph_container.html               # Graph wrapper template
│   │   ├── stat_card.html                     # Stat card template
│   │   └── legend.html                        # Graph legend template
│   ├── package.json                           # Added: chart.js dependency
│   └── 📁 tests/
│       └── test_dashboard_ui.py
│
├── 📁 09_privacy_security/                      # RENAMED (was 10) - SIMPLIFIED
│   ├── __init__.py
│   ├── anonymizer.py
│   ├── encryptor.py
│   ├── key_manager.py
│   ├── access_control.py
│   ├── audit_logger.py
│   ├── privacy_policy.md
│   ├── security_guidelines.md
│   └── 📁 tests/
│       ├── test_anonymization.py
│       ├── test_encryption.py
│       └── test_access_control.py
│   # REMOVED: federated_learning/, on_device/ (optional)
│
├── 📁 10_deployment/                            # RENAMED (was 11) - LOCAL TESTING ONLY
│   ├── local_setup.md                          # NEW: Complete guide
│   ├── local_run.bat                           # NEW: Windows one-click
│   ├── local_run.sh                            # NEW: Linux/Mac one-click
│   ├── 📁 scripts/
│   │   ├── setup_local.sh                      # NEW: venv + pip
│   │   ├── run_api.sh                          # NEW: uvicorn --reload
│   │   ├── run_tests.sh                        # NEW: pytest 15_tests/
│   │   ├── db_init.sh                          # NEW: sqlite3 local.db
│   │   └── health_check.py                     # UNCHANGED
│   └── local_checklist.md                      # NEW: Verification steps
│   # REMOVED: docker/, kubernetes/, cloud/ folders
│
├── 📁 11_notebooks/                             # RENAMED (was 12) - SIMPLIFIED
│   ├── 📁 01_eda/
│   ├── 📁 02_feature_engineering/
│   ├── 📁 03_model_training/
│   ├── 📁 04_evaluation/
│   └── 📁 05_deployment/                        # Renumbered (was 06)
│
├── 📁 12_documentation/                         # RENAMED (was 13)
├── 📁 13_experiments/                           # RENAMED (was 14)
├── 📁 14_scripts/                               # RENAMED (was 15)
├── 📁 15_tests/                                 # RENAMED (was 16)
│   ├── conftest.py
│   ├── test_end_to_end.py
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   ├── test_api_integration.py
│   └── test_dashboard_integration.py
│
├── 📁 16_config/                                # RENAMED (was 17) + local_config.yaml
│   ├── config.yaml
│   ├── logging_config.json
│   ├── model_params.yaml
│   ├── preprocessing_params.yaml
│   ├── deployment_config.yaml
│   └── local_config.yaml                       # NEW
│
├── 📁 17_logs/                                  # RENAMED (was 18)
│   ├── training.log
│   ├── evaluation.log
│   ├── api.log
│   └── dashboard.log
│
├── 📁 18_utils/                                 # RENAMED (was 19)
│   ├── __init__.py
│   ├── logger.py
│   ├── file_handler.py
│   ├── json_handler.py
│   ├── visualization.py
│   └── constants.py
│
└── .vscode/                                     # NEW: VS Code config
    ├── launch.json                             # FastAPI debug F5
    └── settings.json                           # Python path
