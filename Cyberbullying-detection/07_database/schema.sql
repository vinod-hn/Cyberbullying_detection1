-- Cyberbullying Detection System - SQLite Schema
-- This file is for reference; tables are created via SQLAlchemy ORM

-- Enable foreign keys (SQLite requires this explicitly)
PRAGMA foreign_keys = ON;

-- Messages table: stores original text messages
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    source TEXT DEFAULT 'api',
    language TEXT DEFAULT 'en',
    metadata TEXT,  -- JSON stored as text
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_source ON messages(source);

-- Predictions table: stores model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,
    message_id INTEGER NOT NULL,
    model_type TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    is_cyberbullying INTEGER DEFAULT 0,
    probabilities TEXT,  -- JSON stored as text
    inference_time_ms REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(predicted_label);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_type);
CREATE INDEX IF NOT EXISTS idx_predictions_is_bullying ON predictions(is_cyberbullying);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);

-- Users table: user profiles and dashboard authentication
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE NOT NULL,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT,
    display_name TEXT,
    role TEXT DEFAULT 'viewer',
    is_active INTEGER DEFAULT 1,
    risk_score REAL DEFAULT 0.0,
    total_messages INTEGER DEFAULT 0,
    flagged_messages INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active DATETIME
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_risk_score ON users(risk_score);

-- Alerts table: high-severity predictions requiring attention
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT UNIQUE NOT NULL,
    prediction_id INTEGER NOT NULL,
    user_id INTEGER,
    severity TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending',
    reason TEXT,
    resolved_by TEXT,
    resolved_at DATETIME,
    resolution_notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at DATETIME,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

-- Audit logs table: system activity tracking
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id TEXT UNIQUE NOT NULL,
    user_id INTEGER,
    ip_address TEXT,
    user_agent TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    request_data TEXT,  -- JSON stored as text
    response_status INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id);

-- Feedback table: user feedback on predictions
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id TEXT UNIQUE NOT NULL,
    prediction_id TEXT NOT NULL,
    is_correct INTEGER NOT NULL,
    correct_label TEXT,
    comments TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_prediction ON feedback(prediction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_is_correct ON feedback(is_correct);

-- Sample queries for reference:

-- Get recent cyberbullying predictions
-- SELECT p.*, m.text 
-- FROM predictions p 
-- JOIN messages m ON p.message_id = m.id 
-- WHERE p.is_cyberbullying = 1 
-- ORDER BY p.created_at DESC 
-- LIMIT 10;

-- Get prediction distribution by label
-- SELECT predicted_label, COUNT(*) as count 
-- FROM predictions 
-- GROUP BY predicted_label 
-- ORDER BY count DESC;

-- Get pending alerts with prediction details
-- SELECT a.*, p.predicted_label, p.confidence, m.text 
-- FROM alerts a 
-- JOIN predictions p ON a.prediction_id = p.id 
-- JOIN messages m ON p.message_id = m.id 
-- WHERE a.status = 'pending' 
-- ORDER BY a.created_at DESC;

-- Get user activity stats
-- SELECT 
--     DATE(created_at) as date,
--     COUNT(*) as total_predictions,
--     SUM(CASE WHEN is_cyberbullying = 1 THEN 1 ELSE 0 END) as bullying_count
-- FROM predictions 
-- GROUP BY DATE(created_at) 
-- ORDER BY date DESC;
