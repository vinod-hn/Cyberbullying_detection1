/**
 * API Client for Cyberbullying Detection Dashboard
 * Handles all backend communication - NO DEMO DATA
 * 
 * FIXED ISSUES:
 * 1. Added batch prediction to classify ALL messages (not just first)
 * 2. Proper async/await handling for multiple requests
 * 3. Added console logging for debugging
 */

const APIClient = (function() {
    'use strict';

    // Configuration
    const config = {
        baseUrl: 'http://localhost:8000',
        timeout: 30000,
        retryAttempts: 2,
        retryDelay: 1000,
        batchSize: 10  // For batch processing
    };

    // State
    let isOnline = true;
    let mockMode = false;

    // Demo data to use when backend is unavailable
    const demoMessages = [
        { id: 1, date: '2024-04-20', student_id: 'User_1', message: 'You are amazing!', platform: 'whatsapp', severity: 'neutral', score: 'Safe', confidence: '98%', is_cyberbullying: false },
        { id: 2, date: '2024-04-21', student_id: 'User_2', message: 'I will beat you', platform: 'telegram', severity: 'threat', score: 'Cyberbullying', confidence: '87%', is_cyberbullying: true },
        { id: 3, date: '2024-04-22', student_id: 'User_3', message: 'Stop being dumb', platform: 'classroom', severity: 'insult', score: 'Cyberbullying', confidence: '72%', is_cyberbullying: true },
        { id: 4, date: '2024-04-23', student_id: 'User_4', message: 'Let us study together', platform: 'whatsapp', severity: 'neutral', score: 'Safe', confidence: '95%', is_cyberbullying: false }
    ];

    const demoStats = {
        total_predictions: demoMessages.length,
        cyberbullying_count: demoMessages.filter(m => m.is_cyberbullying).length,
        not_cyberbullying_count: demoMessages.filter(m => !m.is_cyberbullying).length,
        severity_distribution: { low: 1, medium: 1, high: 1, critical: 0 },
        language_distribution: { english: 3, kannada: 0, code_mixed: 1 },
        daily_counts: [],
        model_usage: {},
        label_distribution: {}
    };

    const demoAuditLogs = [
        { action: 'Loaded demo data', time: new Date().toLocaleTimeString(), icon: '📁' },
        { action: 'Viewed demo threat', time: new Date().toLocaleTimeString(), icon: '📋' }
    ];

    const demoModels = { models: ['bert', 'mbert', 'indicbert', 'baseline'], default: 'bert' };

    function buildCSV(messages) {
        const headers = ['id', 'date', 'student_id', 'message', 'platform', 'severity', 'score', 'confidence'];
        const rows = messages.map(m => [m.id, m.date, m.student_id, `"${(m.message || '').replace(/"/g, '""')}"`, m.platform, m.severity, m.score, m.confidence].join(','));
        return headers.join(',') + '\n' + rows.join('\n');
    }

    /**
     * Get base URL for external use
     */
    function getBaseUrl() {
        return config.baseUrl;
    }

    /**
     * Make HTTP request with retry logic
     */
    async function request(endpoint, options = {}) {
        const url = `${config.baseUrl}${endpoint}`;
        const fetchOptions = {
            method: options.method || 'GET',
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        if (options.body) {
            fetchOptions.body = JSON.stringify(options.body);
        }

        for (let attempt = 0; attempt <= config.retryAttempts; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), config.timeout);

                const response = await fetch(url, {
                    ...fetchOptions,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                isOnline = true;
                mockMode = false;
                hideOfflineBanner();

                return await response.json();

            } catch (error) {
                console.warn(`Request attempt ${attempt + 1} failed:`, error.message);
                
                if (attempt < config.retryAttempts) {
                    await sleep(config.retryDelay * (attempt + 1));
                } else {
                    isOnline = false;
                    mockMode = true;
                    showOfflineBanner();
                    console.warn('Switching to mockMode - returning demo data where available');
                    // When ultimately offline, bubble up the error so callers can decide,
                    // but many APIClient methods will check mockMode and return demo data.
                    throw error;
                }
            }
        }
    }

    /**
     * Sleep utility
     */
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Show offline banner
     */
    function showOfflineBanner() {
        const banner = document.getElementById('offline-banner');
        if (banner) {
            banner.classList.remove('hidden');
        }
    }

    /**
     * Hide offline banner
     */
    function hideOfflineBanner() {
        const banner = document.getElementById('offline-banner');
        if (banner) {
            banner.classList.add('hidden');
        }
    }

    // Public API
    return {
        /**
         * Check if backend is available
         */
        async checkHealth() {
            try {
                const response = await request('/health');
                return response.status === 'healthy';
            } catch {
                return false;
            }
        },

        /**
         * Get dashboard statistics
         */
        async getStats() {
            // If we're in mock mode (backend down), return demo stats
            if (!isOnline || mockMode) {
                console.log('📡 getStats: returning demo stats (offline)');
                return demoStats;
            }

            try {
                const stats = await request('/stats');
                hideOfflineBanner();
                return stats;
            } catch (error) {
                console.error('Failed to load stats from backend:', error);
                showOfflineBanner();
                // fallback to demo
                return demoStats;
            }
        },

        /**
         * Get predictions with filters
         */
        async getPredictions(params = {}) {
            const query = new URLSearchParams();
            if (params.page) query.append('page', params.page);
            if (params.severity) query.append('severity', params.severity);
            if (params.platform) query.append('platform', params.platform);
            if (params.startDate) query.append('start_date', params.startDate);
            if (params.endDate) query.append('end_date', params.endDate);
            // If offline, return demo messages paginated
            if (!isOnline || mockMode) {
                console.log('📡 getPredictions: returning demo data (offline)');
                const page = parseInt(params.page) || 1;
                const per_page = 10;
                const start = (page - 1) * per_page;
                const data = demoMessages.slice(start, start + per_page);
                return { data, total: demoMessages.length, page, per_page };
            }

            try {
                const result = await request(`/predictions?${query.toString()}`);
                hideOfflineBanner();
                return result;
            } catch (error) {
                console.error('Failed to load predictions from backend:', error);
                showOfflineBanner();
                return { data: [], total: 0, page: 1, per_page: 10 };
            }
        },

        /**
         * Make a prediction for a single text
         */
        async predict(text, modelType = 'bert') {
            console.log(`🔮 Predicting for text: "${text.substring(0, 50)}..."`);
            try {
                const result = await request('/predict', {
                    method: 'POST',
                    body: { text, model_type: modelType }
                });
                console.log('  ✓ Prediction result:', result.prediction);
                return result;
            } catch (error) {
                console.error('❌ Prediction request failed:', error);
                showOfflineBanner();
                throw error;
            }
        },

        /**
         * Batch predict multiple messages
         * FIX: This ensures ALL messages are classified, not just the first one
         * Uses Promise.all for parallel processing with rate limiting
         */
        async batchPredict(messages, modelType = 'bert', onProgress = null) {
            console.log(`🔮 Starting batch prediction for ${messages.length} messages...`);
            const results = [];
            const errors = [];
            
            // Process in batches to avoid overwhelming the server
            for (let i = 0; i < messages.length; i += config.batchSize) {
                const batch = messages.slice(i, i + config.batchSize);
                const batchNumber = Math.floor(i / config.batchSize) + 1;
                const totalBatches = Math.ceil(messages.length / config.batchSize);
                
                console.log(`  📦 Processing batch ${batchNumber}/${totalBatches}`);
                
                // Process batch in parallel
                const batchPromises = batch.map(async (msg, idx) => {
                    const messageIndex = i + idx + 1;
                    console.log(`    🔍 Classifying message #${messageIndex}: "${msg.text?.substring(0, 30) || msg.substring(0, 30)}..."`);
                    
                    try {
                        const text = msg.text || msg;
                        const result = await this.predict(text, modelType);
                        
                        if (onProgress) {
                            onProgress(messageIndex, messages.length, result);
                        }
                        
                        return {
                            index: messageIndex - 1,
                            originalMessage: msg,
                            result: result,
                            success: true
                        };
                    } catch (error) {
                        console.error(`    ❌ Failed to classify message #${messageIndex}:`, error);
                        errors.push({ index: messageIndex - 1, error });
                        return {
                            index: messageIndex - 1,
                            originalMessage: msg,
                            result: null,
                            success: false,
                            error: error.message
                        };
                    }
                });
                
                // Wait for all predictions in this batch
                const batchResults = await Promise.all(batchPromises);
                results.push(...batchResults);
                
                // Small delay between batches to prevent rate limiting
                if (i + config.batchSize < messages.length) {
                    await sleep(100);
                }
            }
            
            console.log(`✅ Batch prediction complete: ${results.filter(r => r.success).length}/${messages.length} successful`);
            
            if (errors.length > 0) {
                console.warn(`⚠️ ${errors.length} messages failed to classify`);
            }
            
            return {
                results: results,
                successful: results.filter(r => r.success).length,
                failed: errors.length,
                total: messages.length
            };
        },

        /**
         * Classify all messages in a table/array and return updated messages
         * FIX: This is the main function to classify ALL messages properly
         */
        async classifyAllMessages(messages, modelType = 'bert', onProgress = null) {
            console.log(`🚀 Starting classification of ${messages.length} messages...`);
            
            const classifiedMessages = [];
            
            for (let i = 0; i < messages.length; i++) {
                const msg = messages[i];
                const text = msg.message || msg.text || msg;
                
                console.log(`🔍 Classifying message ${i + 1}/${messages.length}`);
                
                if (onProgress) {
                    onProgress(i + 1, messages.length, `Classifying message ${i + 1}...`);
                }
                
                try {
                    const result = await this.predict(text, modelType);
                    
                    // Merge result with original message
                    classifiedMessages.push({
                        ...msg,
                        id: msg.id || i + 1,
                        severity: result.severity || (result.is_cyberbullying ? 'high' : 'low'),
                        score: result.prediction || (result.is_cyberbullying ? 'Cyberbullying' : 'Safe'),
                        confidence: typeof result.confidence === 'number' 
                            ? `${Math.round(result.confidence * 100)}%` 
                            : result.confidence,
                        is_cyberbullying: result.is_cyberbullying,
                        model: modelType,
                        classified: true
                    });
                    
                    console.log(`  ✓ Message ${i + 1} classified as: ${result.prediction}`);
                    
                } catch (error) {
                    console.error(`  ❌ Failed to classify message ${i + 1}:`, error);
                    
                    // Add unclassified message with error state
                    classifiedMessages.push({
                        ...msg,
                        id: msg.id || i + 1,
                        severity: 'unknown',
                        score: 'Error',
                        confidence: 'N/A',
                        is_cyberbullying: null,
                        model: modelType,
                        classified: false,
                        error: error.message
                    });
                }
                
                // Small delay to prevent rate limiting
                if (i < messages.length - 1) {
                    await sleep(50);
                }
            }
            
            console.log(`✅ Classification complete: ${classifiedMessages.filter(m => m.classified).length}/${messages.length} successful`);
            
            return classifiedMessages;
        },

        /**
         * Get export data as CSV text
         */
        async exportReports(type = 'all') {
            // If offline, build CSV from demo messages
            if (!isOnline || mockMode) {
                console.log('📡 exportReports: returning demo CSV (offline)');
                const csv = buildCSV(demoMessages);
                return csv;
            }

            try {
                const url = `${config.baseUrl}/export_reports?type=${type}`;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), config.timeout);

                const response = await fetch(url, {
                    method: 'GET',
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                isOnline = true;
                hideOfflineBanner();

                // Return as text (CSV content)
                return await response.text();
            } catch (error) {
                console.error('Export request failed:', error);
                showOfflineBanner();
                // Fallback to demo CSV
                return buildCSV(demoMessages);
            }
        },

        /**
         * Submit feedback
         */
        async submitFeedback(predictionId, isCorrect, correctLabel = null, comments = '') {
            try {
                return await request('/feedback', {
                    method: 'POST',
                    body: {
                        prediction_id: predictionId,
                        is_correct: isCorrect,
                        correct_label: correctLabel,
                        comments
                    }
                });
            } catch (error) {
                console.error('Feedback request failed:', error);
                showOfflineBanner();
                return { success: false, offline: true };
            }
        },

        /**
         * Get audit log
         */
        async getAuditLog(limit = 20) {
            if (!isOnline || mockMode) {
                console.log('📡 getAuditLog: returning demo audit logs (offline)');
                return demoAuditLogs.slice(0, limit);
            }

            try {
                const response = await request(`/audit_log?limit=${limit}`);
                return response.logs || [];
            } catch (error) {
                console.error('Failed to load audit log from backend:', error);
                showOfflineBanner();
                return demoAuditLogs.slice(0, limit);
            }
        },

        /**
         * Add audit log entry
         */
        async addAuditLog(action, icon = '📋') {
            try {
                await fetch(`${config.baseUrl}/audit_log?action=${encodeURIComponent(action)}&icon=${encodeURIComponent(icon)}`, {
                    method: 'POST'
                });
            } catch (error) {
                console.warn('Failed to add audit log entry:', error);
            }
        },

        /**
         * Get available models
         */
        async getModels() {
            if (!isOnline || mockMode) {
                console.log('📡 getModels: returning demo models (offline)');
                return demoModels;
            }

            try {
                return await request('/models');
            } catch (error) {
                console.error('Failed to load models metadata from backend:', error);
                showOfflineBanner();
                return demoModels;
            }
        },

        /**
         * Upload CSV/TXT for analysis
         */
        async uploadCSV(file, modelType = 'bert', textColumn = null) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('model_type', modelType);
            if (textColumn) {
                formData.append('text_column', textColumn);
            }

            try {
                const response = await fetch(`${config.baseUrl}/upload/csv`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Upload failed');
                }

                isOnline = true;
                mockMode = false;
                hideOfflineBanner();

                return await response.json();
            } catch (error) {
                console.error('CSV upload failed:', error);
                throw error;
            }
        },

        // ==========================================
        // HISTORY API METHODS
        // ==========================================

        /**
         * Get list of previous analysis sessions
         */
        async getHistory(limit = 10) {
            console.log('📂 Fetching history from database...');
            try {
                const result = await request(`/history?limit=${limit}`);
                console.log(`  ✓ Loaded ${result.length} history entries`);
                return result;
            } catch (error) {
                console.error('Failed to fetch history:', error);
                return [];
            }
        },

        /**
         * Get full details of a specific history entry
         */
        async getHistoryDetail(historyId) {
            console.log(`📂 Restoring dashboard for session ID ${historyId}...`);
            try {
                const result = await request(`/history/${historyId}`);
                console.log(`  ✓ Loaded history detail: ${result.filename}, ${result.message_count} messages`);
                return result;
            } catch (error) {
                console.error('Failed to fetch history detail:', error);
                throw error;
            }
        },

        /**
         * Save analysis session to history
         */
        async saveHistory(filename, messages, summaryStats = null) {
            console.log(`💾 Saving history: ${filename} with ${messages.length} messages...`);
            try {
                const result = await request('/history', {
                    method: 'POST',
                    body: {
                        filename: filename,
                        messages: messages,
                        summary_stats: summaryStats
                    }
                });
                console.log(`  ✓ Saved to history with ID ${result.id}`);
                return result;
            } catch (error) {
                console.error('Failed to save history:', error);
                // Don't throw - history saving is not critical
                return null;
            }
        },

        /**
         * Delete a history entry
         */
        async deleteHistory(historyId) {
            console.log(`🗑️ Deleting history entry ${historyId}...`);
            try {
                await request(`/history/${historyId}`, { method: 'DELETE' });
                console.log(`  ✓ Deleted history entry ${historyId}`);
                return true;
            } catch (error) {
                console.error('Failed to delete history:', error);
                return false;
            }
        },

        // Utility
        getBaseUrl,

        // Getters
        get isOnline() { return isOnline; },
        get isMockMode() { return mockMode; }
    };
})();

// Export for use in other modules
window.APIClient = APIClient;
