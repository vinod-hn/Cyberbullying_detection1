/**
 * CSV Upload Module for Cyberbullying Detection Dashboard
 * Handles file upload, validation, preview, and analysis
 */

const CSVUpload = (function() {
    'use strict';

    // State
    let state = {
        selectedFile: null,
        fileData: null,
        analysisResults: null,
        isProcessing: false
    };

    // DOM Elements cache
    let elements = {};

    /**
     * Initialize CSV Upload module
     */
    function init() {
        console.log('Initializing CSV Upload module...');
        cacheElements();
        setupEventListeners();
    }

    /**
     * Cache DOM elements
     */
    function cacheElements() {
        elements = {
            // Buttons
            uploadBtn: document.getElementById('upload-csv-btn'),
            analyzeBtn: document.getElementById('analyze-btn'),
            cancelBtn: document.getElementById('cancel-upload'),
            removeFileBtn: document.getElementById('remove-file'),
            retryBtn: document.getElementById('retry-btn'),
            loadResultsBtn: document.getElementById('load-results-btn'),
            closeModalBtn: document.getElementById('close-csv-modal'),
            
            // Modal
            modal: document.getElementById('csv-upload-modal'),
            
            // Sections
            uploadSection: document.getElementById('upload-section'),
            previewSection: document.getElementById('preview-section'),
            processingSection: document.getElementById('processing-section'),
            resultsSection: document.getElementById('results-section'),
            errorSection: document.getElementById('error-section'),
            
            // Upload elements
            dropzone: document.getElementById('dropzone'),
            fileInput: document.getElementById('csv-file-input'),
            
            // File info
            fileName: document.getElementById('selected-file-name'),
            fileSize: document.getElementById('selected-file-size'),
            
            // Selects
            modelSelect: document.getElementById('model-select'),
            textColumnSelect: document.getElementById('text-column-select'),
            
            // Preview table
            previewThead: document.getElementById('preview-thead'),
            previewTbody: document.getElementById('preview-tbody'),
            
            // Progress
            progressFill: document.getElementById('progress-fill'),
            progressText: document.getElementById('progress-text'),
            
            // Results
            resultTotal: document.getElementById('result-total'),
            resultCyberbullying: document.getElementById('result-cyberbullying'),
            resultSafe: document.getElementById('result-safe'),
            resultTime: document.getElementById('result-time'),
            
            // Severity bars
            barCritical: document.getElementById('bar-critical'),
            barHigh: document.getElementById('bar-high'),
            barMedium: document.getElementById('bar-medium'),
            barLow: document.getElementById('bar-low'),
            countCritical: document.getElementById('count-critical'),
            countHigh: document.getElementById('count-high'),
            countMedium: document.getElementById('count-medium'),
            countLow: document.getElementById('count-low'),
            
            // Error
            errorMessage: document.getElementById('error-message')
        };
    }

    /**
     * Setup event listeners
     */
    function setupEventListeners() {
        // Open modal
        if (elements.uploadBtn) {
            elements.uploadBtn.addEventListener('click', openModal);
        }

        // Close modal
        if (elements.closeModalBtn) {
            elements.closeModalBtn.addEventListener('click', closeModal);
        }
        if (elements.cancelBtn) {
            elements.cancelBtn.addEventListener('click', closeModal);
        }
        if (elements.modal) {
            elements.modal.addEventListener('click', (e) => {
                if (e.target === elements.modal) closeModal();
            });
        }

        // Dropzone
        if (elements.dropzone) {
            elements.dropzone.addEventListener('click', () => elements.fileInput?.click());
            elements.dropzone.addEventListener('dragover', handleDragOver);
            elements.dropzone.addEventListener('dragleave', handleDragLeave);
            elements.dropzone.addEventListener('drop', handleDrop);
        }

        // File input
        if (elements.fileInput) {
            elements.fileInput.addEventListener('change', handleFileSelect);
        }

        // Remove file
        if (elements.removeFileBtn) {
            elements.removeFileBtn.addEventListener('click', removeFile);
        }

        // Analyze
        if (elements.analyzeBtn) {
            elements.analyzeBtn.addEventListener('click', analyzeCSV);
        }

        // Retry
        if (elements.retryBtn) {
            elements.retryBtn.addEventListener('click', resetToUpload);
        }

        // Load results
        if (elements.loadResultsBtn) {
            elements.loadResultsBtn.addEventListener('click', loadResultsToDashboard);
        }

        // Keyboard
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !elements.modal?.classList.contains('hidden')) {
                closeModal();
            }
        });
    }

    /**
     * Open upload modal
     */
    function openModal() {
        elements.modal?.classList.remove('hidden');
        resetToUpload();
        document.body.style.overflow = 'hidden';
    }

    /**
     * Close upload modal
     */
    function closeModal() {
        elements.modal?.classList.add('hidden');
        document.body.style.overflow = '';
        if (!state.isProcessing) {
            resetState();
        }
    }

    /**
     * Reset to upload section
     */
    function resetToUpload() {
        hideAllSections();
        elements.uploadSection?.classList.remove('hidden');
        elements.analyzeBtn?.classList.remove('hidden');
        elements.loadResultsBtn?.classList.add('hidden');
        if (elements.analyzeBtn) elements.analyzeBtn.disabled = true;
    }

    /**
     * Reset state
     */
    function resetState() {
        state = {
            selectedFile: null,
            fileData: null,
            analysisResults: null,
            isProcessing: false
        };
        if (elements.fileInput) elements.fileInput.value = '';
    }

    /**
     * Hide all sections
     */
    function hideAllSections() {
        elements.uploadSection?.classList.add('hidden');
        elements.previewSection?.classList.add('hidden');
        elements.processingSection?.classList.add('hidden');
        elements.resultsSection?.classList.add('hidden');
        elements.errorSection?.classList.add('hidden');
    }

    /**
     * Handle drag over
     */
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        elements.dropzone?.classList.add('dragover');
    }

    /**
     * Handle drag leave
     */
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        elements.dropzone?.classList.remove('dragover');
    }

    /**
     * Handle drop
     */
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        elements.dropzone?.classList.remove('dragover');

        const files = e.dataTransfer?.files;
        if (files && files.length > 0) {
            processFile(files[0]);
        }
    }

    /**
     * Handle file select from input
     */
    function handleFileSelect(e) {
        const files = e.target.files;
        if (files && files.length > 0) {
            processFile(files[0]);
        }
    }

    /**
     * Process selected file
     */
    async function processFile(file) {
        // Validate file type (CSV or TXT)
        const fileName = file.name.toLowerCase();
        if (!fileName.endsWith('.csv') && !fileName.endsWith('.txt')) {
            showError('Invalid file format. Please upload a CSV or TXT file.');
            return;
        }

        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            showError('File too large. Maximum size is 10MB.');
            return;
        }

        state.selectedFile = file;

        // Update file info
        if (elements.fileName) {
            elements.fileName.textContent = file.name;
        }
        if (elements.fileSize) {
            elements.fileSize.textContent = `(${formatFileSize(file.size)})`;
        }

        // Parse and preview CSV
        try {
            await parseCSV(file);
            showPreview();
        } catch (error) {
            showError(`Failed to parse CSV: ${error.message}`);
        }
    }

    /**
     * Parse CSV or TXT file
     */
    async function parseCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            const isTxt = file.name.toLowerCase().endsWith('.txt');
            
            reader.onload = (e) => {
                try {
                    const text = e.target.result;
                    const lines = text.split('\n').filter(line => line.trim());
                    
                    if (lines.length < 1) {
                        reject(new Error('File must have at least one line of data'));
                        return;
                    }

                    let headers, data;

                    if (isTxt) {
                        // TXT file: each line is a message
                        headers = ['line_number', 'message'];
                        data = lines.slice(0, 100).map((line, idx) => ({
                            line_number: idx + 1,
                            message: line.trim()
                        }));
                    } else {
                        // CSV file: parse headers and rows
                        if (lines.length < 2) {
                            reject(new Error('CSV must have at least a header and one data row'));
                            return;
                        }

                        headers = parseCSVLine(lines[0]);
                        data = [];
                        for (let i = 1; i < lines.length && i <= 100; i++) {
                            const values = parseCSVLine(lines[i]);
                            if (values.length === headers.length) {
                                const row = {};
                                headers.forEach((h, idx) => {
                                    row[h] = values[idx];
                                });
                                data.push(row);
                            }
                        }
                    }

                    state.fileData = {
                        headers,
                        data,
                        totalRows: isTxt ? lines.length : lines.length - 1,
                        isTxt
                    };

                    // Populate text column select
                    populateTextColumnSelect(headers);

                    resolve(state.fileData);
                } catch (err) {
                    reject(err);
                }
            };

            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    /**
     * Parse a single CSV line
     */
    function parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;

        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        result.push(current.trim());
        
        return result;
    }

    /**
     * Populate text column select with auto-detection
     */
    function populateTextColumnSelect(headers) {
        if (!elements.textColumnSelect) return;

        // Priority text column names
        const textColumnPriority = ['message', 'text', 'content', 'chat', 'comment', 'msg', 'body', 'post', 'tweet'];
        // Metadata columns to skip
        const metadataColumns = ['name', 'username', 'user', 'sender', 'from', 'author', 'time', 'timestamp', 
                                  'datetime', 'date', 'created_at', 'id', 'chat_id', 'message_id', 'student_id',
                                  'user_id', 'platform', 'source', 'phone', 'email', 'number'];

        elements.textColumnSelect.innerHTML = '<option value="">Auto-detect</option>';
        
        let detectedColumn = null;
        
        headers.forEach(header => {
            const option = document.createElement('option');
            option.value = header;
            option.textContent = header;
            elements.textColumnSelect.appendChild(option);
            
            // Auto-detect best text column
            const headerLower = header.toLowerCase();
            if (!detectedColumn) {
                if (textColumnPriority.some(tc => headerLower.includes(tc))) {
                    detectedColumn = header;
                }
            }
        });

        // If detected, pre-select it
        if (detectedColumn) {
            elements.textColumnSelect.value = detectedColumn;
            console.log(`Auto-detected text column: ${detectedColumn}`);
        }
    }

    /**
     * Show preview section
     */
    function showPreview() {
        hideAllSections();
        elements.previewSection?.classList.remove('hidden');
        
        if (elements.analyzeBtn) {
            elements.analyzeBtn.disabled = false;
        }

        renderPreviewTable();
    }

    /**
     * Render preview table (highlighting the message column)
     */
    function renderPreviewTable() {
        if (!state.fileData || !elements.previewThead || !elements.previewTbody) return;

        const { headers, data, isTxt } = state.fileData;
        
        // Get the selected/detected text column
        const selectedCol = elements.textColumnSelect?.value || '';
        const textColumnPriority = ['message', 'text', 'content', 'chat', 'comment', 'msg', 'body'];
        
        // Determine which column is the message column
        let messageCol = selectedCol;
        if (!messageCol) {
            for (const col of textColumnPriority) {
                if (headers.map(h => h.toLowerCase()).includes(col)) {
                    messageCol = headers[headers.map(h => h.toLowerCase()).indexOf(col)];
                    break;
                }
            }
        }

        // Render headers (highlight message column)
        elements.previewThead.innerHTML = `
            <tr>
                ${headers.map(h => `<th class="${h === messageCol ? 'highlight-col' : ''}">${escapeHtml(h)}${h === messageCol ? ' ✓' : ''}</th>`).join('')}
            </tr>
        `;

        // Render first 5 rows (highlight message column)
        const previewData = data.slice(0, 5);
        elements.previewTbody.innerHTML = previewData.map(row => `
            <tr>
                ${headers.map(h => `<td class="${h === messageCol ? 'highlight-col' : ''}">${escapeHtml(truncate(row[h] || '', 50))}</td>`).join('')}
            </tr>
        `).join('');
    }

    /**
     * Remove selected file
     */
    function removeFile() {
        resetState();
        resetToUpload();
    }

    /**
     * Analyze CSV
     * FIX: Added proper logging and ensures ALL messages get classified
     */
    async function analyzeCSV() {
        if (!state.selectedFile || state.isProcessing) return;

        console.log('🔍 Starting CSV analysis...');
        state.isProcessing = true;
        
        hideAllSections();
        elements.processingSection?.classList.remove('hidden');
        elements.analyzeBtn?.classList.add('hidden');

        // Progress tracking
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 5;
            if (progress > 85) progress = 85;
            updateProgress(progress, 'Analyzing messages...');
        }, 300);

        try {
            const formData = new FormData();
            formData.append('file', state.selectedFile);
            formData.append('model_type', elements.modelSelect?.value || 'bert');
            
            const textColumn = elements.textColumnSelect?.value;
            if (textColumn) {
                formData.append('text_column', textColumn);
            }

            console.log('📤 Uploading file to backend for analysis...');
            const response = await fetch(`${APIClient.getBaseUrl()}/upload/csv`, {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Analysis failed');
            }

            const results = await response.json();
            
            // Verify ALL messages were classified
            console.log(`✅ Analysis complete. Checking results...`);
            console.log(`   Total rows: ${results.analyzed_rows}`);
            console.log(`   Results returned: ${results.results?.length || 0}`);
            
            // FIX: Ensure all messages have classification results
            if (results.results && results.results.length > 0) {
                const classifiedCount = results.results.filter(r => r.prediction !== undefined).length;
                const unclassifiedCount = results.results.length - classifiedCount;
                
                console.log(`   Classified: ${classifiedCount}`);
                if (unclassifiedCount > 0) {
                    console.warn(`   ⚠️ Unclassified: ${unclassifiedCount} - These may need re-processing`);
                }
            }
            
            state.analysisResults = results;

            updateProgress(100, 'Complete!');
            console.log('✅ CSV analysis finished successfully');
            
            setTimeout(() => {
                showResults(results);
            }, 500);

        } catch (error) {
            clearInterval(progressInterval);
            console.error('Analysis error:', error);
            showError(error.message || 'Failed to analyze CSV');
        } finally {
            state.isProcessing = false;
        }
    }

    /**
     * Update progress bar
     */
    function updateProgress(percent, text) {
        if (elements.progressFill) {
            elements.progressFill.style.width = `${percent}%`;
        }
        if (elements.progressText) {
            elements.progressText.textContent = text;
        }
    }

    /**
     * Show results
     */
    function showResults(results) {
        hideAllSections();
        elements.resultsSection?.classList.remove('hidden');
        elements.loadResultsBtn?.classList.remove('hidden');
        
        // Update summary stats
        if (elements.resultTotal) {
            elements.resultTotal.textContent = results.analyzed_rows;
        }
        if (elements.resultCyberbullying) {
            elements.resultCyberbullying.textContent = results.cyberbullying_count;
        }
        if (elements.resultSafe) {
            elements.resultSafe.textContent = results.not_cyberbullying_count;
        }
        if (elements.resultTime) {
            elements.resultTime.textContent = `${Math.round(results.processing_time_ms)}ms`;
        }

        // Update severity bars
        const severity = results.severity_distribution || {};
        const total = results.analyzed_rows || 1;

        updateSeverityBar('critical', severity.critical || 0, total);
        updateSeverityBar('high', severity.high || 0, total);
        updateSeverityBar('medium', severity.medium || 0, total);
        updateSeverityBar('low', severity.low || 0, total);
    }

    /**
     * Update severity bar
     */
    function updateSeverityBar(level, count, total) {
        const barEl = elements[`bar${capitalize(level)}`];
        const countEl = elements[`count${capitalize(level)}`];
        
        const percent = (count / total) * 100;
        
        if (barEl) {
            barEl.style.width = `${percent}%`;
        }
        if (countEl) {
            countEl.textContent = count;
        }
    }

    /**
     * Load results to dashboard
     * FIX: Now ensures all messages have proper classification data
     */
    function loadResultsToDashboard() {
        if (!state.analysisResults) {
            console.error('❌ No analysis results to load');
            return;
        }

        console.log('📊 Loading results to dashboard...');
        const results = state.analysisResults;

        // Transform results for dashboard - ensure ALL messages are processed
        const messages = results.results.map((r, idx) => {
            // FIX: Ensure each message has proper classification fields
            const message = {
                id: r.row_id || idx + 1,
                date: new Date().toISOString().split('T')[0],
                student_id: r.student_id || `User_${r.row_id || idx + 1}`,
                message: r.original_text || r.text || '',
                platform: r.platform || 'csv_upload',
                severity: r.severity || (r.is_cyberbullying ? 'high' : 'low'),
                score: r.prediction || (r.is_cyberbullying ? 'Cyberbullying' : 'Safe'),
                confidence: typeof r.confidence === 'number' 
                    ? `${Math.round(r.confidence * 100)}%` 
                    : (r.confidence || 'N/A'),
                prediction: r.prediction,
                model: r.model_used || 'bert',
                is_cyberbullying: r.is_cyberbullying,
                raw_confidence: r.confidence
            };
            
            console.log(`  ✓ Message ${idx + 1}: ${message.severity} (${message.confidence})`);
            return message;
        });

        console.log(`📋 Processed ${messages.length} messages for dashboard`);
        
        // Verify all messages have classifications
        const classified = messages.filter(m => m.severity !== 'unknown' && m.score !== 'N/A');
        const unclassified = messages.filter(m => m.severity === 'unknown' || m.score === 'N/A');
        
        if (unclassified.length > 0) {
            console.warn(`⚠️ ${unclassified.length} messages may not be fully classified`);
        }
        console.log(`✅ ${classified.length}/${messages.length} messages fully classified`);

        // Update dashboard with new data
        if (window.Dashboard) {
            Dashboard.loadCSVResults(messages, results);
        }

        // Close modal
        closeModal();

        // Show notification
        showNotification(`✅ Loaded ${messages.length} analyzed messages to dashboard`);
    }

    /**
     * Show error
     */
    function showError(message) {
        hideAllSections();
        elements.errorSection?.classList.remove('hidden');
        
        if (elements.errorMessage) {
            elements.errorMessage.textContent = message;
        }
        
        elements.analyzeBtn?.classList.remove('hidden');
        if (elements.analyzeBtn) elements.analyzeBtn.disabled = true;
    }

    /**
     * Show notification
     */
    function showNotification(message) {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = 'toast-notification';
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('show');
        }, 100);

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // Utility functions
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function truncate(str, len) {
        if (!str) return '';
        return str.length > len ? str.substring(0, len) + '...' : str;
    }

    function capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    // Public API
    return {
        init,
        openModal,
        closeModal,
        getResults: () => state.analysisResults
    };
})();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', CSVUpload.init);

// Export for other modules
window.CSVUpload = CSVUpload;
