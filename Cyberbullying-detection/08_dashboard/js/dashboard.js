/**
 * Dashboard Main Controller
 * Orchestrates all dashboard components and handles user interactions
 * 
 * FIXED ISSUES:
 * 1. Buttons not working - Added proper event listeners with console logs
 * 2. History not saving - Implemented localStorage persistence
 * 3. State resetting on navigation - Auto-save and restore implemented
 */

const Dashboard = (function() {
    'use strict';

    // Storage key for localStorage persistence
    const STORAGE_KEY = 'cyberbullying_dashboard_state';
    const HISTORY_KEY = 'cyberbullying_history';
    
    // Default state structure
    const defaultState = {
        currentPage: 1,
        pageSize: 10,
        totalPages: 1,
        sortColumn: 'score',
        sortDirection: 'desc',
        filters: {
            startDate: '2024-01-01',
            endDate: '2024-04-24',
            platform: '',
            severity: ''
        },
        messages: [],
        selectedMessage: null,
        currentTab: 'current',  // Track active tab
        lastUpdated: null
    };

    // State - initialized from localStorage or defaults
    let state = loadState() || { ...defaultState };

    // Platform icons (inline SVG data)
    const platformIcons = {
        whatsapp: '<svg viewBox="0 0 24 24" width="20" height="20" fill="#25D366"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z"/></svg>',
        telegram: '<svg viewBox="0 0 24 24" width="20" height="20" fill="#0088cc"><path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/></svg>',
        classroom: '<svg viewBox="0 0 24 24" width="20" height="20" fill="#4285F4"><path d="M12 3c1.73 0 3.14 1.41 3.14 3.14S13.73 9.29 12 9.29 8.86 7.87 8.86 6.14 10.27 3 12 3m0 15.43c2.27 0 4.86-1.14 4.86-2.57V14.3c0-1.64-3.27-2.57-4.86-2.57S7.14 12.66 7.14 14.3v1.56c0 1.43 2.59 2.57 4.86 2.57M5.57 8.86C6.82 8.86 7.86 9.9 7.86 11.14s-1.04 2.29-2.29 2.29c-1.24 0-2.28-1.04-2.28-2.29 0-1.24 1.04-2.28 2.28-2.28m1.72 7.71v-1.43c0-.87.39-1.66 1-2.28-.17-.02-.35-.03-.53-.03-1.07 0-3.18.53-3.18 1.59v2.15h2.71m11.14-7.71c1.24 0 2.29 1.04 2.29 2.28 0 1.25-1.05 2.29-2.29 2.29s-2.28-1.04-2.28-2.29c0-1.24 1.04-2.28 2.28-2.28m2.29 7.71v-2.15c0-1.06-2.11-1.59-3.18-1.59-.18 0-.36.01-.53.03.61.62 1 1.41 1 2.28v1.43h2.71z"/></svg>'
    };

    // ===========================================
    // STATE PERSISTENCE FUNCTIONS
    // FIX: History was resetting because state was only in memory
    // Now using localStorage to persist across page loads/navigation
    // ===========================================

    /**
     * Save current state to localStorage
     */
    function saveState() {
        try {
            const stateToSave = {
                ...state,
                lastUpdated: new Date().toISOString()
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(stateToSave));
            console.log('💾 State saved to localStorage', { 
                messages: state.messages.length,
                page: state.currentPage,
                tab: state.currentTab 
            });
        } catch (error) {
            console.error('Failed to save state:', error);
        }
    }

    /**
     * Load state from localStorage
     */
    function loadState() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                const parsed = JSON.parse(saved);
                console.log('📂 State loaded from localStorage', {
                    messages: parsed.messages?.length || 0,
                    lastUpdated: parsed.lastUpdated
                });
                return parsed;
            }
        } catch (error) {
            console.error('Failed to load state:', error);
        }
        return null;
    }

    /**
     * Restore UI elements from saved state
     */
    function restoreUIFromState() {
        console.log('🔄 Restoring UI from saved state...');
        
        // Restore filter values
        const startDate = document.getElementById('start-date');
        const endDate = document.getElementById('end-date');
        const platformFilter = document.getElementById('platform-filter');
        const severityFilter = document.getElementById('severity-filter');
        
        if (startDate && state.filters.startDate) {
            startDate.value = state.filters.startDate;
        }
        if (endDate && state.filters.endDate) {
            endDate.value = state.filters.endDate;
        }
        if (platformFilter && state.filters.platform) {
            platformFilter.value = state.filters.platform;
        }
        if (severityFilter && state.filters.severity) {
            severityFilter.value = state.filters.severity;
        }
        
        // Restore active tab
        if (state.currentTab) {
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.dataset.tab === state.currentTab) {
                    btn.classList.add('active');
                }
            });
        }
        
        // Render table with saved messages
        if (state.messages && state.messages.length > 0) {
            renderTable();
            updatePagination();
            console.log(`✅ Restored ${state.messages.length} messages from history`);
        }
    }

    /**
     * Save message to history (both localStorage and database)
     */
    async function saveToHistory(messages, filename = null) {
        try {
            // Save to localStorage (fallback)
            const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
            const timestamp = new Date().toISOString();
            const generatedFilename = filename || `upload_${timestamp.replace(/[:.]/g, '-')}.csv`;
            
            const newEntry = {
                timestamp: timestamp,
                filename: generatedFilename,
                messages: messages,
                filters: { ...state.filters }
            };
            history.unshift(newEntry);
            // Keep last 10 history entries
            if (history.length > 10) {
                history.pop();
            }
            localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
            console.log('📚 Saved to localStorage history:', messages.length, 'messages');
            
            // Also save to database via API
            try {
                // Build summary stats
                const severityCounts = { low: 0, medium: 0, high: 0, critical: 0 };
                let cbCount = 0;
                let safeCount = 0;
                
                messages.forEach(msg => {
                    const sev = (msg.severity || 'low').toLowerCase();
                    if (severityCounts[sev] !== undefined) {
                        severityCounts[sev]++;
                    }
                    if (msg.is_cyberbullying || (msg.severity !== 'low' && msg.severity !== 'neutral')) {
                        cbCount++;
                    } else {
                        safeCount++;
                    }
                });
                
                const summaryStats = {
                    total: messages.length,
                    cyberbullying: cbCount,
                    safe: safeCount,
                    severity_distribution: severityCounts
                };
                
                await APIClient.saveHistory(generatedFilename, messages, summaryStats);
                console.log('✅ Saved to database history via API');
                
            } catch (apiError) {
                console.warn('⚠️ Could not save to database (API may be offline):', apiError.message);
            }
            
        } catch (error) {
            console.error('Failed to save to history:', error);
        }
    }

    /**
     * Get history entries
     */
    function getHistory() {
        try {
            return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        } catch (error) {
            console.error('Failed to load history:', error);
            return [];
        }
    }

    /**
     * Clear all saved state
     */
    function clearState() {
        localStorage.removeItem(STORAGE_KEY);
        localStorage.removeItem(HISTORY_KEY);
        state = { ...defaultState };
        console.log('🗑️ State cleared');
    }

    /**
     * Initialize dashboard
     */
    async function init() {
        console.log('🚀 Initializing Dashboard...');

        // Apply Chart.js defaults
        ChartConfig.applyGlobalDefaults();

        // Initialize charts
        PieChart.init();
        LineChart.init();
        BarChart.init();

        // Setup event listeners
        setupEventListeners();
        
        // Restore UI from saved state first
        restoreUIFromState();

        // Load initial data (only if no saved messages or if we need fresh data)
        if (!state.messages || state.messages.length === 0) {
            await loadData();
        } else {
            console.log('📊 Using restored state, skipping initial data load');
            // Still update charts with existing data
            try {
                const stats = await APIClient.getStats();
                updateCharts(stats);
            } catch (error) {
                console.warn('Could not refresh stats:', error);
            }
        }
        
        // Save state on page unload
        window.addEventListener('beforeunload', saveState);
        
        // Auto-save periodically
        setInterval(saveState, 30000); // Save every 30 seconds

        console.log('✅ Dashboard initialized');
    }

    /**
     * Setup all event listeners
     * FIX: Added console logs and ensured all buttons have proper handlers
     */
    function setupEventListeners() {
        console.log('🔧 Setting up event listeners...');
        
        // Apply filters button
        // FIX: Button wasn't providing feedback - now logs actions
        const applyBtn = document.getElementById('apply-filters');
        if (applyBtn) {
            applyBtn.addEventListener('click', function(e) {
                e.preventDefault();
                console.log('🔍 Apply Filters button clicked');
                handleApplyFilters();
            });
            console.log('  ✓ Apply Filters button listener attached');
        } else {
            console.warn('  ✗ Apply Filters button not found!');
        }

        // Date inputs
        const startDate = document.getElementById('start-date');
        const endDate = document.getElementById('end-date');
        if (startDate) {
            startDate.addEventListener('change', function() {
                console.log('📅 Start date changed:', this.value);
                updateFilters();
                saveState();
            });
        }
        if (endDate) {
            endDate.addEventListener('change', function() {
                console.log('📅 End date changed:', this.value);
                updateFilters();
                saveState();
            });
        }

        // Filter selects - save state on change and immediately apply filters
        const platformFilter = document.getElementById('platform-filter');
        const severityFilter = document.getElementById('severity-filter');
        const severityAll = document.getElementById('severity-all');
        
        if (platformFilter) {
            platformFilter.addEventListener('change', function() {
                console.log('🔗 Platform filter changed:', this.value);
                updateFilters();
                state.currentPage = 1;
                renderTable();
                updatePagination();
                updateChartsFromMessages();
                saveState();
            });
        }
        if (severityFilter) {
            severityFilter.addEventListener('change', function() {
                console.log('⚠️ Severity filter changed:', this.value);
                updateFilters();
                state.currentPage = 1;
                renderTable();
                updatePagination();
                updateChartsFromMessages();
                saveState();
            });
        }
        if (severityAll) {
            severityAll.addEventListener('change', function() {
                console.log('📋 Severity-all filter changed:', this.value);
                updateFilters();
                state.currentPage = 1;
                renderTable();
                updatePagination();
                updateChartsFromMessages();
                saveState();
            });
        }

        // Tab navigation - FIX: preserve state when switching tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                console.log('📑 Tab clicked:', this.dataset.tab);
                handleTabClick(e);
            });
        });
        console.log('  ✓ Tab button listeners attached');

        // Table sorting
        document.querySelectorAll('.data-table th.sortable').forEach(th => {
            th.addEventListener('click', function(e) {
                console.log('🔀 Sort column clicked:', this.dataset.sort);
                handleSort(e);
            });
        });

        // Table row click (open modal)
        const tbody = document.getElementById('messages-tbody');
        if (tbody) {
            tbody.addEventListener('click', handleRowClick);
        }

        // Pagination - FIX: Now preserves state
        const prevPage = document.getElementById('prev-page');
        const nextPage = document.getElementById('next-page');
        if (prevPage) {
            prevPage.addEventListener('click', function(e) {
                e.preventDefault();
                console.log('⬅️ Previous page clicked, current:', state.currentPage);
                changePage(-1);
            });
            console.log('  ✓ Previous page button listener attached');
        }
        if (nextPage) {
            nextPage.addEventListener('click', function(e) {
                e.preventDefault();
                console.log('➡️ Next page clicked, current:', state.currentPage);
                changePage(1);
            });
            console.log('  ✓ Next page button listener attached');
        }

        // Page number buttons
        document.querySelectorAll('.page-num').forEach(btn => {
            btn.addEventListener('click', handlePageClick);
        });

        // Export button and dropdown
        const exportBtn = document.getElementById('export-btn');
        const exportMenu = document.getElementById('export-menu');
        if (exportBtn && exportMenu) {
            exportBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('📤 Export button clicked');
                exportMenu.classList.toggle('hidden');
            });
            console.log('  ✓ Export button listener attached');

            document.querySelectorAll('.dropdown-item').forEach(item => {
                item.addEventListener('click', function(e) {
                    console.log('📥 Export option clicked:', this.dataset.export);
                    handleExport(e);
                });
            });

            // Close dropdown on outside click
            document.addEventListener('click', () => {
                exportMenu.classList.add('hidden');
            });
        } else {
            console.warn('  ✗ Export button or menu not found - will create dynamically');
            createExportButton();
        }

        // Modal
        const closeModal = document.getElementById('close-modal');
        const modalOverlay = document.getElementById('message-modal');
        const dismissBtn = document.getElementById('modal-dismiss');

        if (closeModal) closeModal.addEventListener('click', closeMessageModal);
        if (dismissBtn) dismissBtn.addEventListener('click', closeMessageModal);
        if (modalOverlay) {
            modalOverlay.addEventListener('click', (e) => {
                if (e.target === modalOverlay) closeMessageModal();
            });
        }

        // Chart navigation
        const prevWeek = document.getElementById('prev-week');
        const nextWeek = document.getElementById('next-week');
        if (prevWeek) {
            prevWeek.addEventListener('click', function() {
                console.log('📊 Previous week clicked');
                LineChart.previousWeek();
            });
        }
        if (nextWeek) {
            nextWeek.addEventListener('click', function() {
                console.log('📊 Next week clicked');
                LineChart.nextWeek();
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', handleKeyboard);

        // Search box
        const searchBox = document.getElementById('table-search');
        if (searchBox) {
            searchBox.addEventListener('input', debounce(handleSearch, 300));
        }
        
        // Clear state button (for debugging/reset)
        const clearStateBtn = document.getElementById('clear-state-btn');
        if (clearStateBtn) {
            clearStateBtn.addEventListener('click', function(e) {
                e.preventDefault();
                console.log('🗑️ Clear state button clicked');
                if (confirm('This will clear all saved data and history. Are you sure?')) {
                    clearState();
                    location.reload();
                }
            });
            console.log('  ✓ Clear state button listener attached');
        }
        
        // Confidence filter
        const confidenceFilter = document.getElementById('confidence-filter');
        if (confidenceFilter) {
            confidenceFilter.addEventListener('change', function() {
                console.log('📊 Confidence filter changed:', this.value);
                updateFilters();
                state.currentPage = 1;
                renderTable();
                updatePagination();
                updateChartsFromMessages();
                saveState();
            });
            console.log('  ✓ Confidence filter listener attached');
        }
        
        // History dropdown
        setupHistoryDropdown();
        
        console.log('✅ All event listeners set up');
    }
    
    /**
     * Setup history dropdown functionality
     */
    function setupHistoryDropdown() {
        const historyBtn = document.getElementById('history-dropdown-btn');
        const historyMenu = document.getElementById('history-menu');
        const refreshBtn = document.getElementById('refresh-history');
        
        if (historyBtn && historyMenu) {
            historyBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('📂 History dropdown clicked');
                historyMenu.classList.toggle('hidden');
                
                // Load history when dropdown opens
                if (!historyMenu.classList.contains('hidden')) {
                    loadHistoryDropdown();
                }
            });
            
            // Close on outside click
            document.addEventListener('click', function(e) {
                if (!historyMenu.contains(e.target) && e.target !== historyBtn) {
                    historyMenu.classList.add('hidden');
                }
            });
            
            console.log('  ✓ History dropdown listener attached');
        }
        
        if (refreshBtn) {
            refreshBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('🔄 Refresh history clicked');
                loadHistoryDropdown();
            });
        }
    }
    
    /**
     * Load history items into dropdown
     */
    async function loadHistoryDropdown() {
        const historyList = document.getElementById('history-list');
        if (!historyList) return;
        
        historyList.innerHTML = '<div class="history-loading" style="padding: 20px; text-align: center; color: #666;">Loading history...</div>';
        
        try {
            console.log('📂 Loading history from database...');
            const history = await APIClient.getHistory(10);
            
            if (!history || history.length === 0) {
                historyList.innerHTML = `
                    <div class="history-empty">
                        <div class="history-empty-icon">📭</div>
                        <p>No previous sessions found</p>
                        <p style="font-size: 12px;">Upload a chat file to get started</p>
                    </div>
                `;
                return;
            }
            
            historyList.innerHTML = history.map(entry => `
                <div class="history-item" data-history-id="${entry.id}" onclick="Dashboard.loadHistoryEntry(${entry.id})">
                    <div class="history-item-header">
                        <span class="history-filename">📄 ${entry.filename}</span>
                        <span class="history-date">${formatHistoryDate(entry.uploaded_at)}</span>
                    </div>
                    <div class="history-stats">
                        <span class="history-stat">
                            📝 ${entry.message_count} messages
                        </span>
                        <span class="history-stat stat-cb">
                            ⚠️ ${entry.cyberbullying_count} flagged
                        </span>
                        <span class="history-stat stat-safe">
                            ✅ ${entry.neutral_count} safe
                        </span>
                    </div>
                </div>
            `).join('');
            
            console.log(`  ✓ Loaded ${history.length} history entries`);
            
        } catch (error) {
            console.error('Failed to load history:', error);
            historyList.innerHTML = `
                <div class="history-empty">
                    <div class="history-empty-icon">❌</div>
                    <p>Failed to load history</p>
                    <p style="font-size: 12px;">${error.message}</p>
                </div>
            `;
        }
    }
    
    /**
     * Load a specific history entry and restore dashboard
     */
    async function loadHistoryEntry(historyId) {
        console.log(`📂 Restoring dashboard for session ID ${historyId}...`);
        
        // Close dropdown
        const historyMenu = document.getElementById('history-menu');
        if (historyMenu) historyMenu.classList.add('hidden');
        
        try {
            const detail = await APIClient.getHistoryDetail(historyId);
            
            if (!detail || !detail.json_results) {
                throw new Error('No data found for this session');
            }
            
            // Update state with restored data
            state.messages = detail.json_results;
            state.currentPage = 1;
            state.totalPages = Math.ceil(detail.json_results.length / state.pageSize);
            
            console.log(`  ✓ Restored ${detail.json_results.length} messages from history`);
            
            // Update UI
            renderTable();
            updatePagination();
            
            // Update charts
            if (detail.summary_stats) {
                updateCharts(detail.summary_stats);
            } else {
                updateChartsFromMessages();
            }
            
            // Update interventions
            StatsCards.updateInterventions(state.messages);
            
            // Add audit entry
            StatsCards.addAuditEntry(`Loaded history: ${detail.filename}`, '📂');
            
            // Save state
            saveState();
            
            // Show success message
            showToast(`Loaded: ${detail.filename} (${detail.message_count} messages)`);
            
            console.log('✅ Dashboard restored from history');
            
        } catch (error) {
            console.error('Failed to load history entry:', error);
            showToast('Failed to load history: ' + error.message, 'error');
        }
    }
    
    /**
     * Format date for history display
     */
    function formatHistoryDate(dateStr) {
        try {
            const date = new Date(dateStr);
            const now = new Date();
            const diffMs = now - date;
            const diffMins = Math.floor(diffMs / 60000);
            const diffHours = Math.floor(diffMs / 3600000);
            const diffDays = Math.floor(diffMs / 86400000);
            
            if (diffMins < 1) return 'Just now';
            if (diffMins < 60) return `${diffMins}m ago`;
            if (diffHours < 24) return `${diffHours}h ago`;
            if (diffDays < 7) return `${diffDays}d ago`;
            
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        } catch {
            return dateStr;
        }
    }
    
    /**
     * Show toast notification
     */
    function showToast(message, type = 'success') {
        let toast = document.querySelector('.toast-notification');
        if (!toast) {
            toast = document.createElement('div');
            toast.className = 'toast-notification';
            document.body.appendChild(toast);
        }
        
        toast.textContent = message;
        toast.style.background = type === 'error' ? '#e74c3c' : '#27ae60';
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    /**
     * Create export button dynamically if missing from HTML
     */
    function createExportButton() {
        const controlsLeft = document.querySelector('.controls-left');
        if (!controlsLeft) return;
        
        // Check if button already exists
        if (document.getElementById('export-btn')) return;
        
        const exportWrapper = document.createElement('div');
        exportWrapper.className = 'export-dropdown';
        exportWrapper.innerHTML = `
            <button id="export-btn" class="btn btn-secondary">
                📤 Export
            </button>
            <div id="export-menu" class="dropdown-menu hidden">
                <button class="dropdown-item" data-export="csv">📄 Export as CSV</button>
                <button class="dropdown-item" data-export="pdf">📑 Export as PDF</button>
                <button class="dropdown-item" data-export="all">📊 Export All Data</button>
            </div>
        `;
        
        controlsLeft.appendChild(exportWrapper);
        
        // Attach listeners
        const exportBtn = document.getElementById('export-btn');
        const exportMenu = document.getElementById('export-menu');
        
        exportBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('📤 Export button clicked');
            exportMenu.classList.toggle('hidden');
        });
        
        document.querySelectorAll('.dropdown-item').forEach(item => {
            item.addEventListener('click', handleExport);
        });
        
        document.addEventListener('click', () => {
            exportMenu.classList.add('hidden');
        });
        
        console.log('  ✓ Export button created dynamically');
    }

    /**
     * Load all dashboard data
     * FIX: Now saves state after loading
     */
    async function loadData() {
        console.log('📡 Loading dashboard data...');
        try {
            // Load stats
            const stats = await APIClient.getStats();
            updateCharts(stats);
            console.log('  ✓ Stats loaded');

            // Load messages
            const predictions = await APIClient.getPredictions({
                page: state.currentPage,
                severity: state.filters.severity,
                platform: state.filters.platform,
                startDate: state.filters.startDate,
                endDate: state.filters.endDate
            });

            state.messages = predictions.data || predictions;
            state.totalPages = Math.ceil((predictions.total || state.messages.length) / state.pageSize);
            
            console.log(`  ✓ Loaded ${state.messages.length} messages`);

            renderTable();
            updatePagination();

            // Update sidebar
            StatsCards.updateInterventions(state.messages);
            
            // Fetch audit log from backend
            StatsCards.fetchAuditLog();
            
            // Save state after loading new data
            saveState();
            console.log('✅ Dashboard data loaded and state saved');

        } catch (error) {
            console.error('❌ Error loading data:', error);
        }
    }

    /**
     * Update all charts with stats data
     * FIX: Ensure charts are reinitialized with proper data
     */
    function updateCharts(stats) {
        console.log('📊 Updating charts with stats:', stats);
        
        // Ensure stats has proper structure
        if (!stats || typeof stats !== 'object') {
            console.warn('⚠️ No stats provided for charts');
            return;
        }
        
        // Update donut/pie chart with label distribution
        if (stats.label_distribution && Object.keys(stats.label_distribution).length > 0) {
            console.log('  📊 Pie chart: label_distribution =', stats.label_distribution);
            PieChart.updateFromStats(stats);
        } else if (stats.severity_distribution) {
            // Fallback to severity distribution
            console.log('  📊 Pie chart: Using severity_distribution fallback');
            PieChart.updateFromStats({ label_distribution: stats.severity_distribution });
        }
        
        // Update line chart with daily data or generate from current session
        if (stats.daily_counts && stats.daily_counts.length > 0) {
            LineChart.updateFromStats(stats);
        } else if (stats.total_predictions > 0) {
            // Generate daily data for current session
            const today = new Date();
            const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
            const currentDay = days[today.getDay()];
            
            // Show today's data only
            const dailyStats = {
                daily_counts: [
                    { date: today.toISOString().split('T')[0], cyberbullying: stats.cyberbullying_count || 0 }
                ]
            };
            LineChart.updateFromStats(dailyStats);
        }
        
        // Update bar chart with monthly trend
        if (stats.monthly_trend && Object.keys(stats.monthly_trend).length > 0) {
            BarChart.updateFromStats(stats);
        } else if (stats.total_predictions > 0) {
            // Show current month data
            const currentMonth = new Date().toLocaleString('en-US', { month: 'short' });
            BarChart.updateFromStats({
                monthly_trend: { [currentMonth]: stats.total_predictions }
            });
        }
        
        console.log('✅ Charts updated');
    }

    /**
     * Render the messages table
     * FIX: Now properly paginates local data with filters applied
     */
    function renderTable() {
        const tbody = document.getElementById('messages-tbody');
        if (!tbody) return;

        // Apply local filters first
        const filtered = applyLocalFilters([...state.messages]);
        
        // Sort messages
        const sorted = sortMessages(filtered);
        
        // Update total pages based on filtered count
        state.totalPages = Math.ceil(sorted.length / state.pageSize) || 1;
        
        // Update count display
        const filteredCountEl = document.getElementById('filtered-count');
        const totalCountEl = document.getElementById('total-count');
        if (filteredCountEl) filteredCountEl.textContent = sorted.length;
        if (totalCountEl) totalCountEl.textContent = state.messages.length;
        
        // Calculate pagination slice for local data
        const startIndex = (state.currentPage - 1) * state.pageSize;
        const endIndex = startIndex + state.pageSize;
        const paginatedMessages = sorted.slice(startIndex, endIndex);
        
        console.log(`📋 Rendering table: showing ${paginatedMessages.length} of ${sorted.length} filtered messages (page ${state.currentPage})`);

        if (paginatedMessages.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" style="text-align: center; padding: 40px; color: #666;">
                        <p>📭 No messages to display.</p>
                        <p>Try uploading a CSV file or adjusting your filters.</p>
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = paginatedMessages.map(msg => {
            // Handle missing or undefined values gracefully
            const severity = msg.severity || 'unknown';
            const score = msg.score || 'N/A';
            const platform = msg.platform || 'unknown';
            
            // Fix confidence display - prevent double percent
            let confidence = msg.confidence || 'N/A';
            if (typeof confidence === 'string') {
                confidence = confidence.replace(/%%/g, '%'); // Remove double percent
            }
            
            return `
            <tr data-id="${msg.id}" role="button" tabindex="0" aria-label="View message details">
                <td>${msg.date || 'N/A'}</td>
                <td><strong>${msg.student_id || 'Unknown'}</strong></td>
                <td class="message-text">${truncateText(msg.message || '', 40)}</td>
                <td>
                    <span class="platform-badge platform-${platform}">
                        ${platformIcons[platform] || ''}
                        ${capitalize(platform)}
                    </span>
                </td>
                <td>
                    <span class="severity-badge severity-${severity.toLowerCase()}">
                        ${severity.toUpperCase()}
                    </span>
                </td>
                <td class="score-value">${score}</td>
                <td class="confidence-value">${confidence}</td>
            </tr>
        `}).join('');
    }

    /**
     * Sort messages based on current state
     */
    function sortMessages(messages) {
        return messages.sort((a, b) => {
            let valueA = a[state.sortColumn];
            let valueB = b[state.sortColumn];

            // Handle date sorting
            if (state.sortColumn === 'date') {
                valueA = new Date(valueA);
                valueB = new Date(valueB);
            }

            if (state.sortDirection === 'asc') {
                return valueA > valueB ? 1 : -1;
            } else {
                return valueA < valueB ? 1 : -1;
            }
        });
    }

    /**
     * Update pagination UI
     */
    function updatePagination() {
        const pageNumbers = document.getElementById('page-numbers');
        if (!pageNumbers) return;

        let html = '';
        const totalPages = Math.max(state.totalPages, 1);

        // Generate page numbers
        for (let i = 1; i <= Math.min(4, totalPages); i++) {
            html += `<button class="page-num ${i === state.currentPage ? 'active' : ''}">${i}</button>`;
        }

        if (totalPages > 5) {
            html += '<span class="page-ellipsis">..</span>';
            html += '<span class="page-ellipsis">..</span>';
            html += `<button class="page-num ${totalPages === state.currentPage ? 'active' : ''}">${totalPages}</button>`;
        }

        pageNumbers.innerHTML = html;

        // Re-attach event listeners
        pageNumbers.querySelectorAll('.page-num').forEach(btn => {
            btn.addEventListener('click', handlePageClick);
        });
    }

    /**
     * Handle filter updates - now saves state and includes score type filter and confidence filter
     */
    function updateFilters() {
        state.filters.startDate = document.getElementById('start-date')?.value || '';
        state.filters.endDate = document.getElementById('end-date')?.value || '';
        state.filters.platform = document.getElementById('platform-filter')?.value || '';
        state.filters.severity = document.getElementById('severity-filter')?.value || '';
        state.filters.scoreType = document.getElementById('severity-all')?.value || '';
        state.filters.confidence = document.getElementById('confidence-filter')?.value || '';
        console.log('🔧 Filters updated:', state.filters);
        saveState();
    }

    /**
     * Apply filters to local messages array
     */
    function applyLocalFilters(messages) {
        if (!messages || messages.length === 0) return [];
        
        return messages.filter(msg => {
            // Filter by severity level
            if (state.filters.severity) {
                const msgSeverity = (msg.severity || '').toLowerCase();
                if (msgSeverity !== state.filters.severity.toLowerCase()) {
                    return false;
                }
            }
            
            // Filter by score type (insult, threat, harassment, neutral)
            if (state.filters.scoreType) {
                const msgScore = (msg.score || msg.prediction || '').toLowerCase();
                if (msgScore !== state.filters.scoreType.toLowerCase()) {
                    return false;
                }
            }
            
            // Filter by platform
            if (state.filters.platform) {
                const msgPlatform = (msg.platform || '').toLowerCase();
                if (msgPlatform !== state.filters.platform.toLowerCase()) {
                    return false;
                }
            }
            
            // Filter by confidence score range
            if (state.filters.confidence) {
                // Parse confidence value (handle both "85%" and 0.85 formats)
                let confValue = msg.confidence;
                if (typeof confValue === 'string') {
                    confValue = parseFloat(confValue.replace('%', ''));
                } else if (typeof confValue === 'number' && confValue <= 1) {
                    confValue = confValue * 100;
                }
                
                if (isNaN(confValue)) return true; // Don't filter if confidence is unknown
                
                const [min, max] = state.filters.confidence.split('-').map(Number);
                if (confValue < min || confValue > max) {
                    return false;
                }
            }
            
            return true;
        });
    }

    /**
     * Handle apply filters button
     * FIX: Now filters local data without server reload
     */
    async function handleApplyFilters() {
        console.log('🔍 Applying filters...', state.filters);
        
        // Update filters from UI
        updateFilters();
        
        // Show loading indicator
        const applyBtn = document.getElementById('apply-filters');
        if (applyBtn) {
            applyBtn.disabled = true;
            applyBtn.textContent = 'Filtering...';
        }
        
        try {
            // Reset to first page when filtering
            state.currentPage = 1;
            
            // Re-render table with filters applied
            renderTable();
            updatePagination();
            
            // Update charts based on filtered data
            updateChartsFromMessages();
            
            StatsCards.addAuditEntry(`Filtered: ${state.filters.severity || state.filters.scoreType || 'all'}`, '🔍');
            console.log('✅ Filters applied successfully');
        } catch (error) {
            console.error('❌ Error applying filters:', error);
        } finally {
            if (applyBtn) {
                applyBtn.disabled = false;
                applyBtn.textContent = 'Apply';
            }
        }
    }
    
    /**
     * Update charts from current messages data
     */
    function updateChartsFromMessages() {
        const filtered = applyLocalFilters([...state.messages]);
        
        // Build stats from messages
        const severityCounts = { low: 0, medium: 0, high: 0, critical: 0 };
        const labelCounts = {};
        let cbCount = 0;
        let safeCount = 0;
        
        filtered.forEach(msg => {
            // Count by severity
            const sev = (msg.severity || 'low').toLowerCase();
            if (severityCounts[sev] !== undefined) {
                severityCounts[sev]++;
            }
            
            // Count by prediction label
            const label = (msg.score || msg.prediction || 'unknown').toLowerCase();
            labelCounts[label] = (labelCounts[label] || 0) + 1;
            
            // Count cyberbullying vs safe
            if (msg.is_cyberbullying || (label !== 'neutral' && label !== 'safe')) {
                cbCount++;
            } else {
                safeCount++;
            }
        });
        
        const stats = {
            total_predictions: filtered.length,
            cyberbullying_count: cbCount,
            not_cyberbullying_count: safeCount,
            severity_distribution: severityCounts,
            label_distribution: labelCounts,
            daily_counts: [],
            monthly_trend: { 'Current': filtered.length }
        };
        
        updateCharts(stats);
        StatsCards.updateInterventions(filtered);
    }

    /**
     * Handle tab click
     * FIX: Now preserves state when switching tabs
     */
    function handleTabClick(e) {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        e.target.classList.add('active');

        const tab = e.target.dataset.tab;
        state.currentTab = tab;
        console.log('📑 Switched to tab:', tab);
        
        // Save state to preserve tab selection
        saveState();
        
        // Load appropriate data based on tab
        if (tab === 'history') {
            loadHistoryView();
        } else {
            // Current alerts - reload current data
            loadData();
        }
    }

    /**
     * Load history view from saved history
     */
    function loadHistoryView() {
        const history = getHistory();
        console.log('📚 Loading history view:', history.length, 'entries');
        
        if (history.length === 0) {
            // Show empty state message
            const tbody = document.getElementById('messages-tbody');
            if (tbody) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="7" style="text-align: center; padding: 40px; color: #666;">
                            <p>📜 No history entries yet.</p>
                            <p>History is saved automatically when you analyze messages.</p>
                        </td>
                    </tr>
                `;
            }
            return;
        }
        
        // Load most recent history entry
        const latestEntry = history[0];
        state.messages = latestEntry.messages || [];
        state.totalPages = Math.ceil(state.messages.length / state.pageSize);
        renderTable();
        updatePagination();
    }

    /**
     * Handle sort column click
     */
    function handleSort(e) {
        const column = e.target.dataset.sort;
        if (!column) return;

        if (state.sortColumn === column) {
            state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            state.sortColumn = column;
            state.sortDirection = 'desc';
        }

        renderTable();
    }

    /**
     * Handle table row click
     */
    function handleRowClick(e) {
        const row = e.target.closest('tr');
        if (!row) return;

        const id = parseInt(row.dataset.id);
        const message = state.messages.find(m => m.id === id);

        if (message) {
            openMessageModal(message);
        }
    }

    /**
     * Open message detail modal
     */
    function openMessageModal(message) {
        state.selectedMessage = message;

        const modal = document.getElementById('message-modal');
        
        // Populate all modal fields
        const setField = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value || '-';
        };
        
        // Set all fields
        setField('modal-student-id', message.student_id);
        setField('modal-date', message.date);
        setField('modal-message', message.message);
        setField('modal-platform', capitalize(message.platform || 'unknown'));
        setField('modal-score', message.score || message.prediction || '-');
        setField('modal-model', (message.model || 'BERT').toUpperCase());
        
        // Handle confidence - remove double percent
        let confidence = message.confidence || message.raw_confidence || 'N/A';
        if (typeof confidence === 'number') {
            confidence = Math.round(confidence * 100) + '%';
        } else if (typeof confidence === 'string') {
            // Remove any existing % and add fresh one
            confidence = confidence.replace(/%/g, '').trim();
            if (!isNaN(parseFloat(confidence))) {
                confidence = confidence + '%';
            }
        }
        setField('modal-confidence', confidence);
        
        // Set severity with styling
        const modalSeverity = document.getElementById('modal-severity');
        if (modalSeverity) {
            modalSeverity.textContent = (message.severity || 'unknown').toUpperCase();
            modalSeverity.className = `severity-badge severity-${(message.severity || 'unknown').toLowerCase()}`;
        }
        
        // Generate explanation
        const modalExplanation = document.getElementById('modal-explanation');
        if (modalExplanation) {
            modalExplanation.textContent = generateExplanation(message);
        }

        if (modal) modal.classList.remove('hidden');

        StatsCards.addAuditEntry(`Viewed ${message.severity}`, '📋');
    }

    /**
     * Close message modal
     */
    function closeMessageModal() {
        const modal = document.getElementById('message-modal');
        if (modal) modal.classList.add('hidden');
        state.selectedMessage = null;
    }

    /**
     * Generate explanation for prediction
     */
    function generateExplanation(message) {
        const explanations = {
            threat: 'This message contains threatening language indicating potential harm. Immediate attention may be required.',
            harassment: 'This message shows patterns of harassment including repeated targeting or exclusionary language.',
            insult: 'This message contains insulting or demeaning language intended to hurt the recipient.',
            neutral: 'This message appears to be normal communication without harmful content.'
        };

        return explanations[message.severity.toLowerCase()] || 'Classification based on language patterns and context analysis.';
    }

    /**
     * Handle page change - now saves state
     */
    function changePage(delta) {
        const newPage = state.currentPage + delta;
        if (newPage >= 1 && newPage <= state.totalPages) {
            console.log(`📄 Changing page from ${state.currentPage} to ${newPage}`);
            state.currentPage = newPage;
            renderTable();
            updatePagination();
            saveState();
        } else {
            console.log(`📄 Cannot change page: ${newPage} is out of bounds (1-${state.totalPages})`);
        }
    }

    /**
     * Handle page number click
     */
    function handlePageClick(e) {
        const page = parseInt(e.target.textContent);
        if (!isNaN(page) && page !== state.currentPage) {
            console.log(`📄 Page number clicked: ${page}`);
            state.currentPage = page;
            renderTable();
            updatePagination();
            saveState();
        }
    }

    /**
     * Handle export
     * FIX: Now exports comprehensive report with all message details
     */
    async function handleExport(e) {
        const type = e.target.dataset.export;
        console.log(`📤 Starting export: ${type}`);

        try {
            // Show loading state on button
            e.target.textContent = 'Exporting...';
            e.target.disabled = true;
            
            let csvData = '';
            const messages = state.messages || [];
            
            if (messages.length === 0) {
                alert('No data to export. Please upload and analyze a file first.');
                return;
            }
            
            // Generate comprehensive CSV report
            const headers = [
                'Date',
                'Student ID', 
                'Message',
                'Platform',
                'Severity',
                'Prediction',
                'Confidence',
                'Model',
                'Is Cyberbullying'
            ];
            
            // Add summary section at top
            const summary = generateExportSummary(messages);
            csvData = summary + '\n\n';
            
            // Add headers
            csvData += headers.join(',') + '\n';
            
            // Add each message row
            messages.forEach(msg => {
                const row = [
                    escapeCSV(msg.date || ''),
                    escapeCSV(msg.student_id || ''),
                    escapeCSV(msg.message || ''),
                    escapeCSV(msg.platform || ''),
                    escapeCSV(msg.severity || ''),
                    escapeCSV(msg.score || msg.prediction || ''),
                    escapeCSV(String(msg.confidence || '').replace(/%%/g, '%')),
                    escapeCSV(msg.model || 'BERT'),
                    msg.is_cyberbullying ? 'Yes' : 'No'
                ];
                csvData += row.join(',') + '\n';
            });

            // Create download
            const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cyberbullying_report_${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            StatsCards.addAuditEntry('Exported report', '📋');
            console.log(`✅ Export completed: ${messages.length} messages`);

        } catch (error) {
            console.error('❌ Export failed:', error);
            alert('Export failed: ' + error.message);
        } finally {
            // Restore button state
            e.target.disabled = false;
            const originalText = {
                'csv': '📄 Export as CSV',
                'pdf': '📑 Export as PDF',
                'all': '📊 Export All Data'
            };
            e.target.textContent = originalText[type] || '📤 Export';
        }

        // Close dropdown
        document.getElementById('export-menu')?.classList.add('hidden');
    }
    
    /**
     * Generate summary section for export
     */
    function generateExportSummary(messages) {
        const total = messages.length;
        const severityCounts = { low: 0, medium: 0, high: 0, critical: 0 };
        let cbCount = 0;
        
        messages.forEach(msg => {
            const sev = (msg.severity || 'low').toLowerCase();
            if (severityCounts[sev] !== undefined) severityCounts[sev]++;
            if (msg.is_cyberbullying) cbCount++;
        });
        
        const summary = [
            '=== CYBERBULLYING DETECTION REPORT ===',
            `Generated: ${new Date().toLocaleString()}`,
            '',
            '=== SUMMARY ===',
            `Total Messages Analyzed: ${total}`,
            `Cyberbullying Detected: ${cbCount} (${total ? Math.round(cbCount/total*100) : 0}%)`,
            `Safe Messages: ${total - cbCount} (${total ? Math.round((total-cbCount)/total*100) : 0}%)`,
            '',
            '=== SEVERITY BREAKDOWN ===',
            `Critical: ${severityCounts.critical}`,
            `High: ${severityCounts.high}`,
            `Medium: ${severityCounts.medium}`,
            `Low: ${severityCounts.low}`,
            '',
            '=== DETAILED MESSAGE DATA ==='
        ];
        
        return summary.join('\n');
    }
    
    /**
     * Escape CSV field value
     */
    function escapeCSV(value) {
        if (value === null || value === undefined) return '';
        const str = String(value);
        // If contains comma, newline, or quote, wrap in quotes and escape existing quotes
        if (str.includes(',') || str.includes('\n') || str.includes('"')) {
            return '"' + str.replace(/"/g, '""') + '"';
        }
        return str;
    }

    /**
     * Handle search
     */
    function handleSearch(e) {
        const query = e.target.value.toLowerCase();

        if (!query) {
            renderTable();
            return;
        }

        const filtered = state.messages.filter(msg =>
            msg.message.toLowerCase().includes(query) ||
            msg.student_id.toLowerCase().includes(query)
        );

        const tbody = document.getElementById('messages-tbody');
        if (!tbody) return;

        tbody.innerHTML = filtered.map(msg => `
            <tr data-id="${msg.id}" role="button" tabindex="0">
                <td>${msg.date}</td>
                <td><strong>${msg.student_id}</strong></td>
                <td class="message-text">${truncateText(msg.message, 40)}</td>
                <td>
                    <span class="platform-badge platform-${msg.platform}">
                        ${platformIcons[msg.platform] || ''}
                        ${capitalize(msg.platform)}
                    </span>
                </td>
                <td>
                    <span class="severity-badge severity-${msg.severity.toLowerCase()}">
                        ${msg.severity.toUpperCase()}
                    </span>
                </td>
                <td class="score-value">${msg.score}</td>
                <td class="confidence-value">${msg.confidence}</td>
            </tr>
        `).join('');
    }

    /**
     * Handle keyboard shortcuts
     */
    function handleKeyboard(e) {
        // Escape closes modal
        if (e.key === 'Escape') {
            closeMessageModal();
        }

        // Arrow keys for pagination
        if (e.key === 'ArrowLeft' && e.ctrlKey) {
            changePage(-1);
        }
        if (e.key === 'ArrowRight' && e.ctrlKey) {
            changePage(1);
        }
    }

    // Utility functions
    function truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    function capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    /**
     * Load CSV analysis results into dashboard
     * FIX: Now saves results to history and state
     */
    function loadCSVResults(messages, analysisResults) {
        console.log('📊 Loading CSV results to dashboard:', messages.length, 'messages');
        
        // Update state with new messages
        state.messages = messages;
        state.currentPage = 1;
        state.totalPages = Math.ceil(messages.length / state.pageSize);
        
        // Render the table with new data
        renderTable();
        updatePagination();
        
        // Build proper stats from actual message data for charts
        const severityCounts = { low: 0, medium: 0, high: 0, critical: 0 };
        const labelCounts = {};
        let cbCount = 0;
        let safeCount = 0;
        
        messages.forEach(msg => {
            // Count by severity
            const sev = (msg.severity || 'low').toLowerCase();
            if (severityCounts[sev] !== undefined) {
                severityCounts[sev]++;
            } else {
                severityCounts['low']++;
            }
            
            // Count by prediction label
            const label = (msg.score || msg.prediction || 'neutral').toLowerCase();
            labelCounts[label] = (labelCounts[label] || 0) + 1;
            
            // Count cyberbullying vs safe
            if (msg.is_cyberbullying || (label !== 'neutral' && label !== 'safe' && label !== 'not_cyberbullying')) {
                cbCount++;
            } else {
                safeCount++;
            }
        });
        
        const stats = {
            total_predictions: messages.length,
            cyberbullying_count: cbCount,
            not_cyberbullying_count: safeCount,
            severity_distribution: severityCounts,
            label_distribution: labelCounts,
            daily_counts: [],
            monthly_trend: { 'Current Session': messages.length },
            model_usage: {}
        };
        
        // Count by model
        const modelUsed = messages[0]?.model || 'bert';
        stats.model_usage[modelUsed] = messages.length;
        
        console.log('📈 Built stats for charts:', stats);
        
        // Update charts
        updateCharts(stats);
        
        // Update interventions sidebar
        StatsCards.updateInterventions(messages);
        
        // Hide offline banner since we have data
        const banner = document.getElementById('offline-banner');
        if (banner) {
            banner.classList.add('hidden');
        }
        
        // Save to state and history
        saveState();
        saveToHistory(messages);
        
        // Show success message
        console.log('✅ Dashboard updated with CSV results and saved to history');
    }

    // Public API
    return {
        init,
        loadData,
        loadCSVResults,
        saveState,
        loadState,
        clearState,
        getHistory,
        loadHistoryEntry,
        loadHistoryDropdown,
        state: () => state
    };
})();

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    Dashboard.init();
});

// Export
window.Dashboard = Dashboard;
