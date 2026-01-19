/**
 * Stats Cards Module
 * Handles summary statistics cards and intervention suggestions
 * UPDATED: Dynamic intervention suggestions based on severity distribution
 */

const StatsCards = (function() {
    'use strict';

    // Default stats
    const defaultStats = {
        totalMessages: 1247,
        bullying: 687,
        safe: 560,
        accuracy: 99.86
    };

    /**
     * Update intervention suggestions based on data
     * ENHANCED: More detailed and dynamic suggestions
     */
    function updateInterventions(predictions) {
        const suggestionList = document.getElementById('suggestion-list');
        const alertBanner = document.getElementById('severity-alert-banner');
        const alertBannerText = document.getElementById('alert-banner-text');
        const interventionStats = document.getElementById('intervention-stats');
        
        if (!suggestionList) return;

        const total = predictions.length;
        if (total === 0) {
            // No data state
            suggestionList.innerHTML = `
                <li class="suggestion-item safe">
                    <span class="bullet green"></span>
                    No messages to analyze
                </li>
            `;
            if (alertBanner) alertBanner.classList.add('hidden');
            if (interventionStats) interventionStats.innerHTML = '';
            return;
        }

        // Count severities
        const threatCount = predictions.filter(p => 
            p.severity?.toLowerCase() === 'threat' || 
            p.score?.toLowerCase() === 'threat' ||
            p.prediction?.toLowerCase() === 'threat'
        ).length;

        const harassmentCount = predictions.filter(p => 
            p.severity?.toLowerCase() === 'harassment' || 
            p.score?.toLowerCase() === 'harassment' ||
            p.prediction?.toLowerCase() === 'harassment'
        ).length;

        const insultCount = predictions.filter(p => 
            p.severity?.toLowerCase() === 'insult' || 
            p.score?.toLowerCase() === 'insult' ||
            p.prediction?.toLowerCase() === 'insult'
        ).length;

        const neutralCount = predictions.filter(p => 
            p.severity?.toLowerCase() === 'neutral' || 
            p.severity?.toLowerCase() === 'low' ||
            p.score?.toLowerCase() === 'neutral' ||
            p.prediction?.toLowerCase() === 'neutral' ||
            !p.is_cyberbullying
        ).length;

        const highSeverity = threatCount + harassmentCount;
        const mediumSeverity = insultCount;
        const cyberbullyingTotal = total - neutralCount;
        
        // Calculate percentages
        const highPercent = total > 0 ? ((highSeverity / total) * 100).toFixed(1) : 0;
        const mediumPercent = total > 0 ? ((mediumSeverity / total) * 100).toFixed(1) : 0;
        const safePercent = total > 0 ? ((neutralCount / total) * 100).toFixed(1) : 0;

        // Generate suggestions based on severity distribution
        const suggestions = [];

        // HIGH SEVERITY (> 30% threats/harassment OR any threats)
        if (highPercent > 30 || threatCount > 0) {
            if (alertBanner) {
                alertBanner.classList.remove('hidden');
                if (alertBannerText) {
                    alertBannerText.textContent = threatCount > 0 
                        ? `⚠️ ${threatCount} Threat${threatCount > 1 ? 's' : ''} Detected!`
                        : `⚠️ High Severity: ${highPercent}%`;
                }
            }
            
            suggestions.push({
                text: '🚨 Immediate action recommended',
                bullet: 'red',
                priority: 'critical',
                description: 'Notify moderator or counselor immediately'
            });
            
            if (threatCount > 0) {
                suggestions.push({
                    text: '👁️ Review threat messages urgently',
                    bullet: 'red',
                    priority: 'critical',
                    description: `${threatCount} potential threat(s) found`
                });
            }
            
            suggestions.push({
                text: '🔒 Consider temporary restrictions',
                bullet: 'orange',
                priority: 'warning',
                description: 'May need to limit user access'
            });
        }
        // MEDIUM SEVERITY (> 20% insults or some harassment)
        else if (mediumPercent > 20 || harassmentCount > 0) {
            if (alertBanner) {
                alertBanner.classList.remove('hidden');
                if (alertBannerText) {
                    alertBannerText.textContent = `⚡ Moderate Concerns Detected`;
                }
            }
            
            suggestions.push({
                text: '👀 Monitor conversation closely',
                bullet: 'orange',
                priority: 'warning',
                description: 'Watch for escalation patterns'
            });
            
            suggestions.push({
                text: '💬 Provide behavioral guidance',
                bullet: 'orange',
                priority: 'warning',
                description: 'Consider sending a warning message'
            });
            
            suggestions.push({
                text: '📋 Document incidents',
                bullet: 'yellow',
                priority: 'info',
                description: 'Keep records for future reference'
            });
        }
        // LOW SEVERITY (mostly neutral)
        else if (safePercent > 70) {
            if (alertBanner) alertBanner.classList.add('hidden');
            
            suggestions.push({
                text: '✅ No immediate action required',
                bullet: 'green',
                priority: 'safe',
                description: 'Conversation appears healthy'
            });
            
            if (mediumSeverity > 0) {
                suggestions.push({
                    text: '📊 Routine monitoring recommended',
                    bullet: 'blue',
                    priority: 'info',
                    description: 'Some mild concerns detected'
                });
            }
        }
        // DEFAULT
        else {
            if (alertBanner) alertBanner.classList.add('hidden');
            
            suggestions.push({
                text: '📊 Standard monitoring',
                bullet: 'blue',
                priority: 'info',
                description: 'Review periodically'
            });
            
            suggestions.push({
                text: '✅ No urgent action needed',
                bullet: 'green',
                priority: 'safe',
                description: 'Continue normal operations'
            });
        }

        // Render suggestions
        suggestionList.innerHTML = suggestions.map(s => `
            <li class="suggestion-item ${s.priority}" title="${s.description}">
                <span class="bullet ${s.bullet}"></span>
                ${s.text}
            </li>
        `).join('');

        // Update intervention stats
        if (interventionStats) {
            interventionStats.innerHTML = `
                <div class="intervention-stat-item">
                    <span>Total Messages:</span>
                    <span class="intervention-stat-value">${total}</span>
                </div>
                <div class="intervention-stat-item">
                    <span>Threats:</span>
                    <span class="intervention-stat-value danger">${threatCount}</span>
                </div>
                <div class="intervention-stat-item">
                    <span>Harassment:</span>
                    <span class="intervention-stat-value danger">${harassmentCount}</span>
                </div>
                <div class="intervention-stat-item">
                    <span>Insults:</span>
                    <span class="intervention-stat-value warning">${insultCount}</span>
                </div>
                <div class="intervention-stat-item">
                    <span>Neutral:</span>
                    <span class="intervention-stat-value safe">${neutralCount}</span>
                </div>
                <div class="intervention-stat-item">
                    <span>Risk Level:</span>
                    <span class="intervention-stat-value ${highPercent > 30 ? 'danger' : mediumPercent > 20 ? 'warning' : 'safe'}">
                        ${highPercent > 30 ? 'HIGH' : mediumPercent > 20 ? 'MEDIUM' : 'LOW'}
                    </span>
                </div>
            `;
        }

        console.log(`📊 Intervention suggestions updated: ${suggestions.length} suggestions, Risk: ${highPercent > 30 ? 'HIGH' : mediumPercent > 20 ? 'MEDIUM' : 'LOW'}`);
    }

    /**
     * Update audit log display
     */
    function updateAuditLog(logs) {
        const auditList = document.getElementById('audit-log');
        if (!auditList) return;

        const logItems = logs || [
            { action: 'Exported report', time: '10:32 AM', icon: '📋' },
            { action: 'Viewed threat', time: '10:35 AM', icon: '📋' }
        ];

        auditList.innerHTML = logItems.slice(0, 5).map(log => `
            <li class="audit-item">
                <span class="audit-icon">${log.icon}</span>
                <span class="audit-text">${log.action}</span>
                <span class="audit-time">${log.time}</span>
            </li>
        `).join('');
    }

    /**
     * Fetch audit log from backend and update display
     */
    async function fetchAuditLog() {
        const auditList = document.getElementById('audit-log');
        if (!auditList) return;

        try {
            const logs = await APIClient.getAuditLog(5);
            if (logs && logs.length > 0) {
                updateAuditLog(logs);
            } else {
                // Show empty state
                auditList.innerHTML = `
                    <li class="audit-item">
                        <span class="audit-icon">📋</span>
                        <span class="audit-text">No recent activity</span>
                        <span class="audit-time">--</span>
                    </li>
                `;
            }
        } catch (error) {
            console.warn('Failed to fetch audit log:', error);
        }
    }

    /**
     * Add entry to audit log (both UI and backend)
     */
    async function addAuditEntry(action, icon = '📋') {
        const auditList = document.getElementById('audit-log');
        if (!auditList) return;

        const now = new Date();
        const time = now.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
        });

        const newItem = document.createElement('li');
        newItem.className = 'audit-item';
        newItem.innerHTML = `
            <span class="audit-icon">${icon}</span>
            <span class="audit-text">${action}</span>
            <span class="audit-time">${time}</span>
        `;

        // Insert at beginning
        auditList.insertBefore(newItem, auditList.firstChild);

        // Remove old items if more than 5
        while (auditList.children.length > 5) {
            auditList.removeChild(auditList.lastChild);
        }

        // Also log to backend (fire and forget)
        try {
            await APIClient.addAuditLog(action, icon);
        } catch (e) {
            // Ignore errors for audit logging
        }
    }

    /**
     * Calculate and display summary stats
     */
    function updateSummary(stats) {
        // This could update stat cards if we add them
        console.log('Stats summary:', stats);
    }

    /**
     * Format large numbers
     */
    function formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    /**
     * Format percentage
     */
    function formatPercent(num) {
        return num.toFixed(1) + '%';
    }

    // Public API
    return {
        updateInterventions,
        updateAuditLog,
        fetchAuditLog,
        addAuditEntry,
        updateSummary,
        formatNumber,
        formatPercent
    };
})();

// Export
window.StatsCards = StatsCards;
