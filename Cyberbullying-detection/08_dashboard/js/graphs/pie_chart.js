/**
 * Pie/Donut Chart Module
 * Renders the Severity Distribution donut chart
 */

const PieChart = (function() {
    'use strict';

    let chartInstance = null;
    const canvasId = 'severity-donut-chart';

    // Color mapping for labels
    const labelColors = {
        'Neutral': '#4CAF50',
        'Insult': '#FFC107',
        'Harassment': '#F44336',
        'Threat': '#ef4444',
        'Aggression': '#ff5722',
        'Hate': '#e91e63',
        'Toxicity': '#9c27b0',
        'Cyberbullying': '#d32f2f',
        'Other': '#607d8b',
        'No Data': '#e0e0e0'
    };

    // Default data
    const defaultData = {
        labels: ['No Data'],
        values: [100]
    };

    /**
     * Initialize the donut chart
     */
    function init(data = null) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error('Donut chart canvas not found');
            return null;
        }

        // Destroy existing chart
        if (chartInstance) {
            chartInstance.destroy();
        }

        const chartData = data || defaultData;
        const ctx = canvas.getContext('2d');
        
        // Get colors for each label
        const colors = chartData.labels.map(label => labelColors[label] || '#607d8b');

        chartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: chartData.labels,
                datasets: [{
                    data: chartData.values,
                    backgroundColor: colors,
                    borderColor: '#ffffff',
                    borderWidth: 3,
                    hoverBorderWidth: 4,
                    hoverOffset: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1,
                cutout: '55%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(26, 26, 46, 0.95)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                return ` ${label}: ${value}%`;
                            }
                        }
                    }
                }
            }
        });

        return chartInstance;
    }

    /**
     * Update chart with new data
     */
    function update(data) {
        if (!chartInstance) {
            return init(data);
        }

        // Get colors for labels
        const colors = data.labels.map(label => labelColors[label] || '#607d8b');

        chartInstance.data.labels = data.labels;
        chartInstance.data.datasets[0].data = data.values;
        chartInstance.data.datasets[0].backgroundColor = colors;
        chartInstance.update('active');
    }

    /**
     * Update from API stats
     * Expects backend StatisticsResponse with label_distribution.
     */
    function updateFromStats(stats) {
        // If no stats or empty distribution, show default empty state
        if (!stats) {
            update({ labels: ['No Data'], values: [100] });
            return;
        }

        // Prefer label_distribution from backend; fall back to severity_distribution if needed
        const distribution = stats.label_distribution || stats.severity_distribution;
        if (!distribution || Object.keys(distribution).length === 0) {
            // Show empty state with placeholder
            update({ labels: ['No Data'], values: [100] });
            return;
        }

        const labels = [];
        const values = [];
        const total = Object.values(distribution).reduce((a, b) => a + b, 0) || 0;
        if (!total) {
            update({ labels: ['No Data'], values: [100] });
            return;
        }

        // Map labels and calculate percentages - show each category separately
        const labelMap = {
            // Prediction labels
            neutral: 'Neutral',
            safe: 'Neutral',
            'not_cyberbullying': 'Neutral',
            'not cyberbullying': 'Neutral',
            insult: 'Insult',
            harassment: 'Harassment',
            threat: 'Threat',
            aggression: 'Aggression',
            hate: 'Hate',
            toxicity: 'Toxicity',
            cyberbullying: 'Cyberbullying',
            other: 'Other',
            // Severity labels (when using severity_distribution)
            low: 'Neutral',
            medium: 'Insult',
            high: 'Harassment',
            critical: 'Threat'
        };

        // Aggregate by mapped category
        const categoryTotals = {};

        for (const [key, count] of Object.entries(distribution)) {
            const normalizedKey = key.toLowerCase();
            const mappedLabel = labelMap[normalizedKey] || 'Other';
            categoryTotals[mappedLabel] = (categoryTotals[mappedLabel] || 0) + count;
        }

        // Convert to percentages and arrays
        for (const [label, count] of Object.entries(categoryTotals)) {
            const percentage = Math.round((count / total) * 100);
            if (percentage > 0) {
                labels.push(label);
                values.push(percentage);
            }
        }

        // If somehow we got no labels, show placeholder
        if (labels.length === 0) {
            update({ labels: ['No Data'], values: [100] });
            return;
        }

        update({ labels, values });
        updateLegend(labels, values);
    }

    /**
     * Update the legend display
     */
    function updateLegend(labels, values) {
        const legendContainer = document.getElementById('donut-legend');
        if (!legendContainer) return;

        const colors = {
            'Neutral': '#4CAF50',
            'Insult': '#FFC107',
            'Harassment': '#F44336',
            'Threat': '#ef4444',
            'Aggression': '#ff5722',
            'Hate': '#e91e63',
            'Toxicity': '#9c27b0',
            'Cyberbullying': '#d32f2f',
            'Other': '#607d8b',
            'No Data': '#9e9e9e'
        };

        legendContainer.innerHTML = labels.map((label, i) => `
            <div class="legend-item">
                <span class="legend-color" style="background: ${colors[label] || '#607d8b'};"></span>
                <span class="legend-label">${label}: ${values[i]}%</span>
            </div>
        `).join('');
    }

    /**
     * Destroy the chart
     */
    function destroy() {
        if (chartInstance) {
            chartInstance.destroy();
            chartInstance = null;
        }
    }

    /**
     * Get chart instance
     */
    function getInstance() {
        return chartInstance;
    }

    // Public API
    return {
        init,
        update,
        updateFromStats,
        destroy,
        getInstance
    };
})();

// Export
window.PieChart = PieChart;
