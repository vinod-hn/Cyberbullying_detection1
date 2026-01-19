/**
 * Line Chart Module
 * Renders the Daily Alerts Trend line chart
 */

const LineChart = (function() {
    'use strict';

    let chartInstance = null;
    const canvasId = 'daily-line-chart';

    // Default data matching the screenshot (Mon-Sun)
    const defaultData = {
        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        values: [12, 19, 15, 22, 18, 25, 20]
    };

    /**
     * Initialize the line chart
     */
    function init(data = null) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error('Line chart canvas not found');
            return null;
        }

        // Destroy existing chart
        if (chartInstance) {
            chartInstance.destroy();
        }

        const chartData = data || defaultData;
        const ctx = canvas.getContext('2d');

        // Create gradient for area fill
        const gradient = ctx.createLinearGradient(0, 0, 0, 200);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: [{
                    label: 'Daily Alerts',
                    data: chartData.values,
                    borderColor: ChartConfig.colors.lineBlue,
                    backgroundColor: gradient,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#ffffff',
                    pointBorderColor: ChartConfig.colors.lineBlue,
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointHoverBackgroundColor: ChartConfig.colors.lineBlue,
                    pointHoverBorderColor: '#ffffff',
                    pointHoverBorderWidth: 2
                }]
            },
            options: {
                ...ChartConfig.lineOptions,
                scales: {
                    ...ChartConfig.lineOptions.scales,
                    y: {
                        ...ChartConfig.lineOptions.scales.y,
                        min: 0,
                        max: Math.max(...chartData.values) + 5,
                        ticks: {
                            ...ChartConfig.lineOptions.scales.y.ticks,
                            stepSize: 5
                        }
                    }
                },
                plugins: {
                    ...ChartConfig.lineOptions.plugins,
                    tooltip: {
                        ...ChartConfig.lineOptions.plugins.tooltip,
                        callbacks: {
                            title: function(context) {
                                return context[0].label;
                            },
                            label: function(context) {
                                return ` Alerts: ${context.parsed.y}`;
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

        chartInstance.data.labels = data.labels;
        chartInstance.data.datasets[0].data = data.values;
        
        // Update y-axis max
        const maxValue = Math.max(...data.values);
        chartInstance.options.scales.y.max = maxValue + 5;
        
        chartInstance.update('active');
    }

    /**
     * Update from API stats
     * Expects backend StatisticsResponse.daily_counts.
     */
    function updateFromStats(stats) {
        // If no stats or no daily counts, show default week with zeros
        if (!stats || !Array.isArray(stats.daily_counts) || stats.daily_counts.length === 0) {
            // Show last 7 days with zero values
            const days = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
            update({ labels: days, values: [0, 0, 0, 0, 0, 0, 0] });
            return;
        }

        const sorted = [...stats.daily_counts].sort((a, b) => {
            return new Date(a.date) - new Date(b.date);
        });

        // Take last 7 days or available data
        const recent = sorted.slice(-7);

        const labels = recent.map(entry => {
            const d = new Date(entry.date);
            return d.toLocaleDateString('en-US', { weekday: 'short' });
        });

        // Use cyberbullying counts for alerts; fall back to total predictions
        const values = recent.map(entry => (
            typeof entry.cyberbullying === 'number' ? entry.cyberbullying : (entry.predictions || 0)
        ));

        update({ labels, values });
    }

    /**
     * Update date range display
     */
    function updateDateRange(startDate, endDate) {
        const navLabel = document.querySelector('.chart-navigation .nav-label');
        if (navLabel) {
            const start = new Date(startDate).toLocaleDateString('en-US', { 
                month: 'long', 
                day: 'numeric' 
            });
            const end = new Date(endDate).toLocaleDateString('en-US', { 
                month: 'short', 
                day: 'numeric',
                year: 'numeric'
            });
            navLabel.textContent = `${start} 📅 - ${end}`;
        }
    }

    /**
     * Navigate to previous week
     */
    function previousWeek() {
        // Shift data window left (in real implementation, fetch new data)
        console.log('Navigate to previous week');
        // Could trigger API call with new date range
    }

    /**
     * Navigate to next week
     */
    function nextWeek() {
        // Shift data window right (in real implementation, fetch new data)
        console.log('Navigate to next week');
        // Could trigger API call with new date range
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
        updateDateRange,
        previousWeek,
        nextWeek,
        destroy,
        getInstance
    };
})();

// Export
window.LineChart = LineChart;
