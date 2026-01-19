/**
 * Bar Chart Module
 * Renders the Monthly Trend bar chart
 */

const BarChart = (function() {
    'use strict';

    let chartInstance = null;
    const canvasId = 'monthly-bar-chart';

    // Default data matching the screenshot (Jan-May)
    const defaultData = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        values: [245, 380, 420, 480, 560]
    };

    /**
     * Initialize the bar chart
     */
    function init(data = null) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error('Bar chart canvas not found');
            return null;
        }

        // Destroy existing chart
        if (chartInstance) {
            chartInstance.destroy();
        }

        const chartData = data || defaultData;
        const ctx = canvas.getContext('2d');

        // Create gradient for bars
        const gradient = ctx.createLinearGradient(0, 0, 0, 200);
        gradient.addColorStop(0, ChartConfig.colors.primaryBlue);
        gradient.addColorStop(1, ChartConfig.colors.lineBlue);

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: chartData.labels,
                datasets: [{
                    label: 'Monthly Alerts',
                    data: chartData.values,
                    backgroundColor: ChartConfig.colors.primaryBlue,
                    hoverBackgroundColor: ChartConfig.colors.lineBlue,
                    borderRadius: 4,
                    borderSkipped: false,
                    barThickness: 'flex',
                    maxBarThickness: 40
                }]
            },
            options: {
                ...ChartConfig.barOptions,
                scales: {
                    ...ChartConfig.barOptions.scales,
                    y: {
                        ...ChartConfig.barOptions.scales.y,
                        min: 0,
                        max: Math.max(...chartData.values) + 100,
                        ticks: {
                            ...ChartConfig.barOptions.scales.y.ticks,
                            stepSize: 100
                        }
                    }
                },
                plugins: {
                    ...ChartConfig.barOptions.plugins,
                    tooltip: {
                        ...ChartConfig.barOptions.plugins.tooltip,
                        callbacks: {
                            title: function(context) {
                                return context[0].label + ' 2024';
                            },
                            label: function(context) {
                                return ` Total Alerts: ${context.parsed.y}`;
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
        chartInstance.options.scales.y.max = Math.ceil(maxValue / 100) * 100 + 100;
        
        chartInstance.update('active');
    }

    /**
     * Update from API stats
     * Expects backend StatisticsResponse.monthly_trend.
     */
    function updateFromStats(stats) {
        // If no stats or no monthly trend, show default months with zeros
        if (!stats || !stats.monthly_trend || Object.keys(stats.monthly_trend).length === 0) {
            update({ labels: ['Jan'], values: [0] });
            return;
        }

        const monthlyData = stats.monthly_trend;
        const labels = Object.keys(monthlyData);
        const values = Object.values(monthlyData);

        update({ labels, values });
    }

    /**
     * Highlight specific month
     */
    function highlightMonth(monthIndex) {
        if (!chartInstance) return;

        const colors = chartInstance.data.datasets[0].data.map((_, i) => 
            i === monthIndex ? ChartConfig.colors.lineBlue : ChartConfig.colors.primaryBlue
        );

        chartInstance.data.datasets[0].backgroundColor = colors;
        chartInstance.update('active');
    }

    /**
     * Reset highlights
     */
    function resetHighlight() {
        if (!chartInstance) return;

        chartInstance.data.datasets[0].backgroundColor = ChartConfig.colors.primaryBlue;
        chartInstance.update('active');
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
        highlightMonth,
        resetHighlight,
        destroy,
        getInstance
    };
})();

// Export
window.BarChart = BarChart;
