/**
 * Chart Configuration for Cyberbullying Detection Dashboard
 * Defines global Chart.js settings and color palettes
 */

const ChartConfig = (function() {
    'use strict';

    // Color palette matching the design
    const colors = {
        // Severity colors
        neutral: '#4CAF50',
        insult: '#FFC107',
        harassment: '#F44336',
        threat: '#ef4444',
        other: '#9C27B0',
        
        // Chart colors
        primaryBlue: '#3b82f6',
        lineBlue: '#60a5fa',
        barBlue: '#3b82f6',
        
        // Background colors (with alpha)
        neutralBg: 'rgba(76, 175, 80, 0.1)',
        insultBg: 'rgba(255, 193, 7, 0.1)',
        harassmentBg: 'rgba(244, 67, 54, 0.1)',
        threatBg: 'rgba(239, 68, 68, 0.1)',
        otherBg: 'rgba(156, 39, 176, 0.1)',
        
        // UI colors
        grid: '#e5e7eb',
        text: '#6b7280',
        textDark: '#1a1a2e'
    };

    // Donut chart colors in order - expanded for more categories
    const donutColors = [
        colors.neutral,    // Green - Neutral/Safe
        colors.insult,     // Yellow - Insult
        colors.harassment, // Red - Harassment
        colors.threat,     // Dark Red - Threat
        '#ff5722',         // Deep Orange - Aggression
        '#e91e63',         // Pink - Hate
        '#9c27b0',         // Purple - Toxicity
        '#607d8b',         // Blue Grey - Other
        '#795548'          // Brown - Additional
    ];

    // Default chart options
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false // We use custom legends
            },
            tooltip: {
                enabled: true,
                backgroundColor: 'rgba(26, 26, 46, 0.95)',
                titleColor: '#ffffff',
                bodyColor: '#ffffff',
                padding: 12,
                cornerRadius: 8,
                displayColors: true,
                boxPadding: 4
            }
        }
    };

    // Donut/Pie chart specific options
    const donutOptions = {
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
                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                        const percentage = ((value / total) * 100).toFixed(1);
                        return `${label}: ${percentage}%`;
                    }
                }
            }
        }
    };

    // Line chart specific options
    const lineOptions = {
        ...defaultOptions,
        scales: {
            x: {
                grid: {
                    display: false
                },
                ticks: {
                    color: colors.text,
                    font: {
                        size: 11
                    }
                }
            },
            y: {
                beginAtZero: true,
                grid: {
                    color: colors.grid,
                    drawBorder: false
                },
                ticks: {
                    color: colors.text,
                    font: {
                        size: 11
                    },
                    stepSize: 5
                }
            }
        },
        elements: {
            line: {
                tension: 0.4,
                borderWidth: 2
            },
            point: {
                radius: 4,
                hoverRadius: 6,
                backgroundColor: '#ffffff',
                borderWidth: 2
            }
        },
        plugins: {
            ...defaultOptions.plugins,
            tooltip: {
                ...defaultOptions.plugins.tooltip,
                intersect: false,
                mode: 'index'
            }
        }
    };

    // Bar chart specific options
    const barOptions = {
        ...defaultOptions,
        scales: {
            x: {
                grid: {
                    display: false
                },
                ticks: {
                    color: colors.text,
                    font: {
                        size: 11
                    }
                }
            },
            y: {
                beginAtZero: true,
                grid: {
                    color: colors.grid,
                    drawBorder: false
                },
                ticks: {
                    color: colors.text,
                    font: {
                        size: 11
                    }
                }
            }
        },
        plugins: {
            ...defaultOptions.plugins,
            tooltip: {
                ...defaultOptions.plugins.tooltip
            }
        }
    };

    // Chart.js global defaults
    function applyGlobalDefaults() {
        if (typeof Chart !== 'undefined') {
            Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
            Chart.defaults.font.size = 12;
            Chart.defaults.color = colors.text;
            Chart.defaults.plugins.legend.display = false;
        }
    }

    // Create gradient for line charts
    function createGradient(ctx, colorStart, colorEnd) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 200);
        gradient.addColorStop(0, colorStart);
        gradient.addColorStop(1, colorEnd);
        return gradient;
    }

    // Get severity color
    function getSeverityColor(severity) {
        const severityLower = severity.toLowerCase();
        return colors[severityLower] || colors.other;
    }

    // Get severity background color
    function getSeverityBgColor(severity) {
        const severityLower = severity.toLowerCase();
        return colors[`${severityLower}Bg`] || colors.otherBg;
    }

    // Public API
    return {
        colors,
        donutColors,
        defaultOptions,
        donutOptions,
        lineOptions,
        barOptions,
        applyGlobalDefaults,
        createGradient,
        getSeverityColor,
        getSeverityBgColor
    };
})();

// Apply global defaults when loaded
if (typeof Chart !== 'undefined') {
    ChartConfig.applyGlobalDefaults();
}

// Export
window.ChartConfig = ChartConfig;
