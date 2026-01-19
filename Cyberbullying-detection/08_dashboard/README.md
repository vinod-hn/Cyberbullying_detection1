# Cyberbullying Detection Dashboard

A modern, responsive operator dashboard for monitoring and analyzing cyberbullying incidents in educational environments.

## 🚀 Quick Start

### Option 1: Open Directly
Simply open `index.html` in a modern web browser (Chrome, Firefox, Edge, Safari).

```bash
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

### Option 2: With Local Server (Recommended)
For full API integration, start the backend server first:

```bash
# From project root
cd Cyberbullying-detection
pip install -r 06_api/requirements_api.txt
python run_api.py
```

Then open http://localhost:8000 in your browser.

### Option 3: Python HTTP Server
```bash
cd 08_dashboard
python -m http.server 8080
```
Open http://localhost:8080 in your browser.

## 📁 File Structure

```
08_dashboard/
├── index.html              # Main dashboard entry point
├── README.md               # This file
│
├── css/
│   ├── main.css           # Core styles, variables, reset
│   ├── dashboard.css      # Layout components
│   ├── graphs.css         # Chart-specific styles
│   └── responsive.css     # Mobile/tablet breakpoints
│
├── js/
│   ├── dashboard.js       # Main orchestration script
│   ├── api_client.js      # Backend API communication
│   ├── chart_config.js    # Chart.js configuration
│   └── graphs/
│       ├── pie_chart.js   # Severity distribution donut
│       ├── line_chart.js  # Daily alerts trend
│       ├── bar_chart.js   # Monthly trend
│       └── stats_cards.js # Sidebar statistics
│
├── components/
│   ├── graph_container.html  # Chart wrapper template
│   ├── stat_card.html        # Statistics card template
│   └── legend.html           # Chart legend template
│
├── assets/
│   └── images/
│       ├── whatsapp.svg      # Platform icon
│       ├── telegram.svg      # Platform icon
│       ├── classroom.svg     # Platform icon
│       ├── severity_threat.svg
│       ├── severity_harassment.svg
│       ├── severity_insult.svg
│       └── severity_neutral.svg
│
└── tests/
    └── test_dashboard_ui.py  # Selenium UI tests
```

## ✨ Features

### Interactive Charts
- **Severity Distribution (Donut)**: Shows breakdown of threat, harassment, insult, and neutral classifications
- **Daily Alerts Trend (Line)**: Weekly pattern of detected incidents
- **Monthly Trend (Bar)**: Historical view of incident volumes

### Data Table
- Sortable columns (click headers)
- Pagination with page navigation
- Platform badges (WhatsApp, Telegram, Classroom)
- Severity badges with color coding
- Confidence scores

### Filters
- Date range selection
- Platform filter
- Severity filter
- Real-time search

### Sidebar
- **Intervention Suggestions**: AI-recommended actions for flagged messages
- **Audit & Compliance Log**: Track all operator actions

### Modal Details
- Full message view
- Prediction explanation
- Feedback submission
- Escalation options

### Offline Mode
The dashboard works offline with mock data when the API is unavailable.

## 🎨 Customization

### Colors
Edit CSS variables in `css/main.css`:

```css
:root {
    --primary-color: #4f46e5;
    --threat-color: #FF5252;
    --harassment-color: #FFA000;
    --insult-color: #FFD600;
    --neutral-color: #4CAF50;
}
```

### API Endpoint
Update the base URL in `js/api_client.js`:

```javascript
const BASE_URL = 'http://localhost:8000';
```

## 🧪 Testing

### Install Test Dependencies
```bash
pip install pytest selenium
```

### Run Tests
```bash
cd 08_dashboard
python -m pytest tests/ -v
```

### Manual Testing Checklist
- [ ] Page loads without errors
- [ ] All three charts render
- [ ] Table populates with data
- [ ] Filters update table
- [ ] Clicking row opens modal
- [ ] Modal closes on X or outside click
- [ ] Export button works
- [ ] Responsive on tablet (768px)
- [ ] Responsive on mobile (375px)

## 🔌 API Integration

The dashboard expects these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stats` | GET | Dashboard statistics |
| `/predictions` | GET | Message predictions list |
| `/predict` | POST | Analyze new message |
| `/export/reports` | POST | Export data as CSV |
| `/feedback` | POST | Submit operator feedback |

### Stats Response Format
```json
{
    "severity_counts": {
        "threat": 42,
        "harassment": 78,
        "insult": 95,
        "neutral": 185
    },
    "daily_alerts": {
        "Mon": 25,
        "Tue": 32,
        ...
    },
    "monthly_trend": {
        "Jan": 120,
        "Feb": 145,
        ...
    }
}
```

## 🌐 Browser Support

| Browser | Version |
|---------|---------|
| Chrome  | 90+     |
| Firefox | 88+     |
| Edge    | 90+     |
| Safari  | 14+     |

## 📱 Responsive Breakpoints

| Device | Width | Layout |
|--------|-------|--------|
| Desktop | 1200px+ | 3-column charts, table + sidebar |
| Laptop | 992px | 2-column charts |
| Tablet | 768px | Stacked layout |
| Mobile | 480px | Single column, compact |

## ♿ Accessibility

- WCAG 2.1 AA compliant
- Keyboard navigation support
- Screen reader compatible
- Reduced motion support
- High contrast colors

## 🔧 Troubleshooting

### Charts not rendering
1. Check browser console for errors
2. Verify Chart.js CDN is accessible
3. Clear browser cache

### API connection failed
1. Check if backend server is running
2. Verify `BASE_URL` in api_client.js
3. Dashboard will use mock data as fallback

### Styles not loading
1. Check file paths are correct
2. Verify CSS files exist
3. Clear browser cache

## 📄 License

This project is part of the Cyberbullying Detection System.
See main project LICENSE for details.

## 🙏 Credits

- [Chart.js](https://www.chartjs.org/) - Visualization library
- [Simple Icons](https://simpleicons.org/) - Platform icons
- Design inspired by modern dashboard best practices
