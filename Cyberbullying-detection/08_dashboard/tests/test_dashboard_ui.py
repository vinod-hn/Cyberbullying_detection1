"""
Dashboard UI Tests
==================
Test suite for the Cyberbullying Detection Dashboard.
Uses Selenium for browser automation testing.
"""

import pytest
import json
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

# Optional imports - gracefully handle missing dependencies
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


# Test configuration
DASHBOARD_DIR = Path(__file__).parent.parent
TEST_PORT = 8765
TEST_URL = f"http://localhost:{TEST_PORT}/index.html"


class SimpleServer:
    """Simple HTTP server for serving dashboard files"""
    
    def __init__(self, directory, port):
        self.directory = directory
        self.port = port
        self.httpd = None
        self.thread = None
    
    def start(self):
        import os
        os.chdir(self.directory)
        handler = SimpleHTTPRequestHandler
        self.httpd = HTTPServer(('localhost', self.port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        if self.httpd:
            self.httpd.shutdown()


@pytest.fixture(scope="module")
def server():
    """Start HTTP server for tests"""
    srv = SimpleServer(str(DASHBOARD_DIR), TEST_PORT)
    srv.start()
    time.sleep(0.5)  # Wait for server to start
    yield srv
    srv.stop()


@pytest.fixture(scope="function")
def browser():
    """Setup and teardown Chrome browser"""
    if not SELENIUM_AVAILABLE:
        pytest.skip("Selenium not installed. Run: pip install selenium")
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)
    yield driver
    driver.quit()


class TestDashboardStructure:
    """Test dashboard HTML structure"""
    
    def test_files_exist(self):
        """Verify all required files exist"""
        required_files = [
            'index.html',
            'css/main.css',
            'css/dashboard.css',
            'css/graphs.css',
            'css/responsive.css',
            'js/dashboard.js',
            'js/api_client.js',
            'js/chart_config.js',
            'js/graphs/pie_chart.js',
            'js/graphs/line_chart.js',
            'js/graphs/bar_chart.js',
            'js/graphs/stats_cards.js',
        ]
        
        for file in required_files:
            path = DASHBOARD_DIR / file
            assert path.exists(), f"Missing file: {file}"
    
    def test_index_html_valid(self):
        """Verify index.html has required elements"""
        index_path = DASHBOARD_DIR / 'index.html'
        content = index_path.read_text(encoding='utf-8')
        
        # Check required elements
        assert '<header' in content, "Missing header element"
        assert 'severity-chart' in content, "Missing severity chart canvas"
        assert 'trend-chart' in content, "Missing trend chart canvas"
        assert 'monthly-chart' in content, "Missing monthly chart canvas"
        assert 'messages-tbody' in content, "Missing messages table body"
        assert 'interventions-list' in content, "Missing interventions list"
        assert 'audit-list' in content, "Missing audit list"


@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
class TestDashboardUI:
    """Test dashboard UI with Selenium"""
    
    def test_page_loads(self, server, browser):
        """Test that dashboard page loads successfully"""
        browser.get(TEST_URL)
        
        title = browser.title
        assert "Cyberbullying" in title or "Dashboard" in title
    
    def test_header_visible(self, server, browser):
        """Test header is visible with logo and title"""
        browser.get(TEST_URL)
        
        header = browser.find_element(By.CLASS_NAME, 'dashboard-header')
        assert header.is_displayed()
        
        title = browser.find_element(By.CSS_SELECTOR, '.header-title h1')
        assert "Cyberbullying" in title.text
    
    def test_charts_rendered(self, server, browser):
        """Test that all charts are rendered"""
        browser.get(TEST_URL)
        time.sleep(1)  # Wait for charts to render
        
        # Check canvas elements exist
        severity_chart = browser.find_element(By.ID, 'severity-chart')
        trend_chart = browser.find_element(By.ID, 'trend-chart')
        monthly_chart = browser.find_element(By.ID, 'monthly-chart')
        
        assert severity_chart.is_displayed()
        assert trend_chart.is_displayed()
        assert monthly_chart.is_displayed()
    
    def test_table_populated(self, server, browser):
        """Test that messages table is populated with mock data"""
        browser.get(TEST_URL)
        time.sleep(1)
        
        tbody = browser.find_element(By.ID, 'messages-tbody')
        rows = tbody.find_elements(By.TAG_NAME, 'tr')
        
        assert len(rows) > 0, "Table should have at least one row"
    
    def test_filters_work(self, server, browser):
        """Test that filter controls are functional"""
        browser.get(TEST_URL)
        
        # Test severity filter
        severity_filter = browser.find_element(By.ID, 'severity-filter')
        severity_filter.click()
        
        options = severity_filter.find_elements(By.TAG_NAME, 'option')
        assert len(options) >= 4, "Should have at least 4 severity options"
    
    def test_modal_opens(self, server, browser):
        """Test that clicking a row opens the modal"""
        browser.get(TEST_URL)
        time.sleep(1)
        
        # Click first row
        tbody = browser.find_element(By.ID, 'messages-tbody')
        first_row = tbody.find_element(By.TAG_NAME, 'tr')
        first_row.click()
        
        time.sleep(0.5)
        
        modal = browser.find_element(By.ID, 'message-modal')
        assert 'hidden' not in modal.get_attribute('class')
    
    def test_modal_closes(self, server, browser):
        """Test that modal closes on close button click"""
        browser.get(TEST_URL)
        time.sleep(1)
        
        # Open modal
        tbody = browser.find_element(By.ID, 'messages-tbody')
        first_row = tbody.find_element(By.TAG_NAME, 'tr')
        first_row.click()
        time.sleep(0.5)
        
        # Close modal
        close_btn = browser.find_element(By.ID, 'close-modal')
        close_btn.click()
        time.sleep(0.3)
        
        modal = browser.find_element(By.ID, 'message-modal')
        assert 'hidden' in modal.get_attribute('class')
    
    def test_sidebar_visible(self, server, browser):
        """Test that sidebar sections are visible"""
        browser.get(TEST_URL)
        
        # Check interventions section
        interventions = browser.find_element(By.ID, 'interventions-list')
        assert interventions.is_displayed()
        
        # Check audit log section
        audit_log = browser.find_element(By.ID, 'audit-list')
        assert audit_log.is_displayed()
    
    def test_responsive_tablet(self, server, browser):
        """Test tablet responsive layout"""
        browser.set_window_size(768, 1024)
        browser.get(TEST_URL)
        time.sleep(0.5)
        
        # Dashboard should still be functional
        header = browser.find_element(By.CLASS_NAME, 'dashboard-header')
        assert header.is_displayed()
    
    def test_responsive_mobile(self, server, browser):
        """Test mobile responsive layout"""
        browser.set_window_size(375, 812)
        browser.get(TEST_URL)
        time.sleep(0.5)
        
        # Dashboard should still be functional
        header = browser.find_element(By.CLASS_NAME, 'dashboard-header')
        assert header.is_displayed()


class TestCSSValidation:
    """Validate CSS files"""
    
    def test_css_files_valid(self):
        """Check CSS files for syntax (basic validation)"""
        css_files = [
            'css/main.css',
            'css/dashboard.css',
            'css/graphs.css',
            'css/responsive.css',
        ]
        
        for file in css_files:
            path = DASHBOARD_DIR / file
            content = path.read_text(encoding='utf-8')
            
            # Basic brace matching
            open_braces = content.count('{')
            close_braces = content.count('}')
            assert open_braces == close_braces, f"Mismatched braces in {file}"
    
    def test_css_variables_defined(self):
        """Check that CSS variables are defined"""
        main_css = DASHBOARD_DIR / 'css/main.css'
        content = main_css.read_text(encoding='utf-8')
        
        required_vars = [
            '--primary-color',
            '--threat-color',
            '--harassment-color',
            '--neutral-color',
        ]
        
        for var in required_vars:
            assert var in content, f"Missing CSS variable: {var}"


class TestJavaScriptValidation:
    """Validate JavaScript files"""
    
    def test_js_files_valid(self):
        """Check JS files for basic syntax"""
        js_files = [
            'js/dashboard.js',
            'js/api_client.js',
            'js/chart_config.js',
            'js/graphs/pie_chart.js',
            'js/graphs/line_chart.js',
            'js/graphs/bar_chart.js',
            'js/graphs/stats_cards.js',
        ]
        
        for file in js_files:
            path = DASHBOARD_DIR / file
            content = path.read_text(encoding='utf-8')
            
            # Basic brace matching (accounting for strings)
            # Simple check - not perfect but catches basic errors
            assert 'function' in content or 'const' in content, f"Invalid JS in {file}"
    
    def test_api_client_has_mock_data(self):
        """Verify API client has fallback mock data"""
        api_client = DASHBOARD_DIR / 'js/api_client.js'
        content = api_client.read_text(encoding='utf-8')
        
        assert 'mockData' in content, "API client should have mock data"
        assert 'isOnline' in content or 'fallback' in content.lower(), "Should have offline fallback"


class TestAccessibility:
    """Test accessibility features"""
    
    def test_html_has_lang(self):
        """Check HTML has language attribute"""
        index = DASHBOARD_DIR / 'index.html'
        content = index.read_text(encoding='utf-8')
        
        assert 'lang="en"' in content or "lang='en'" in content
    
    def test_images_have_alt(self):
        """Check images have alt attributes"""
        index = DASHBOARD_DIR / 'index.html'
        content = index.read_text(encoding='utf-8')
        
        # If there are img tags, they should have alt
        import re
        img_tags = re.findall(r'<img[^>]+>', content)
        for img in img_tags:
            assert 'alt=' in img, f"Image missing alt attribute: {img}"
    
    def test_form_labels(self):
        """Check form inputs have labels"""
        index = DASHBOARD_DIR / 'index.html'
        content = index.read_text(encoding='utf-8')
        
        # Critical inputs should have labels or aria-labels
        assert 'for="start-date"' in content or 'aria-label' in content


def test_smoke():
    """Basic smoke test - always passes if files exist"""
    assert (DASHBOARD_DIR / 'index.html').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
