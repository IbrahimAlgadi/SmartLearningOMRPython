import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from playwright.sync_api import sync_playwright
import pathlib

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    uri = (_ROOT / "sheets" / "answer_page_ar.html").resolve().as_uri()
    page.goto(uri, wait_until="networkidle")
    # Take screenshot of the full page
    page.screenshot(path="test_blank_ar.png", full_page=True)
    browser.close()
print("Created test_blank_ar.png")