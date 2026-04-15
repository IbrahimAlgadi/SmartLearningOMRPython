from playwright.sync_api import sync_playwright
import pathlib

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    uri = pathlib.Path("answer_page_ar.html").resolve().as_uri()
    page.goto(uri, wait_until="networkidle")
    # Take screenshot of the full page
    page.screenshot(path="test_blank_ar.png", full_page=True)
    browser.close()
print("Created test_blank_ar.png")