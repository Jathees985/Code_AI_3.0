from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# --- Selenium Setup ---
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

service = Service("chromedriver-win64/chromedriver.exe")  # Change if chromedriver not in PATH
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Open the page
    url = "https://www.jac-lang.org/learn/introduction/"
    driver.get(url)

    # Wait for page to load main content
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "main"))
    )

    time.sleep(2)  # Ensure JavaScript-rendered content is loaded

    # Get all visible text inside main content
    main_content = driver.find_element(By.TAG_NAME, "main")
    page_text = main_content.text

    # Get all code snippets separately
    code_snippets = driver.find_elements(By.TAG_NAME, "code")
    code_blocks = [code.text for code in code_snippets if code.text.strip()]

    # Output results
    print("\n===== PAGE TEXT =====\n")
    print(page_text)

    print("\n===== CODE SNIPPETS =====\n")
    for idx, snippet in enumerate(code_blocks, start=1):
        print(f"--- Code Block {idx} ---\n{snippet}\n")

finally:
    driver.quit()
