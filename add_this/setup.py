from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--user-data-dir=/Users/aparna/selenium-profile")
driver = webdriver.Chrome(options=chrome_options)
driver.get("https://chat.openai.com/")

