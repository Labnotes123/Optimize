import time
import pandas as pd
import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import subprocess
import sys

# Ensure dependencies are installed
def install_missing_packages():
    required_packages = ["selenium", "webdriver-manager", "pandas", "streamlit"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_missing_packages()

# Function to extract data
def scrape_biological_variation():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    base_url = "https://biologicalvariation.eu/bv_specifications?page={}"  # Adjust for pagination
    page = 1
    data = []
    
    while True:
        driver.get(base_url.format(page))
        time.sleep(3)
        
        # Extract Measurand, Specification (View Details), Reference
        measurands = driver.find_elements(By.CSS_SELECTOR, "td:nth-child(1)")
        specifications = driver.find_elements(By.CSS_SELECTOR, "td:nth-child(2) button")
        references = driver.find_elements(By.CSS_SELECTOR, "td:nth-child(3)")
        
        if not measurands:
            break  # Stop if no more data
        
        for i in range(len(measurands)):
            measurand = measurands[i].text.strip()
            reference = references[i].text.strip() if i < len(references) else ""
            
            # Click on View Details
            within_subject = "N/A"
            between_subject = "N/A"
            try:
                driver.execute_script("arguments[0].click();", specifications[i])
                time.sleep(3)
                
                # Extract Estimates of Within Subject & Between Subject
                within_subject_elem = driver.find_elements(By.CSS_SELECTOR, "td:contains('Estimates of Within Subject') + td")
                between_subject_elem = driver.find_elements(By.CSS_SELECTOR, "td:contains('Estimates of Between Subject') + td")
                
                if within_subject_elem:
                    within_subject = within_subject_elem[0].text.strip()
                if between_subject_elem:
                    between_subject = between_subject_elem[0].text.strip()
                
                driver.back()
                time.sleep(3)
            except Exception as e:
                print(f"Error extracting details: {e}")
            
            data.append([measurand, reference, within_subject, between_subject])
        
        page += 1  # Move to next page
    
    driver.quit()
    
    df = pd.DataFrame(data, columns=["Measurand", "Reference", "Estimates of Within Subject", "Estimates of Between Subject"])
    return df

# Streamlit UI
st.title("Biological Variation Data Scraper")
if st.button("Start Scraping"):
    df = scrape_biological_variation()
    st.write(df)
    st.download_button("Download Data", df.to_csv(index=False), "biological_variation_data.csv", "text/csv")
