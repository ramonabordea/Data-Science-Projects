# Check if required packages are installed, if not install them
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ['selenium', 'python-dotenv', 'webdriver-manager']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        install_package(package)

# Now import the required packages
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_driver():
    """Set up Chrome driver with options"""
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    
    # Initialize the Chrome driver with automatic ChromeDriver installation
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def login_to_linkedin(driver, email, password):
    """Login to LinkedIn"""
    driver.get("https://www.linkedin.com/login")
    
    # Wait for email field and enter email
    email_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "username"))
    )
    email_field.send_keys(email)
    
    # Enter password
    password_field = driver.find_element(By.ID, "password")
    password_field.send_keys(password)
    
    # Click login button
    login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    login_button.click()
    
    time.sleep(3)

def get_about_section(driver, profile_url):
    """Get the About section content"""
    driver.get(profile_url)
    
    try:
        about_section = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div#about"))
        )
        
        # Expand the About section if needed
        try:
            see_more_button = about_section.find_element(By.CSS_SELECTOR, "button.inline-show-more-text__button")
            see_more_button.click()
            time.sleep(1)
        except:
            pass
        
        # Get the text content
        about_text = about_section.find_element(By.CSS_SELECTOR, "div.inline-show-more-text").text
        return about_text
        
    except Exception as e:
        print(f"Error getting About section: {str(e)}")
        return None

def main():
    # Get LinkedIn credentials
    EMAIL = input("Enter your LinkedIn email: ")
    PASSWORD = input("Enter your LinkedIn password: ")
    
    profile_url = "https://www.linkedin.com/in/ramona-bordea/"
    
    try:
        print("Setting up browser...")
        driver = setup_driver()
        
        print("Logging in to LinkedIn...")
        login_to_linkedin(driver, EMAIL, PASSWORD)
        
        print("Getting About section...")
        about_text = get_about_section(driver, profile_url)
        
        if about_text:
            print("\nAbout Section:")
            print("-" * 50)
            print(about_text)
            print("-" * 50)
        else:
            print("Could not find About section")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        print("Closing browser...")
        driver.quit()

if __name__ == "__main__":
    main()
