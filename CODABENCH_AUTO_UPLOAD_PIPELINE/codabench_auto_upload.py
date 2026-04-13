"""
CodaBench Automated File Upload Script
Automatically uploads files one at a time to CodaBench submission system
"""

import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

# Path to the folder containing your submission files
SUBMISSION_FOLDER = r"C:\Users\neeld\Downloads\CS780_PROJECT\SUBMISSIONS\approach_7"

# CodaBench competition URL (the participants tab)
CODABENCH_URL = "https://www.codabench.org/competitions/YOUR_COMPETITION_ID/?tab=participate"

# File extensions to upload (modify based on what you're submitting)
VALID_EXTENSIONS = ['.zip']  

# Time to wait between uploads (seconds) - adjust if needed
WAIT_BETWEEN_UPLOADS = 3

# Maximum time to wait for elements to load (seconds)
TIMEOUT = 20

# ============================================================================
# SCRIPT
# ============================================================================

def setup_driver():
    """Initialize and return Chrome WebDriver"""
    options = webdriver.ChromeOptions()
    
    # Keep browser open if you want to see what's happening
    # options.add_argument('--headless')  # Uncomment to run in background
    
    # Use existing Chrome profile (to maintain login session)
    # Uncomment and modify path if you want to use your existing Chrome profile:
    # options.add_argument("user-data-dir=/path/to/chrome/profile")
    
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    return driver


def get_submission_files(folder_path):
    """Get list of files to submit from the folder"""
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
            files.append(file)
    
    files.sort()  # Sort alphabetically
    return files


def wait_for_element(driver, by, value, timeout=TIMEOUT):
    """Wait for an element to be present and return it"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except TimeoutException:
        print(f"Timeout waiting for element: {by}={value}")
        return None


def wait_for_clickable(driver, by, value, timeout=TIMEOUT):
    """Wait for an element to be clickable and return it"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((by, value))
        )
        return element
    except TimeoutException:
        print(f"Timeout waiting for clickable element: {by}={value}")
        return None


def upload_file(driver, file_path, upload_number):
    """Upload a single file to CodaBench"""
    print(f"\n{'='*60}")
    print(f"Upload #{upload_number}: {file_path.name}")
    print(f"{'='*60}")
    
    try:
        # Option 1: Look for "Submit" or "Upload" button
        # Try multiple common selectors
        button_selectors = [
            (By.XPATH, "//button[contains(text(), 'Submit')]"),
            (By.XPATH, "//button[contains(text(), 'Upload')]"),
            (By.XPATH, "//a[contains(text(), 'Submit')]"),
            (By.CSS_SELECTOR, "button.submit-btn"),
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.ID, "submit-button"),
        ]
        
        submit_button = None
        for by, selector in button_selectors:
            try:
                submit_button = wait_for_clickable(driver, by, selector, timeout=5)
                if submit_button:
                    print(f"Found submit button using: {selector}")
                    break
            except:
                continue
        
        if not submit_button:
            print("Could not find submit button. Please inspect the page and update selectors.")
            return False
        
        # Click the submit button
        submit_button.click()
        print("Clicked submit button")
        time.sleep(2)  # Wait for upload dialog
        
        # Option 2: Find file input element
        # Try multiple common selectors for file input
        file_input_selectors = [
            (By.CSS_SELECTOR, "input[type='file']"),
            (By.XPATH, "//input[@type='file']"),
            (By.NAME, "file"),
            (By.ID, "file-upload"),
        ]
        
        file_input = None
        for by, selector in file_input_selectors:
            try:
                file_input = wait_for_element(driver, by, selector, timeout=5)
                if file_input:
                    print(f"Found file input using: {selector}")
                    break
            except:
                continue
        
        if not file_input:
            print("Could not find file input element. Checking for visible inputs...")
            # Sometimes the input is hidden, try to find any file input
            try:
                file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
            except NoSuchElementException:
                print("ERROR: No file input found on page!")
                return False
        
        # Upload the file
        file_input.send_keys(str(file_path.absolute()))
        print(f"File path sent to input: {file_path.absolute()}")
        time.sleep(1)
        
        # Look for confirm/upload button after file selection
        confirm_selectors = [
            (By.XPATH, "//button[contains(text(), 'Upload')]"),
            (By.XPATH, "//button[contains(text(), 'Confirm')]"),
            (By.XPATH, "//button[contains(text(), 'Submit')]"),
            (By.CSS_SELECTOR, "button[type='submit']"),
        ]
        
        confirm_button = None
        for by, selector in confirm_selectors:
            try:
                confirm_button = wait_for_clickable(driver, by, selector, timeout=3)
                if confirm_button and confirm_button.is_displayed():
                    print(f"Found confirm button using: {selector}")
                    break
            except:
                continue
        
        if confirm_button:
            confirm_button.click()
            print("Clicked confirm button")
        
        # Wait for upload to complete
        print("Waiting for upload to complete...")
        time.sleep(WAIT_BETWEEN_UPLOADS)
        
        # Check for success message (optional)
        try:
            success_indicators = [
                (By.XPATH, "//*[contains(text(), 'success')]"),
                (By.XPATH, "//*[contains(text(), 'Success')]"),
                (By.XPATH, "//*[contains(text(), 'uploaded')]"),
                (By.CLASS_NAME, "alert-success"),
            ]
            
            for by, selector in success_indicators:
                try:
                    success_elem = driver.find_element(by, selector)
                    if success_elem.is_displayed():
                        print("✓ Upload successful!")
                        break
                except:
                    continue
        except:
            pass
        
        print(f"Completed upload of {file_path.name}")
        return True
        
    except Exception as e:
        print(f"ERROR during upload: {str(e)}")
        return False


def main():
    """Main execution function"""
    print("CodaBench Automated File Upload Script")
    print("="*60)
    
    # Get files to upload
    try:
        files = get_submission_files(SUBMISSION_FOLDER)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    if not files:
        print(f"No files found in {SUBMISSION_FOLDER}")
        print(f"Looking for files with extensions: {VALID_EXTENSIONS}")
        return
    
    print(f"\nFound {len(files)} file(s) to upload:")
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file.name}")
    
    # Initialize browser
    print("\nInitializing browser...")
    driver = setup_driver()
    
    try:
        # Navigate to CodaBench
        print(f"\nNavigating to: {CODABENCH_URL}")
        driver.get(CODABENCH_URL)
        
        # Wait for user to confirm they're logged in
        print("\n" + "="*60)
        print("IMPORTANT: Make sure you are logged in to CodaBench!")
        print("="*60)
        input("Press ENTER when you're ready to start uploading...")
        
        # Upload each file
        successful = 0
        failed = 0
        
        for i, file in enumerate(files, 1):
            success = upload_file(driver, file, i)
            if success:
                successful += 1
            else:
                failed += 1
                print(f"Failed to upload {file.name}")
                
                # Ask if user wants to continue
                response = input("\nContinue with next file? (y/n): ")
                if response.lower() != 'y':
                    break
        
        # Summary
        print("\n" + "="*60)
        print("UPLOAD SUMMARY")
        print("="*60)
        print(f"Total files: {len(files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print("="*60)
        
        # Keep browser open
        input("\nPress ENTER to close browser...")
        
    finally:
        driver.quit()
        print("Browser closed.")


if __name__ == "__main__":
    main()
