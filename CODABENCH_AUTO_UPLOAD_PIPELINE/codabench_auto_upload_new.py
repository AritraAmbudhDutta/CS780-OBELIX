"""
CodaBench Automated File Upload Script
Automatically uploads files one at a time to CodaBench submission system
Competition: Reinforcement Learning Challenge! CS780
"""

import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
import getpass

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

# Path to the folder containing your submission files
SUBMISSION_FOLDER = r"C:\Users\neeld\Downloads\CS780_PROJECT\SUBMISSIONS\approach_7"

# CodaBench credentials (leave empty to be prompted)
CODABENCH_USERNAME = ""  # Your CodaBench username/email
CODABENCH_PASSWORD = ""  # Your CodaBench password (leave empty for secure input)

# Competition URLs
CODABENCH_LOGIN_URL = "https://www.codabench.org/accounts/login/"
COMPETITION_ID = "14572"
COMPETITION_URL = f"https://www.codabench.org/competitions/{COMPETITION_ID}/"
SUBMISSION_TAB_URL = f"https://www.codabench.org/competitions/{COMPETITION_ID}/#/participate-tab"

# File extensions to upload (modify based on what you're submitting)
VALID_EXTENSIONS = ['.zip']  

# Time to wait between uploads (seconds) - adjust if needed
WAIT_BETWEEN_UPLOADS = 5

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


def login_to_codabench(driver, username, password):
    """Log in to CodaBench"""
    print("\n" + "="*60)
    print("LOGGING IN TO CODABENCH")
    print("="*60)
    
    try:
        # Navigate to login page
        print(f"Navigating to login page...")
        driver.get(CODABENCH_LOGIN_URL)
        time.sleep(2)
        
        # Find username/email field
        username_selectors = [
            (By.ID, "id_username"),
            (By.NAME, "username"),
            (By.CSS_SELECTOR, "input[type='text']"),
            (By.CSS_SELECTOR, "input[type='email']"),
            (By.XPATH, "//input[@placeholder='Username' or @placeholder='Email']"),
        ]
        
        username_field = None
        for by, selector in username_selectors:
            try:
                username_field = wait_for_element(driver, by, selector, timeout=5)
                if username_field:
                    print(f"Found username field")
                    break
            except:
                continue
        
        if not username_field:
            print("ERROR: Could not find username/email field")
            return False
        
        # Find password field
        password_field = wait_for_element(driver, By.CSS_SELECTOR, "input[type='password']")
        if not password_field:
            print("ERROR: Could not find password field")
            return False
        
        # Enter credentials
        print("Entering credentials...")
        username_field.clear()
        username_field.send_keys(username)
        time.sleep(0.5)
        
        password_field.clear()
        password_field.send_keys(password)
        time.sleep(0.5)
        
        # Find and click login button
        login_button_selectors = [
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.XPATH, "//button[contains(text(), 'Log') or contains(text(), 'Sign')]"),
            (By.ID, "submit-id-submit"),
            (By.NAME, "submit"),
        ]
        
        login_button = None
        for by, selector in login_button_selectors:
            try:
                login_button = wait_for_clickable(driver, by, selector, timeout=3)
                if login_button:
                    break
            except:
                continue
        
        if not login_button:
            # Try submitting form with Enter key
            print("Submitting with Enter key...")
            password_field.send_keys(Keys.RETURN)
        else:
            print("Clicking login button...")
            login_button.click()
        
        # Wait for login to complete
        time.sleep(3)
        
        # Check if login was successful
        if "login" in driver.current_url.lower():
            print("ERROR: Login failed - still on login page")
            print("Please check your credentials")
            return False
        
        print("✓ Login successful!")
        return True
        
    except Exception as e:
        print(f"ERROR during login: {str(e)}")
        return False


def upload_file(driver, file_path, upload_number):
    """Upload a single file to CodaBench"""
    print(f"\n{'='*60}")
    print(f"Upload #{upload_number}: {file_path.name}")
    print(f"{'='*60}")
    
    try:
        # Navigate to submission tab
        print("Navigating to submission tab...")
        driver.get(SUBMISSION_TAB_URL)
        time.sleep(3)
        
        # Try to find the upload/submit button on the participate tab
        button_selectors = [
            # Common button selectors for CodaBench
            (By.XPATH, "//button[contains(text(), 'Submit') or contains(text(), 'submit')]"),
            (By.XPATH, "//button[contains(text(), 'Upload') or contains(text(), 'upload')]"),
            (By.XPATH, "//a[contains(text(), 'Submit') or contains(text(), 'submit')]"),
            (By.CSS_SELECTOR, "button.btn-primary"),
            (By.CSS_SELECTOR, "button.submit-btn"),
            (By.CSS_SELECTOR, ".upload-button"),
            (By.XPATH, "//button[contains(@class, 'submit')]"),
            (By.XPATH, "//input[@value='Submit']"),
        ]
        
        submit_button = None
        for by, selector in button_selectors:
            try:
                submit_button = wait_for_clickable(driver, by, selector, timeout=5)
                if submit_button and submit_button.is_displayed():
                    print(f"✓ Found submit button")
                    break
            except:
                continue
        
        if not submit_button:
            print("\nWARNING: Could not find submit button automatically.")
            print("Please inspect the page to find the correct selector.")
            print("Current page URL:", driver.current_url)
            input("\nManually click the submit button, then press ENTER to continue...")
        else:
            # Click the submit button
            submit_button.click()
            print("Clicked submit button")
        
        time.sleep(2)  # Wait for upload dialog/modal
        
        # Find file input element - try multiple approaches
        file_input_selectors = [
            (By.CSS_SELECTOR, "input[type='file']"),
            (By.XPATH, "//input[@type='file']"),
            (By.NAME, "file"),
            (By.ID, "file-upload"),
            (By.ID, "id_file"),
            (By.NAME, "submission_file"),
        ]
        
        file_input = None
        for by, selector in file_input_selectors:
            try:
                # Try to find visible inputs first
                elements = driver.find_elements(by, selector)
                for elem in elements:
                    # File inputs are often hidden, so we don't check visibility
                    file_input = elem
                    print(f"✓ Found file input")
                    break
                if file_input:
                    break
            except:
                continue
        
        if not file_input:
            print("\nERROR: Could not find file input element")
            print("The page structure might have changed.")
            return False
        
        # Upload the file
        print(f"Uploading file: {file_path.name}")
        file_input.send_keys(str(file_path.absolute()))
        print(f"✓ File path sent to input")
        time.sleep(2)
        
        # Look for confirm/submit button after file selection
        confirm_selectors = [
            (By.XPATH, "//button[contains(text(), 'Upload') or contains(text(), 'Submit')]"),
            (By.XPATH, "//button[contains(text(), 'Confirm')]"),
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.CSS_SELECTOR, ".btn-primary"),
            (By.XPATH, "//button[@type='submit']"),
        ]
        
        # Wait a moment for any file validation
        time.sleep(1)
        
        confirm_button = None
        for by, selector in confirm_selectors:
            try:
                buttons = driver.find_elements(by, selector)
                for btn in buttons:
                    if btn.is_displayed() and btn.is_enabled():
                        confirm_button = btn
                        print(f"✓ Found confirm button")
                        break
                if confirm_button:
                    break
            except:
                continue
        
        if confirm_button:
            print("Clicking confirm/submit button...")
            confirm_button.click()
            time.sleep(1)
        else:
            print("No confirm button found - file may auto-upload")
        
        # Wait for upload to complete
        print("Waiting for upload to complete...")
        time.sleep(WAIT_BETWEEN_UPLOADS)
        
        # Check for success/error messages
        try:
            # Look for success indicators
            success_indicators = [
                (By.XPATH, "//*[contains(text(), 'success') or contains(text(), 'Success')]"),
                (By.XPATH, "//*[contains(text(), 'submitted') or contains(text(), 'Submitted')]"),
                (By.CLASS_NAME, "alert-success"),
                (By.CLASS_NAME, "success"),
            ]
            
            for by, selector in success_indicators:
                try:
                    success_elem = driver.find_element(by, selector)
                    if success_elem.is_displayed():
                        print("✓ Upload successful!")
                        return True
                except:
                    continue
            
            # Look for error indicators
            error_indicators = [
                (By.XPATH, "//*[contains(text(), 'error') or contains(text(), 'Error')]"),
                (By.XPATH, "//*[contains(text(), 'failed') or contains(text(), 'Failed')]"),
                (By.CLASS_NAME, "alert-danger"),
                (By.CLASS_NAME, "error"),
            ]
            
            for by, selector in error_indicators:
                try:
                    error_elem = driver.find_element(by, selector)
                    if error_elem.is_displayed():
                        print(f"⚠ Upload error detected: {error_elem.text}")
                        return False
                except:
                    continue
                    
        except:
            pass
        
        print(f"✓ Completed upload of {file_path.name}")
        return True
        
    except Exception as e:
        print(f"ERROR during upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function"""
    print("="*60)
    print("CodaBench Automated File Upload Script")
    print("Competition: Reinforcement Learning Challenge! CS780")
    print("="*60)
    
    # Get credentials
    username = CODABENCH_USERNAME
    password = CODABENCH_PASSWORD
    
    if not username:
        print("\nPlease enter your CodaBench credentials:")
        username = input("Username/Email: ").strip()
    
    if not password:
        password = getpass.getpass("Password: ")
    
    if not username or not password:
        print("ERROR: Username and password are required")
        return
    
    # Get files to upload
    try:
        files = get_submission_files(SUBMISSION_FOLDER)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease update SUBMISSION_FOLDER in the script configuration")
        return
    
    if not files:
        print(f"\nNo files found in {SUBMISSION_FOLDER}")
        print(f"Looking for files with extensions: {VALID_EXTENSIONS}")
        return
    
    print(f"\nFound {len(files)} file(s) to upload:")
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file.name}")
    
    # Confirm before proceeding
    response = input("\nProceed with upload? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    # Initialize browser
    print("\nInitializing browser...")
    driver = setup_driver()
    
    try:
        # Login to CodaBench
        login_success = login_to_codabench(driver, username, password)
        
        if not login_success:
            print("\nLogin failed. Please check your credentials and try again.")
            input("Press ENTER to close browser...")
            return
        
        # Navigate to competition
        print(f"\nNavigating to competition...")
        driver.get(COMPETITION_URL)
        time.sleep(2)
        
        print("\n" + "="*60)
        print("READY TO START UPLOADING")
        print("="*60)
        print("The browser will now navigate to the submission tab")
        print("and upload each file automatically.")
        print("\nIf you see any issues, you can stop the script (Ctrl+C)")
        input("\nPress ENTER to start uploading...")
        
        # Upload each file
        successful = 0
        failed = 0
        
        for i, file in enumerate(files, 1):
            success = upload_file(driver, file, i)
            if success:
                successful += 1
            else:
                failed += 1
                print(f"\n⚠ Failed to upload {file.name}")
                
                # Ask if user wants to continue
                response = input("\nContinue with next file? (y/n): ")
                if response.lower() != 'y':
                    print("Stopping upload process")
                    break
            
            # Small delay between files
            if i < len(files):
                print(f"\nWaiting {WAIT_BETWEEN_UPLOADS} seconds before next upload...")
                time.sleep(WAIT_BETWEEN_UPLOADS)
        
        # Summary
        print("\n" + "="*60)
        print("UPLOAD SUMMARY")
        print("="*60)
        print(f"Total files: {len(files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print("="*60)
        
        # Navigate to submissions to see results
        print("\nNavigating to your submissions...")
        driver.get(SUBMISSION_TAB_URL)
        time.sleep(2)
        
        # Keep browser open
        print("\nCheck your submissions on the page above.")
        input("\nPress ENTER to close browser...")
        
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()
        print("\nBrowser closed.")


if __name__ == "__main__":
    main()